import gc
import time
import torch
import torch.nn as nn

from progress.bar import Bar
from torch.cuda.amp import GradScaler


# =====================================================
# For timing utilities
# =====================================================
def start_timer():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    return start_time


def end_timer_and_print(start_time, local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = %.5f gigabytes" % (torch.cuda.max_memory_allocated()/1e9))
    print(torch.cuda.get_device_name())


# =====================================================
# For Hutchinson's Method
# =====================================================
def get_trace_hut(model, data, criterion, n_v, channelwise=False, layerwise=False, trace_FP16=True):
    """
    Compute the trace of hessian using Hutchinson's method
    This approach requires computing only the application of the Hessian to a random input vector
    This has the same cost as backpropagating the gradient

    Rademacher vector v is a list of tensors that follows size of parameter tensors.
    Hessian vector product Hv is a tuple of tensors that follows size of parameter tensors.
    Final result trace_vHv is a 3D tensor containing 300 x vHv for each channel in each layer.

    Supports faster computation of Hessian trace in FP16
    """

    assert not (channelwise and layerwise)
    model.eval()

    # Initialize gradient scaler
    if trace_FP16:
        scaler = GradScaler()
    else:
        scaler = None

    # Get data to CUDA if possible
    inputs, targets = data
    inputs, targets = inputs.cuda(), targets.cuda()

    # Forward pass in FP32 or FP16
    start_time = start_timer()
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=trace_FP16):
        outputs = model(inputs)  # output is float16 because conv and linear layers autocast to float16
        loss = criterion(outputs, targets)  # loss is float32 because crossentropy layers autocast to float32

    params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)

    if trace_FP16:
        gradsH = torch.autograd.grad(scaler.scale(loss), params, create_graph=True)

        # Unscale 1st order gradients using ordinary division
        inv_scale = 1. / scaler.get_scale()
        gradsH = [gradient_tensor * inv_scale for gradient_tensor in gradsH]

    else:
        # Backward pass in FP32
        gradsH = torch.autograd.grad(loss, params, create_graph=True)

    if channelwise:
        trace_vhv = [[[] for _ in range(p.size(0))] for p in params]
    elif layerwise:
        trace_vhv = [[] for _ in params]
    else:
        trace_vhv = []

    bar = Bar('Computing trace', max=n_v)

    for i in range(n_v):
        bar.suffix = f'({i + 1}/{n_v}) |ETA: {bar.elapsed_td}<{bar.eta_td}'
        bar.next()

        if trace_FP16:
            scaler.update(scaler.get_scale() / 2 ** 8)  # Value was tuned by hand manually to avoid NaN/infs from occuring

            # Sampling a random vector from the Rademacher Distribution
            v = [torch.randint_like(p, high=2, device='cuda').float() * 2 - 1 for p in params]

            # Calculate 2nd order gradients in FP16
            Hv = hessian_vector_product(scaler.scale(gradsH), params, v, stop_criterion=(i == (n_v - 1)))

        else:
            # Sampling a random vector from the Rademacher Distribution
            v = [torch.randint_like(p, high=2, device='cuda').float() * 2 - 1 for p in params]

            # Calculate 2nd order gradients in FP32
            Hv = hessian_vector_product(gradsH, params, v, stop_criterion=(i == (n_v - 1)))

        v = [vi.detach().cpu() for vi in v]
        Hv = [Hvi.detach().cpu() for Hvi in Hv]

        with torch.no_grad():
            if channelwise:
                for layer_i in range(len(Hv)):
                    for channel_i in range(Hv[layer_i].size(0)):
                        trace_vhv[layer_i][channel_i].append(Hv[layer_i][channel_i].flatten().dot
                        (v[layer_i][channel_i].flatten()).item())
            elif layerwise:
                for Hv_i in range(len(Hv)):
                    trace_vhv[Hv_i].append(Hv[Hv_i].flatten().dot(v[Hv_i].flatten()).item())
            else:
                trace_vhv.append(group_product(Hv, v).item())
    bar.finish()

    if trace_FP16:
        end_timer_and_print(start_time, "Mixed Precision Trace Estimator:")
    else:
        end_timer_and_print(start_time, "Default Precision Trace Estimator:")

    return trace_vhv


def hessian_vector_product(gradsH, params, v, stop_criterion=False):
    """
    Compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, retain_graph=not stop_criterion)

    return hv


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def fetch_mat_weights(layer):
    if isinstance(layer, nn.Conv2d):
        weight = layer.weight
        weight = weight.view(weight.size(0), -1)
        if layer.bias is not None:
            weight = torch.cat([weight, layer.bias.unsqueeze(1)], 1)
    elif isinstance(layer, nn.Linear):
        weight = layer.weight
        if layer.bias is not None:
            weight = torch.cat([weight, layer.bias.unsqueeze(1)], 1)
    else:
        raise NotImplementedError

    return weight

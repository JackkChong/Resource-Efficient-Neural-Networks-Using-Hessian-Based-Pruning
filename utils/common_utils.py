import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable


# =====================================================
# For learning rate schedule
# =====================================================
class CosineLRScheduler(object):
    def __init__(self, epochs, start_lr):
        self.epochs = epochs
        self.start_lr = start_lr

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = 0.5 * (1 + np.cos(np.pi * iteration / self.epochs)) * self.start_lr
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


# =====================================================
# For criterion
# =====================================================
class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing. """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


# =====================================================
# For computing compression ratio and FLOPs
# =====================================================
def count_parameters(model):
    """ The number of trainable parameters. """

    return sum(p.numel() for p in model.parameters())


def compute_ratio(model, total):
    pruned_numel = count_parameters(model)
    ratio = 100. * pruned_numel / total
    return ratio, pruned_numel


def compute_model_param_flops(model=None, input_res=32, multiply_adds=True):
    # noinspection PyUnusedLocal
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        flops = (kernel_ops * (
            2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    # noinspection PyUnusedLocal
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    # noinspection PyUnusedLocal
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    # noinspection PyUnusedLocal
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    # noinspection PyUnusedLocal
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    def _rm_hooks(model):
        for m in model.modules():
            m._forward_hooks = OrderedDict()

    list_conv = []
    list_linear = []
    list_bn = []
    list_relu = []
    list_pooling = []

    foo(model)
    input = Variable(torch.rand(1, 3, input_res, input_res), requires_grad=True)
    input = input.cuda()
    model = model.eval()
    _ = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print('Number of FLOPs: %.2fG' % (total_flops / 1e9))
    _rm_hooks(model)

    return total_flops

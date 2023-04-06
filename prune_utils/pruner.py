import os
import math
import time
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import OrderedDict
from torch.cuda.amp import GradScaler
from prune_utils.hutchinson import get_trace_hut, fetch_mat_weights
from prune_utils.dependencies import get_layer_dependencies


# =====================================================
# For Hessian pruner
# =====================================================
class HessianPruner:

    def __init__(self, model, optimizer, lr_scheduler, prune_ratio,
                 prune_ratio_limit, batch_averaged, use_patch, fix_layers,
                 fix_rotation, hessian_mode, use_decompose):

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.prune_ratio = prune_ratio
        self.prune_ratio_limit = prune_ratio_limit
        self.batch_averaged = batch_averaged
        self.use_patch = use_patch
        self.fix_layers = fix_layers
        self.fix_rotation = fix_rotation
        self.hessian_mode = hessian_mode
        self.use_decompose = use_decompose
        self.known_modules = {'Linear', 'Conv2d'}
        self.cfg = ""
        self.modules = []
        self.importances = {}
        self.W_pruned = {}
        self.steps = 0

        if self.use_decompose:
            self.known_modules = {'Conv2d'}

    def make_pruned_model(self, hessdata, criterion, n_v, trace_directory, network, trace_FP16):
        self._prepare_model()
        self.init_step()
        self._compute_hessian_importance(hessdata, criterion, n_v, trace_directory, trace_FP16)
        self._do_prune(self.prune_ratio, network)
        self._build_pruned_model()

        self._rm_hooks()
        self._clear_buffer()

        self.cfg = str(self.model)

        return self.cfg

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
        self.modules = self.modules[self.fix_layers:]

    def init_step(self):
        self.steps = 0

    def _compute_hessian_importance(self, hessdata, criterion, n_v, trace_directory, trace_FP16):
        if self.hessian_mode == 'trace':

            # Set requires_grad for convolution layers and linear layers only
            for m in self.model.parameters():
                shape_list = [2, 4]
                if self.use_decompose:
                    shape_list = [4]
                if len(m.shape) in shape_list:
                    m.requires_grad = True
                else:
                    m.requires_grad = False

            trace_dir = trace_directory
            if os.path.exists(trace_directory):
                print(f"Loading trace...\n")
                results = np.load(trace_dir, allow_pickle=True)
            else:
                results = get_trace_hut(self.model, hessdata, criterion, n_v, channelwise=True, layerwise=False, trace_FP16=trace_FP16)
                np.save(trace_dir, np.array(results, dtype=object))

            for m in self.model.parameters():
                m.requires_grad = True

            channel_trace, weighted_trace = [], []
            for k, layer in enumerate(results):
                channel_trace.append(torch.zeros(len(layer)))
                weighted_trace.append(torch.zeros(len(layer)))

                # Calculate average of vHv for each channel
                for cnt, channel in enumerate(layer):
                    channel_trace[k][cnt] = sum(channel) / len(channel)
            
            for k, m in enumerate(self.modules):
                tmp = []

                # Calculate second-order sensitivity using Hessian trace for each channel
                for cnt, channel in enumerate(m.weight.data):
                    tmp.append((channel_trace[k][cnt] * channel.detach().norm() ** 2 / channel.numel()).cpu().item())

                self.importances[m] = (tmp, len(tmp))
                self.W_pruned[m] = fetch_mat_weights(m)

    def _do_prune(self, prune_ratio, network):
        """
        all_importances is an array containing loss perturbations
        for 8080 filters from Conv layer and 10 output channels from Linear layer in WideResNet-26-8
        for 784 filters from Conv layer and 10 output channels from Linear layer in ResNet-20
        """

        # Get threshold
        all_importances = []
        for m in self.modules:
            imp_m = self.importances[m]
            imps = imp_m[0]
            all_importances += imps
        all_importances = sorted(all_importances)
        idx = int(prune_ratio * len(all_importances))
        threshold = all_importances[idx]
        print('=> The threshold is: %.5f (%d/%d)\n' % (threshold, idx, len(all_importances)))

        # Check for NaN/infs
        if math.isnan(threshold) or math.isinf(threshold):
            raise Exception("Threshold is NaN/infs during Hutchinson trace computation!")

        # Displays possible prune ratios
        print("Possible prune ratios:\n")
        for i in range(3):
            next_idx = idx + (i + 1)
            valid_prune_ratio = (next_idx+1)/len(all_importances)
            print("%.5f for %d" % (valid_prune_ratio, next_idx))
        print("\n")
        for i in range(3):
            next_idx = idx - (i + 1)
            valid_prune_ratio = (next_idx+1)/len(all_importances)
            print("%.5f for %d" % (valid_prune_ratio, next_idx))
        print("\n")

        # Add buffers
        for module in self.model.modules():
            if module.__class__.__name__ == "Conv2d":
                module.register_buffer('out_indices', torch.zeros(1))
                module.register_buffer('in_indices', torch.zeros(1))
            elif module.__class__.__name__ == "Linear":
                module.register_buffer('out_indices', torch.zeros(1))
                module.register_buffer('in_indices', torch.zeros(1))

        # Do pruning
        print('=> Conducting network pruning. Max: %.5f, Min: %.5f, Threshold: %.5f' %
              (max(all_importances), min(all_importances), threshold))

        for idx, m in enumerate(self.modules):
            imp_m = self.importances[m]
            n_r = imp_m[1]
            row_imps = imp_m[0]
            row_indices = filter_indices(row_imps, threshold)
            r_ratio = 1 - len(row_indices) / n_r

            # Compute row indices
            if r_ratio > self.prune_ratio_limit:
                r_threshold = get_threshold(row_imps, self.prune_ratio_limit)
                row_indices = filter_indices(row_imps, r_threshold)
                print('* row indices empty!')

            # For the last linear layer, set row indices to be number of output classes
            if isinstance(m, nn.Linear) and idx == len(self.modules) - 1:
                row_indices = list(range(self.W_pruned[m].size(0)))

            m.out_indices = torch.IntTensor(row_indices)
            m.in_indices = torch.IntTensor([0, 1, 2])
            m.is_pruned = True

        update_indices(self.model, network)

    def _build_pruned_model(self):
        for m_name, m in self.model.named_modules():

            if isinstance(m, nn.BatchNorm2d):
                idxs = m.in_indices.tolist()
                m.num_features = len(idxs)
                m.weight.data = m.weight.data[idxs]
                m.bias.data = m.bias.data[idxs].clone()
                m.running_mean = m.running_mean[idxs].clone()
                m.running_var = m.running_var[idxs].clone()
                m.weight.grad = None
                m.bias.grad = None

            elif isinstance(m, nn.Conv2d):
                out_indices = m.out_indices.tolist()
                in_indices = m.in_indices.tolist()

                m.weight.data = m.weight.data[out_indices, :, :, :][:, in_indices, :, :].clone()

                if m.bias is not None:
                    m.bias.data = m.bias.data[out_indices]
                    m.bias.grad = None

                m.in_channels = len(in_indices)
                m.out_channels = len(out_indices)
                m.weight.grad = None

            elif isinstance(m, nn.Linear):
                out_indices = m.out_indices.tolist()
                in_indices = m.in_indices.tolist()

                m.weight.data = m.weight.data[out_indices, :][:, in_indices].clone()

                if m.bias is not None:
                    m.bias.data = m.bias.data[out_indices].clone()
                    m.bias.grad = None

                m.in_features = len(in_indices)
                m.out_features = len(out_indices)
                m.weight.grad = None

    def _rm_hooks(self):
        for m in self.model.modules():
            classname = m.__class__.__name__
            if classname in self.known_modules:
                m._backward_hooks = OrderedDict()
                m._forward_pre_hooks = OrderedDict()

    def _clear_buffer(self):
        self.modules = []

    def finetune_model(self, epoch, criterion, trainloader, finetune_FP16):
        self.model = self.model.train()
        self.model = self.model.cpu()
        self.model = self.model.cuda()
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0

        self.lr_scheduler(self.optimizer, epoch)

        desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            self.lr_scheduler.get_lr(self.optimizer), 0, 0, correct, total))
        prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)

        scaler = GradScaler(enabled=finetune_FP16)

        for batch_idx, (inputs, targets) in prog_bar:
            # Get data to CUDA if possible
            inputs, targets = inputs.cuda(), targets.cuda()

            # forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=finetune_FP16):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

            # backward pass
            self.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # update weights and biases
            scaler.step(self.optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (self.lr_scheduler.get_lr(self.optimizer), train_loss / (batch_idx + 1),
                     100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

        print(f'Finetune Loss: {train_loss / total}')
        print(f'Finetune Acc: {np.around(correct / total * 100, 2)}')

    def test_model(self, epoch, criterion, testloader, test_FP16, log_directory, best_acc):
        self.model = self.model.eval()
        self.model = self.model.cpu()
        self.model = self.model.cuda()
        test_loss = 0
        correct = 0
        total = 0

        desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            self.lr_scheduler.get_lr(self.optimizer), 0, 0, correct, total))
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in prog_bar:
                # Get data to CUDA if possible
                inputs, targets = inputs.cuda(), targets.cuda()

                # forward pass
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=test_FP16):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    self.lr_scheduler.get_lr(self.optimizer), test_loss / (batch_idx + 1), 100. * correct / total,
                    correct, total))
                prog_bar.set_description(desc, refresh=True)

        print(f'Test Loss: {test_loss / total}')
        print(f'Test Acc: {np.around(correct / total * 100, 2)}')

        # save checkpoint
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'acc': acc,
                'epoch': epoch + 1,
                'loss': loss
            }

            torch.save(state, log_directory)
            best_acc = acc
        
        return best_acc

    def speed_model(self, dataloader):
        """ Test the speed of the model """

        self.model = self.model.eval()
        self.model = self.model.cpu()
        self.model = self.model.cuda()

        # Warm up
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                _ = self.model(inputs)
                if batch_idx == 999:
                    break

        # Measure time
        start = time.time()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                _ = self.model(inputs)
                if batch_idx == 999:
                    break
        end = time.time()

        return end - start


# =====================================================
# For pruning threshold and indices
# =====================================================
def get_threshold(values, percentage):
    v_sorted = sorted(values)
    n = int(len(values) * percentage)
    threshold = v_sorted[n]
    return threshold


def filter_indices(values, threshold):
    """ To obtain indices of filters that pass the threshold """

    indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            indices.append(idx)
    if len(indices) < 1:
        # we make it at least 1 filter in each layer
        indices = [0]
    return indices


def update_indices(model, network):
    print("Updating indices...\n")
    dependencies = get_layer_dependencies(model, network)
    update_in_indices(dependencies)


def update_in_indices(dependencies):
    for m, deps in dependencies.items():
        if len(deps) > 0:
            indices = set()
            for d in deps:
                indices = indices.union(d.out_indices.tolist())
            m.in_indices = torch.IntTensor(sorted(list(indices)))

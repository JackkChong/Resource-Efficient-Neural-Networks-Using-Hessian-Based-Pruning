import os
import gc
import math
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from progress.bar import Bar
from inspect import isfunction
from torchvision import transforms
from collections import OrderedDict
from torch.autograd import Variable
from pytorch_model_summary import summary
from torch.cuda.amp import GradScaler

# Globals
BEST_ACC = 0
AFFINE = True
start_time = 0.0


# =====================================================
# For timing utilities
# =====================================================
def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = %.5f gigabytes" % (torch.cuda.max_memory_allocated()/1e9))
    print(torch.cuda.get_device_name())


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


class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """

    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print('=> Using a preset learning rate schedule:\n')
        self.for_once = True

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = self.decay_schedule.get(iteration, param_group['lr'])
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


# =====================================================
# For data loaders and data transforms
# =====================================================
def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=1, root='data'):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader


def get_hessianloader(dataset, hessian_batch_size, num_workers=1, root='data'):
    transform_hessian = get_transforms_hessian(dataset)
    hessianset = None

    if dataset == 'cifar10':
        hessianset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_hessian)

    elif dataset == 'cifar100':
        hessianset = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform_hessian)

    assert hessianset is not None, 'Error, no dataset %s' % dataset

    hessian_loader = torch.utils.data.DataLoader(hessianset, batch_size=hessian_batch_size, shuffle=False,
                                                 num_workers=num_workers)

    return hessian_loader


def get_transforms(dataset):
    transform_train = None
    transform_test = None

    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset

    return transform_train, transform_test


def get_transforms_hessian(dataset):
    transform_hessian = None

    if dataset == 'cifar10':
        transform_hessian = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    if dataset == 'cifar100':
        transform_hessian = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    assert transform_hessian is not None, 'Error, no dataset %s' % dataset

    return transform_hessian


# =====================================================
# For ResNet model architecture
# =====================================================
class CIFARResNet(nn.Module):
    """
    ResNet model for CIFAR from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    num_classes : int, default 10
        Number of classification classes.
    """

    def __init__(self,
                 channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):

        super(CIFARResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ))

        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()

            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels

            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_pool", nn.AvgPool2d(kernel_size=8, stride=1))
        self.output = nn.Linear(in_features=in_channels, out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)

        return x


class ResUnit(nn.Module):
    """
    ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bias=False,
                 use_bn=True):

        super(ResUnit, self).__init__()

        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            use_bn=use_bn)

        if self.resize_identity:
            # noinspection PyTypeChecker
            self.identity_conv = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=bias,
                use_bn=use_bn,
                activation=None)

        self.activ = nn.ReLU(inplace=True)

        self.speed_tensor = None
        self.create_flag = True
        self.speed_tensor_indices = [[], []]
        self.residual_indices = []
        self.output_indices = []

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x

        x = self.body(x)

        # Enable the residual to be added properly if residual unit was pruned
        is_pruned = hasattr(self.body.conv1.conv, 'is_pruned')
        if is_pruned:
            # Initialize indices list with output indices
            indices = [self.body.conv2.conv.out_indices.tolist()]

            # Append residual indices into indices list
            if hasattr(self, "identity_conv"):
                indices.append(self.identity_conv.conv.out_indices.tolist())
            else:
                indices.append(self.body.conv1.conv.in_indices.tolist())

            # n_c refers to the number of channels for the speed tensor
            n_c = len(set(indices[0] + indices[1]))

            # all_indices refers to the list of all unique indices between the output and residual
            all_indices = list(set(indices[0] + indices[1]))

            # Create speed tensor on the 1st forward call
            if self.create_flag or (not set(self.speed_tensor_indices[1]) == set(all_indices)) or (
                    not set(self.speed_tensor_indices[0]) == set(self.body.conv1.conv.in_indices)) or (
                    self.speed_tensor.size(0) < x.size(0)):

                self.speed_tensor_indices[0] = self.body.conv1.conv.in_indices.tolist()
                self.speed_tensor_indices[1] = all_indices
                self.create_flag = False

                self.residual_indices = []
                self.output_indices = []

                for i in range(n_c):
                    idx = all_indices[i]

                    # Index belongs to both output and residual
                    if idx in indices[0] and idx in indices[1]:
                        self.residual_indices.append(i)
                        self.output_indices.append(i)

                    # Index belongs to output only
                    elif idx in indices[0]:
                        self.output_indices.append(i)

                    # Index belongs to residual only
                    elif idx in indices[1]:
                        self.residual_indices.append(i)

                self.speed_tensor = torch.zeros(x.size(0), n_c, identity.size(2), identity.size(3)).cuda()

            # Perform the addition between output and residual
            tmp_tensor = self.speed_tensor[:x.size(0), :, :, :] + 0.  # +0 is used for preventing copy issue
            tmp_tensor[:, self.residual_indices, :, :] += identity
            tmp_tensor[:, self.output_indices, :, :] += x
            x = tmp_tensor

        else:
            x = x + identity

        x = self.activ(x)

        return x


class ResBlock(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bias=False,
                 use_bn=True):
        super(ResBlock, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            use_bn=use_bn)

        # noinspection PyTypeChecker
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            use_bn=use_bn,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=False,
                 use_bn=True,
                 activation=(lambda: nn.ReLU(inplace=True))):

        super(ConvBlock, self).__init__()

        self.use_bn = use_bn
        self.activate = (activation is not None)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)

        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)

        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)

        if self.use_bn:
            x = self.bn(x)

        if self.activate:
            x = self.activ(x)

        return x


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    Returns:
    -------
    nn.Module
        Activation layer.
    """

    assert (activation is not None)

    if isfunction(activation):
        return activation()

    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise NotImplementedError()

    else:
        assert (isinstance(activation, nn.Module))
        return activation


# =====================================================
# For WideResNet model architecture
# =====================================================
class WideResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, widening_factor=1, dropRate=0.3):
        super(WideResNet, self).__init__()

        _outputs = [16, 16 * widening_factor, 32 * widening_factor, 64 * widening_factor]

        self.in_planes = _outputs[0]

        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, _outputs[1], num_blocks[0], dropRate, stride=1)
        self.layer2 = self._make_layer(block, _outputs[2], num_blocks[1], dropRate, stride=2)
        self.layer3 = self._make_layer(block, _outputs[3], num_blocks[2], dropRate, stride=2)
        self.bn1 = nn.BatchNorm2d(_outputs[3], affine=AFFINE)
        self.fc = nn.Linear(_outputs[3], num_classes)

        self._weights_init()

    def _make_layer(self, block, planes, num_blocks, dropRate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropRate))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _weights_init(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class BasicBlock_WRN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, dropRate=0.0):
        super(BasicBlock_WRN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, affine=AFFINE)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes, affine=AFFINE)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

        self.in_planes = in_planes

        self.speed_tensor = None
        self.create_flag = True
        self.speed_tensor_indices = [[], []]
        self.residual_indices = []
        self.output_indices = []

    def forward(self, x):
        x = self.relu1(self.bn1(x))

        if self.equalInOut:
            residual = x
        else:
            residual = self.convShortcut(x)

        x = self.conv1(x)
        x = self.relu2(self.bn2(x))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)

        x = self.conv2(x)

        # Enable the residual to be added properly if residual unit was pruned
        is_pruned = hasattr(self.conv1, 'is_pruned')
        if is_pruned:
            # Initialize indices list with output indices
            indices = [self.conv2.out_indices.tolist()]

            # Append residual indices into indices list
            if self.convShortcut is not None:
                indices.append(self.convShortcut.out_indices.tolist())
            else:
                indices.append(self.conv1.in_indices.tolist())

            # n_c refers to the number of channels for the speed tensor
            n_c = len(set(indices[0] + indices[1]))

            # all_indices refers to the list of all unique indices between the output and residual
            all_indices = list(set(indices[0] + indices[1]))

            # Create speed tensor on the 1st forward call
            if self.create_flag or (not set(self.speed_tensor_indices[1]) == set(all_indices)) or (
                    not set(self.speed_tensor_indices[0]) == set(self.conv1.in_indices)) or (
                    self.speed_tensor.size(0) < x.size(0)):

                self.speed_tensor_indices[0] = self.conv1.in_indices.tolist()
                self.speed_tensor_indices[1] = all_indices
                self.create_flag = False

                self.residual_indices = []
                self.output_indices = []

                for i in range(n_c):
                    idx = all_indices[i]

                    # Index belongs to both output and residual
                    if idx in indices[0] and idx in indices[1]:
                        self.residual_indices.append(i)
                        self.output_indices.append(i)

                    # Index belongs to output only
                    elif idx in indices[0]:
                        self.output_indices.append(i)

                    # Index belongs to residual only
                    elif idx in indices[1]:
                        self.residual_indices.append(i)

                self.speed_tensor = torch.zeros(x.size(0), n_c, residual.size(2), residual.size(3)).cuda()
            
            # Perform the addition between output and residual
            tmp_tensor = self.speed_tensor[:x.size(0), :, :, :] + 0.  # +0 is used for preventing copy issue
            tmp_tensor[:, self.residual_indices, :, :] += residual
            tmp_tensor[:, self.output_indices, :, :] += x
            x = tmp_tensor

        else:
            x += residual

        return x


# =====================================================
# For network helper functions
# =====================================================
def get_network(network, depth, dataset, widening_factor):
    if network == 'wideresnet':
        return wideresnet(depth=depth, dataset=dataset, widening_factor=widening_factor)
    elif network == 'resnet':
        return resnet(depth=depth, dataset=dataset)
    else:
        raise NotImplementedError


def wideresnet(depth, dataset, widening_factor):
    assert (depth - 4) % 6 == 0, 'Depth must be = 6n + 4, got %d' % depth

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    n = (depth - 4) // 6
    model = WideResNet(BasicBlock_WRN, [n] * 3, num_classes, widening_factor, dropRate=0.3)

    return model


def resnet(depth, dataset):
    assert (depth - 2) % 6 == 0, 'Depth must be = 6n + 2, got %d' % depth

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    layers = [(depth - 2) // 6] * 3
    channels_per_layers = [16, 32, 64]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    init_block_channels = 16

    model = CIFARResNet(channels=channels, init_block_channels=init_block_channels, num_classes=num_classes)

    return model


def count_parameters(model):
    """ The number of trainable parameters.
    """

    return sum(p.numel() for p in model.parameters())


def compute_ratio(model, total):
    pruned_numel = count_parameters(model)
    ratio = 100. * pruned_numel / total
    return ratio, pruned_numel


def compute_model_param_flops(model=None, input_res=224, multiply_adds=True):
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


# =====================================================
# For criterion
# =====================================================
class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing.
    """

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

    def test_model(self, epoch, criterion, testloader, test_FP16, log_directory):
        global BEST_ACC
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
        if acc > BEST_ACC:
            print('Saving..')
            state = {
                'net': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'acc': acc,
                'epoch': epoch + 1,
                'loss': loss
            }

            torch.save(state, log_directory)
            BEST_ACC = acc

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
    start_timer()
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
        end_timer_and_print("Mixed Precision Trace Estimator:")
    else:
        end_timer_and_print("Default Precision Trace Estimator:")

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


# =====================================================
# For layer and block dependencies
# =====================================================
def get_layer_dependencies(model, network):
    """
    There should be a total of X dependencies in the dictionary

    1 empty dependency for the 1st conv layer
    1 dependency for the last fc layer
    """

    dependencies = OrderedDict()

    if network == "wideresnet":
        dependencies[model.conv1] = []
        prev_modules = [model.conv1]

        update_wideresnet_layer_dependencies(prev_modules, model.layer1, dependencies)
        prev_modules = [model.layer1[-1].conv2]
        if model.layer1[-1].convShortcut is not None:
            prev_modules.append(model.layer1[-1].convShortCut)
        else:
            prev_modules = [model.layer1[-1].conv2] + dependencies[model.layer1[-1].conv1]

        update_wideresnet_layer_dependencies(prev_modules, model.layer2, dependencies)
        prev_modules = [model.layer2[-1].conv2]
        if model.layer2[-1].convShortcut is not None:
            prev_modules.append(model.layer2[-1].convShortcut)
        else:
            prev_modules = [model.layer2[-1].conv2] + dependencies[model.layer2[-1].conv1]

        update_wideresnet_layer_dependencies(prev_modules, model.layer3, dependencies)
        prev_modules = [model.layer3[-1].conv2]
        if model.layer3[-1].convShortcut is not None:
            prev_modules.append(model.layer3[-1].convShortCut)
        else:
            prev_modules = [model.layer3[-1].conv2] + dependencies[model.layer3[-1].conv1]

        dependencies[model.bn1] = prev_modules
        dependencies[model.fc] = prev_modules

    elif network == "resnet":
        dependencies[model.features.init_block.conv] = []
        dependencies[model.features.init_block.bn] = [model.features.init_block.conv]

        prev_modules = [model.features.init_block.conv]
        update_resnet_stage_dependencies(prev_modules, model.features.stage1, dependencies)

        prev_modules = [model.features.stage1[-1].body.conv2.conv]
        if hasattr(model.features.stage1[-1], "identity_conv"):
            prev_modules.append(model.features.stage1[-1].identity_conv)
        else:
            prev_modules = [model.features.stage1[-1].body.conv2.conv] + dependencies[
                model.features.stage1[-1].body.conv1.conv]
        update_resnet_stage_dependencies(prev_modules, model.features.stage2, dependencies)

        prev_modules = [model.features.stage2[-1].body.conv2.conv]
        if hasattr(model.features.stage2[-1], "identity_conv"):
            prev_modules.append(model.features.stage2[-1].identity_conv)
        else:
            prev_modules = [model.features.stage2[-1].body.conv2.conv] + dependencies[
                model.features.stage2[-1].body.conv1.conv]
        update_resnet_stage_dependencies(prev_modules, model.features.stage3, dependencies)

        prev_modules = [model.features.stage3[-1].body.conv2.conv]
        if hasattr(model.features.stage3[-1], "identity_conv"):
            prev_modules.append(model.features.stage3[-1].identity_conv)
        else:
            prev_modules = [model.features.stage3[-1].body.conv2.conv] + dependencies[
                model.features.stage3[-1].body.conv1.conv]

        dependencies[model.output] = prev_modules

    return dependencies


def update_wideresnet_layer_dependencies(prev_modules, layer, dependencies):
    num_blocks = len(layer)
    for block_idx in range(num_blocks):
        block = layer[block_idx]
        update_wideresnet_block_dependencies(prev_modules, block, dependencies)
        prev_modules = [block.conv2]
        if block.convShortcut is not None:
            prev_modules.append(block.convShortcut)
        else:
            prev_modules.extend(dependencies[block.conv1])


def update_wideresnet_block_dependencies(prev_modules, block, dependencies):
    for m in prev_modules:
        assert isinstance(m, (nn.Conv2d, nn.Linear)), 'Only conv or linear layer can be previous modules.'

    dependencies[block.bn1] = prev_modules
    dependencies[block.conv1] = prev_modules
    dependencies[block.bn2] = [block.conv1]
    dependencies[block.conv2] = [block.conv1]

    if block.convShortcut is not None:
        dependencies[block.convShortcut] = prev_modules


def update_resnet_stage_dependencies(prev_modules, stage, dependencies):
    num_units = len(stage)
    for unit_idx in range(num_units):
        unit = stage[unit_idx]
        update_resnet_unit_dependencies(prev_modules, unit, dependencies)
        prev_modules = [unit.body.conv2.conv]
        if hasattr(unit, "identity_conv"):
            prev_modules.append(unit.identity_conv.conv)
        else:
            prev_modules.extend(dependencies[unit.body.conv1.conv])


def update_resnet_unit_dependencies(prev_modules, unit, dependencies):
    for m in prev_modules:
        assert isinstance(m, (nn.Conv2d, nn.Linear)), 'Only conv or linear layer can be previous modules.'

    dependencies[unit.body.conv1.conv] = prev_modules
    dependencies[unit.body.conv1.bn] = [unit.body.conv1.conv]
    dependencies[unit.body.conv2.conv] = [unit.body.conv1.conv]
    dependencies[unit.body.conv2.bn] = [unit.body.conv2.conv]

    if hasattr(unit, "identity_conv"):
        dependencies[unit.identity_conv.conv] = prev_modules
        dependencies[unit.identity_conv.bn] = [unit.identity_conv.conv]


# =====================================================
# For main functions
# =====================================================
def main(args):
    global BEST_ACC

    cudnn.benchmark = True

    # Initialize model architecture
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset,
                      widening_factor=args.widening_factor)
    print(summary(net, torch.zeros(1, 3, 32, 32), show_input=True, show_hierarchical=False))

    # Loading pre-trained model from checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.load_checkpoint, map_location=args.device)
    print("Acc: %.2f%%, Epoch: %d, Loss: %.4f\n" % (checkpoint['acc'], checkpoint['epoch'], checkpoint['loss']))
    state_dict = checkpoint['net']
    net.load_state_dict(state_dict)
    total_parameters = count_parameters(net)
    net = net.cuda()

    # Initialize data loaders
    trainloader, testloader = get_dataloader(dataset=args.dataset, train_batch_size=args.batch_size,
                                             test_batch_size=args.batch_size,
                                             num_workers=args.num_workers)

    # Add buffers
    for module in net.modules():
        if module.__class__.__name__ == "Conv2d":
            module.register_buffer('out_indices', torch.zeros(1))
            module.register_buffer('in_indices', torch.zeros(1))
        elif module.__class__.__name__ == "Linear":
            module.register_buffer('out_indices', torch.zeros(1))
            module.register_buffer('in_indices', torch.zeros(1))

    # Initialize hessian loader
    hess_data = []
    hessianloader = get_hessianloader(args.dataset, args.hessian_batch_size)
    for data, label in hessianloader:
        hess_data = (data, label)

    # Initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)

    # Initialize learning rate scheduler
    lr_scheduler = CosineLRScheduler(epochs=args.epochs, start_lr=args.learning_rate)

    # Initialize criterion
    criterion = nn.CrossEntropyLoss()

    # Initialize pruner
    pruner = HessianPruner(net, optimizer, lr_scheduler, args.prune_ratio, args.prune_ratio_limit, args.batch_averaged,
                           args.use_patch, args.fix_layers, args.fix_rotation, args.hessian_mode, args.use_decompose)

    total_flops = 0
    # Calculate FLOPS before pruning
    if args.dataset == 'cifar10' or 'cifar100':
        total_flops = compute_model_param_flops(pruner.model, 32)
    elif args.dataset == 'imagenet':
        total_flops = compute_model_param_flops(pruner.model, 224)

    # Conduct pruning
    _ = pruner.make_pruned_model(hess_data, criterion, args.n_v, args.trace_directory, args.network, args.trace_FP16)

    # Track the compression ratio
    compression_ratio, pruned_numel = compute_ratio(pruner.model, total_parameters)
    print("Compression ratio: %.2f%%(%d/%d)\n" % (compression_ratio, pruned_numel, total_parameters))

    pruned_flops = 0
    # Calculate FLOPS after pruning
    if args.dataset == 'cifar10' or 'cifar100':
        pruned_flops = compute_model_param_flops(pruner.model, 32)
    elif args.dataset == 'imagenet':
        pruned_flops = compute_model_param_flops(pruner.model, 224)

    print("Remained flops: %.2f%%" % (pruned_flops / total_flops * 100))

    # Start finetuning
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        pruner.finetune_model(epoch, criterion, trainloader, args.finetune_FP16)
        pruner.test_model(epoch, criterion, testloader, args.test_FP16, args.log_directory)

    # Print model size and best accuracy
    print("%.2f MB" % (os.path.getsize(args.log_directory) / 1e6))
    print("Best accuracy is %.2f%%" % (BEST_ACC))


if __name__ == "__main__":
    # Fetch args
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="cifar10", type=str)
    parser.add_argument('--network', default="resnet", type=str)
    parser.add_argument('--depth', default=32, type=int)
    parser.add_argument('--widening_factor', default=1, type=int)

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--learning_rate', default=0.0512, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--nesterov', default=True, type=bool)

    parser.add_argument('--finetune_FP16', default=False, type=bool)
    parser.add_argument('--test_FP16', default=False, type=bool)

    parser.add_argument('--n_v', default=300, type=int)
    parser.add_argument('--hessian_batch_size', default=512, type=int)
    parser.add_argument('--prune_ratio', default=0.77214, type=float)
    parser.add_argument('--prune_ratio_limit', default=0.95, type=float)
    parser.add_argument('--batch_averaged', default=True, type=bool)
    parser.add_argument('--use_patch', default=False, type=bool)
    parser.add_argument('--fix_layers', default=0, type=int)
    parser.add_argument('--fix_rotation', default=False, type=bool)
    parser.add_argument('--hessian_mode', default="trace", type=str)
    parser.add_argument('--use_decompose', default=False, type=bool)
    parser.add_argument('--trace_FP16', default=False, type=bool)

    parser.add_argument('--load_checkpoint', default="cifar10_result//resnet_32_best.pth.tar", type=str)
    parser.add_argument('--log_directory', default="HAP_cifar10_result/resnet_32_pruned.pth.tar", type=str)
    parser.add_argument('--trace_directory', default="HAP_cifar10_result/resnet_32_pruned.npy", type=str)

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)

    args = parser.parse_args()

    main(args)

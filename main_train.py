import os
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
from inspect import isfunction
from torchvision import transforms
from collections import OrderedDict
from torch.autograd import Variable
from torch.cuda.amp import GradScaler
from pytorch_model_summary import summary

# Globals
BEST_ACC = 0
AFFINE = True


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
def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=1, root='data', returnset=False):
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

    if returnset:
        return trainset, testset
    else:
        return trainloader, testloader


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
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
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

    def forward(self, x):

        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x

        x = self.body(x)
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

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, dropRate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropRate))
            self.in_planes = planes

        return nn.Sequential(*layers)

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
        x += residual

        return x


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()


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
# For training and testing
# =====================================================
def train(epoch, net, lr_scheduler, optimizer, criterion, trainloader, train_FP16):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)

    desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)

    scaler = GradScaler(enabled=train_FP16)

    for batch_idx, (inputs, targets) in prog_bar:
        # Get data to CUDA if possible
        inputs, targets = inputs.cuda(), targets.cuda()

        # forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_FP16):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # update weights and biases
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    print(f'Train Loss: {train_loss / total}')
    print(f'Train Acc: {np.around(correct / total * 100, 2)}')


def test(epoch, net, lr_scheduler, optimizer, criterion, testloader, test_FP16, log_directory):
    global BEST_ACC
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in prog_bar:
            # Get data to CUDA if possible
            inputs, targets = inputs.cuda(), targets.cuda()

            # forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=test_FP16):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                lr_scheduler.get_lr(optimizer), test_loss / (batch_idx + 1), 100. * correct / total,
                correct, total))
            prog_bar.set_description(desc, refresh=True)

        print(f'Test Loss: {test_loss / total}')
        print(f'Test Acc: {np.around(correct / total * 100, 2)}')

    # save checkpoint
    acc = 100. * correct / total
    if acc > BEST_ACC:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch + 1,
            'loss': loss,
        }

        torch.save(state, log_directory)
        BEST_ACC = acc


def main(args):
    global BEST_ACC

    cudnn.benchmark = True

    # Initialize model architecture
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset, widening_factor=args.widening_factor)
    print(summary(net, torch.zeros(1, 3, 32, 32), show_input=True, show_hierarchical=False))
    net = net.cuda()

    # Initialize data loaders
    trainloader, testloader = get_dataloader(dataset=args.dataset, train_batch_size=args.batch_size, test_batch_size=args.batch_size,
                                             num_workers=args.num_workers)

    # Initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=args.nesterov)

    # Initialize learning rate scheduler
    lr_scheduler = CosineLRScheduler(epochs=args.epochs, start_lr=args.learning_rate)

    # Initialize criterion
    criterion = LabelSmoothing(args.smoothing)

    # Calculate FLOPS
    if args.dataset == 'cifar10' or 'cifar100':
        compute_model_param_flops(net, 32)
    elif args.dataset == 'imagenet':
        compute_model_param_flops(net, 224)
    else:
        raise NotImplementedError

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print('==> Resuming from checkpoint...\n')
        checkpoint = torch.load(f'{args.resume}', map_location=args.device)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        BEST_ACC = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('==> Loaded %s trained on %s\n' % (args.network, args.dataset))
        print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%\n' % (start_epoch, BEST_ACC))

    # Iterate over epochs
    for epoch in range(start_epoch, args.epochs):
        train(epoch, net, lr_scheduler, optimizer, criterion, trainloader, args.train_FP16)
        test(epoch, net, lr_scheduler, optimizer, criterion, testloader, args.test_FP16, args.log_directory)

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
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--nesterov', default=True, type=bool)
    parser.add_argument('--smoothing', default=0.0, type=float)

    parser.add_argument('--train_FP16', default=True, type=bool)
    parser.add_argument('--test_FP16', default=True, type=bool)

    parser.add_argument('--log_directory', default="cifar10_result/resnet_32_best.pth.tar", type=str)
    parser.add_argument('--resume', '-r', default=None, type=str)

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    
    args = parser.parse_args()

    main(args)

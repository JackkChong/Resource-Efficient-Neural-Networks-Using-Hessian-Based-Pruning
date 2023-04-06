import torch
import torch.nn as nn
import torch.nn.init as init
from inspect import isfunction


# ============================================================
# For ResNet / WideResNet model architecture
# ============================================================
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
        is_pruned = hasattr(self.body.conv1.conv, 'in_indices')
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


# ============================================================
# For ResNet / WideResNet model architecture after HAP
# ============================================================
class CIFARResNet_HAP(nn.Module):
    """
    ResNet model for CIFAR from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    state_dict : dictionary of saved parameters
        Contains information on the shape of parameter tensors.
    stages : int
        Number of stages
    units : int
        Number of units
    num_classes : int, default 10
        Number of classification classes.
    """

    def __init__(self, state_dict, stages, units, in_channels=3, in_size=(32, 32), num_classes=10):
        super(CIFARResNet_HAP, self).__init__()

        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ConvBlock_HAP(
            in_channels=in_channels,
            out_channels=state_dict["features.init_block.conv.weight"].shape[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ))

        for i in range(stages):
            stage = nn.Sequential()

            for j in range(units):
                if (j == 0) and (i != 0):
                    stride = 2
                    identity_conv_out_channels = \
                        state_dict["features.stage{}.unit{}.identity_conv.conv.weight".format(i + 1, j + 1)].shape[0]
                else:
                    stride = 1
                    if "features.stage{}.unit{}.identity_conv.conv.weight".format(i + 1, j + 1) in state_dict:
                        identity_conv_out_channels = \
                        state_dict["features.stage{}.unit{}.identity_conv.conv.weight".format(i + 1, j + 1)].shape[0]
                    else:
                        identity_conv_out_channels = None

                stage.add_module("unit{}".format(j + 1), ResUnit_HAP(
                    conv1_in_channels=
                    state_dict["features.stage{}.unit{}.body.conv1.conv.weight".format(i + 1, j + 1)].shape[1],
                    conv1_out_channels=
                    state_dict["features.stage{}.unit{}.body.conv1.conv.weight".format(i + 1, j + 1)].shape[0],
                    conv2_out_channels=
                    state_dict["features.stage{}.unit{}.body.conv2.conv.weight".format(i + 1, j + 1)].shape[0],
                    identity_conv_out_channels=identity_conv_out_channels,
                    stride=stride))

            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_pool", nn.AvgPool2d(kernel_size=8, stride=1))
        self.output = nn.Linear(in_features=state_dict["output.weight"].shape[1], out_features=num_classes)

        self.output.register_buffer('out_indices', torch.zeros(state_dict["output.weight"].shape[0]))
        self.output.register_buffer('in_indices', torch.zeros(state_dict["output.weight"].shape[1]))

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


class ResUnit_HAP(nn.Module):
    """
    ResNet unit with residual connection and support for HAP.

    Parameters:
    ----------
    conv1_in_channels : int
        Number of input channels for the 1st convolution layer.
    conv1_out_channels : int
        Number of output channels for the 1st convolution layer.
    conv2_out_channels : int
        Number of output channels for the 2nd convolution layer.
    identity_conv_out_channels : int
        Number of output channels for the shortcut convolution layer.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    """

    def __init__(self,
                 conv1_in_channels,
                 conv1_out_channels,
                 conv2_out_channels,
                 identity_conv_out_channels,
                 stride,
                 bias=False,
                 use_bn=True):

        super(ResUnit_HAP, self).__init__()

        if identity_conv_out_channels is not None:
            self.resize_identity = True
        else:
            self.resize_identity = False

        self.body = ResBlock_HAP(
            conv1_in_channels=conv1_in_channels,
            conv1_out_channels=conv1_out_channels,
            conv2_out_channels=conv2_out_channels,
            stride=stride,
            bias=bias,
            use_bn=use_bn)

        if self.resize_identity:
            # noinspection PyTypeChecker
            self.identity_conv = ConvBlock_HAP(
                in_channels=conv1_in_channels,
                out_channels=identity_conv_out_channels,
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
        is_pruned = hasattr(self.body.conv1.conv, 'in_indices')
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


class ResBlock_HAP(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit with support for HAP.

    Parameters:
    ----------
    conv1_in_channels : int
        Number of input channels for the 1st convolution layer.
    conv1_out_channels : int
        Number of output channels for the 1st convolution layer.
    conv2_out_channels : int
        Number of output channels for the 2nd convolution layer.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    """

    def __init__(self,
                 conv1_in_channels,
                 conv1_out_channels,
                 conv2_out_channels,
                 stride,
                 bias=False,
                 use_bn=True):
        super(ResBlock_HAP, self).__init__()

        self.conv1 = ConvBlock_HAP(
            in_channels=conv1_in_channels,
            out_channels=conv1_out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            use_bn=use_bn)

        # noinspection PyTypeChecker
        self.conv2 = ConvBlock_HAP(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
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


class ConvBlock_HAP(nn.Module):
    """
    Standard convolution block with Batch normalization, activation and extra buffers

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

        super(ConvBlock_HAP, self).__init__()

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

        self.conv.register_buffer('out_indices', torch.zeros(out_channels))
        self.conv.register_buffer('in_indices', torch.zeros(in_channels))

    def forward(self, x):
        x = self.conv(x)

        if self.use_bn:
            x = self.bn(x)

        if self.activate:
            x = self.activ(x)

        return x
    
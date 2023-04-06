import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_utils.quant_functions import *


# =====================================================
# For quantized modules
# =====================================================
class QuantAct(nn.Module):
    """
    Class to quantize given activations

    Parameters:
    ----------
    activation_bit : int, default 8
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.99
        Momentum for updating the activation quantization range.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    act_percentile : float, default 0
        The percentile to set up quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    fixed_point_quantization : bool, default False
        Whether to skip deployment-oriented operations and use fixed-point rather than integer-only quantization.
    """

    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.99,
                 full_precision_flag=False,
                 running_stat=True,
                 quant_mode="symmetric",
                 fix_flag=False,
                 act_percentile=0,
                 fixed_point_quantization=False):

        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.fix_flag = fix_flag
        self.act_percentile = act_percentile
        self.fixed_point_quantization = fixed_point_quantization

        self.act_function = None

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        self.register_buffer('pre_weight_scaling_factor', torch.ones(1))
        self.register_buffer('identity_weight_scaling_factor', torch.ones(1))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "full_precision_flag={2}, quant_mode={3}, Act_min: {4:.2f}, " \
               "Act_max: {5:.2f})".format(self.__class__.__name__, self.activation_bit,
                                          self.full_precision_flag, self.quant_mode, self.x_min.item(),
                                          self.x_max.item())

    def fix(self):
        """ Fix the activation range by setting running stat to False """

        self.running_stat = False
        self.fix_flag = True

    def unfix(self):
        """ Unfix the activation range by setting running stat to True """

        self.running_stat = True
        self.fix_flag = False

    def forward(self, target, pre_act_scaling_factor=None, pre_weight_scaling_factor=None, identity=None,
                identity_scaling_factor=None, identity_weight_scaling_factor=None, x=None, residual_indices=None, output_indices=None):
        """
        target: the activation that we need to quantize

        pre_act_scaling_factor: the scaling factor of the previous activation quantization layer
        pre_weight_scaling_factor: the scaling factor of the previous weight quantization layer
        identity: if True, we need to consider the identity branch
        identity_scaling_factor: the scaling factor of the previous activation quantization of identity
        identity_weight_scaling_factor: the scaling factor of the weight quantization layer in the identity branch
        x: output from the main convolution pathway

        Note that there are two cases for identity branch:
        (1) identity branch directly connect to the input featuremap
        (2) identity branch contains convolutional layers that operate on the input featuremap
        """

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.act_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        # Calculate the quantization range of the activations
        if self.running_stat:
            if self.act_percentile == 0:
                x_min = target.data.min()
                x_max = target.data.max()
            elif self.quant_mode == 'symmetric':
                x_min, x_max = get_percentile_min_max(target.detach().view(-1), 100 - self.act_percentile,
                                                      self.act_percentile, output_tensor=True)
            elif self.quant_mode == 'asymmetric':
                # Note that our asymmetric quantization is implemented using scaled unsigned integers without zero_points,
                # that is to say our asymmetric quantization should always be after ReLU, which makes
                # the minimum value to be always 0. As a result, if we use percentile mode for asymmetric quantization,
                # the lower_percentile will be set to 0 in order to make sure the final x_min is 0.
                x_min, x_max = get_percentile_min_max(target.detach().view(-1), 0, self.act_percentile, output_tensor=True)
            else:
                raise ValueError("Quantization range not set!")

            if self.x_min == self.x_max:
                # Initialization
                self.x_min += x_min
                self.x_max += x_max
            elif self.act_range_momentum == -1:
                self.x_min = min(self.x_min, x_min)
                self.x_max = max(self.x_max, x_max)
            else:
                # Use momentum to update the quantization range
                self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

        # Perform the quantization
        if not self.full_precision_flag:
            if self.quant_mode == 'symmetric':
                self.act_scaling_factor = symmetric_linear_quantization_params(self.activation_bit,
                                                                               self.x_min, self.x_max, False)        
            else:
                # Note that our asymmetric quantization is implemented using scaled unsigned integers
                # without zero_point shift. As a result, asymmetric quantization should be after ReLU,
                # and the self.act_zero_point should be 0.
                self.act_scaling_factor, self.act_zero_point = asymmetric_linear_quantization_params(
                    self.activation_bit, self.x_min, self.x_max, True)
            
            if (pre_act_scaling_factor is None) or (self.fixed_point_quantization is True):
                # For input quantization or fixed point quantization
                quant_act_int = self.act_function(target, self.activation_bit, self.act_scaling_factor)
            else:
                if identity is None:
                    if pre_weight_scaling_factor is None:
                        pre_weight_scaling_factor = self.pre_weight_scaling_factor

                    quant_act_int = fixedpoint_fn.apply(target, self.activation_bit, self.quant_mode,
                                                        self.act_scaling_factor, 0, pre_act_scaling_factor,
                                                        pre_weight_scaling_factor)

                else:
                    if identity_weight_scaling_factor is None:
                        identity_weight_scaling_factor = self.identity_weight_scaling_factor

                    quant_act_int = fixedpoint_fn.apply(target, self.activation_bit, self.quant_mode,
                                                        self.act_scaling_factor, 1, pre_act_scaling_factor,
                                                        pre_weight_scaling_factor,
                                                        identity, identity_scaling_factor,
                                                        identity_weight_scaling_factor, x, residual_indices, output_indices)

            correct_output_scale = self.act_scaling_factor.view(-1)

            return quant_act_int * correct_output_scale, self.act_scaling_factor

        else:
            return target


class QuantBnConv2d(nn.Module):
    """
    Class to quantize given convolutional layer weights, with support for both folded BN and separate BN.

    Parameters:
    ----------
    weight_bit : int, default 8
        Bitwidth for quantized weights.
    bias_bit : int, default 32
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default True
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to set up quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    fix_BN : bool, default True
        Whether to fix BN statistics during training.
    fix_BN_threshold: int, default None
        When to start training with folded BN.
    """

    def __init__(self,
                 weight_bit=8,
                 bias_bit=32,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=True,
                 fix_flag=False,
                 weight_percentile=0,
                 fix_BN=True,
                 fix_BN_threshold=None):
        super(QuantBnConv2d, self).__init__()

        self.weight_bit = weight_bit
        self.full_precision_flag = full_precision_flag
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.quant_mode = quant_mode
        self.fix_BN = fix_BN
        self.training_BN_mode = fix_BN
        self.fix_BN_threshold = fix_BN_threshold
        self.counter = 1

        self.out_channels = None
        self.conv = None
        self.bn = None
        self.weight_function = None

    def set_param(self, conv, bn):
        self.out_channels = conv.out_channels
        self.register_buffer('convbn_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(conv.weight.data))
        self.register_buffer('bias_integer', torch.zeros_like(bn.bias))

        self.conv = conv
        self.bn = bn
        self.bn.momentum = 0.99

    def __repr__(self):
        conv_s = super(QuantBnConv2d, self).__repr__()
        s = "({0}, weight_bit={1}, bias_bit={2}, groups={3}, wt-channel-wise={4}, wt-percentile={5}, quant_mode={6})".format(
            conv_s, self.weight_bit, self.bias_bit, self.conv.groups, self.per_channel, self.weight_percentile,
            self.quant_mode)
        return s

    def fix(self):
        """ Fix the BN statistics by setting fix_BN to True """

        self.fix_flag = True
        self.fix_BN = True

    def unfix(self):
        """ Change the mode (fixed or not) of BN statistics to its original status """

        self.fix_flag = False
        self.fix_BN = self.training_BN_mode

    def forward(self, x, pre_act_scaling_factor=None):
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        else:
            raise ValueError("Unknown quant mode: {}".format(self.quant_mode))

        # Determine whether to fold BN or not
        if self.fix_flag is False:
            self.counter += 1

            if (self.fix_BN_threshold is None) or (self.counter < self.fix_BN_threshold):
                self.fix_BN = self.training_BN_mode
            else:
                if self.counter == self.fix_BN_threshold:
                    print("Start Training with Folded BN")
                self.fix_BN = True

        # Run the forward without folding BN
        if self.fix_BN is False:
            raise NotImplementedError
        
            # w_transform = self.conv.weight.data.contiguous().view(self.conv.out_channels, -1)
            # w_min = w_transform.min(dim=1).values
            # w_max = w_transform.max(dim=1).values

            # conv_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_channel)
            # weight_integer = self.weight_function(self.conv.weight, self.weight_bit, conv_scaling_factor)
            # conv_output = F.conv2d(x, weight_integer, self.conv.bias, self.conv.stride, self.conv.padding,
            #                        self.conv.dilation, self.conv.groups) * conv_scaling_factor.view(1, -1, 1, 1)

            # batch_mean = torch.mean(conv_output, dim=(0, 2, 3))
            # batch_var = torch.var(conv_output, dim=(0, 2, 3))

            # # Update mean and variance in running stats
            # self.bn.running_mean = self.bn.running_mean.detach() * self.bn.momentum + (
            #         1 - self.bn.momentum) * batch_mean
            # self.bn.running_var = self.bn.running_var.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_var

            # output_factor = self.bn.weight.view(1, -1, 1, 1) / torch.sqrt(batch_var + self.bn.eps).view(1, -1, 1, 1)
            # output = output_factor * (conv_output - batch_mean.view(1, -1, 1, 1)) + self.bn.bias.view(1, -1, 1, 1)

            # return output, conv_scaling_factor.view(-1) * output_factor.view(-1)

        # Fold BN and fix running statistics
        else:
            running_std = torch.sqrt(self.bn.running_var.detach() + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            scaled_weight = self.conv.weight * scale_factor.reshape([self.conv.out_channels, 1, 1, 1])

            if self.conv.bias is not None:
                scaled_bias = self.conv.bias
            else:
                scaled_bias = torch.zeros_like(self.bn.running_mean)

            scaled_bias = (scaled_bias - self.bn.running_mean.detach()) * scale_factor + self.bn.bias

            bias_scaling_factor = 0
            if not self.full_precision_flag:
                if self.per_channel:
                    w_transform = scaled_weight.data.contiguous().view(self.conv.out_channels, -1)

                    if self.weight_percentile == 0:
                        w_min = w_transform.min(dim=1).values
                        w_max = w_transform.max(dim=1).values
                    else:
                        lower_percentile = 100 - self.weight_percentile
                        upper_percentile = self.weight_percentile
                        input_length = w_transform.shape[1]

                        lower_index = math.ceil(input_length * lower_percentile * 0.01)
                        upper_index = math.ceil(input_length * upper_percentile * 0.01)

                        w_min = torch.kthvalue(w_transform, k=lower_index, dim=1).values
                        w_max = torch.kthvalue(w_transform, k=upper_index, dim=1).values
                else:
                    if self.weight_percentile == 0:
                        w_min = scaled_weight.data.min()
                        w_max = scaled_weight.data.max()
                    else:
                        w_min, w_max = get_percentile_min_max(scaled_weight.view(-1), 100 - self.weight_percentile,
                                                              self.weight_percentile, output_tensor=True)

                if self.quant_mode == 'symmetric':
                    self.convbn_scaling_factor = symmetric_linear_quantization_params(self.weight_bit,
                                                                                      w_min, w_max, self.per_channel)
                    self.weight_integer = self.weight_function(scaled_weight, self.weight_bit,
                                                               self.convbn_scaling_factor)

                    if self.quantize_bias:
                        bias_scaling_factor = self.convbn_scaling_factor.view(1, -1) * pre_act_scaling_factor.view(1,
                                                                                                                   -1)
                        self.bias_integer = self.weight_function(scaled_bias, self.bias_bit, bias_scaling_factor)
                else:
                    raise Exception('For weight, we only support symmetric quantization.')

            # Quantize the activations using previous activation scaling factor
            pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
            x_int = x / pre_act_scaling_factor

            correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

            # Step 1: Conduct convolution using quantized activations, weights and bias
            # Step 2: Rescale them back to original floating point format
            return (F.conv2d(x_int, self.weight_integer, self.bias_integer, self.conv.stride, self.conv.padding,
                             self.conv.dilation, self.conv.groups) * correct_output_scale, self.convbn_scaling_factor)


class QuantAveragePool2d(nn.Module):
    """
    Quantized Average Pooling Layer

    Parameters:
    ----------
    kernel_size : int, default 7
        Kernel size for average pooling.
    stride : int, default 1
        stride for average pooling.
    padding : int, default 0
        padding for average pooling.
    """

    def __init__(self, kernel_size=7, stride=1, padding=0):
        super(QuantAveragePool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.final_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def set_param(self, pool):
        self.final_pool = pool

    def forward(self, x, x_scaling_factor=None):
        if x_scaling_factor is None:
            return self.final_pool(x)

        x_scaling_factor = x_scaling_factor.view(-1)
        correct_scaling_factor = x_scaling_factor

        x_int = x / correct_scaling_factor
        x_int = ste_round.apply(x_int)
        x_int = self.final_pool(x_int)

        x_int = transfer_float_averaging_to_int_averaging.apply(x_int)

        return x_int * correct_scaling_factor, correct_scaling_factor


class QuantLinear(nn.Module):
    """
    Class to quantize weights of given linear layer

    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to set up quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode='symmetric',
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0,
                 ):
        super(QuantLinear, self).__init__()

        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.counter = 0

        self.in_features = None
        self.out_features = None
        self.weight = None
        self.bias = None
        self.weight_function = None

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={}, quantize_fn={})".format(
            self.weight_bit, self.full_precision_flag, self.quant_mode)

        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('bias_integer', torch.zeros_like(linear.bias))

        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False

    def forward(self, x, prev_act_scaling_factor=None):
        """ Using quantized weights to forward activation x """
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        w = self.weight
        w_transform = w.data.detach()

        # Calculate the quantization range of weights and bias
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
            if self.quantize_bias:
                b_min = self.bias.data
                b_max = self.bias.data
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)
            if self.quantize_bias:
                b_min = self.bias.data.min()
                b_max = self.bias.data.max()

        # Perform the quantization
        bias_scaling_factor = None
        if not self.full_precision_flag:
            if self.quant_mode == 'symmetric':
                self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                              self.per_channel)
                self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.fc_scaling_factor)

                bias_scaling_factor = self.fc_scaling_factor.view(1, -1) * prev_act_scaling_factor.view(1, -1)
                self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor)
            else:
                raise Exception('For weight, we only support symmetric quantization.')
        else:
            w = self.weight
            b = self.bias

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor
        correct_output_scale = bias_scaling_factor[0].view(1, -1)

        return ste_round.apply(
            F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)) * correct_output_scale


# =====================================================
# For quantized architecture
# =====================================================
class Q_ResNet(nn.Module):
    """ Quantized ResNet model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385. """

    def __init__(self, model, channels):
        super(Q_ResNet, self).__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct()
        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(init_block.conv, init_block.bn)
        self.quant_act_int32 = QuantAct()
        self.act = nn.ReLU()

        self.channel = channels
        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=8, stride=1)
        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)
        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)
        x = self.act(x)

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResBlockBn(nn.Module):
    """ Quantized ResNet block with residual path. """

    def __init__(self):
        super(Q_ResBlockBn, self).__init__()
        self.resize_identity = None
        self.quant_act = None
        self.quant_convbn1 = None
        self.quant_act1 = None
        self.quant_convbn2 = None
        self.quant_act_int32 = None
        self.quant_identity_convbn = None

        self.speed_tensor = None
        self.create_flag = True
        self.speed_tensor_indices = [[], []]
        self.residual_indices = None
        self.output_indices = None

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct()
        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)

        self.quant_act1 = QuantAct()
        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):

        identity_act_scaling_factor = None
        identity_weight_scaling_factor = None

        # Pass identity through 1x1 convolution
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)

        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        # Enable the residual to be added properly if residual unit was pruned
        is_pruned = hasattr(self.quant_convbn1.conv, 'in_indices')
        if is_pruned:
            # Initialize indices list with output indices
            indices = [self.quant_convbn2.conv.out_indices.tolist()]

            # Append residual indices into indices list
            if self.resize_identity:
                indices.append(self.quant_identity_convbn.conv.out_indices.tolist())
            else:
                indices.append(self.quant_convbn1.conv.in_indices.tolist())

            # n_c refers to the number of channels for the speed tensor
            n_c = len(set(indices[0] + indices[1]))

            # all_indices refers to the list of all unique indices between the output and residual
            all_indices = list(set(indices[0] + indices[1]))

            # Create speed tensor on the 1st forward call
            if self.create_flag or (not set(self.speed_tensor_indices[1]) == set(all_indices)) or (
                    not set(self.speed_tensor_indices[0]) == set(self.quant_convbn1.conv.in_indices.tolist())) or (
                    self.speed_tensor.size(0) < x.size(0)):

                self.speed_tensor_indices[0] = self.quant_convbn1.conv.in_indices.tolist()
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
            result = tmp_tensor

        else:
            result = x + identity

        if self.resize_identity:
            result, act_scaling_factor = self.quant_act_int32(result, act_scaling_factor, weight_scaling_factor,
                                                              identity,
                                                              identity_act_scaling_factor,
                                                              identity_weight_scaling_factor, x, self.residual_indices, self.output_indices)
        else:
            result, act_scaling_factor = self.quant_act_int32(result, act_scaling_factor, weight_scaling_factor,
                                                              identity,
                                                              scaling_factor_int32, None, x, self.residual_indices, self.output_indices)

        result = nn.ReLU()(result)

        return result, act_scaling_factor


def quantize_resnet(model, depth):
    num_units = (depth - 2) // 6
    units = [num_units] * 3
    
    net = Q_ResNet(model, units)
    
    return net


def quantize_wideresnet(model, depth):
    num_units = (depth - 4) // 6
    units = [num_units] * 3
    
    net = Q_ResNet(model, units)
    
    return net


# =====================================================
# For freezing activation range
# =====================================================
def freeze_model(model):
    """ Freeze the activation range """

    if type(model) == QuantAct:
        model.fix()
    elif type(model) == QuantLinear:
        model.fix()
    elif type(model) == QuantBnConv2d:
        model.fix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)


def unfreeze_model(model):
    """ Unfreeze the activation range """

    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == QuantLinear:
        model.unfix()
    elif type(model) == QuantBnConv2d:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)

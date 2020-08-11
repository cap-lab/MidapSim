from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from acc_utils.attrdict import AttrDict
from generic_op import *

# Note that this builder does not support weight initialization


def get_padding(kernel, dilation, option):
    if isinstance(option, int):
        return (option, option, option, option)
    if isinstance(option, tuple) or isinstance(option, list):
        if len(option) == 4:
            return option
        elif len(option) == 2:
            return (option[0], option[0], option[1], option[1])
        else:
            raise ValueError("Invalid padding: {}".format(option))
    if isinstance(option, str):
        if option.lower() == 'valid':
            return (0, 0, 0, 0)
        if option.lower() == 'same':
            if isinstance(kernel, int):
                k = (kernel, kernel)
            else:
                k = kernel
            pad_t = (k[0] - 1) // 2
            pad_b = k[0] - 1 - pad_t
            pad_l = (k[1] - 1) // 2
            pad_r = k[1] - 1 - pad_l
            return (pad_t * dilation, pad_b * dilation, pad_l * dilation, pad_r * dilation)
    raise ValueError("Invalid padding: {}".format(option))


class ModelBuilder(object):
    def __init__(self, name="custom_model"):
        self.model_dict = OrderedDict()
        self.name_gen = {}
        self.name = name

    def __del__(self):
        del self.model_dict, self.name_gen, self.name

    def set_input_tensor(self, name='input', tensor_shape=(1, 3, 224, 224), input_tensor=None, order='NCHW'):
        if input_tensor is None:
            x = torch.randn(*tensor_shape, requires_grad=False)
            input_tensor = x.detach().numpy()
        else:
            if order == 'WHC':
                input_tensor = input_tensor.transpose(2, 1, 0)
                input_tensor = input_tensor[np.newaxis, :]
            elif order == 'NCHW':
                pass
            else:
                raise ValueError("Unknown input dimension, it should be one of ['NCHW', 'WHC']")
            x = torch.from_numpy(input_tensor)
        data = OperatorBase(name=name, op_type='HEAD',
                            output_tensor=input_tensor)
        self._add_model(x, data)
        return name

    def get_operator_dict(self):
        ret = OrderedDict()
        for key in self.model_dict:
            ret[key] = self.model_dict[key].generic
        return ret

    def _get_name(self, pre_name, name):
        if name is not None:
            return name
        if pre_name in self.name_gen:
            self.name_gen[pre_name] += 1
        else:
            self.name_gen[pre_name] = 1
        return pre_name + str(self.name_gen[pre_name])

    def _add_model(self, output, generic_op):
        self.model_dict[generic_op.name] = AttrDict({'output': output, 'generic': generic_op})

    def _get_act(self, activation):
        if activation is None:
            activation = 'linear'
        activation = activation.lower()
        torch_act = nn.Identity()
        if activation == 'relu':
            torch_act = nn.ReLU()
        elif activation == 'relu6':
            torch_act = nn.ReLU6()
        elif activation == 'leakyrelu':
            torch_act = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'sigmoid':
            torch_act = nn.Sigmoid()
        else:
            activation = None
        return activation, torch_act

    def _get_data_with_pad(self, input_name, pad=None):
        if pad is None or pad == (0, 0, 0, 0):
            return self.model_dict[input_name].output
        else:
            pad_t, pad_b, pad_l, pad_r = pad
            return nn.ZeroPad2d((pad_l, pad_r, pad_t, pad_b))(self.model_dict[input_name].output)

    def Conv(self, input_name, in_c, out_c, k, stride=1, pad=0, dilation=1, groups=1, bias=True, activation='Relu', name=None):
        name = self._get_name('Conv', name)
        pad = get_padding(k, dilation, pad)
        input_data = self._get_data_with_pad(input_name, pad)
        torch_conv = nn.Conv2d(in_c, out_c, k, stride, 0,
                               dilation, groups, bias, padding_mode='zeros')
        activation, torch_act = self._get_act(activation)
        torch_layer = nn.Sequential(torch_conv, torch_act)
        output = torch_layer(input_data)
        generic_op = ConvOp(
            name=name,
            input_layers=[input_name],
            weight=torch_conv.weight.detach().numpy(),
            bias=torch_conv.bias.detach().numpy() if bias else None,
            dilation=dilation,
            group=groups,
            kernel=k,
            stride=stride,
            pad=pad,
            output_tensor=output.detach().numpy(),
            activation=activation
        )
        self._add_model(output, generic_op)
        return name

    def F_Conv(
            self,
            input_name,
            weight=None,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bias=None,
            activation='Relu',
            order='NCHW',
            name=None):
        name = self._get_name('Conv', name)
        if weight is None:
            raise ValueError("weight data is required")
        if order != 'NCHW':
            raise ValueError("NCHW Input is only supported vs. {}".format(order))
        n, c, k_h, k_w = weight.shape
        weight_tensor = torch.from_numpy(weight)
        bias_tensor = None if bias is None else torch.from_numpy(bias)
        k = [k_h, k_w]
        pad = get_padding(k, dilation, pad)
        input_data = self._get_data_with_pad(input_name, pad)
        # print('input shape : {}'.format(input_data.shape))
        conv_output = F.conv2d(input_data, weight_tensor, bias_tensor, stride, 0, dilation, groups)
        activation, torch_act = self._get_act(activation)
        torch_layer = nn.Sequential(torch_act)
        output = torch_layer(conv_output)
        # print('output shape: {}'.format(output.shape))
        generic_op = ConvOp(
            name=name,
            input_layers=[input_name],
            weight=weight,
            bias=bias,
            dilation=dilation,
            group=groups,
            kernel=k,
            stride=stride,
            pad=pad,
            output_tensor=output.detach().numpy(),
            activation=activation
        )
        self._add_model(output, generic_op)
        return name

    def DWConv(self, input_name, in_c, k, stride=1, pad=0, dilation=1, bias=True, activation='Relu', name=None):
        name = self._get_name('DWConv', name)
        return self.Conv(input_name, in_c, in_c, k, stride, pad, dilation, in_c, bias, activation, name)

    def FC(self, input_name, in_c, out_c, bias=True, activation='Linear', name=None):
        name = self._get_name('Linear', name)
        return self.Conv(input_name, in_c, out_c, k=1, bias=bias, activation=activation, name=name)

    def Concat(self, input_layers, axis=1, output_tensor=None, name=None):
        if isinstance(axis, str):
            axis = axis.lower()
        if axis in ['c', 'z']:
            axis = 1
        elif axis in ['w', 'x']:
            axis = 3
        elif axis in ['h', 'y', 2]:
            raise ValueError("y-axis concatenation is not supported in MIDAP")
        if not isinstance(axis, int) or axis > 3:
            raise ValueError("Unknown axis : {}".format(axis))
        name = self._get_name('Concat', name)
        inputs = [self.model_dict[x].output for x in input_layers]

        # Set PyTorch Output
        if output_tensor is None:
            output = torch.cat(inputs, dim=axis)
            output_tensor = output.detach().numpy()
        else:
            output = torch.from_numpy(output_tensor)

        concat_info = [x.shape[axis] for x in inputs]
        generic_op = ConcatOp(name=name, input_layers=input_layers, axis=axis,
                              concat_info=concat_info, output_tensor=output_tensor)
        self._add_model(output, generic_op)
        return name

    def _Pool(self, input_name, pool_type, k, stride=1, pad=0, name=None):
        name = self._get_name(pool_type, name)
        pad_orig = pad
        pad = get_padding(k, 1, pad)
        if pool_type == 'MaxPool':
            input_data = self._get_data_with_pad(input_name, pad)
            pad = 0
        elif pad[0] == pad[1] and pad[2] == pad[3]:
            pad = pad_orig = (pad[0], pad[2])
            input_data = self._get_data_with_pad(input_name, None)
        else:
            raise ValueError(
                "uneven padding is not supported on Pooling operation")
        torch_pool = nn.MaxPool2d(k, stride, pad) if pool_type == 'MaxPool' else nn.AvgPool2d(
            k, stride, pad, count_include_pad=False)
        output = torch_pool(input_data)
        generic_op = PoolOp(
            name=name,
            op_type=pool_type,
            input_layers=[input_name],
            kernel=k,
            stride=stride,
            pad=pad_orig,
            output_tensor=output.detach().numpy()
        )
        self._add_model(output, generic_op)
        return name

    def MaxPool(self, input_name, k, stride=1, pad=0, name=None):
        return self._Pool(input_name, 'MaxPool', k, stride, pad, name)

    def AvgPool(self, input_name, k, stride=1, pad=0, name=None):
        return self._Pool(input_name, 'AveragePool', k, stride, pad, name)

    def GlobalPool(self, input_name, name=None):
        name = self._get_name('GlobalPool', name)
        input_data = self.model_dict[input_name].output
        output = torch.nn.AdaptiveAvgPool2d(1)(input_data)
        generic_op = PoolOp(
            name=name,
            op_type='AveragePool',
            input_layers=[input_name],
            kernel=1,
            stride=1,
            pad=0,
            global_pooling=True,
            output_tensor=output.detach().numpy()
        )
        self._add_model(output, generic_op)
        return name

    def Upsample(self, input_name, scale, algorithm='NN', name=None):
        name = self._get_name('UpsampleNN', name)
        input_data = self.model_dict[input_name].output
        output = torch.nn.UpsamplingNearest2d(scale_factor=scale)(input_data)
        generic_op = UpsampleOp(name=name, input_layers=[
                                input_name], kernel=scale, algorithm=algorithm, output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def ConvTranspose(self, input_name, in_c, out_c, k, stride=1, pad=0, groups=1, bias=True, dilation=1, activation='ReLU', name=None):
        input_data = self.model_dict[input_name].output
        if stride > 1:
            name = self._get_name('UpsampleZero', name)
            w = input_data.new_zeros(stride, stride)
            w[0, 0] = 1
            upsample_output = F.conv_transpose2d(input_data, w.expand(input_data.size(
                1), 1, stride, stride), stride=stride, groups=input_data.size(1))
            # upsample_output = upsample_output[:, :, :-(stride - 1), :-(stride - 1)]
            generic_op = UpsampleOp(name=name, algorithm='Zero', input_layers=[
                                    input_name], kernel=stride, output_tensor=upsample_output.detach().numpy())
            self._add_model(upsample_output, generic_op)
            input_name = name
            name = None
        if not isinstance(pad, str):
            pad = [pad for _ in range(2)] if isinstance(pad, int) else pad
            k = [k, k] if isinstance(k, int) else k
            pad_t = k[0] - 1 - pad[0]
            pad_b = k[0] - 1 - pad[0] - (stride - 1)
            pad_l = k[1] - 1 - pad[1]
            pad_r = k[1] - 1 - pad[1] - (stride - 1)
            pad = (pad_t, pad_b, pad_l, pad_r)
        return self.Conv(input_name, in_c, out_c, k, 1, pad, 1, groups, activation=activation, bias=bias, name=name)

    def Sum(self, input1, input2, activation='Linear', name=None):
        name = self._get_name('Sum', name)
        inputd1, inputd2 = [
            self.model_dict[x].output for x in [input1, input2]]
        output = torch.add(inputd1, inputd2)
        activation, torch_act = self._get_act(activation)
        output = torch_act(output)
        generic_op = SumOp(name=name, input_layers=[
                           input1, input2], activation=activation, output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def Mul(self, input1, input2, name=None):
        name = self._get_name('Mul', name)
        inputd1, inputd2 = [
            self.model_dict[x].output for x in [input1, input2]]
        output = torch.mul(inputd1, inputd2)
        # activation, torch_act = self.get_act(activation)
        # output = torch_act(output)
        generic_op = MulOp(name=name, input_layers=[input1, input2], output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def Crop(self, input1, crop_x=None, crop_y=None, name=None):
        name = self._get_name('Crop', name)
        inputd1 = self.model_dict[input1].output
        output = inputd1.detach().numpy()
        if crop_x is not None:
            x1, x2 = crop_x
            x2 = x2 if x2 <= 0 else x2 - output.shape[-1]
            output = output[:, :, :, x1:output.shape[-1] + x2]
            crop_x = [x1, x2]
        if crop_y is not None:
            y1, y2 = crop_y
            y2 = y2 if y2 <= 0 else y2 - output.shape[-2]
            crop_y = [y1, y2]
            output = output[:, :, y1:output.shape[-2] + y2, :]
        generic_op = Crop(name=name, input_layers=[input1], crop_x=crop_x, crop_y=crop_y, output_tensor=output)
        self._add_model(torch.from_numpy(output), generic_op)
        return name

    def Softmax(self, input1, name=None):
        name = self._get_name('Softmax', name)
        inputd1 = self.model_dict[input1].output
        output = torch.nn.Softmax2d()(inputd1)
        generic_op = OperatorBase(name=name, op_type='Softmax', input_layers=[
                                  input1], output_tensor=output.detach().numpy())
        self._add_model(output, generic_op)
        return name

    def from_generic_op(self, op, new_input=None, name=None):
        if new_input is None:
            new_input = op.input_layers
            name = op.name
        if not isinstance(op, OperatorBase):
            raise ValueError("op must be defined as generic_op")
        print('convert op: {}'.format(op))
        if isinstance(op, ConvOp):
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            return self.from_conv_op(input_name, op, name)
        elif isinstance(op, ConcatOp):
            return self.Concat(new_input, op.axis, name)
        elif isinstance(op, PoolOp):
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            return self.from_pool_op(input_name, op, name)
        elif isinstance(op, UpsampleOp):
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            return self.Upsample(input_name, op.k_h, op.algorithm, name)
        elif isinstance(op, ArithmeticOp):
            if isinstance(new_input, list) and len(new_input) == 2:
                pass
            else:
                raise ValueError("ArithmeticOp input must be a length-2 list")
            input1, input2 = new_input
            if isinstance(op, SumOp):
                return self.Sum(input1, input2, op.activation, name)
            elif isinstance(op, MulOp):
                return self.Mul(input1, input2, op.activation, name)
            else:
                raise ValueError("Unknown ArithmeticOp: {}".format(op))
        elif isinstance(op, Crop):
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            crop_x = op.crop_x
            crop_y = op.crop_y
            return self.Crop(input_name, crop_x, crop_y, name)
        elif op.type == 'Softmax':
            input_name = new_input[0] if isinstance(new_input, list) else new_input
            return self.Softmax(input_name, name)
        else:
            raise ValueError("Unknown Operator: {}".format(op))

    def from_conv_op(self, input_name, conv_op, pad=None, name=None):
        weight = conv_op.weight_origin
        order = conv_op.order
        # print('weight.shape: {}'.format(weight.shape))
        # print('order: {}'.format(order))
        if order == 'NWHC':
            weight = weight.transpose(0, 3, 2, 1)
            # print('transposed weight.shape: {}'.format(weight.shape))
        stride = conv_op.stride
        if not pad:
            pad = [conv_op.pad_t, conv_op.pad_b, conv_op.pad_l, conv_op.pad_r]
        dilation = conv_op.dilation
        groups = weight.shape[0] if conv_op.type == 'Depthwise' else 1
        bias = conv_op.bias_origin
        return self.F_Conv(
            input_name,
            weight,
            stride,
            pad,
            dilation,
            groups,
            bias,
            conv_op.activation,
            'NCHW',
            name
        )

    def from_pool_op(self, input_name, pool_op, pad=None, name=None):
        if pool_op.global_pooling:
            return self.GlobalPool(input_name, name)
        pool_type = pool_op.type
        k = [pool_op.k_h, pool_op.k_w]
        if not pad:
            pad = [pool_op.pad_t, pool_op.pad_b, pool_op.pad_l, pool_op.pad_r]
        stride = pool_op.stride
        return self._Pool(input_name, pool_type, k, stride, pad, name)

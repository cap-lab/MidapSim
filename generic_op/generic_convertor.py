from collections import OrderedDict

import logging
import numpy as np

from acc_utils.errors import _assert
from acc_utils.model_utils import *
from acc_utils.model_utils import _RemoveLegacyPad
from config import cfg

from .concat_op import ConcatOp
from .conv_op import ConvOp
from .convpool_op_base import ConvPoolOpBase
from .fc import FC
from .operator_base import OperatorBase
from .pool_op import PoolOp


class GenericConvertor(object):
    activation_list = ['LeakyRelu', 'Relu', 'Sigmoid', 'Tanh']
    reduction_list = ['Softmax']

    def __init__(self):
        self.operator_dict = OrderedDict()
        self.input_tensor = None
        self.layer_match = {}
        self.type_cnt = {}

    def set_image(self, img_path=None, img_shape=None):
        self.input_tensor = get_input_tensor(img_path, img_shape)

    def set_input_tensor(self, tensor_shape=None):
        self.input_tensor = np.random.rand(*tensor_shape).astype('float32')

    def from_caffe2_model(self, init_pb=None, predict_pb=None):
        _assert(self.input_tensor is not None,
                'Please call "CLS.set_image" first')
        if init_pb is None:
            init_pb = get_abs_path(cfg.SYSTEM.WEIGHTS)
        if predict_pb is None:
            predict_pb = get_abs_path(cfg.SYSTEM.NETWORK)
        init_def = caffe2_pb2.NetDef()
        net_def = caffe2_pb2.NetDef()
        with open(init_pb, 'rb') as f:
            init_def.ParseFromString(f.read())
        with open(predict_pb, 'rb') as f:
            net_def.ParseFromString(f.read())
        workspace.RunNetOnce(init_def)
        workspace.FeedBlob(net_def.op[0].input[0], self.input_tensor)
        workspace.CreateNet(net_def, overwrite=True)
        workspace.RunNetOnce(net_def)
        net_def = _RemoveLegacyPad(net_def, init_def, self.input_tensor.shape)
        workspace.CreateNet(net_def, overwrite=True)
        # workspace.RunNetOnce(net_def)
        # print(net_def)
        self.net_def = net_def
        self._parse_caffe2_network(net_def)
        self.post_process()

    def post_process(self, to_midap_tensor=True, merge_batchnorm=True):
        # print("\n\n__________________________post processing of ordereddict__________________________")
        for key in self.operator_dict:
            op = self.operator_dict[key]
            op.tensor_to_midap_tensor()
            for in_layer in op.input_layers:
                self.operator_dict[in_layer].next.append(key)
            if len(op.input_layers) > 0:
                input_shape = self.operator_dict[op.input_layers[0]].output_tensor.shape
                # print(input_shape)
            if merge_batchnorm and isinstance(op, ConvOp):
                op.merge_normalization()
            if isinstance(op, PoolOp) and op.global_pooling > 0:
                op.k_w, op.k_h = input_shape[:-1]
                op.pad_t, op.pad_b, op.pad_l, op.pad_r = 0, 0, 0, 0
            if isinstance(op, ConvPoolOpBase):  # pad fix
                w, h = input_shape[:-1]
                w_align = (w + op.pad_l + op.pad_r - op.k_w) % op.stride
                h_align = (h + op.pad_t + op.pad_b - op.k_h) % op.stride
                if w_align > 0:
                    op.pad_r -= w_align
                if h_align > 0:
                    op.pad_b -= h_align
        # print("\n\n__________________________post processing of ordereddict__________________________\n\n")

    def _add_operator(self, operator):
        # Operator name generally represents the output name - caffe style
        # and it should not be duplicated
        if operator.name in self.operator_dict:
            raise ValueError(
                "The operator already exists!: {}".format(operator.name))
        self.operator_dict[operator.name] = operator

    def _parse_caffe2_network(self, net_def):
        # See midap_software/midap_model.py
        # insert 'data' layer into the op dict
        self.layer_match[net_def.op[0].input[0]] = 'data'
        data = OperatorBase(name='data', op_type='HEAD',
                            output_tensor=self.input_tensor)
        self._add_operator(data)
        for op in net_def.op:
            workspace.RunOperatorOnce(op)
            _assert(op.input[0] in self.layer_match,
                    'unknown input layer: {}'.format(op.input[0]))
            input_layer = self.layer_match[op.input[0]]
            update_layer = self._check_caffe2_op_update(op)
            if update_layer:
                prev_layer = self.operator_dict[input_layer]
                self.layer_match[op.output[0]] = prev_layer.name
                prev_layer.output_tensor = workspace.FetchBlob(op.output[0])
                # self._add_operator(prev_layer)
                continue
            new_operator = None
            kwargs = self._get_args_from_caffe2_op(op)
            if op.type == 'ConvTranspose':
                # Caffe2 does not support ConvTranspose..... ;;
                pass
            elif 'Conv' in op.type:
                new_operator = ConvOp(op_type=op.type, **kwargs)
            elif 'Pool' in op.type:
                new_operator = PoolOp(op_type=op.type, **kwargs)
            elif op.type == 'FC':
                new_operator = FC(op_type=op.type, **kwargs)
            elif 'Upsample' in op.type:
                new_operator = UpsampleOp(op_type=op.type, **kwargs)
            elif 'Concat' in op.type:
                channel_info = list(map(
                    lambda x: self.operator_dict[x].output_tensor.shape[1], kwargs['input_layers']))
                new_operator = ConcatOp(
                    op_type=op.type, channel_info=channel_info, **kwargs)
            elif op.type in ['Sum', 'Add']:
                new_operator = SumOp(**kwargs)
            else:
                new_operator = OperatorBase(op_type=op.type, **kwargs)
            self._add_operator(new_operator)

    def get_new_operator_name(self, op_type):
        idx = 0
        if op_type in self.type_cnt:
            idx = self.type_cnt[op_type] + 1
            self.type_cnt[op_type] = idx
        else:
            self.type_cnt[op_type] = 1
            idx = 1
        return '{}_{}'.format(op_type, idx)

    def _check_caffe2_op_update(self, op):
        update_layer = False
        input_layer = self.layer_match[op.input[0]]
        if op.type in self.activation_list:
            prev_op = self.operator_dict[input_layer]
            _assert(prev_op.activation is None,
                    'prev_op.activation must be none')
            prev_op.activation = op.type
            update_layer = True
        elif op.type == 'SpatialBN':
            prev_op = self.operator_dict[input_layer]
            _assert(prev_op.bn is None, 'prev_op.bn must be none')
            _assert(isinstance(prev_op, ConvOp), 'invalid bn')
            norm_arr = list(
                map(lambda x: workspace.FetchBlob(x), op.input[1:]))
            for arg in op.arg:
                if arg.name == 'epsilon':
                    np.add(norm_arr[3], arg.f, norm_arr[3])
            np.sqrt(norm_arr[3], norm_arr[3])
            prev_op.bn = np.array(norm_arr)
            update_layer = True
        elif op.type in ['Mul', 'Add']:
            if op.input[1] in self.layer_match:
                pass  # Layerwise multiplication
            else:
                prev_op = self.operator_dict[input_layer]
                _assert(isinstance(prev_op, ConvOp), 'invalid scale op')
                if op.type == 'Mul':
                    prev_op.mul_scale(workspace.FetchBlob(op.input[1]))
                else:
                    prev_op.add_bias(workspace.FetchBlob(op.input[1]))
                update_layer = True
        elif op.type in ['Dropout', 'LRN']:
            update_layer = True
        return update_layer

    def _get_args_from_caffe2_op(self, op):
        new_name = self.get_new_operator_name(op.type)
        kwargs = dict()
        for arg in op.arg:
            kwargs[arg.name] = arg.i
        if 'pad_t' in kwargs:
            kwargs['pad'] = [kwargs['pad_t'], kwargs['pad_b'],
                             kwargs['pad_l'], kwargs['pad_r']]
        if 'kernel_h' in kwargs:
            kwargs['kernel'] = [kwargs['kernel_h'], kwargs['kernel_w']]
        if op.type in ['FC', 'Conv', 'ConvTranspose']:
            kwargs['weight'] = workspace.FetchBlob(op.input[1])
            if len(op.input) > 2:
                kwargs['bias'] = workspace.FetchBlob(op.input[2])
        input_layers = []
        for input_layer in op.input:
            if input_layer in self.layer_match:
                input_layers.append(self.layer_match[input_layer])
        kwargs['input_layers'] = input_layers
        kwargs['output_tensor'] = workspace.FetchBlob(op.output[0])
        kwargs['name'] = new_name
        self.layer_match[op.output[0]] = new_name
        return kwargs

    def print_conv_information(self, factor=1):
        print("===============================================================")
        print("Name.Input Shape.Kernel Shape.Output Shape")
        for key in self.operator_dict:
            op = self.operator_dict[key]
            if isinstance(op, ConvOp):
                input_key = op.input_layers[0]
                input_shape = self.operator_dict[input_key].output_tensor.shape
                input_shape = (input_shape[0] * factor,
                               input_shape[1] * factor, input_shape[2])
                output_shape = op.output_tensor.shape
                output_shape = (
                    output_shape[0] * factor, output_shape[1] * factor, output_shape[2])
                kernel_shape = op.weight.shape
                print("{}.{}.{}.{}".format(
                    key, input_shape, kernel_shape, output_shape))

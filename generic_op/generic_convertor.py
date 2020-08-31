from collections import OrderedDict

import logging
import numpy as np

from acc_utils.model_utils import *
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

    def get_new_operator_name(self, op_type):
        idx = 0
        if op_type in self.type_cnt:
            idx = self.type_cnt[op_type] + 1
            self.type_cnt[op_type] = idx
        else:
            self.type_cnt[op_type] = 1
            idx = 1
        return '{}_{}'.format(op_type, idx)

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

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
from collections import OrderedDict

import numpy as np

from acc_utils.attrdict import AttrDict
from acc_utils.errors import *
from acc_utils.model_utils import *
from config import cfg
from generic_op import ArithmeticOp, ConcatOp, ConvOp, ConvPoolOpBase, Crop, PoolOp, UpsampleOp

from .midap_layer import MidapLayer


class MidapModel(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(MidapModel, self).__init__(*args, **kwargs)
        self.init_layer = []

    def from_generic_op_dict(self, op_dict):
        reduction_logic_on = cfg.MODEL.REDUCTION_LOGIC
        allow_abstract = cfg.MODEL.ALLOW_ABSTRACT_DATA
        self.mapping_layer_dict = {}
        finish = False
        for op_key in op_dict:
            op = op_dict[op_key]
            if op.type == 'HEAD':
                self._add_layer(op)
            elif reduction_logic_on and any([isinstance(op, PoolOp) and op.global_pooling, op.type == 'Softmax']):
                self._add_reduction_layer(op)
            elif any([isinstance(op, UpsampleOp), isinstance(op, Crop) and op.input_layers[0] not in self.init_layer]) and allow_abstract and all([len(op_dict[layer].input_layers) == 1 for layer in op.next]):
                if op.input_layers[0] in self.mapping_layer_dict:
                    raise ValueError("Multi-level tensor virtualization is not supported")
                self.mapping_layer_dict[op.name] = op
            elif any([isinstance(op, ConvOp), isinstance(op, ConcatOp), isinstance(op, ArithmeticOp), isinstance(op, PoolOp)]):
                if finish:
                    raise ModelBuildError("post_processing already started")
                self._add_layer(op)
            # or op.type in ['Add', 'Sum', 'Mul']:
            elif isinstance(op, PoolOp) or isinstance(op, UpsampleOp) or isinstance(op, Crop):
                if finish:
                    raise ModelBuildError("post_processing already started")
                self._update_layer(op)
            else:
                finish = True
                print('{}: Not supported operator type < {} > considered as a post-processing step begins'.format(op.name, op.type))
        self._post_process()

    def _add_layer(self, op):
        mapping_op = None
        for l in op.input_layers:
            if l in self.mapping_layer_dict:
                mapping_op = self.mapping_layer_dict[l]
        if mapping_op is not None:
            if len(op.input_layers) > 1:
                raise ValueError("v1.3.0) Multi-input layers with mapping func are not supported yet")
                # How to support it?
            elif isinstance(op, Crop):
                raise ValueError("Virtual tensor cannot be Cropped on DRAM")
            else:
                op.input_layers = copy.copy(mapping_op.input_layers)
        layer = MidapLayer(op, mapping_op=mapping_op)
        if op.type == 'HEAD':
            layer.output_name = layer.name
            self.init_layer.append(layer.name)
        self[layer.name] = layer

    def _update_layer(self, op):
        input_layer = op.input_layers[0]
        if self[input_layer].set_sub_op(op):
            layer = self[input_layer]
            self[layer.name] = layer
            del self[input_layer]
        else:
            if isinstance(op, UpsampleOp):
                print(op)
                raise ModelBuildError("Isolated Upsampling is not supported")
            self._add_layer(op)

    def log_print(self, func=print):
        func("-" * 80)
        func("|{:^20s}|{:^15s}|{:^20s}|{:^20s}|".format("Name", "Op Type", "Input Shape", "Output Shape"))
        func("-" * 80)
        for l in self.values():
            l.log_print(func)
        func("-" * 80)

    def _add_reduction_layer(self, op):
        if op.input_layers[0] not in self:
            raise ModelBuildError("Unknown reduction logic")
        reduction_target_layer = self[op.input_layers[0]]
        reduction_layer = reduction_target_layer.set_reduction_op(op)
        if reduction_layer is not None:
            self[reduction_layer.name] = reduction_layer
        else:
            if op.type == 'Softmax':
                raise ModelBuildError("Isolated Softmax is not supported")
            self._add_layer(op)

    def _post_process(self):
        self._make_graph()
        if cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER == 'GEMM':
            self._setup_gemm()

        for key in self:
            layer = self[key]
            self._setup_pad_feature_map(layer)

        for l in self.init_layer:
            self[l].setup()
            self[l].write_on_dram = True

        for key in self:
            if key in self.init_layer:
                continue
            layer = self[key]
            if len(layer.next) == 0 or (len(layer.next) > 1 and layer.have_reduction_layer):
                layer.write_on_dram = True  # Default : False
            in_c = self[layer.input[0]].output_tensor.shape[-1]
            self._initialize_addr_mapping(layer, in_c)
            if isinstance(layer.main_op, ConvOp):
                self._setup_pad_wb(layer, in_c)
            if isinstance(layer.main_op, ConcatOp):
                self._setup_concat(layer)
            if layer.reduction:
                if in_c != layer.output_tensor.shape[-1]:
                    pad = in_c - layer.output_tensor.shape[-1]
                    layer.output_tensor = np.pad(layer.output_tensor, ((0, 0), (0, 0), (0, pad)), 'constant')
            layer.input = [self[key] for key in layer.input]
            if isinstance(layer.main_op, ArithmeticOp):
                inl = layer.input
                if inl[0].output_tensor.size != inl[1].output_tensor.size:
                    layer.main_op.broadcast = True
                    layer.load_filter_once = True
                    size1, size2 = [i.output_tensor[:,:,0].size for i in inl]
                    if size1 == 1:
                        inl[0].write_on_dram = True
                        layer.input = [inl[1], inl[0]]
                    elif size2 == 1:
                        inl[1].write_on_dram = True
                        layer.input = [inl[0], inl[1]]
                    else:
                        raise ModelBuildError("unmatching MulOp")
            layer.setup()

    def _make_graph(self):
        for key in self:
            if key in self.init_layer:
                continue
            layer = self[key]
            layer.output_name = key
            # make graph
            for input_layer in layer.input:
                if layer not in self[input_layer].next:
                    self[input_layer].next.append(layer)

    def _setup_gemm(self):
        first_conv_ops = []
        for l in self.init_layer:
            layer = self[l]
            for target in layer.next:
                if isinstance(target.main_op, ConvOp) and target.main_op.weight.shape[-1] <= 3:
                    if len(layer.next) > 1:
                        raise ModelBuildError("Input data should be exists in multiple representations")
                    first_conv_ops.append(target.main_op)
                    data = layer.output_tensor
                    target.gemm_input_shape = data.shape  # XXX for static calculation
                    data = im2col(data, target.main_op)
                    data = data.transpose(2, 1, 0)
                    layer.output_tensor = data
        for main_op in first_conv_ops:
            main_op.type = 'Gemm'
            weight = main_op.weight.transpose(0, 3, 1, 2)  # NWHC -> NCWH
            weight = weight.reshape(weight.shape[0], 1, 1, -1)
            main_op.weight = weight

    def _setup_pad_feature_map(self, layer):
        if layer.main_op.type == 'Concat' or (layer.main_op.type in ['Conv', 'FC'] and layer.mapping_func is None and layer.dilation == 1):
            z_pad = cfg.MIDAP.WMEM.NUM
        else:
            z_pad = cfg.MIDAP.SYSTEM_WIDTH
        for input_layer in layer.input:
            input_layer = self[input_layer]
            input_tensor = input_layer.output_tensor
            pad = input_tensor.shape[-1] % z_pad
            if pad > 0:
                pad = z_pad - pad
                input_layer.output_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, pad)), 'constant')
        if layer.parallel_type is not None:
            # print("parallel-layer: {}".format(layer.main_op))
            pad = layer.output_tensor.shape[-1] % z_pad
            if pad > 0:
                pad = z_pad - pad
                layer.output_tensor = np.pad(layer.output_tensor, ((0, 0), (0, 0), (0, pad)), 'constant')
            # print("Pad: {}, new shape: {}".format(pad, layer.output_tensor.shape))
        layer.offset = (0, layer.output_tensor.shape[-1])

    def _setup_pad_wb(self, layer, in_c):
        weight = layer.main_op.weight
        bias = layer.main_op.bias
        z_pad = in_c - weight.shape[-1]
        if layer.parallel_type is not None:
            n_pad = weight.shape[0] % cfg.MIDAP.SYSTEM_WIDTH
            n_pad = cfg.MIDAP.SYSTEM_WIDTH - n_pad if n_pad > 0 else 0
            z_pad = 0
        else:
            n_pad = weight.shape[0] % cfg.MIDAP.WMEM.NUM
            n_pad = cfg.MIDAP.WMEM.NUM - n_pad if n_pad > 0 else 0
        # NWHC --> z_pad = C-axis padding, n_pad = N-axis padding
        # z_pad --> sync with input channel, n_pad --> hardware-aware padding (output)
        if z_pad > 0:
            weight = np.pad(weight, ((0, 0), (0, 0), (0, 0), (0, z_pad)), 'constant')
        if n_pad > 0:
            weight = np.pad(weight, ((0, n_pad), (0, 0), (0, 0), (0, 0)), 'constant')
            if bias is not None:
                layer.main_op.bias = np.pad(bias, ((0, n_pad)), 'constant')
            output_n_pad = max(0, weight.shape[0] - layer.output_tensor.shape[-1])
            if output_n_pad > 0:
                layer.output_tensor = np.pad(layer.output_tensor, ((0, 0), (0, 0), (0, output_n_pad)), 'constant')
        if layer.main_op.type == 'Gemm':  # 2-dimensional Matrix
            weight = weight.reshape(weight.shape[0], -1)
        elif layer.main_op.type == 'Depthwise':
            weight = weight.transpose(3, 1, 2, 0)  # N x W x H x 1 --> 1 x W x H x N
            weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        else:
            weight = weight.reshape(weight.shape[0], weight.shape[1], -1)  # N x W x HC
            zy = weight.shape[2]
            zy_pad = zy % cfg.MIDAP.SYSTEM_WIDTH
            if zy_pad > 0:
                zy_pad = cfg.MIDAP.SYSTEM_WIDTH - zy_pad
                weight = np.pad(weight, ((0, 0), (0, 0), (0, zy_pad)), 'constant')
        layer.main_op.weight = weight
        if weight.size <= cfg.MIDAP.WMEM.NUM_ENTRIES * cfg.MIDAP.WMEM.NUM * 2 and layer.main_op.type != 'Depthwise':
            layer.load_filter_once = True

    def _setup_concat(self, layer):
        # prev = layer.main_op.output_tensor.shape[-1]
        axis = layer.main_op.axis
        offset_prev = layer.output_tensor.shape[axis]
        for concat_layer, offset in reversed(list(zip(layer.input, layer.main_op.size_info))):
            if len(self[concat_layer].next) == 1:
                self[concat_layer].offset = (offset, offset_prev)
                offset_prev = offset
                self[concat_layer].output_tensor = layer.output_tensor
                self[concat_layer].output_name = layer.name
            else:
                info = AttrDict()
                info.shape = layer.output_tensor.shape
                info.name = layer.name
                info.offset = (0, 0, offset) if axis == 2 else (offset, 0, 0)
                # FIXME extra_output_info is only used in concat layer.
                self[concat_layer].extra_output_info.append(info)

    def _initialize_addr_mapping(self, layer, in_c):
        if isinstance(layer.main_op, ConvPoolOpBase) and \
                in_c % cfg.MIDAP.SYSTEM_WIDTH == 0 and \
                layer.mapping_func is None:
            layer.valid_func = lambda x, y: True
            layer.mapping_func = lambda x, y: (x, y)

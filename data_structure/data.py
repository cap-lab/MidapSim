from midap_software import MidapLayer
from generic_op import ConvOp, UpsampleOp, Crop, ArithmeticOp, PoolOp
from collections import OrderedDict

import math
import numpy as np

class SDataInfo(object):
    def __init__(self, name, data, flip = False, require_dram_space = True):
        self.shape = data.shape
        self.name = name
        self.data = data
        self.flip = flip
        if flip:
            self.data = np.flip(self.data, axis=0)
        self.compare_data = np.zeros(self.shape)
        self.require_dram_space = require_dram_space # Not in use.. for further implementation
        self.diff_arr = np.zeros(self.shape)
        self.diff_cnt = 0

    def get_compare_data(self):
        return self.compare_data

    def check_result(self, offset = None, shape = None, name = None):
        if offset == None:
            offset = (0, 0, 0)
        if shape == None:
            shape = self.shape
        if name == None:
            name = self.name
        end_offset = [o + s for o, s in zip(offset, shape)]
        sx, sy, sz = offset
        dx, dy, dz = end_offset
        n = self.data[sx:dx, sy:dy, sz:dz]
        p = self.compare_data[sx:dx, sy:dy, sz:dz]
        diff = np.abs(n - p)
        abs_arr = np.abs(n) + np.abs(p)
        abs_arr = np.where(abs_arr > 0, abs_arr, 1)
        diff_ratio = np.true_divide(diff, abs_arr)
        diff_arr = np.where(diff_ratio < 0.01, 0, 1)
        diff_value = np.sum(diff_arr)
        ret_str = "Function Simulation result - layer: {}, diff: {} / {}".format(
            name, diff_value, diff_arr.size)
        self.diff_arr = diff_arr
        self.diff_cnt = diff_value
        diff_ratio = diff_value / self.data.size
        return diff_ratio , ret_str

class DataMapping(list):
    def __init__(self, mem_id, head, tail):
        super().__init__([mem_id, head, tail])
        self.mem_id = mem_id
        self.head = head
        self.tail = tail
        # or head idx & size?

class MappingInfo(list):
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.write_on_dram_pivot = shape[0]
        self.yz_plane_size = shape[1] * shape[2]

    def add(self, idx, head, tail):
        mapping = DataMapping(idx, head, tail)
        self.append(mapping)

    def __repr__(self):
        mappinginfo = super().__repr__()
        info = "Name: {}, Shape: {}, DRAM Pivot X: {}".format(
                self.name, self.shape, self.write_on_dram_pivot)
        return info + mappinginfo

class SFMEMInfo(object):
    def __init__(self, **kwargs):
        self.input_mapping = OrderedDict() # MappingInfo
        self.output_mapping = OrderedDict() # Where to write?

    def add_midap_layer_info(self, midap_layer):
        control_info = midap_layer.control_info
        input_name = midap_layer.input[0].output_name
        input_shape = midap_layer.input[0].output_tensor.shape
        output_name = midap_layer.output_name
        output_shape = midap_layer.output_tensor.shape
        if input_name not in self.input_mapping:
            self.input_mapping[input_name] = MappingInfo(input_name, input_shape)
        for info in control_info.input_mapping:
            fmem_idx, (head, tail), flag = info
            self.input_mapping[input_name].add(fmem_idx, head, tail)
        if output_name not in self.output_mapping:
            self.output_mapping[output_name] = MappingInfo(output_name, output_shape)
        for info in control_info.output_mapping:
            fmem_idx, (head, tail) = info
            self.output_mapping[output_name].add(fmem_idx, head, tail)

class SWMEMInfo(object):
    def __init__(self, **kwargs):
        self.filter_name = None # Instead of address
        self.bias_name = None # Instead of address
        self.filter_size = 0 # Filter size
        self.num_filters = 0
        self.compute_type = 0
        self.load_filter_once = False
        self.filter_group_size = 1
        self.prepare_info = None # data name, type, size
        self.prepared = False
        self.reverse_load = False

    def from_midap_layer(self, midap_layer):
        control_info = midap_layer.control_info
        main_op = midap_layer.main_op
        if isinstance(main_op, ConvOp):
            self.filter_name = main_op.name + '_w'
            self.filter_size = main_op.weight[0,:].size
            self.num_filters = main_op.weight.shape[0]
            if main_op.bias is not None:
                self.bias_name = main_op.name + '_b'
            self.load_filter_once = midap_layer.load_filter_once
            if main_op.type == 'Depthwise':
                self.load_filter_once = False
                self.compute_type = 1
                self.num_filters = midap_layer.input[0].output_tensor.shape[-1]
        elif isinstance(main_op, ArithmeticOp):
            in2 = midap_layer.input[1]
            self.filter_name = in2.name
            self.num_filters = in2.output_tensor.shape[0]
            self.filter_size = in2.output_tensor[0,:].size
            self.reverse_load = control_info.input_flip != in2.control_info.flip
            self.compute_type = 2
        elif isinstance(main_op, PoolOp):
            self.num_filters = midap_layer.input[0].output_tensor.shape[-1]
            self.compute_type = 1
            return None
        else:
            raise ValueError("Unknown main operator {}".format(main_op))


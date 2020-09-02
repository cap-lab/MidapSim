import math
import numpy as np
import logging

from midap_software import MidapLayer
from generic_op import ConvOp, UpsampleOp, Crop, ArithmeticOp, PoolOp, ConvPoolOpBase
from config import cfg

from .virtual_tensor import VInputTensor, VOutputTensor
from .data import SFMEMInfo, SWMEMInfo

class SLayerInfo(object):
    def __init__(self, **kwargs):
        self.input = []
        self.name = None
        self.modules = SModule()
        self.control_info = SControlInfo()

    def from_midap_layer(self, midap_layer):
        # Set Input
        self.name = midap_layer.name
        self.set_input(midap_layer)
        self.modules.from_midap_layer(midap_layer)
        self.control_info.from_midap_layer(midap_layer, self.input)
    
    def set_input(self, midap_layer):
        #input tensor
        scale = [1, 1, 1]
        offset = [0, 0, 0]
        mapping_type = 'default'
        if isinstance(midap_layer.mapping_op, UpsampleOp):
            mapping_type = 'valid' if midap_layer.mapping_op.algorithm.lower() == 'zero' else 'linear'
        elif isinstance(midap_layer.mapping_op, Crop) or midap_layer.mapping_func is not None:
            mapping_type = 'linear'
        if mapping_type != 'default':
            scale = [midap_layer.scale_w, midap_layer.scale_h, 1]
            offset = [midap_layer.x_offset[0], midap_layer.y_offset[0], 0]
        data = midap_layer.input[0]
        data_name = data.output_name
        input_orig_shape = data.output_tensor.shape
        w, h, c = input_orig_shape
        w -= sum(midap_layer.x_offset)
        h -= sum(midap_layer.y_offset)
        input_shape = [w, h, c]
        input_shape = [i*j for i, j in zip(scale, input_shape)]
        input_tensor = VInputTensor(flip_x = midap_layer.control_info.input_flip)
        input_tensor.set_tensor(
                name = data_name,
                shape = input_shape,
                orig_shape = input_orig_shape,
                mapping_type = mapping_type,
                offset = offset,
                scale = scale)
        #Input tensor for second input cannot be virtualized yet...
        self.input = [input_tensor]

    def __repr__(self):
        s = "<<<Layer Input Information>>>\n"
        for tensor in self.input:
            s += str(tensor) + '\n'
        s += "<<<Processing Information>>>\n"
        s += "Process: {}\n".format(self.name)
        for idx, module in enumerate(self.modules):
            s += 'Op {}: '.format(idx + 1) + str(module.op)
            if len(module.output) > 0:
                s += 'Output: {}\n'.format(module.output[0])
        s += "Action: {} \n".format(self.control_info.behavior_info)
        s += "Expected input mapping: {}\nExpected output mapping: {}".format(self.control_info.get_input_mapping(), self.control_info.get_output_mapping())
        return s


class SModule(list):
    def __init__(self):
        super().__init__()
        self.num_modules = 0

    def add(self, element):
        if not isinstance(element, ModuleElement):
            raise ValueError("Undefined element type")
        self.append(element)
        self.num_modules += 1

    def from_midap_layer(self, midap_layer):
        # Main module
        main_module = ModuleElement(midap_layer)
        self.add(main_module)
        # Reduction module
        if midap_layer.have_reduction_layer:
            reduction_module = ModuleElement(midap_layer.next[0])
            self.add(reduction_module)

    def get_output(self, module_idx = -1, output_idx = 0):
        self[module_idx].get_output(output_idx)

class ModuleElement(object):
    def __init__(self, input_layer = None):
        self.op = None
        self.processing_type = None
        self.output = []
        self.name = None
        if isinstance(input_layer, MidapLayer):
            self.from_midap_layer(input_layer)
    
    def get_output(self, output_idx = 0):
        return self.output[output_idx]

    def from_midap_layer(self, midap_layer):
        self.op = midap_layer.main_op
        self.name = self.op.name
        if any([isinstance(self.op, ArithmeticOp),
                isinstance(self.op, PoolOp),
                self.op.type in ['Depthwise']]):
            self.processing_type = 'extended'
        elif self.op.type in ['Gemm']:
            self.processing_type = 'gemm'
        elif isinstance(self.op, ConvOp):
            self.processing_type = 'default'

        control_info = midap_layer.control_info
        virtual = False
        if midap_layer.have_reduction_layer and not midap_layer.write_on_dram:
            virtual = True
        data_name = midap_layer.output_name
        # for concat
        offset = [0, 0, 0]
        orig_shape = midap_layer.output_tensor.shape
        offset_temp = list(zip(offset, orig_shape))
        offset_temp[midap_layer.offset_axis] = midap_layer.offset
        shape = [x[1] - x[0] for x in offset_temp]
        offset = [x[0] for x in offset_temp]
        # for upsampling, zero insertion
        sub_op = midap_layer.sub_op
        scale = [1, 1, 1]
        # In future: sub_op should integrate whole output tensor virtualization formats
        # @@@ I think that concat should be included in sub_op
        mapping_type = 'default' if sum(offset) == 0 else 'linear'
        if isinstance(sub_op, UpsampleOp):
            if sub_op.algorithm == 'Zero':
                mapping_type = 'zero'
            else:
                mapping_type = 'linear'
            scale = [sub_op.k_w, sub_op.k_h, 1]
            # shape = [size//scale for size, scale in zip(shape, scale)]
        elif isinstance(sub_op, Crop):
            raise ValueError("Crop for sub_op is a weird case.. should not occur")

        tensor = VOutputTensor(reverse_write = control_info.reverse_write, write_on_dram = midap_layer.write_on_dram, virtual = virtual)
        tensor.set_tensor(
                name = data_name,
                shape = shape,
                orig_shape = orig_shape,
                mapping_type = mapping_type,
                offset = offset,
                scale = scale)
        self.output.append(tensor)
        # Extra outputs: must be saved with un-flipped form
        reverse_write = not control_info.input_flip
        for info in midap_layer.extra_output_info:
            extra_tensor = VOutputTensor(reverse_write = reverse_write)
            extra_data_name = info.name
            extra_orig_shape = info.shape
            extra_tensor.set_tensor(extra_data_name, shape, extra_orig_shape, 'linear', offset, scale)
            self.output.append(extra_tensor)

class Behavior(list):
    def __init__(self, behavior_type, in1, in2, in3):
        super().__init__([behavior_type, in1, in2, in3])
        # Condition : [-1, k in [0 ~ N-1], N = # of FMEM banks - 1]
        # -1 : After the finish of previous behavior
        # k : right after processing x==k th input mapping (corresponding input)
        # --> should be changed to index-based condition
        # behavior : [load/process, name, k]
        # load: load k th input mapping of corresponding data
        # process: process ~ kth input mapping of current layer
        self.type = behavior_type
        if behavior_type == 'LOAD':
            self.cond = in1
            self.input_name = in2
            self.index = in3
        elif behavior_type == 'PROCESS':
            self.idx = in1
            self.min_x = in2
            self.max_x = in3

class SBehaviorInfo(list):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_initial_fragments = 0
        self.num_gemm_rows = 0
        self.min_y, self.max_y = [0, 0]
        self.main_op, self.input_tensor, self.input_mapping = [None, None, None] # Temporal data
        
    def from_midap_layer(self, midap_layer, input_tensors):
        self.input_tensor = input_tensors[0]
        self.main_op = midap_layer.main_op
        control_info = midap_layer.control_info
        self.num_initial_fragments = control_info.num_initial_fragments
        self.input_mapping = control_info.input_mapping
        # Verification`
        input_last_x, _, _ = self.input_tensor.get_loc((self.input_tensor.shape[0] - 1, 0, 0))
        if input_last_x >= self.input_mapping[-1][1][1]:
            raise RuntimeError("Invalid Input mapping: {}, input tensor: {}, op: {}".format(self.input_mapping, self.input_tensor, self.main_op))
        # 
        input_name = self.input_tensor.name
        loaded_inputs = self.num_initial_fragments
        processed_inputs = 0
        self.min_y, self.max_y = self.set_min_max_y()
        if self.main_op.type.lower() == 'gemm':
            self.num_gemm_rows = self.input_tensor.shape[-1] // cfg.MIDAP.SYSTEM_WIDTH
        for action_type, idx in control_info.action:
            if action_type == 'LOAD':
                for input_idx in range(loaded_inputs, idx):
                    self.append(Behavior('LOAD', -1, input_name, input_idx))
                loaded_inputs = idx
            elif action_type == 'PROCESS':
                min_x, max_x = self.get_min_max_x(processed_inputs, idx, idx < loaded_inputs)
                self.append(Behavior('PROCESS', idx, min_x, max_x))
                for input_id in range(processed_inputs, idx):
                    load_flag = self.input_mapping[input_id][-1]
                    if load_flag:
                        cond_x = self.get_output_x(self.input_mapping[input_id + 1][1][0])
                        self.append(Behavior('LOAD', cond_x, input_name, loaded_inputs))
                        loaded_inputs += 1
                processed_inputs = idx
        # Verification
        if processed_inputs != loaded_inputs or loaded_inputs != len(self.input_mapping):
            raise RuntimeError("All Input fragments must be processed.. All {} vs loaded {} vs processed {}".format(len(self.input_mapping), loaded_inputs, processed_inputs))

    
    def set_min_max_y(self):
        main_op = self.main_op
        input_tensor = self.input_tensor
        if main_op.type == 'Gemm':
            return -1, -1
        y_min = 0
        y_max = input_tensor.shape[1]
        if isinstance(main_op, ConvPoolOpBase):
            y_min = -main_op.pad_t
            y_max += main_op.pad_b
            y_max -= main_op.k_h
        else:
            y_max -= 1
        return y_min, y_max

    def get_min_max_x(self, head_idx, tail_idx, next_on_chip=False):
        main_op = self.main_op
        scale = self.input_tensor.scale[0]
        offset = self.input_tensor.offset[0]
        input_shape = self.input_tensor.shape
        head_info = self.input_mapping[head_idx][1]
        tail_info = self.input_mapping[tail_idx - 1][1]
        x_min = max(0, (head_info[0] - offset) * scale)
        x_max = max(0, (tail_info[1] - offset) * scale - 1)
        if isinstance(main_op, ConvPoolOpBase) and main_op.type != 'Gemm':
            x_limit = input_shape[0] - main_op.k_w + main_op.pad_r
            if head_idx == 0:
                x_min -= main_op.pad_l
            if tail_idx == len(self.input_mapping):
                x_max += main_op.pad_r
            if not next_on_chip:
                x_max -= main_op.k_w - 1
            remain = (x_min + main_op.pad_l) % main_op.stride
            if remain > 0:
                x_min = x_min + remain
            x_max = min(x_max, x_limit)
        return x_min, x_max

    def get_output_x(self, x):
        main_op = self.main_op
        if main_op.type == 'Gemm':
            return math.ceil(x / main_op.output_tensor.shape[1])
        elif isinstance(main_op, ConvPoolOpBase):
            return math.ceil((x + main_op.pad_l) / main_op.stride)
        else:
            return x

    def __repr__(self):
        orig_str = super().__repr__()
        return "List of [condition, behavior_type, input_name, input_idx] = " + orig_str

class SControlInfo(object):
    def __init__(self, compiler_control_info = None):
        self.fmem_info = SFMEMInfo()
        self.wmem_info = SWMEMInfo()
        self.behavior_info = SBehaviorInfo()
        # MappingInfo
        if compiler_control_info is None:
            pass
        elif isinstance(compiler_control_info, MidapLayer):
            self.from_midap_layer(compiler_control_info)
        else:
            raise ValueError("Not Implemented Error: Unknown control information")
    
    def from_midap_layer(self, midap_layer, input_tensors):
        #Set SFMEMInfo
        self.fmem_info.add_midap_layer_info(midap_layer)
        if midap_layer.have_reduction_layer:
            reduction_layer = midap_layer.next[0]
            self.fmem_info.add_midap_layer_info(reduction_layer)
        #Set SWMEMInfo
        self.wmem_info.from_midap_layer(midap_layer)
        self.behavior_info.from_midap_layer(midap_layer, input_tensors)

    def get_input_mapping(self, name = None):
        if name is None:
            name = list(self.fmem_info.input_mapping.keys())[0]
        return self.fmem_info.input_mapping[name]

    def get_output_mapping(self, name = None):
        if name is None:
            name = list(self.fmem_info.output_mapping.keys())[0]
        return self.fmem_info.output_mapping[name]

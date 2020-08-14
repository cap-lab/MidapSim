import numpy as np

from generic_op import ConvOp, ConcatOp, Crop 
from midap_software import MidapLayer

from .instruction_components import SLayerInfo
from .data import SDataInfo
from config import cfg

import logging 
class SimulatorInstruction(object):
    def __init__(self, compiler_input = None):
        self.processing_order = []
        self.dram_address_dict = {} # Initial data, nparray
        self.data_info_dict = {} # temporal data (output tensors for each layer), SDataInfo
        self.dram_data = np.zeros(0)
        self.dram_size = 0
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger("debug")
        if compiler_input is not None:
            self.from_compiler_input(**compiler_input)

    def from_compiler_input(self, **kwargs):
        pass
    
    def register_dram_data(self, data_name, data):
        if data_name not in self.dram_address_dict:
            addr = self.dram_size
            self.dram_address_dict[data_name] = addr
            self.dram_size += data.size
            self.dram_data = np.concatenate([self.dram_data, data.reshape(-1)])
            del data

    def dump_dram_data(self, file_name = 'dram.txt'):
        self.dram_data.astype('float32').tofile(file_name)
        self.logger.info("Finished... expected Dump File {} SIZE: {} KiB.. ".format(file_name, 4 * self.dram_data.size / 1024))

class SimulatorInstructionV1(SimulatorInstruction): # from existing compiler input, v1.4.0
    def __init__(self, compiler_input = None, **kwargs):
        super(SimulatorInstructionV1, self).__init__(compiler_input)
        self.output_mapping_dict = {}
    
    def from_compiler_input(self, input_data_list, init_layer_list, path_info):
        for data_name, data in zip(init_layer_list, input_data_list):
            self.register_dram_data(data_name, data.reshape(-1))
        prev_wmem_info = None
        prev_layer = None
        for midap_layer in path_info:
            if midap_layer.reduction:
                pass
            elif isinstance(midap_layer.main_op, ConcatOp):
                pass
            elif isinstance(midap_layer.main_op, Crop):
                # Crop z, y -axis is not supported
                main_op = midap_layer.main_op
                if main_op.crop_y is not None:
                    raise ValueError("op {} is not supported.. y-axis crop".formap(main_op))
                # Too dirty code.. T_T
                input_name = midap_layer.input[0].output_name
                pivot_x = main_op.crop_x[0]
                self.update_pivot(input_name, pivot_x)
                #
                offset = main_op.crop_x[0] * midap_layer.output_tensor[0,:,:].size
                addr = self.dram_address_dict[midap_layer.input[0].output_name] + offset
                self.dram_address_dict[midap_layer.output_name] = addr
                continue
            else:
                layer_info = SLayerInfo()
                layer_info.from_midap_layer(midap_layer)
                wmem_info = layer_info.control_info.wmem_info
                if wmem_info.filter_name is not None:
                    if prev_wmem_info is not None and wmem_info.filter_name != prev_layer:
                        prev_wmem_info.prepare_info = wmem_info
                        wmem_info.prepared = True
                    prev_wmem_info = wmem_info
                prev_layer = midap_layer.output_name
                self.processing_order.append(layer_info)
                self.update_mapping_info(layer_info)
            self.setup_data(midap_layer)
        ct = cfg.DRAM.COMM_TYPE
        if 'DMA' in ct:
            dram_file = cfg.DRAM.DUMP_FILE
            self.dump_dram_data(dram_file)
            if not 'TEST' in ct:
                del self.dram_data # Minimize the simulation memory size

    def update_mapping_info(self, layer_info):
        input_name = layer_info.input[0].name
        if input_name in self.output_mapping_dict:
            nif = layer_info.control_info.behavior_info.num_initial_fragments
            input_mapping = layer_info.control_info.get_input_mapping(input_name)
            if nif < len(input_mapping):
                pivot_x = input_mapping[nif].head
                self.update_pivot(input_name, pivot_x)
        output_mapping = layer_info.control_info.fmem_info.output_mapping
        for on in output_mapping:
            if on in self.output_mapping_dict:
                self.output_mapping_dict[on].append(output_mapping[on])
            else:
                self.output_mapping_dict[on] = [output_mapping[on]]

    def update_pivot(self, input_name, pivot_x):
        if input_name in self.output_mapping_dict:
            oml = self.output_mapping_dict[input_name]
            for omap in oml:
                omap.write_on_dram_pivot = min(omap.write_on_dram_pivot, pivot_x)
            if not self.data_info_dict[input_name].require_dram_space:
                self.data_info_dict[input_name].require_dram_space = True
                data = np.zeros(self.data_info_dict[input_name].data.size)
                self.register_dram_data(input_name, data)

    def setup_data(self, midap_layer):
        #Output tensor
        output_data = midap_layer.output_tensor
        output_name = midap_layer.output_name
        control_info = midap_layer.control_info
        # Reserve DRAM Space
        require_dram_space = False
        if midap_layer.have_reduction_layer and not midap_layer.write_on_dram:
            pass
        elif midap_layer.write_on_dram:
            require_dram_space = True
        else:
            output_mapping = control_info.output_mapping
            if len(output_mapping) == 0:
                require_dram_space = True
            else:
                idx = 0 if control_info.reverse_write else -1
                require_dram_space = output_mapping[idx][-1][-1] < output_data.shape[0]
        if require_dram_space:
            data = np.zeros(output_data.size)
            self.register_dram_data(output_name, data)
            #print("output: {}, size: {}".format(output_name, output_data.size))
        # DRAM Space reservation END
        if output_name not in self.data_info_dict:
            self.data_info_dict[output_name] = SDataInfo(output_name, output_data, flip = control_info.flip, require_dram_space = require_dram_space)

        #Weight, bias
        main_op = midap_layer.main_op
        if isinstance(main_op, ConvOp):
            weight = main_op.weight
            bias = main_op.bias
            weight_name = main_op.name + '_w'
            self.register_dram_data(weight_name, weight.reshape(-1))
            if bias is not None:
                bias_name = main_op.name + '_b'
                self.register_dram_data(bias_name, bias.reshape(-1))

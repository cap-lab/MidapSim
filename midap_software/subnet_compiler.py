from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from config import cfg
from generic_op import FC, ArithmeticOp, ConcatOp, ConvOp, Crop, PoolOp
from logger import init_logger
from midap_software.double_buffer_compiler import DoubleBufferCompiler
from midap_software.hide_mem_latency import HideMemLatency
from midap_software.layer_block import BlockBuilder, LayerBlock
from midap_software.min_mem_access import MinMemAccess


class SubNetCompiler(dict):
    def __init__(self, *args, **kwargs):
        super(SubNetCompiler, self).__init__(*args, **kwargs)

        # Policy
        if cfg.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER == 'MIN_DRAM_ACCESS':
            self.layer_compiler = MinMemAccess()
        elif cfg.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER == 'HIDE_DRAM_LATENCY':
            self.layer_compiler = HideMemLatency()
        elif cfg.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER == 'DOUBLE_BUFFER':
            self.layer_compiler = DoubleBufferCompiler()

        self.model = None

        self.logger = init_logger('SubNetCompiler', logging.INFO)
        self.block_organizer = BlockBuilder()

    def force_setup(self, num_init_banks):
        self.layer_compiler.force_setup_layer(num_init_banks)

    def compile(self, midap_model):
        self.model = midap_model
        self.logger.debug("-" * 80)
        self.logger.debug("|{:^78}|".format('Determine Path & Stationary'))
        self.logger.debug("-" * 80)
        self.logger.debug("|{:^20s}|{:^15s}|{:^20s}|{:^20s}|".format("Name", "Op Type", "Input Shape", "Output Shape"))
        self.logger.debug("-" * 80)
        self.determine_path_and_stationary()
        self.logger.debug("-" * 80)

        self.logger.debug("|{:^78}|".format('Generating Control Code'))
        self.generate_control_info()
        self.logger.debug("-" * 80)

        self.logger.debug("|{:^78}|".format('Post Processing'))
        self.post_process()
        self.logger.debug("-" * 80)

        inputs = [self.model[l].output_tensor for l in self.model.init_layer]
        return inputs, self.processing_order

    def get_model_control_info(self):
        input_tensor_list = [self.model[l].output_tensor for l in self.model.init_layer]
        return input_tensor_list, self.processing_order

    def post_process(self):
        # generate detail control information for each layer
        prev_layer = self.processing_order[0]
        prev_layer.control_info['prepared'] = False
        for layer_info in self.processing_order[1:]:
            layer_info.control_info['prepared'] = False
            if isinstance(layer_info.main_op, ConvOp):
                prev_layer.control_info['prepare'] = layer_info
                prev_layer = layer_info
                layer_info.control_info['prepared'] = True
            if isinstance(layer_info.main_op, ArithmeticOp):
                if layer_info.main_op.broadcast and prev_layer in layer_info.input:
                    prev_layer.control_info['prepare'] = None
                else:
                    prev_layer.control_info['prepare'] = layer_info
                    layer_info.control_info['prepared'] = False
                prev_layer = layer_info
            if isinstance(layer_info.main_op, FC):
                main_op = layer_info.main_op
                input_shape = layer_info.input[0].output_tensor.shape
                main_op.k_w, main_op.k_h = input_shape[0], input_shape[1]
                weight = main_op.weight
                weight = weight.reshape(weight.shape[0], -1, input_shape[0], input_shape[1]).transpose(0, 3, 2, 1)
                weight = weight.reshape(weight.shape[0], input_shape[0], -1)
                if layer_info.control_info.input_flip:
                    weight = np.flip(weight, axis=1)
                main_op.weight = weight

        # load filter group
        for layer_info in self.processing_order:
            last_process = 0
            for action, fragment in layer_info.control_info.action:
                if action == 'LOAD':
                    continue
                fragment_size = fragment - last_process
                last_process = fragment
                filter_load = self.get_filter_load(layer_info, fragment_size)
                layer_info.control_info.filter_group.append(filter_load)

    def get_filter_load(self, layer_info, fragment_size):
        if isinstance(layer_info.main_op, PoolOp):
            return 0
        if cfg.MIDAP.CONTROL_STRATEGY.FILTER_LOAD == 'ONE':
            return 1
        elif cfg.MIDAP.CONTROL_STRATEGY.FILTER_LOAD == 'MAXIMUM':
            target_filter = None
            if isinstance(layer_info.main_op, ConvOp):
                target_filter = layer_info.main_op.weight  # NWHC
            elif isinstance(layer_info.main_op, ArithmeticOp):
                target_filter = layer_info.input[1].output_tensor  # WHC
            else:
                raise ValueError("Undefined filter size")
            filter_size = target_filter[0].size
            num_filter = target_filter.shape[0]
            if layer_info.parallel_type is None:
                num_filter = num_filter // cfg.MIDAP.WMEM.NUM
            maximum_load = min(num_filter, cfg.MIDAP.WMEM.NUM_ENTRIES // filter_size)
            if maximum_load == 0:
                raise ValueError("Too big filter size")
            return maximum_load
        else:
            self.determine_filter_load(layer_info, fragment_size)

    def determine_filter_load(self, layer_info, fragment_size):
        pass  # TODO: Should be implemented

    def determine_path_and_stationary(self):
        input_blob = self.model.init_layer
        input_layer = [self.model[x] for x in input_blob]

        paths = self.block_organizer.make_block_path(input_layer)
        po = []
        for v in paths:
            v.log_print(self.logger.debug)
            if isinstance(v, LayerBlock):
                po.extend(v.get_ordered_path())
            else:
                po.append(v)
        self.processing_order = [v for v in po if v.main_op.type != 'HEAD']

    def generate_control_info(self):
        for idx, layer in enumerate(self.processing_order):
            self.logger.debug("============================================== [ {:^12s} ] ==============================================".format(layer.name))

            if isinstance(layer.main_op, ConcatOp) or isinstance(layer.main_op, Crop):
                continue

            next_layer = self.processing_order[idx + 1] if layer != self.processing_order[-1] else None
            self.layer_compiler.setup_layer(layer, next_layer)
            self.layer_compiler.compile_layer()

            self.logger.debug('Input Mapping:', layer.control_info.input_mapping)
            self.logger.debug('Output Mapping:', layer.control_info.output_mapping)
            self.logger.debug('Action:', layer.control_info.action)
            self.logger.debug("=============================================================================================================\n")
        self.logger.debug("Control Manager: Control info generation is finished")

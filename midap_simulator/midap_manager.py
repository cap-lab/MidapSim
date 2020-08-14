from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import logging.config
import traceback
import numpy as np

from acc_utils.model_utils import *
from generic_op import ConcatOp, Crop
from config import cfg

from .memory_controller import MemoryController
from .pipeline import get_pipeline
from .control_logic import get_control_logic
from .statistics import Stats

class MidapManager():
    # MidapManager processes each layer based on given control sequence
    def __init__(self, simulation_level = 0, *args, **kwargs):
        # initialize submodules (cfg)
        self.stats = Stats()
        self.simulation_level = simulation_level
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger()
        self.initialized = False
        self.initialize()

    def __del__(self):
        del self.stats, self.memory_controller
        if not self.initialized:
            del self.pipeline, self.control_logic, self.data_info_dict

    def initialize(self):
        self.stats.init()
        self.data_info_dict = None
        self.print_stats = True
        # self.control_logic = get_control_logic(self, self.simulation_level)
        self.memory_controller = MemoryController(self)
        self.control_logic = get_control_logic(self, self.simulation_level)
        self.pipeline = get_pipeline(self, self.simulation_level)
        self.initialized = True
        self.diff_cnt = 0

    def simulate(self, simulator_instruction):
        if not self.initialized:
            self.initialize()
        self.initialized = False
        self.data_info_dict = simulator_instruction.data_info_dict
        processing_order = simulator_instruction.processing_order
        self.memory_controller.set_dram_info(simulator_instruction.dram_data, simulator_instruction.dram_address_dict)
        return self._process_network(processing_order)

    def _process_network(self, path_info):
        for idx, layer_info in enumerate(path_info):
            self.process_layer(layer_info, idx)
        self.stats.end_simulation()
        last_diff = self.diff_cnt
        stats = self.stats.global_stats
        latency = stats.CLOCK
        dram_access = stats.READ.DRAM + stats.WRITE.DRAM
        return (last_diff, latency, stats.READ.DRAM2FMEM, stats.READ.DRAM2WMEM)

    def process_layer(self, layer_info, layer_idx=None):  # layer_idx : debugging info
        self.setup_layer(layer_info)
        self.run()
        self.finish()
        self.logger.info('---------------------------------------------------')
    
    def setup_layer(self, layer_info):
        self.layer_info = layer_info
        self.control_info = layer_info.control_info
        self.main_op = layer_info.modules[0].op
        self.pipeline.setup(layer_info.modules)
        self.control_logic.setup(layer_info)
        self.memory_controller.setup(layer_info)
        self.on_chip_input_idx = 0
        self.logger.info(str(layer_info))
    
    def run(self):
        behavior_info = self.control_info.behavior_info
        input_mapping = self.control_info.fmem_info.input_mapping
        for idx, (btype, i1, i2, i3) in enumerate(behavior_info):
            self.logger.info("Processing: {}, {}, {}:{}".format(btype, i1, i2, i3))
            if btype == 'LOAD':
                cond, data_name, load_idx = i1, i2, i3
                fmem_idx, head, tail = input_mapping[data_name][load_idx]
                self.run_pipeline(cond)
                self.memory_controller.load_fmem(fmem_idx, data_name, [head, tail])
            elif btype == 'PROCESS':
                process_idx, head_x, tail_x = i1, i2, i3
                last = len(behavior_info) - 1  == idx
                self.run_pipeline(-1)
                self.control_logic.set_generator(head_x, tail_x, self.on_chip_input_idx, last)
                self.on_chip_input_idx = process_idx

    def run_pipeline(self, cond):
        for dataflow, simulation_info in self.control_logic.generator:
            running_info, simulated_cycle = simulation_info
            self.stats.increase_cycle(simulated_cycle)
            out_dataflow = self.pipeline.run(dataflow)
            output_phase = out_dataflow.phase
            last_filter, x = running_info.last_filter, running_info.x
            del running_info, out_dataflow
            if output_phase == 3: # Layer processing is finished
                self.logger.info("Layer processing is finished")
                break
            if last_filter and x == cond:
                self.logger.info("Interrupt condition is met")
                break

    def finish(self):
        self.finish_pipeline()
        self.memory_controller.sync()
        self.control_logic.sync()
        # Check Functional Result
        if self.simulation_level == 0:
            self.diff_cnt = 0
            for m in self.layer_info.modules:
                for data in m.output:
                    diff_ratio, ret_str = self.data_info_dict[data.name].check_result(data.offset, data.shape, m.name)
                    self.logger.info(ret_str)
                    if diff_ratio > 0.001:
                        self.logger.warning("Diff Ratio > 0.1%... Please check the following layers ) Difference might be caused by zero-padded data")
                    self.diff_cnt += self.data_info_dict[data.name].diff_cnt
            # Check End
        self.stats.set_macs(self.main_op.get_macs())
        self.stats.update(self.layer_info.name, self.print_stats)
    
    def finish_pipeline(self):
        self.run_pipeline(-1)
        self.control_logic.set_finish_generator()
        self.run_pipeline(-1)
        self.stats.increase_cycle(self.pipeline.get_skipped_pipeline_stages())

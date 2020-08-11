from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from collections import deque

import numpy as np

from config import cfg
from generic_op import ArithmeticOp, PoolOp

from .stage import Stage
from .dataflow import generate_dataflow_info

class FWriteStage(Stage):
    def initialize(self):
        self.memory_controller = self.manager.memory_controller
        self.stats = self.manager.stats
        self.save_buf = np.zeros(self.system_width)
        self.concurrency = self.num_wmem
        self.save_info = generate_dataflow_info()
        self.write_queue = deque()
        self.write_offset = 0
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger("debug")
    
    def setup(self, modules): # microcontroller
        self.data_info_dict = self.manager.data_info_dict
        self.output_mapping = self.manager.control_info.fmem_info.output_mapping
        self.main_output = modules[0].output
        self.reduction_output = [] if len(modules) <= 1 else modules[1].output
        self.concurrency = self.num_wmem
        op = modules[0].op
        if any([isinstance(op, ArithmeticOp), isinstance(op, PoolOp), op.type == 'Depthwise']):
            self.concurrency = self.system_width
        # Write property
        mo = self.main_output
        cc = self.concurrency
        self.write_size = 0
        self.main_write_size = cc
        if len(mo) == 0:
            pass
        elif mo[0].offset[-1] % cc != 0 or mo[0].shape[-1] % cc != 0:
            self.main_write_size = min(self.num_wmem, self.system_width)

        ro = self.reduction_output
        self.reduction_write_size = cc
        if len(ro) == 0:
            pass
        elif ro[0].offset[-1] % cc != 0 or ro[0].shape[-1] % cc != 0:
            self.reduction_write_size = min(self.num_wmem, self.system_width)

    def run(self, info):
        output_info = self.set_write(info)
        self.write()
        return output_info

    def set_write(self, info):
        while len(self.write_queue) > 0:
            if not any([info.phase == 3, info.last]):
                return info
            self.stats.wait_writing()
            self.write()
        output_info = self.save_info
        self.save_info = info
        self.save_buf[:] = self.input_buf[:]
        self.write_offset = 0
        if not info.last:
            return output_info
        elif info.phase == 1:
            output_tensor = self.main_output
            self.write_size = self.main_write_size
        elif info.phase == 2:
            output_tensor = self.reduction_output
            self.write_size = self.reduction_write_size
        else:
            return output_info
        loc = (info.out_x, info.out_y, info.out_z)
        for t in output_tensor:
            locs = t.get_output_loc(loc)
            guard = t.shape[-1] + t.offset[-1]
            for loc in locs:
                self.write_queue.append((t, loc, guard))
        return output_info

    def get_fmem_info(self, x, mapping):
        fmem_idx = -1
        for idx, head, tail in mapping:
            if head <= x and x < tail:
                fmem_idx = idx
                effective_x = x - head
                return fmem_idx, effective_x
        return -1, -1

    def write(self):
        if not self.save_info.last:
            if len(self.write_queue) > 0:
                self.logger.error("Write Queue: {}".format(self.write_queue))
                self.logger.error("Save_info: {}".format(self.save_info))
                self.logger.error("wo, ws = {}, {}".format(self.write_offset, self.write_size, self.concurrency))
                self.logger.error("Write Queue should not be empty in this case!")
                raise ValueError()
            return None
        if len(self.write_queue) == 0:
            return None
        if self.save_info.phase in [0, 3]:
            if len(self.write_queue) > 0:
                self.logger.error("Write Queue must be empty for such info {}".format(self.save_info))
                raise RuntimeError()
            return None
        output_tensor, loc, guard = self.write_queue[0]
        on = output_tensor.name
        wo = self.write_offset
        ws = self.write_size
        x, y, head = loc
        head += wo
        tail = head + ws
        # For functinality checking
        self.write_data_info(on, x, y, head, tail, self.save_buf[wo:wo+ws])
        #self.data_info_dict[on].compare_data[x, y, head:tail] = self.save_buf[wo:wo+ws]
        # Check Data
        if not output_tensor.virtual:
            fmem_idx = -1
            fmem_loc = None
            output_mapping = None
            write_on_dram = output_tensor.write_on_dram
            if on in self.output_mapping:
                output_mapping = self.output_mapping[on]
                fmem_idx, effective_x = self.get_fmem_info(x, output_mapping)
                fmem_loc = (effective_x, y, head)
                if not write_on_dram:
                    write_on_dram = x >= output_mapping.write_on_dram_pivot
            # Write on DRAM
            if write_on_dram: 
                dram_addr = output_tensor.get_address((x, y, head))
                #self.logger.debug("WRITE ON DRAM, address: {}".format(dram_addr))
                self.memory_controller.write_dram(on, dram_addr, self.save_buf[wo:wo+ws])
            # Write on FMEM
            if fmem_idx >= 0:
                address = output_tensor.get_address(fmem_loc)
                #self.logger.debug("WRITE ON FMEM, address: {}, {}".format(fmem_idx, address))
                self.memory_controller.write_fmem(fmem_idx, address, self.save_buf[wo:wo+ws])
            # Check
        self.write_offset += ws
        if self.write_offset >= self.concurrency or tail >= guard:  # Finish writing
            self.write_queue.popleft()
            self.write_offset = 0

    def write_data_info(self, output_name, x, y, head, tail, data):
        self.data_info_dict[output_name].compare_data[x, y, head:tail] = data[:]

class VWriteStage(FWriteStage):
    def run(self, info):
        self.set_write(info)
        while len(self.write_queue) > 0:
            self.write()
        return info
    
    def write_data_info(self, *args, **kwargs):
        pass

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import logging

from acc_utils.model_utils import *
from config import cfg

from .virtual_memory_manager import VMemoryManager, TVMemoryManager
from .dma_memory_manager import DMemoryManager, TDMemoryManager

def get_memory_manager(manager):
    ct = cfg.DRAM.COMM_TYPE
    if ct == 'VIRTUAL':
        return TVMemoryManager(manager)
    if ct == 'TEST_DMA':
        return TDMemoryManager(manager)
    elif ct == 'DMA':
        return DMemoryManager(manager)

class MemoryController():
    def __init__(self, manager):
        self.manager = manager
        self.memory_manager = get_memory_manager(manager)
        self.system_width = cfg.MIDAP.SYSTEM_WIDTH
        self.filter_name = None
        self.bias_name = None
        # 0 : normal / 1: depthwise / 2: arithmetic
        self.compute_type = 0
        # normal - load num_wmem filters at once
        # depthwise - load 1 filter at once
        # tensor - load wmem_size
        self.input_tensor = None
        self.num_filters = 0
        self.filter_idx_pivot = 0
        self.load_filter_once = False
        self.all_filters_on_wmem = False
        self.filter_size = 0
        self.prepare_info = None
        self.num_wmem = cfg.MIDAP.WMEM.NUM
        self.filter_group_size = 1
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('debug')

    def __del__(self):
        del self.memory_manager

    def set_dram_info(self, dram_data, dram_address_dict):
        self.memory_manager.set_dram_info(dram_data, dram_address_dict)

    def setup(self, layer_info):
        control_info = layer_info.control_info
        # Load fmem information
        self.input_mapping = control_info.fmem_info.input_mapping
        # Load wmem information
        self.wmem_info = control_info.wmem_info
        wi = self.wmem_info
        self.compute_type = wi.compute_type
        # Filter information
        self.filter_name = wi.filter_name
        self.bias_name = wi.bias_name
        self.filter_size = wi.filter_size
        self.num_filters = wi.num_filters
        self.use_extended_cim = False
        if self.filter_size % self.system_width != 0:
            raise ValueError("Filter size is not padded as well")
        if self.compute_type == 0: # Conv
            self.filter_set_size = self.num_wmem
        elif self.compute_type == 1: # DWConv, Pool
            self.filter_set_size = self.system_width
            self.use_extended_cim = True
        elif self.compute_type == 2: # ArithmeticOp
            self.filter_set_size = 1
            self.use_extended_cim = True
        self.num_filters = self.num_filters // self.filter_set_size
        # Group related terms
        self.filter_group_size = wi.filter_group_size
        self.num_filter_groups = div_ceil(self.num_filters, self.filter_group_size)
        self.filter_idx = -1
        # Filter loading configuration
        self.reverse_load = wi.reverse_load
        self.load_filter_once = wi.load_filter_once
        self.all_filters_on_wmem = self.filter_name is None
        # preparation information
        if self.filter_name is not None and not self.wmem_info.prepared:
            self.memory_manager.reset_wmem()
            self.load_wmem(False)
            if self.bias_name is not None:
                self.memory_manager.load_bmmem(self.bias_name, wi.num_filters)
        self.prepare_info = wi.prepare_info
        self.setup_bmmem()
    
    def sync(self):
        wait_sync = self.memory_manager.sync()
        self.manager.stats.wait_write_dram(wait_sync)
        
    # WMEM related functions
    def set_next(self, last_use=False):
        self.filter_idx = (self.filter_idx + 1) % self.num_filters
        last_filter = self.filter_idx + 1 == self.num_filters
        if self.compute_type == 0:
            switch_wmem = False
            if self.filter_idx % self.filter_group_size == 0:
                switch_wmem = True
            if self.all_filters_on_wmem and self.filter_idx == 0:
                switch_wmem = self.num_filter_groups % 2 == 0
            if switch_wmem:
                self.memory_manager.switch_wmem()
            self.load_wmem(last_use)
        elif self.compute_type == 1 and not self.all_filters_on_wmem:
            self.memory_manager.switch_wmem()
            self.load_wmem(True)
        elif self.compute_type == 2:
            self.memory_manager.switch_wmem()
            self.load_wmem(last_use)
        self.logger.debug("WMEM IN USE: {}".format(self.memory_manager.wmem_in_use))
        filter_idx = self.filter_idx * self.filter_set_size
        return last_filter, filter_idx

    def load_wmem(self, last_use = False):
        filter_size = self.filter_size
        filter_set_size = self.filter_set_size
        load = self.filter_idx % self.filter_group_size == 0 or self.filter_idx == -1
        if not load:
            return None
        load_filter_idx = self.filter_idx + self.filter_group_size
        load_prepare = False
        if (self.compute_type == 1 and last_use) or load_filter_idx >= self.num_filters:
            if self.load_filter_once or self.compute_type == 1:
                self.all_filters_on_wmem = True
            load_prepare = last_use
            load_filter_idx = 0
        self.logger.debug("Load_WMEM: Load filter idx : {} to wmem {}, load_prepare : {}, all_filters_on_wmem: {}".format(load_filter_idx, (self.memory_manager.wmem_in_use + 1) % 2,  load_prepare, self.all_filters_on_wmem))
        if not load_prepare:
            if self.all_filters_on_wmem:
                return None
            next_group_size = min(self.num_filters - load_filter_idx, self.filter_group_size)
            if self.compute_type == 0:
                wmem_pivot = 0 if not self.load_filter_once else self.filter_group_size * (load_filter_idx // (2 * self.filter_group_size))
                self.load_filter(
                        self.compute_type,
                        self.filter_name,
                        self.filter_size,
                        next_group_size,
                        wmem_pivot,
                        load_filter_idx
                        )
            elif self.compute_type == 1:
                self.load_filter(
                        self.compute_type,
                        self.filter_name,
                        self.filter_size,
                        1, 0, 0,
                        )
            elif self.compute_type == 2:
                filter_idx_pivot = self.num_filters - load_filter_idx - next_group_size if self.reverse_load else load_filter_idx
                self.load_filter(
                        self.compute_type,
                        self.filter_name,
                        self.filter_size,
                        next_group_size,
                        0,
                        filter_idx_pivot,
                        self.reverse_load
                        )
        elif self.prepare_info is None:
            return None
        else:
            self.logger.debug("Load Prepare Info")
            pi = self.prepare_info
            if pi.compute_type == 2:
                load_idx = 0 if not pi.reverse_load else pi.num_filters - pi.filter_group_size
                self.load_filter(
                        pi.compute_type,
                        pi.filter_name,
                        pi.filter_size,
                        pi.filter_group_size,
                        0,
                        load_idx,
                        pi.reverse_load
                        )
            else:
                self.load_filter(
                        pi.compute_type,
                        pi.filter_name,
                        pi.filter_size,
                        pi.filter_group_size,
                        0,
                        0,
                        )

    def load_filter(
            self,
            compute_type,
            filter_name,
            filter_size,
            group_size,
            wmem_pivot,
            filter_idx_pivot,
            reverse_load = False
            ):
        self.logger.debug("Function call: Load Filter")
        self.logger.debug("Time: {}, compute_type: {}, filter_name: {}, filter_size: {}, group_size: {}, wmem_pivot: {}, filter_idx_pivot: {}, reverse_load: {}".format(self.manager.stats.total_cycle(), compute_type, filter_name, filter_size, group_size, wmem_pivot, filter_idx_pivot, reverse_load))
        if compute_type == 0:
            for g in range(group_size):
                for wmem_idx in range(self.num_wmem):
                    continuous_request = not (wmem_idx == self.num_wmem - 1)
                    self.memory_manager.load_wmem(
                            wmem_idx = wmem_idx,
                            filter_name = filter_name,
                            filter_size = filter_size,
                            wmem_offset = (wmem_pivot + g) * filter_size,
                            dram_offset = (filter_idx_pivot + g) * filter_size * self.num_wmem + wmem_idx * filter_size,
                            continuous_request = continuous_request)
                    self.manager.stats.read_dram2wmem(filter_size)
        elif compute_type == 1:
            self.memory_manager.load_wmem(
                    wmem_idx = 0,
                    filter_name = filter_name,
                    filter_size = filter_size,
                    wmem_offset = 0,
                    dram_offset = 0,
                    )
            self.manager.stats.read_dram2wmem(filter_size)
        else:
            for g in range(group_size):
                if reverse_load:
                    pivot = filter_idx_pivot + group_size - g - 1
                else:
                    pivot = filter_idx_pivot + g
                self.memory_manager.load_wmem(
                        wmem_idx = 0,
                        filter_name = filter_name,
                        filter_size = filter_size,
                        wmem_offset = g * filter_size,
                        dram_offset = pivot * filter_size
                        )
                self.manager.stats.read_dram2wmem(filter_size)

    def load_wbuf(self, wbuf, row):
        pivot = 0
        if self.load_filter_once:
            pivot = (self.filter_idx // (self.filter_group_size * 2)) * self.filter_group_size
        pivot += self.filter_idx % self.filter_group_size
        address = self.system_width * row + pivot * self.filter_size
        wait_time = self.memory_manager.read_wmem(wbuf, self.use_extended_cim, address)
        self.manager.stats.wait_dram2wmem(wait_time)
        self.manager.stats.wmem_read()

    def load_fmem(self, fmem_idx, data_name, info):
        inp = self.input_mapping[data_name]
        data_size = (info[1] - info[0]) * inp.yz_plane_size
        data_address = inp.yz_plane_size * info[0]
        self.logger.debug("Time {}: LOAD FMEM bank {}: DATA {}, address {}, size {}".format(self.manager.stats.total_cycle(), fmem_idx, data_name, data_address, data_size))
        self.memory_manager.load_fmem(
                fmem_idx,
                data_name,
                data_size,
                0,
                data_address,
                )
        self.manager.stats.read_dram2fmem(data_size)

    def load_fbuf(self, fbuf, bank_idx, row):
        address = self.system_width * row
        wait_time = self.memory_manager.read_fmem(fbuf, bank_idx, address)
        self.manager.stats.wait_dram2fmem(wait_time)
        self.manager.stats.fmem_read()

    def write_fmem(self, bank_idx, address, data):
        # set debug info
        wait_time = self.memory_manager.write_fmem(bank_idx, address, data)
        self.manager.stats.wait_write_dram(wait_time)

    def write_dram(self, data_name, address, data):  # DRAM Write
        wait_time = self.memory_manager.write_dram(data_name, address, data) 
        self.manager.stats.wait_memory(wait_time)
        self.manager.stats.dram_write(data.size)

    """ BMMEM related """
    def setup_bmmem(self):
        if self.bias_name is not None:
            self.memory_manager.switch_bmmem()
        if self.prepare_info is not None and self.prepare_info.bias_name is not None:
            pi = self.prepare_info
            self.memory_manager.load_bmmem(pi.bias_name, pi.num_filters)
            self.manager.stats.dram_read(pi.num_filters)

    def load_bbuf(self, bbuf, address):
        wait_time = self.memory_manager.read_bmmem(bbuf, self.use_extended_cim, address)
        self.manager.stats.wait_memory(wait_time)

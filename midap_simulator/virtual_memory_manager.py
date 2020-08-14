from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import math

from acc_utils.model_utils import *
from config import cfg
from .memory_manager import MemoryManager

class VMemoryManager(MemoryManager):
    def __init__(self, manager):
        super().__init__(manager)
        # Set DRAM constraints
        self.bus_policy = cfg.MIDAP.BUS_POLICY
        self.dram_constants = [cfg.DRAM.CAS, cfg.DRAM.PAGE_DELAY, cfg.DRAM.REFRESH_DELAY]
        self.dram_offsets = [cfg.DRAM.PAGE_OFFSET, cfg.DRAM.RESET_OFFSET, cfg.DRAM.RESET_PERIOD]
        self.dram_offsets = [cfg.DRAM.FREQUENCY * i for i in self.dram_offsets]
        self.dram_page_size = cfg.DRAM.PAGE_SIZE
        self.dram_bandwidth = (cfg.SYSTEM.BANDWIDTH * 1000 // cfg.SYSTEM.FREQUENCY) // cfg.SYSTEM.DATA_SIZE
        self.dram_latency_type = cfg.LATENCY.LATENCY_TYPE.lower()
        self.include_dram_write = cfg.DRAM.INCLUDE_DRAM_WRITE
        self.fmem_valid_timer = [-1 for _ in range(self.num_fmem)]
        self.wmem_valid_timer = [-1, -1]
        self.bmmem_valid_timer = [0, 0]
        self.fifo_end_timer = 0
        self.continuous_request_size = 0
    
    def get_dram_write_latency(self, size):
        cas, pg_dly, rst_dly = self.dram_constants 
        return math.ceil(pg_dly * math.ceil(size / self.dram_page_size))
    
    def get_dram_read_latency(self, size, continuous_request = False):
        if continuous_request:
            self.continuous_request_size += size
            return 0
        elif self.continuous_request_size > 0:
            size += self.continuous_request_size
            self.continuous_request_size = 0
        if self.dram_latency_type == 'worst':
            cas, pg_dly, rst_dly = self.dram_constants
            pg_ofs, rst_ofs, rst_prd = self.dram_offsets  # Deprecated
            predict = cas + pg_dly * (size // self.dram_page_size) + \
                rst_dly * math.ceil(size / rst_prd)
            return math.ceil(predict)
        elif self.dram_latency_type == 'exact':
            cas, pg_dly, rst_dly = self.dram_constants
            pg_ofs, rst_ofs, rst_prd = self.dram_offsets  # Deprecated
            predict = cas + pg_dly * math.ceil((size - pg_ofs) / self.dram_page_size) + \
                rst_dly * max(0, math.ceil((size - rst_ofs) / rst_prd))
            return math.ceil(predict)
        else:
            raise ValueError("Unknown latency type!: " + self.dram_latency_type)
    
    def reset_wmem(self):
        self.wmem_in_use = -1
    
    def read_dram_data(self, name, offset, size):
        return self.dram_dict[name][offset:offset + size]

    def load_wmem(self, wmem_idx, filter_name, filter_size = 0, wmem_offset = 0, dram_offset = 0, continuous_request = False):
        wmem_not_in_use = (self.wmem_in_use + 1) % 2
        self.logger.debug("Load data [{}] - addr {}, size {} to WMEM {}, offset {}".format(filter_name, dram_offset, filter_size, (wmem_not_in_use, wmem_idx), wmem_offset))
        if wmem_idx != 0 and wmem_offset + filter_size > self.wmem_size:
            self.logger.error("WMEM Size: {} vs Requested Address: {}".format(self.wmem_size, wmem_offset + filter_size))
            raise ValueError("Wrong Address")
        self.wmem[wmem_not_in_use, wmem_idx, wmem_offset:wmem_offset + filter_size] = \
                self.read_dram_data(filter_name, dram_offset, filter_size)
        # DRAM Access time
        self.update_wmem_timer(wmem_not_in_use, filter_size, continuous_request)

    def update_wmem_timer(self, wid, filter_size, continuous_request):
        current_time = self.manager.stats.total_cycle()
        load_start_time = max(current_time, max(self.wmem_valid_timer))
        expected_transfer_time = self.get_dram_read_latency(filter_size, continuous_request)
        if expected_transfer_time == 0:
            self.logger.debug("Transfer size cumulation...")
            return None
        self.wmem_valid_timer[wid] = load_start_time + expected_transfer_time
        for bid in range(len(self.bmmem_valid_timer)):
            if self.bmmem_valid_timer[bid] > load_start_time:
                self.bmmem_valid_timer[bid] += expected_transfer_time
                #self.logger.debug("WMEM Load precede BMMEM {} Load.. delayed to {}".format(bid, self.bmmem_valid_timer[bid]))
        for fid in range(len(self.fmem_valid_timer)):
            if self.fmem_valid_timer[fid] > load_start_time:
                self.fmem_valid_timer[fid] += expected_transfer_time
                #self.logger.debug("WMEM Load precede FMEM {} Load.. delayed to {}".format(fid, self.fmem_valid_timer[fid]))
        self.fifo_end_timer = max(self.wmem_valid_timer[wid], self.fifo_end_timer + expected_transfer_time)
        self.logger.debug("WMEM {} timer is updated to {}, fifo_end_timer = {}".format(wid, self.wmem_valid_timer[wid], self.fifo_end_timer))
    
    def load_fmem(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0):
        self.logger.debug("Load data [{}] - addr {}, size {} to FMEM {}".format(data_name, dram_offset, data_size, fmem_idx))
        self.fmem[fmem_idx, fmem_offset:fmem_offset+data_size] = \
                self.read_dram_data(data_name, dram_offset, data_size)
        # DRAM Access Time
        self.update_fifo_timer(data_size)
        self.fmem_valid_timer[fmem_idx] = self.fifo_end_timer 
        self.logger.debug("FMEM {} timer & fifo_end_timer is updated to {}".format(fmem_idx, self.fifo_end_timer))
    
    def update_fifo_timer(self, data_size):
        current_time = self.manager.stats.total_cycle()
        load_start_time = max(current_time, self.fifo_end_timer)
        expected_transfer_time = self.get_dram_read_latency(data_size)
        self.fifo_end_timer = load_start_time + expected_transfer_time

    def load_bmmem(self, bias_name, bias_size):
        bmmem_not_in_use = (self.bmmem_in_use + 1) % 2
        self.bmmem[bmmem_not_in_use, : bias_size] = self.read_dram_data(bias_name, 0, bias_size)
        #DRAM Access Time
        self.update_fifo_timer(bias_size)
        self.bmmem_valid_timer[bmmem_not_in_use] = self.fifo_end_timer
        self.logger.debug("BMMEM {} timer & fifo_end_timer is updated to {}".format(bmmem_not_in_use, self.fifo_end_timer))

    def read_wmem(self, buf, extended_cim, address):
        # Latency
        current_time = self.manager.stats.total_cycle()
        time_gap = max(0, self.wmem_valid_timer[self.wmem_in_use] - current_time)
        if time_gap > 0:
            self.logger.debug("Time {}: WMEM {} load delay occured & time_gap = {}".format(current_time, self.wmem_in_use, time_gap))
            self.wmem_valid_timer[self.wmem_in_use] = 0
        # End
        if not extended_cim and address + self.system_width > self.wmem_size:
            self.logger.error("WMEM Size: {} vs Requested Address: {}".format(self.wmem_size, address + self.system_width))
            raise ValueError("Wrong Address")
        if address + self.system_width > self.extended_wmem_size:
            self.logger.error("Extended WMEM Size: {} vs Requested Address: {}".format(self.extended_wmem_size, address + self.system_width))
            raise ValueError("Wrong Address")
        data_set_size = 1 if extended_cim else self.num_wmem
        buf[:data_set_size,:self.system_width] = \
                self.wmem[self.wmem_in_use, :data_set_size, address:address+self.system_width]
        return time_gap

    def read_fmem(self, buf, bank_idx, address):
        current_time = self.manager.stats.total_cycle()
        time_gap = max(0, self.fmem_valid_timer[bank_idx] - current_time)
        if time_gap > 0:
            self.fmem_valid_timer[bank_idx] = 0
            self.logger.debug("Time {}: FMEM {} load delay occured & time_gap = {}".format(current_time, bank_idx, time_gap))
        buf[:self.system_width] = self.fmem[bank_idx, address:address+self.system_width]
        return time_gap

    def read_bmmem(self, buf, extended_cim, address):
        current_time = self.manager.stats.total_cycle()
        time_gap = max(0, self.bmmem_valid_timer[self.bmmem_in_use] - current_time)
        if time_gap > 0:
            self.bmmem_valid_timer[self.bmmem_in_use] = 0
            self.logger.debug("Time {}: BMMEM {} load delay occured & time_gap = {}".format(current_time, self.bmmem_in_use, time_gap))
        size = self.system_width if extended_cim else self.num_wmem
        buf[:size] = self.bmmem[self.bmmem_in_use, address:address + size]
        return time_gap
    
    def write_fmem(self, bank_idx, address, data):
        current_time = self.manager.stats.total_cycle()
        time_gap = max(0, self.fmem_valid_timer[bank_idx] - current_time)
        if time_gap > 0:
            self.fmem_valid_timer[bank_idx] = 0
        self.fmem[bank_idx, address:address+data.size] = data[:]
        return time_gap

    def write_dram(self, data_name, offset, data):
        if self.include_dram_write:
            write_start_time = max(self.manager.stats.total_cycle(), self.fifo_end_timer)
            write_delay = self.get_dram_write_latency(data.size)
            self.fifo_end_timer = write_start_time + write_delay
        self.dram_dict[data_name][offset:offset+data.size] = data[:]
        return 0

class TVMemoryManager(VMemoryManager): # Use dump memory file test
    def __init__(self, manager):
        super().__init__(manager)
    
    def read_dram_data(self, name, offset, size):
        dram_address = self.dram_dict[name] + offset
        return self.dram_data[dram_address:dram_address + size]

    def write_dram(self, data_name, offset, data):
        if self.include_dram_write:
            write_start_time = max(self.manager.stats.total_cycle(), self.fifo_end_timer)
            write_delay = self.get_dram_write_latency(data.size)
            self.fifo_end_timer = write_start_time + write_delay
        dram_address = self.dram_dict[data_name] + offset
        self.dram_data[dram_address:dram_address + data.size] = data[:]
        return 0

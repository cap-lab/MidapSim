from __future__ import absolute_import, division, print_function, unicode_literals

import math
import logging

import numpy as np

from acc_utils.model_utils import *
from config import cfg

class MemoryManager():
    def __init__(self, manager):
        # Set FMEM constraints
        self.manager = manager
        self.num_fmem = cfg.MIDAP.FMEM.NUM
        self.fmem_size = cfg.MIDAP.FMEM.NUM_ENTRIES
        self.fmem = np.zeros([self.num_fmem, self.fmem_size], dtype = np.float32)
        # input = 0, output = 1, else = -1
        self.system_width = cfg.MIDAP.SYSTEM_WIDTH
        # Set WMEM constraints
        self.num_wmem = cfg.MIDAP.WMEM.NUM
        self.wmem_size = cfg.MIDAP.WMEM.NUM_ENTRIES
        self.extended_wmem_size = cfg.MIDAP.WMEM.ECIM_NUM_ENTIRES
        # double buffered WMEM
        self.wmem = np.zeros([2, self.num_wmem, self.extended_wmem_size], dtype = np.float32)
        self.wmem_in_use = -1
        # BMMEM setting
        self.bmmem_size = cfg.MIDAP.BMMEM.NUM_ENTRIES
        # double buffered BMEM
        self.bmmem = np.zeros([2, self.bmmem_size], dtype = np.float32)
        self.bmmem_in_use = -1
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('debug')
        self.dram_data = None
        self.dram_dict = None

    def set_dram_info(self, dram_data, dram_dict):
        self.dram_data = dram_data
        self.dram_dict = dram_dict
    
    def sync(self):
        return 0

    def switch_wmem(self):
        self.wmem_in_use = (self.wmem_in_use + 1) % 2
        self.logger.debug("Switch_WMEM: to "+str(self.wmem_in_use))
    
    def switch_bmmem(self):
        self.bmmem_in_use = (self.bmmem_in_use + 1) % 2
        self.logger.debug("Switch_BMMEM: to "+str(self.bmmem_in_use))

    def reset_wmem(self):
        self.wmem_in_use = -1

    def load_wmem(self, wmem_idx, filter_name, filter_size = 0, wmem_offset = 0, dram_offset = 0):
        pass

    def load_fmem(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0):
        pass

    def load_bmmem(self, bias_name, bias_size):
        pass

    def read_wmem(self, buf, extended_cim, address):
        return 0

    def read_fmem(self, buf, bank_idx, address):
        return 0

    def read_bmmem(self, buf, extended_cim, address):
        return 0
    
    def write_fmem(self, bank_idx, address, data):
        return 0

    def write_dram(self, data_name, address, data):
        return 0


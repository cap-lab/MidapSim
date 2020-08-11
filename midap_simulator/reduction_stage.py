from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from acc_utils.attrdict import AttrDict
from acc_utils.errors import *
from acc_utils.model_utils import *
from config import cfg
from generic_op import ArithmeticOp, PoolOp

from .stage import Stage
from .dataflow import generate_dataflow_info


class FReductionStage(Stage):
    def initialize(self):
        self.reduction_type = None
        self.reduction_info = None
        self.reduction_buf = np.zeros(cfg.MIDAP.REDUCTION.NUM_ENTRIES)
        self.reduction_value = 0
        self.bypass_flag = True
        self.output_buf = np.zeros(self.system_width)
        self.concurrency = self.num_wmem
        self.input_buf = None
        self.logger = self.manager.logger

    def setup(self, modules):
        op = modules[0].op
        self.concurrency = self.num_wmem
        if any([isinstance(op, ArithmeticOp), isinstance(op, PoolOp), op.type == 'Depthwise']):
            self.concurrency = self.system_width
        self.reduction_type = 0 # No reduction
        self.reduction_value = 0
        if modules.num_modules > 1:
            reduction_op = modules[1].op
        else:
            reduction_op = None
        if reduction_op is None:
            pass
        elif isinstance(reduction_op, PoolOp):
            self.reduction_type = 1
            self.reduction_value = reduction_op.k_w * reduction_op.k_h
        elif reduction_op.type == 'Softmax':
            self.reduction_type = 2
        else:
            raise ValueError("Unknown Reduction operation {}".format(layer_info.main_op))

    def run(self, dataflow_info):
        info = dataflow_info
        input_buf = self.input_buf
        if self.reduction_type == 0 or info.phase in [0, 3]:
            pass
        if info.phase == 1: # Phase 1
            self.output_buf[:] = input_buf[:] # bypass
            if not info.last:
                pass
            elif self.reduction_type == 1: # update reduction buf
                out_x, out_y, filter_idx = info.out_x, info.out_y, info.out_z
                if out_x == 0 and out_y == 0:
                    self.reduction_buf[filter_idx:filter_idx + self.concurrency] = input_buf[:self.concurrency]
                else:
                    self.reduction_buf[filter_idx:filter_idx + self.concurrency] = np.add(self.reduction_buf[filter_idx:filter_idx + self.concurrency], input_buf[:self.concurrency])
                #self.logger.debug("reduction update - loc: {}, updated data: {}".format((info.out_x, info.out_y, info.out_z), self.reduction_buf[filter_idx:filter_idx+4]))
            elif self.reduction_type == 2:
                filter_idx = info.out_z
                self.reduction_buf[filter_idx: filter_idx + self.concurrency] = np.exp(input_buf[:self.concurrency])
                self.reduction_value += np.sum(self.reduction_buf[filter_idx:filter_idx + self.concurrency])
        elif info.phase == 2: # Phase 2
            self.output_buf[:self.concurrency] = np.true_divide(self.reduction_buf[info.out_z:info.out_z + self.concurrency], self.reduction_value)
        return info

class VReductionStage(Stage):
    def initialize(self):
        self.output_buf = np.zeros(self.system_width)
    def run(self, info):
        return info

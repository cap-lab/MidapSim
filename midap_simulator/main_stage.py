import numpy as np
import logging

from generic_op import ConvOp, PoolOp, ArithmeticOp, SumOp
from .stage import Stage
from .dataflow import generate_dataflow_info
from config import cfg

class FMainStage(Stage): # Cycle - level functional simulation
    def __init__(self, manager):
        super().__init__(manager)
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger("debug")
    
    def initialize(self):
        super().initialize()
        self.output_buf = np.zeros(self.system_width)
        # Add pipeline stage & required components
        self.dataflow_info_buf = [generate_dataflow_info() for _ in range(4)] 
        # Stage 1
        self.wmem_last_load = -1
        self.wbuf_mem = np.zeros((self.num_wmem, self.system_width))
        self.fbuf_mem = np.zeros(self.system_width)
        # Stage 2
        self.wbuf = np.zeros((self.num_wmem, self.system_width))
        self.fbuf = np.zeros(self.system_width * 2)
        self.broadcast_fbuf = np.zeros(self.system_width)
        # Stage 3
        self.logic_type = 0
        self.use_extended_cim = False
        self.alu_buf = np.zeros([self.num_wmem, self.system_width])
        # Stage 4
        self.adder_count = 0
        self.adder_output_buf = np.zeros(self.system_width)
        self.csatree_buf = np.zeros(self.num_wmem)
        self.accumulator_buf = np.zeros(self.system_width)
        # Stage 5
        self.activation = None
        self.add_bias = False
        self.loaded_bias = -1
        self.bias_buf = np.zeros(self.system_width)
        self.bias_output_buf = np.zeros(self.system_width)
        self.concurrency = self.num_wmem

    def setup(self, modules):
        op = modules[0].op
        self.use_extended_cim = False
        self.concurrency = self.num_wmem
        self.logic_type = 0
        if any([isinstance(op, ArithmeticOp), isinstance(op, PoolOp), op.type == 'Depthwise']):
            self.use_extended_cim = True
            self.concurrency = self.system_width
        if isinstance(op, PoolOp):
            self.logic_type = 2 if op.type.lower() == 'maxpool' else 1
        elif isinstance(op, SumOp):
            self.logic_type = 3
        self.adder_count = 0
        self.activation = op.activation
        self.add_bias = isinstance(op, ConvOp) and op.bias is not None
        self.loaded_bias = -1
        self.wmem_last_load = (-1, -1)

    def run(self, dataflow_info):
        output_dataflow_info = self.do_activator(self.dataflow_info_buf[3])
        self.dataflow_info_buf[3] = self.do_adder(self.dataflow_info_buf[2])
        self.dataflow_info_buf[2] = self.do_alu(self.dataflow_info_buf[1])
        self.dataflow_info_buf[1] = self.do_broadcast(self.dataflow_info_buf[0])
        self.dataflow_info_buf[0] = self.do_load(dataflow_info)
        return output_dataflow_info

    def do_load(self, info):
        if info.phase in [0, 2, 3]:
            return info
        fmem_row = info.fmem_row
        fmem_idx = info.fmem_idx
        wmem_row = info.wmem_row
        if fmem_row > -1:
            self.memory_controller.load_fbuf(self.fbuf_mem, fmem_idx, fmem_row)
        if wmem_row > -1 and self.wmem_last_load != (info.out_z, wmem_row):
            self.memory_controller.load_wbuf(self.wbuf_mem, wmem_row)
            self.wmem_last_load = (info.out_z, wmem_row)
        self.logger.debug("Input info: {}".format(info))
        #self.logger.debug("fbuf_mem: {}".format(self.fbuf_mem[0:6]))
        #self.logger.debug("wbuf_mem: {}".format(self.wbuf_mem[0,0:6]))
        return info

    def do_broadcast(self, info):
        if info.phase in [0, 2, 3]:
            return info
        offset = info.broadcast_offset
        delete_f = info.delete_foffset
        delete_b = info.delete_boffset
        # Shift
        self.fbuf[self.system_width:] = self.fbuf[:self.system_width]
        self.fbuf[0:self.system_width] = self.fbuf_mem[:]
        # Alignment submodule
        if offset > 0:
            self.broadcast_fbuf[:offset] = self.fbuf[-offset:]
            self.broadcast_fbuf[offset:] = self.fbuf[:self.system_width - offset]
        else:
            self.broadcast_fbuf[:] = self.fbuf[:self.system_width]
        if delete_b > 0:
            self.broadcast_fbuf[delete_b:] = np.zeros(
                self.system_width - delete_b)
        if delete_f > 0:
            self.broadcast_fbuf[:delete_f] = np.zeros(delete_f)
        # Load WMEM
        self.wbuf[:, :] = self.wbuf_mem
        return info

    def do_alu(self, info):
        if info.phase in [0, 2, 3]:
            return info
        alu_buf = self.alu_buf[0] if self.use_extended_cim else self.alu_buf
        wbuf = self.wbuf[0] if self.use_extended_cim else self.wbuf
        if self.logic_type == 0:
            alu_buf[:] = np.multiply(self.broadcast_fbuf, wbuf)
        elif self.logic_type in [1, 2]:
            alu_buf[:] = self.broadcast_fbuf[:]
        elif self.logic_type == 3:
            alu_buf[:] = np.add(self.broadcast_fbuf, wbuf)
        return info

    def do_adder(self, info):
        if info.phase in [0, 2, 3] or info.junk:
            pass
        elif not self.use_extended_cim:
            partial_sum = np.sum(self.alu_buf, axis=1)
            # #self.logger.debug("CSATree - loc: {}, partial_sum: {}".format((info.out_x, info.out_y, info.out_z), np.sum(partial_sum)))
            if info.reset:
                self.csatree_buf[:] = partial_sum[:]
            else:
                self.csatree_buf = np.add(self.csatree_buf, partial_sum)
            if info.last:
                self.adder_output_buf[0:self.num_wmem] = self.csatree_buf[:]
        else:
            # Extended CIM Logic
            alu_buf = self.alu_buf[0]
            if info.reset:
                self.accumulator_buf[:] = alu_buf[:]
                self.adder_count = 1
            elif self.logic_type in [0, 1]: # Avgpool, Depthwise
                self.accumulator_buf[:] = np.add(self.accumulator_buf, alu_buf)
                self.adder_count += 1
            elif self.logic_type == 2: # Maxpool
                self.accumulator_buf[:] = np.maximum(self.accumulator_buf, alu_buf)
            else:
                raise ValueError("Not a possible scenario")
            if info.last:
                if self.logic_type == 1:
                    self.adder_output_buf[:] = np.true_divide(self.accumulator_buf, self.adder_count)
                else:
                    self.adder_output_buf[:] = self.accumulator_buf[:]
            # #self.logger.debug("Extended CIM - loc: {}, accumulator_buf: {}".format((info.out_x, info.out_y, info.out_z), self.accumulator_buf[:4]))
        return info

    def do_activator(self, info):
        if info.phase in [0, 2, 3] or info.junk:
            pass
        elif info.last:
            if self.add_bias:
                if self.loaded_bias != info.out_z:
                    self.memory_controller.load_bbuf(self.bias_buf, info.out_z)
                    self.loaded_bias = info.out_z
                self.bias_output_buf[:] = np.add(self.adder_output_buf, self.bias_buf)
            else:
                self.bias_output_buf[:] = self.adder_output_buf[:]
            if self.activation is not None:
                if self.activation.lower() == 'leakyrelu':
                    self.output_buf[:] = np.where(self.bias_output_buf > 0, self.bias_output_buf, self.bias_output_buf * 0.01)
                elif 'relu' in self.activation.lower():
                    self.output_buf[:] = np.maximum(self.bias_output_buf, 0)
                    if self.activation.lower() == 'relu6':
                        self.output_buf[:] = np.minimum(self.output_buf, 6)
                elif self.activation.lower() == 'sigmoid':
                    self.output_buf[:] = 1 / (1 + np.exp(-self.bias_output_buf))
                else:
                    raise ValueError("Unknown acitvation {}".format(self.activation))
            else:
                self.output_buf[:] = self.bias_output_buf[:]
            if self.use_extended_cim:
                pass
                #self.logger.debug("Activator - info: {}".format(info))
                #self.logger.debug("Activator - data: {}".format(self.output_buf[0]))
        return info

class VMainStage(Stage):
    def __init__(self, manager):
        super().__init__(manager)
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger("debug")

    def initialize(self):
        super().initialize()
        self.skipped_pipeline_stages = 5
        self.buffer = np.zeros([self.num_wmem, self.system_width])
        self.output_buf = np.zeros(self.system_width)
    
    def setup(self, modules):
        op = modules[0].op
        self.use_extended_cim = False
        self.concurrency = self.num_wmem
        self.logic_type = 0
        if any([isinstance(op, ArithmeticOp), isinstance(op, PoolOp), op.type == 'Depthwise']):
            self.use_extended_cim = True
            self.concurrency = self.system_width
        if isinstance(op, PoolOp):
            self.logic_type = 2 if op.type.lower() == 'maxpool' else 1
        elif isinstance(op, SumOp):
            self.logic_type = 3
        self.add_bias = isinstance(op, ConvOp) and op.bias is not None
        self.loaded_bias = -1
        self.wmem_last_load = (-1, -1)

    def run(self, info):
        if info.phase in [0, 2, 3]:
            return info
        fmem_row = info.fmem_row
        fmem_idx = info.fmem_idx
        wmem_row = info.wmem_row
        if fmem_row > -1:
            self.memory_controller.load_fbuf(self.buffer[0], fmem_idx, fmem_row)
        if wmem_row > -1 and self.wmem_last_load != (info.out_z, wmem_row):
            self.memory_controller.load_wbuf(self.buffer, wmem_row)
            self.wmem_last_load = (info.out_z, wmem_row)
        if info.last and self.add_bias and self.loaded_bias != info.out_z:
            self.memory_controller.load_bbuf(self.buffer[0], 0)
            self.loaded_bias = info.out_z
        return info



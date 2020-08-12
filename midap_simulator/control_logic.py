from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import logging.config

import numpy as np

from acc_utils.attrdict import AttrDict
from acc_utils.model_utils import *
from config import cfg
from generic_op import ConvOp, ConvPoolOpBase, PoolOp, ArithmeticOp
from .dataflow import generate_dataflow_info

def get_control_logic(manager, simulation_level):
    if simulation_level == -1:
        return ControlLogic(manager)
    if simulation_level == 0:
        return ControlLogicLv0(manager)
    if simulation_level == 1:
        return ControlLogicLv1(manager)
    if simulation_level == 2:
        return ControlLogicLv2(manager)

class RunningInfo():
    def __init__(self, x = -2, last_filter = False):
        self.x = x
        self.last_filter = last_filter

class ControlLogic():
    def __init__(self, manager):
        # Initialize System Configuration
        self.manager = manager
        self.memory_controller = manager.memory_controller
        self.system_width = cfg.MIDAP.SYSTEM_WIDTH
        self.num_wmem = cfg.MIDAP.WMEM.NUM
        self.num_fmem = cfg.MIDAP.FMEM.NUM
        self.concurrency = self.num_wmem
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger()
    
    def setup(self, layer_info):
        control_info = layer_info.control_info
        self.input_tensor = layer_info.input[0]
        self.input_mapping = control_info.get_input_mapping(self.input_tensor.name)
        self.head_y = control_info.behavior_info.min_y
        self.tail_y = control_info.behavior_info.max_y
        self.num_gemm_rows = control_info.behavior_info.num_gemm_rows
        self.shape = self.input_tensor.shape
        self.modules = layer_info.modules
        self.mm = self.modules[0]
        self.main_op = self.mm.op
        self.output_shape = self.mm.output[0].orig_shape
        if isinstance(self.main_op, PoolOp) and self.main_op.global_pooling == 1:
            self.main_op.k_w, self.main_op.k_h = self.shape[:-1]
        self.rm = None if self.modules.num_modules <= 1 else self.modules[1]
        self.reduction_op = None if self.rm is None else self.rm.op
        self.skipped_cycles = 0
        self.output_loc = (0, 0, 0)
        self.input_pivot_idx = 0 # (From) Which input fragment is on FMEM
        self.concurrency = self.num_wmem
        if any([isinstance(self.main_op, ArithmeticOp), isinstance(self.main_op, PoolOp), self.main_op.type == 'Depthwise']):
            self.concurrency = self.system_width
        self.generator = self.default_generator()
        self.save_last_run = None
        self.loaded_fmem = (-1, -1)
    
    def set_generator(self, head_x, tail_x, pivot_idx, last):
        self.input_pivot_idx = pivot_idx
        self.logger.info("Set Generator: from x={} to {}".format(head_x, tail_x))
        if self.main_op.type.lower() == 'gemm':
            self.generator = iter(self.gemm_generator(head_x, tail_x, last))
        elif isinstance(self.main_op, ConvPoolOpBase):
            self.generator = iter(self.convpool_generator(head_x, tail_x, last))
        elif isinstance(self.main_op, ArithmeticOp):
            self.generator = iter(self.arithmetic_generator(head_x, tail_x, last))
        else:
            raise RuntimeError("Unexpected main operator: {}".format(self.main_op))
    
    def set_finish_generator(self):
        self.generator = self.finish_generator()
    
    def set_next(self, last):
        if self.skipped_cycles > 0:
            self.manager.stats.increase_cycle(self.skipped_cycles)
            self.skipped_cycles = 0
        return self.memory_controller.set_next(last)

    def generate(self, dataflow, last_filter = False):
        running_info = RunningInfo(x = self.output_loc[0], last_filter = last_filter)
        simulated_cycle = 1 + self.skipped_cycles
        self.skipped_cycles = 0
        self.loaded_fmem = (dataflow.fmem_idx, dataflow.fmem_row)
        return (dataflow, [running_info, simulated_cycle])

    def gemm_generator(self, head_x, tail_x, last):
        last_filter = False
        while not last_filter:
            last_filter, filter_idx = self.set_next(last)
            worker = self.gemm_worker(head_x, tail_x, filter_idx)
            for dataflow in worker:
                yield self.generate(dataflow, last_filter)

    def convpool_generator(self, head_x, tail_x, last):
        # generate (dataflow, [running_info, simulated_cycle]) #
        main_op = self.main_op
        s= main_op.stride
        last_filter = False
        while not last_filter:
            last_filter, filter_idx = self.set_next(last)
            for x in range(head_x, tail_x + 1, s):
                for y in range(self.head_y, self.tail_y + 1, s):
                    worker = self.convpool_worker(x, y, filter_idx)
                    for dataflow in worker:
                        yield self.generate(dataflow, last_filter)
        self.generator = self.default_generator()

    def arithmetic_generator(self, head_x, tail_x, last):
        # generate (dataflow, [running_info, simulated_cycle]) #
        for x in range(head_x, tail_x + 1):
            _, _ = self.set_next(last and x == tail_x)
            for dataflow in self.arithmetic_worker(x):
                yield self.generate(dataflow, True)
        self.generator = self.default_generator()
    
    def finish_generator(self):
        if self.reduction_op is not None:
            for dataflow in self.reduction_worker():
                yield self.generate(dataflow, False)
        dataflow = generate_dataflow_info(phase = 3)
        yield self.generate(dataflow, False)
        while True:
            dataflow = generate_dataflow_info()
            yield self.generate(dataflow, False)

    def convpool_worker(self, x, y, filter_idx): # generate dataflow
        # Generate Output Location
        main_op = self.main_op
        pivot_x = x + main_op.pad_l
        pivot_y = y + main_op.pad_t
        s = main_op.stride
        if pivot_x % s != 0 or pivot_y % s != 0:
            raise ValueError("Wrong conv location: ({}, {}) for main_op : {}".format(x, y, main_op))
        out_x, out_y = (pivot_x // s , pivot_y // s)
        self.output_loc = (out_x, out_y, filter_idx)
        worker = self.default_worker
        if isinstance(main_op, PoolOp) or main_op.type.lower() == 'depthwise':
            worker = self.depthwise_worker
        elif isinstance(main_op, ConvOp):
            worker = self.conv_yz_worker if self.input_tensor.mapping_type == 'default' else self.conv_z_worker
        for dataflow in self.working(worker(x, y, filter_idx)):
            yield dataflow
    
    def working(self, worker):
        for dataflow in worker:
            yield dataflow

    def default_generator(self, **kwargs):
        #self.logger.debug("default_generator is called")
        if not isinstance(self, ControlLogicLv1):
            yield self.generate(generate_dataflow_info())
        self.generator = self.default_generator()
    
    def default_worker(self, **kwargs):
        #self.logger.debug("default_worker is called")
        yield generate_dataflow_info()

    def gemm_worker(self, head_x, tail_x, filter_idx):
        self.logger.warning("gemm_worker is not implemented yet")
        yield generate_dataflow_info()
        self.generator = self.default_generator()

    def conv_z_worker(self, in_x, in_y, *args, **kwargs):
        self.logger.warning("conv_z_worker is not implemented yet")
        yield generate_dataflow_info()

    def conv_yz_worker(self, in_x, in_y, *args, **kwargs):
        self.logger.warning("conv_yz_worker is not implemented yet")
        yield generate_dataflow_info()

    def depthwise_worker(self, in_x, in_y, filter_idx):
        self.logger.warning("depthwise_worker is not implemented yet")
        yield generate_dataflow_info()

    def arithmetic_worker(self, x, *args, **kwargs): # generate dataflow
        self.logger.warning("arithmetic_worker is not implemented yet")
        yield generate_dataflow_info()
    
    def reduction_worker(self):
        self.logger.warning("reduction_worker is not implemented yet")
        yield generate_dataflow_info()

    def get_fmem_info(self, x):
        input_mapping = self.input_mapping
        for fmem_idx, head, tail in input_mapping[self.input_pivot_idx:]:
            if head <= x and x < tail:
                return fmem_idx, x - head

    def sync(self):
        self.manager.stats.increase_cycle(self.skipped_cycles)

class ControlLogicLv0(ControlLogic):
    def __init__(self, manager):
        super().__init__(manager)
    
    def gemm_worker(self, head_x, tail_x, filter_idx):
        out_w, out_h, _ = self.output_shape
        for x in range(head_x, tail_x + 1):
            fmem_idx, effective_x = self.get_fmem_info(x)
            w = x // out_h
            h = x % out_h
            self.output_loc = (w, h, filter_idx)
            reset = True
            for row_offset in range(self.num_gemm_rows):
                last = row_offset == self.num_gemm_rows - 1
                row = effective_x * self.num_gemm_rows + row_offset
                yield generate_dataflow_info(
                        phase = 1,
                        loc=(w, h),
                        filter_idx=filter_idx,
                        fmem_idx=fmem_idx,
                        fmem_row=row,
                        wmem_row=row_offset,
                        reset=reset,
                        last=last,
                        )
                reset = False

    def depthwise_worker(self, in_x, in_y, filter_idx):
        main_op = self.main_op
        load_weight = False
        dilation = 1
        if isinstance(main_op, ConvOp):
            load_weight = True
            dilation = main_op.dilation
        k_h, k_w = main_op.k_h, main_op.k_w
        in_w, in_h, in_c = self.input_tensor.shape
        row = filter_idx // self.system_width
        _ , real_h, real_c = self.input_tensor.orig_shape
        yz_plane_size = real_h * real_c
        reset = True
        dataflow_info = None
        for kx in range(k_w):
            x = in_x + kx * dilation
            if x < 0 or x >= in_w:
                continue
            for ky in range(k_h):
                y = in_y + ky * dilation
                if y < 0 or y >= in_h:
                    continue
                if not self.input_tensor.valid(x, y):
                    if not cfg.MIDAP.EFFICENT_LOGIC:
                        self.skipped_cycles += 1
                    continue
                mapped_x, mapped_y, _ = self.input_tensor.get_loc((x, y, 0))
                fmem_idx, effective_x = self.get_fmem_info(mapped_x)
                # Error Checking
                fmem_row = (effective_x * yz_plane_size +
                            mapped_y * real_c) // self.system_width + row
                wmem_row = -1
                if load_weight:
                    wmem_row = (kx * k_h * real_c + ky *
                                real_c) // self.system_width + row
                if dataflow_info is not None:
                    yield dataflow_info
                dataflow_info = generate_dataflow_info(
                        phase = 1,
                        loc=self.output_loc,
                        fmem_idx=fmem_idx,
                        fmem_row=fmem_row,
                        wmem_row=wmem_row,
                        reset=reset
                        )
                reset = False
        if dataflow_info is not None:
            dataflow_info.last = True
            yield dataflow_info

    def conv_z_worker(self, in_x, in_y, *args, **kwargs):
        main_op = self.main_op
        dilation = main_op.dilation
        k_h, k_w = main_op.k_h, main_op.k_w
        real_w, real_h, real_c = self.input_tensor.orig_shape
        row_per_channel = real_c // self.system_width
        yz_plane_rows = real_h * row_per_channel
        in_w, in_h, in_c = self.input_tensor.shape
        reset = True
        dataflow_info = None
        for kx in range(k_w):
            x = in_x + kx * dilation
            if x < 0 or x >= in_w:
                continue
            for ky in range(k_h):
                y = in_y + ky * dilation
                if y < 0 or y >= in_h:
                    continue
                if not self.input_tensor.valid(x, y):
                    if not cfg.MIDAP.EFFICENT_LOGIC:
                        stats.increase_cycle()
                    continue
                mapped_x, mapped_y, _ = self.input_tensor.get_loc((x, y, 0))
                fmem_idx, effective_x = self.get_fmem_info(mapped_x)
                fmem_start_row = effective_x * yz_plane_rows + mapped_y * row_per_channel
                wmem_start_row = kx * k_h * row_per_channel + ky * row_per_channel
                for row in range(row_per_channel):
                    if dataflow_info is not None:
                        yield dataflow_info
                    dataflow_info = generate_dataflow_info(
                            phase = 1,
                            loc=self.output_loc,
                            fmem_idx=fmem_idx,
                            fmem_row=fmem_start_row + row,
                            wmem_row=wmem_start_row + row,
                            reset=reset,
                            )
                    reset = False
        if dataflow_info is not None:
            dataflow_info.last = True
        yield dataflow_info

    def conv_yz_worker(self, in_x, in_y, *args, **kwargs): # default input tensor type
        main_op = self.main_op
        k_h, k_w = main_op.k_h, main_op.k_w
        in_w, in_h, in_c = self.input_tensor.orig_shape
        row_per_kernel_yz = div_ceil(in_c * k_h, self.system_width)
        yz_plane_size = in_h * in_c
        reset = True
        for kx in range(k_w): #ZY-plane wise multiplication w/ alignment submodule
            x = in_x + kx
            if x < 0 or x >= in_w:
                continue
            last_x = (x == in_w - 1) or (kx == k_w - 1)
            fmem_idx, effective_x = self.get_fmem_info(x)
            # wmem configuration
            start_ky = max(0, -in_y)
            end_ky = min(k_h, in_h - in_y)
            wmem_start_row = kx * row_per_kernel_yz
            wmem_start_row += start_ky * in_c // self.system_width  # when pad skipped
            wmem_offset = (start_ky * in_c) % self.system_width
            fmem_start_address = effective_x * yz_plane_size + (in_y + start_ky) * in_c
            fmem_offset = fmem_start_address % self.system_width
            fmem_start_row = fmem_start_address // self.system_width
            fmem_last_row = div_ceil(
                effective_x * yz_plane_size + (in_y + end_ky) * in_c, self.system_width) - 1
            bubble = wmem_offset - fmem_offset
            num_rows = div_ceil(end_ky * in_c, self.system_width) - \
                (start_ky * in_c // self.system_width)
            self.save_last_run = (fmem_start_row, wmem_start_row, num_rows, fmem_offset,
                                  wmem_offset, end_ky, k_h, row_per_kernel_yz, fmem_last_row, 0)
            for row in range(num_rows):
                if bubble < 0:
                    if self.loaded_fmem != (fmem_idx, fmem_start_row):
                        yield generate_dataflow_info(
                                phase = 1,
                                loc=self.output_loc,
                                fmem_idx=fmem_idx,
                                fmem_row=fmem_start_row,
                                reset=reset,
                                junk=True
                                )
                    bubble = self.system_width + bubble
                    fmem_start_row += 1
                fmem_row = fmem_last_row if fmem_last_row < fmem_start_row + \
                    row else fmem_start_row + row
                wmem_row = wmem_start_row + row
                cnt = 0
                last = False
                if row == num_rows - 1:
                    cnt = (end_ky * in_c) % self.system_width
                    last = last_x
                yield generate_dataflow_info(
                        phase = 1,
                        loc=self.output_loc,
                        fmem_idx=fmem_idx,
                        fmem_row=fmem_row,
                        wmem_row=wmem_row,
                        broadcast_offset=bubble,
                        delete_foffset=wmem_offset,
                        delete_boffset=cnt,
                        reset=reset,
                        last=last,
                        )
                wmem_offset = 0
                reset = False

    def arithmetic_worker(self, x, *args, **kwargs):
        main_op = self.main_op
        _, in_h, in_c = self.shape
        in_x, _, _ = self.input_tensor.get_loc((x,0,0))
        fmem_idx, effective_x = self.get_fmem_info(x)
        row_per_channel = in_c // self.system_width
        row_per_yz_plane = in_h * in_c // self.system_width
        for y in range(in_h):
            fmem_start_row = self.input_tensor.get_address((effective_x, y, 0)) // self.system_width
            for row in range(row_per_channel): # I'm not sure that input virtualization can be applied on arithmetic operators
                z = row * self.system_width
                wmem_row = row_per_channel * y + row if not main_op.broadcast else row
                self.output_loc = (x, y, z)
                yield generate_dataflow_info(
                        phase = 1,
                        loc=self.output_loc,
                        reset=True,
                        last=True,
                        fmem_idx=fmem_idx,
                        fmem_row=fmem_start_row + row,
                        wmem_row=wmem_row
                        )

    def reduction_worker(self):
        filter_idx = 0
        while filter_idx < self.output_shape[-1]:
            self.output_loc = (0, 0, filter_idx)
            yield generate_dataflow_info(
                    phase = 2,
                    loc = self.output_loc,
                    last = True
                    )
            filter_idx += self.concurrency


class ControlLogicLv1(ControlLogicLv0): # Only generate 1 signal per output pixel
    def working(self, worker):
        for dataflow in worker:
            if dataflow.phase in [0, 1, 2] and not dataflow.last:
                self.skipped_cycles += 1
                if dataflow.phase == 1: # Update statistics
                    if dataflow.fmem_row >= 0:
                        self.manager.stats.fmem_read()
                    if dataflow.wmem_row >= 0:
                        self.manager.stats.wmem_read()
            else:
                yield dataflow

class ControlLogicLv2(ControlLogicLv1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_lambda_x = lambda x: True
        self.skip_write = not cfg.DRAM.INCLUDE_DRAM_WRITE and cfg.DRAM.COMM_TYPE == 'VIRTUAL'

    def setup(self, layer_info):
        super().setup(layer_info)
        self.output_mapping = layer_info.control_info.fmem_info.output_mapping
        self.main_output = layer_info.modules[0].output
        self.write_lambda_x = self.get_write_lambda_x()

    def set_generator(self, *args, **kwargs):
        super().set_generator(*args, **kwargs)
        self.read_crit_idx = -1

    def get_and_set_read_crit_x(self):
        if self.read_crit_idx == -1:
            read_crit_x = self.input_mapping[self.input_pivot_idx][0]
        elif self.read_crit_idx >= len(self.input_mapping):
            read_crit_x = self.input_mapping[-1][2]
        else:
            read_crit_x = self.input_mapping[self.read_crit_idx][2]
        self.read_crit_idx += 1
        return read_crit_x
    
    def get_write_lambda_x(self):
        if len(self.main_output) > 1:
            return lambda x: True
        out = self.main_output[0]
        if out.virtual:
            return lambda x: False
        if out.write_on_dram:
            return lambda x: True
        crit_x = self.output_mapping[out.name].write_on_dram_pivot
        if crit_x >= out.shape[0]:
            return lambda x: True
        crit_x -= out.offset[0]
        if out.reverse_write:
            crit_x = out.orig_shape[0] - crit_x - 1
        if self.main_op.type.lower() == 'gemm':
            _, out_h, _ = out.orig_shape
            crit_x = out_h * crit_x
        elif isinstance(self.main_op, ConvOp):
            crit_x *= self.main_op.stride
            crit_x -= self.main_op.pad_l
        if out.reverse_write:
            return lambda x: x <= crit_x
        else:
            return lambda x: x >= crit_x

    def gemm_worker(self, head_x, tail_x, filter_idx):
        read_crit_x = tail_x + 1
        if filter_idx == 0:
            read_crit_x = self.get_and_set_read_crit_x()
        out_w, out_h, _ = self.output_shape
        skipped_mem_read = 0
        first_wmem_read = True
        for x in range(head_x, tail_x + 1):
            read_flag = False
            write_flag = self.write_lambda_x(x)
            if x == read_crit_x:
                read_flag = True
                read_crit_x = self.get_and_set_read_crit_x()
            yield_flag = read_flag or (not self.skip_write and write_flag) or first_wmem_read
            first_wmem_read = False
            if not yield_flag:
                self.skipped_cycles += self.num_gemm_rows
                skipped_mem_read += self.num_gemm_rows
                if write_flag:
                    self.manager.stats.dram_write()
            else: 
                self.skipped_cycles += self.num_gemm_rows - 1
                skipped_mem_read += self.num_gemm_rows - 1
                fmem_idx, effective_x = self.get_fmem_info(x)
                w = x // out_h
                h = x % out_h
                self.output_loc = (w, h, filter_idx)
                fmem_row = effective_x * self.num_gemm_rows
                yield generate_dataflow_info(
                        phase = 1,
                        loc = self.output_loc,
                        fmem_idx = fmem_idx,
                        fmem_row = fmem_row,
                        wmem_row = 0,
                        reset = True,
                        last = True,
                        )
        self.manager.stats.fmem_read(skipped_mem_read)
        self.manager.stats.wmem_read(skipped_mem_read)
    
    def convpool_generator(self, head_x, tail_x, last):
        main_op = self.main_op
        s= main_op.stride
        last_filter = False
        read_crit_x = self.get_and_set_read_crit_x()
        yz_worker_flag = self.input_tensor.mapping_type == 'default'
        while not last_filter:
            last_filter, filter_idx = self.set_next(last)
            first_wmem_read = True
            for x in range(head_x, tail_x + 1, s):
                read_flag = False
                write_flag = self.write_lambda_x(x)
                orig_x, _, _ = self.input_tensor.get_loc((min(self.input_tensor.shape[0], x + main_op.k_w) - 1, 0, 0))
                if orig_x >= read_crit_x:
                    read_flag = True
                    read_crit_x = self.get_and_set_read_crit_x()
                for y in range(self.head_y, self.tail_y + 1, s):
                    if yz_worker_flag or read_flag or (not self.skip_write and write_flag) or first_wmem_read:
                        self.logger.debug("LV2 Logic generate dataflow for loc {}, flags = {}".format((x, y, filter_idx), (read_flag, write_flag, first_wmem_read)))
                        worker = self.convpool_worker(x, y, filter_idx)
                        for dataflow in worker:
                            yield self.generate(dataflow, last_filter)
                        read_flag = False
                        first_wmem_read = False
                    else:
                        self.update_stats(x, y)
                        if write_flag:
                            self.manager.stats.dram_write()
        self.generator = self.default_generator()

    def update_stats(self, x, y, include_last = True):
        main_op = self.main_op
        access_count = 0
        wmem_factor = 1
        if isinstance(main_op, ArithmeticOp) or main_op.type.lower() == 'gemm':
            raise ValueError("main operation {} must not call this function".format(main_op))
        elif isinstance(main_op, ConvPoolOpBase) and self.input_tensor.mapping_type != 'default':
            it = self.input_tensor
            wmem_factor = 0
            read_factor = 1
            if isinstance(main_op, ConvOp):
                read_factor = it.shape[-1] // self.system_width
                wmem_factor = 1
            if main_op.type.lower() == 'depthwise':
                read_factor = 1
            min_x, max_x = max(x, 0), min(x + main_op.k_w - 1, it.shape[0] - 1)
            min_y, max_y = max(y, 0), min(y + main_op.k_h - 1, it.shape[1] - 1)
            if self.input_tensor.mapping_type == 'valid':
                it = self.input_tensor
                scale_x, scale_y, _ = it.scale
                x_access_count = max_x // scale_x - (min_x - 1) // scale_x
                y_access_count = max_y // scale_y - (min_y - 1) // scale_y
            else:
                x_access_count = max_x - min_x + 1
                y_access_count = max_y - min_y + 1
            access_count = x_access_count * y_access_count * read_factor
        elif all([isinstance(main_op, ConvOp), self.input_tensor.mapping_type == 'default']):
            main_op = self.main_op
            k_h, k_w = main_op.k_h, main_op.k_w
            in_w, in_h, in_c = self.input_tensor.orig_shape
            row_per_kernel_yz = div_ceil(in_c * k_h, self.system_width)
            yz_plane_size = in_h * in_c
            in_x, in_y = x, y
            for kx in range(k_w): #ZY-plane wise multiplication w/ alignment submodule
                x = in_x + kx
                if x < 0 or x >= in_w:
                    continue
                last_x = (x == in_w - 1) or (kx == k_w - 1)
                fmem_idx, effective_x = self.get_fmem_info(x)
                # wmem configuration
                start_ky = max(0, -in_y)
                end_ky = min(k_h, in_h - in_y)
                wmem_offset = (start_ky * in_c) % self.system_width
                fmem_start_address = effective_x * yz_plane_size + (in_y + start_ky) * in_c
                fmem_offset = fmem_start_address % self.system_width
                bubble = wmem_offset - fmem_offset
                access_count += div_ceil(end_ky * in_c, self.system_width) - (start_ky * in_c // self.system_width)
                if bubble < 0:
                    access_count += 1
        else:
            raise ValueError("Not Classified main operation {}".format(main_op))
        self.logger.debug("Computation time for ({},{}) is estimated as {} cycles".format(x, y, access_count))
        if not include_last:
            access_count -= 1
        self.manager.stats.fmem_read(access_count)
        self.manager.stats.wmem_read(access_count * wmem_factor)
        self.skipped_cycles += access_count
    
    def depthwise_worker(self, in_x, in_y, filter_idx):
        main_op = self.main_op
        orig_x, _, _ = self.input_tensor.get_loc((min(self.input_tensor.shape[0], in_x + main_op.k_w) - 1, 0, 0))
        fmem_idx, _ = self.get_fmem_info(orig_x)
        yield generate_dataflow_info(
                phase = 1,
                loc = self.output_loc,
                fmem_idx = fmem_idx,
                fmem_row = 0,
                wmem_row = 0 if isinstance(self.main_op, ConvOp) else -1,
                reset = True,
                last = True
                )
        self.update_stats(in_x, in_y, False)
    
    def conv_z_worker(self, in_x, in_y, filter_idx):
        main_op = self.main_op
        orig_x, _, _ = self.input_tensor.get_loc((min(self.input_tensor.shape[0], in_x + main_op.k_w) - 1, 0, 0))
        fmem_idx, _ = self.get_fmem_info(orig_x)
        yield generate_dataflow_info(
                phase = 1,
                loc = self.output_loc,
                fmem_idx = fmem_idx,
                fmem_row = 0,
                wmem_row = 0,
                reset = True,
                last = True
                )
        self.update_stats(in_x, in_y, False)

    def arithmetic_worker(self, x, *args, **kwargs):
        main_op = self.main_op
        _, in_h, in_c = self.shape
        in_x, _, _ = self.input_tensor.get_loc((x,0,0))
        fmem_idx, effective_x = self.get_fmem_info(x)
        row_per_channel = in_c // self.system_width
        access_count = row_per_channel * in_h - 1
        yield generate_dataflow_info(
                phase = 1,
                loc=self.output_loc,
                fmem_idx=fmem_idx,
                fmem_row=0,
                wmem_row=0,
                reset=True,
                last=True,
                )
        self.skipped_cycles += access_count
        self.manager.stats.fmem_read(access_count)
        self.manager.stats.wmem_read(access_count)


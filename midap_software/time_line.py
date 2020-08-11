from __future__ import division, print_function

import copy
import heapq
import itertools
import logging
import sys
from math import ceil

from acc_utils.attrdict import AttrDict
from acc_utils.model_utils import div_ceil
from config import cfg
from generic_op import ArithmeticOp, ConvOp, PoolOp
from logger import init_logger

from .static_estimator import estimator


class DRAM(object):
    def __init__(self, layer):
        self.layer = layer
        self.fmem_request = []
        self.wmem_request = []

        self.timeline = []
        self.valid_fmem = {}
        self.valid_wmem = {}

        self.num_cim = cfg.MIDAP.WMEM.NUM

        self.dram_constants = [cfg.DRAM.CAS, cfg.DRAM.PAGE_DELAY, cfg.DRAM.REFRESH_DELAY]
        self.dram_offsets = [cfg.DRAM.PAGE_OFFSET, cfg.DRAM.RESET_OFFSET, cfg.DRAM.RESET_PERIOD]
        self.dram_offsets = [cfg.DRAM.FREQUENCY * i for i in self.dram_offsets]
        self.dram_page_size = cfg.DRAM.PAGE_SIZE

        self.bandwidth = cfg.SYSTEM.BANDWIDTH * 1000 // cfg.SYSTEM.FREQUENCY
        self.time = 0

        self.wcounter = itertools.count(1)
        self.req_counter = itertools.count(1)
        self._set_unit_time(layer)

    def _set_unit_time(self, layer):
        op = layer.main_op

        # fmem
        # input_size = layer.get_input_size()
        input_size_per_bank = layer.get_fsize_per_bank()
        last_input_size = layer.get_rest_fsize()
        # print(input_size_per_bank, last_input_size, file=sys.stderr)

        # 2.2012x - 90.363
        self.unit_size = AttrDict()
        typical_size = input_size_per_bank
        last_size = last_input_size if last_input_size != 0 else typical_size
        # 0: whole bank 1: partial bank
        self.unit_size['fmem'] = [typical_size, last_size]

        self.unit_time = AttrDict()
        typical_time = div_ceil(input_size_per_bank, self.bandwidth)
        last_time = div_ceil(last_input_size, self.bandwidth) if last_input_size != 0 else typical_time
        # 0: whole bank 1: partial bank
        self.unit_time['fmem'] = [typical_time, last_time]

        # wmem
        if isinstance(op, ConvOp) and op.type == 'Depthwise':
            weight_size = op.weight.size
            self.unit_time['wmem'] = div_ceil(weight_size, self.bandwidth)
            self.unit_size['wmem'] = weight_size
        elif isinstance(op, ConvOp):
            # weight_size = op.orig_weight_size
            weight_size = op.weight.size
            one_filter_size = self.layer.get_filter_size()
            filter_size = one_filter_size * self.num_cim
            self.unit_time['wmem'] = div_ceil(filter_size, self.bandwidth)
            self.unit_size['wmem'] = filter_size
        else:
            self.unit_time['wmem'] = 0
            self.unit_size['wmem'] = 0

        # print('[ Input Size ]', input_size, '[ Size / Bank ]', input_size_per_bank, '[ Mem Time ]', self.unit_time, file=sys.stderr)

    def _calc_fmem_load_cycle(self, num_bank, is_last=False):
        if is_last:
            cycle = (num_bank - 1) * \
                self.unit_time.fmem[0] + self.unit_time.fmem[1]
        else:
            cycle = num_bank * self.unit_time.fmem[0]
        return cycle

    def _calc_fmem_load_size(self, num_bank, is_last=False):
        if is_last:
            size = (num_bank - 1) * self.unit_size.fmem[0] + self.unit_size.fmem[1]
        else:
            size = num_bank * self.unit_size.fmem[0]
        return size

    def request_fmem(self, current_time, fragments, is_last=False):
        self._update_queue(current_time)
        for f in fragments:
            data_cycle = self._calc_fmem_load_cycle(1, (is_last if f == fragments[-1] else False))
            size = data_cycle * self.bandwidth
            req_count = next(self.req_counter)
            req_start_time = int(current_time + self.get_dram_latency(size))
            start_time = max(self.fmem_request[-1][0], req_start_time) if self.fmem_request else req_start_time
            load_request = (start_time, 2, req_count, tuple(f), data_cycle)
            # heapq.heappush(self.fmem_request, load_request)
            self.fmem_request.append(load_request)
            heapq.heappush(self.timeline, load_request)
        # print('[ Request FMEM ]', current_time, fragments, self.timeline, file=sys.stderr)

    def write(self, current_time, size, num):
        self._update_queue(current_time)
        req_count = next(self.req_counter)
        time = div_ceil(size, self.bandwidth) * num
        req_start_time = current_time - time
        start_time = max(self.fmem_request[-1][0], req_start_time) if self.fmem_request else req_start_time
        req = (start_time, 2, req_count, -1, time)
        # heapq.heappush(self.fmem_request, req)
        self.fmem_request.append(req)
        heapq.heappush(self.timeline, req)

    def get_dram_latency(self, size):
        import math
        cas, pg_dly, ref_dly = self.dram_constants
        _, _, rst_prd = self.dram_offsets  # Deprecated
        predict = cas + pg_dly * (size // self.dram_page_size) + ref_dly * math.ceil(size / rst_prd)
        return predict

    def request_wmem(self, current_time):
        idx = next(self.wcounter)
        # print(self.layer.name, self.layer.is_weight_in_wmem, idx, div_ceil(self.layer.main_op.weight.shape[0], cfg.MIDAP.WMEM.NUM))
        if self.layer.is_weight_in_wmem and idx > div_ceil(self.layer.main_op.weight.shape[0], cfg.MIDAP.WMEM.NUM):
            # print(self.layer.name, idx, self.layer.main_op.weight.shape[0] // cfg.MIDAP.WMEM.NUM)
            return idx

        if self.time > current_time:
            self.time = current_time
        self._update_queue(current_time)
        req_count = next(self.req_counter)

        one_filter_size = self.layer.get_filter_size()
        filter_size = one_filter_size * self.num_cim
        load_request = (int(current_time + self.get_dram_latency(filter_size)), 1, req_count, idx, self.unit_time.wmem)
        self.wmem_request.append(load_request)
        heapq.heappush(self.timeline, load_request)
        # print('[ {:,} | Request WMEM ]'.format(current_time), idx, self.timeline, file=sys.stderr)
        return idx

    def _update_queue(self, current_time):
        time = self.time

        while time < current_time:
            if not self.timeline:
                break

            v = self.timeline[0]
            s, req_type, req_id, f, t = v
            start_time = max(time, s)
            time = start_time + t

            if req_type == 2:  # fmem request and write
                next_w_req = self.wmem_request[0] if self.wmem_request else None
                if (not next_w_req or next_w_req[0] > current_time) and time > current_time:
                    self.timeline[0] = (s, 2, req_id, f, t - max(current_time - start_time, 0))
                    time = current_time
                    break
                elif next_w_req and time > next_w_req[0]:
                    del self.timeline[0]
                    del_req = None
                    for req in self.timeline:
                        if req[3] == next_w_req[3]:
                            del_req = req
                            break
                    self.timeline.remove(del_req)
                    self.timeline.append(tuple([s, *next_w_req[1:]]))
                    self.timeline.append((s, 2, req_id, f, t - max(next_w_req[0] - start_time, 0)))
                    heapq.heapify(self.timeline)
                    time = start_time + max(next_w_req[0] - start_time, 0)
                else:
                    if f != -1:
                        self.valid_fmem[f] = time
                    assert self.fmem_request[0][2] == self.timeline[0][2]
                    del self.fmem_request[0]
                    heapq.heappop(self.timeline)
            elif req_type == 1:  # wmem request
                if time > current_time:
                    self.timeline[0] = (s, 1, req_id, f, t - max(current_time - start_time, 0))
                    time = current_time
                    break
                else:
                    assert self.wmem_request[0][2] == self.timeline[0][2]
                    self.valid_wmem[f] = time
                    del self.wmem_request[0]
                    heapq.heappop(self.timeline)
        self.time = time

    def get_fmem_delay_time(self, current_time, fragment):
        self._update_queue(current_time)

        f = tuple(fragment[1])
        flag = True
        for f_req in self.fmem_request:
            if f == f_req[3]:
                flag = False
                break
        if flag:
            return 0

        copy_dram = self.copy()
        copy_dram._update_queue(sys.maxsize)
        end_time = copy_dram.valid_fmem[f]
        copy_dram.delete()
        return max(end_time - current_time, 0)

    def get_wmem_delay_time(self, current_time, filter_num):
        layer = self.layer
        if isinstance(layer.main_op, ArithmeticOp) or isinstance(layer.main_op, PoolOp):
            return 0

        self._update_queue(current_time)

        flag = True
        for w_req in self.wmem_request:
            if filter_num == w_req[3]:
                flag = False
                break
        if flag:
            return 0

        copy_dram = self.copy()
        copy_dram._update_queue(sys.maxsize)
        end_time = copy_dram.valid_wmem[filter_num]
        copy_dram.delete()
        return max(end_time - current_time, 0)

    def copy(self):
        copy_dram = copy.copy(self)
        copy_dram.time = copy.copy(self.time)
        copy_dram.wcounter = copy.copy(self.wcounter)
        copy_dram.req_counter = copy.copy(self.req_counter)
        copy_dram.timeline = copy.copy(self.timeline)
        copy_dram.fmem_request = copy.copy(self.fmem_request)
        copy_dram.wmem_request = copy.copy(self.wmem_request)
        copy_dram.valid_wmem = copy.copy(self.valid_wmem)
        copy_dram.valid_fmem = copy.copy(self.valid_fmem)
        return copy_dram

    def delete(self):
        del self.timeline
        del self.time, self.wcounter, self.req_counter
        del self.fmem_request, self.wmem_request
        del self.valid_fmem, self.valid_wmem
        del self


class CIM(object):
    def __init__(self, layer, dram=None):
        self.layer = layer
        self.num_cim = cfg.MIDAP.WMEM.NUM
        self.blocking_time = 0

        self.logger = init_logger('CIM', logging.INFO)

        self._set_unit_time(layer)
        self.wcounter = itertools.count(1)

    def _set_unit_time(self, layer):
        self.unit_time = AttrDict()

        op = layer.main_op
        if isinstance(op, ConvOp) and op.type == 'Depthwise':
            self.filter_num = 1
        elif isinstance(op, ConvOp):
            weight_size = op.weight.size
            one_filter_size = self.layer.get_filter_size()
            filter_size = one_filter_size * self.num_cim
            self.filter_num = int(div_ceil(weight_size, filter_size))
        else:
            self.filter_num = 1

        input_size = layer.get_input_size()
        input_size_per_bank = layer.get_fsize_per_bank()
        last_input_size = layer.get_rest_fsize()

        static_cycle = estimator.calc_layer(layer)
        typical_time = int(
            ceil(static_cycle * (input_size_per_bank / input_size)))
        last_time = int(ceil(static_cycle * (last_input_size / input_size))
                        ) if last_input_size != 0 else typical_time
        # 0: whole bank 1: partial bank
        self.unit_time = [typical_time / self.filter_num, last_time / self.filter_num]

        self.logger.debug('{} [ Input Size ] {} [ Size / Bank ] {} / {} [ Processing Time ] {}'.format(layer.name, input_size, input_size_per_bank, last_input_size, self.unit_time))

    def _calc_process_cycle(self, num_bank, is_last=False):
        if is_last:
            cycle = (num_bank - 1) * self.unit_time[0] + self.unit_time[1]
        else:
            cycle = num_bank * self.unit_time[0]
        return cycle

    def post_process(self, time, dram, fragments, bound):
        layer = self.layer
        op = layer.main_op
        reverse_write = layer.control_info.reverse_write
        num_write = 0
        out_shape = layer.get_output_shape()
        for f in fragments:
            # FIXME fragments can be reduced by timeline module.. it is not considered.
            out_x = layer.get_output_x(f[1], not (f != fragments[-1]))
            if bound != -1 and ((reverse_write and out_x < bound) or (not reverse_write and out_x >= bound)):
                rest_x = ((bound - out_x) if reverse_write else out_x - bound + 1)
                if op.type != 'Gemm':
                    num_write += max(out_shape[1] * int((f[1][1] - f[1][0]) / op.stride) - (out_shape[1] * rest_x), 0)
                else:
                    num_write += max(f[1][1] - f[1][0] - (out_shape[1] * rest_x), 0)
                break
            num_write += (out_shape[1] * int((f[1][1] - f[1][0]) / op.stride) if op.type != 'Gemm' else f[1][1] - f[1][0])

        # size = 64 if op.type == 'Depthwise' or isinstance(op, PoolOp) else 16  # Conv 16 / Depthwise & Pool 64
        # dram.write(time, size, num_write)

    def predict_block_time(self, current_time, fragments_list, bound, is_last, dram):
        op = self.layer.main_op
        time = current_time
        copy_dram = dram.copy()
        total_delay_time = 0
        wcounter = copy.copy(self.wcounter)

        for fragments in fragments_list:
            self.logger.debug('============= [ {} ] Predict Process: {} {} ============'.format(current_time, fragments, self.unit_time))
            req_idx = next(wcounter)
            w_delay_time = copy_dram.get_wmem_delay_time(time, req_idx)
            time += w_delay_time
            total_delay_time += w_delay_time
            self.logger.debug('[ {} ] Predict Process: 1st -> W{} delay {}'.format(time, req_idx, w_delay_time))
            for f in fragments:
                f_delay_time = copy_dram.get_fmem_delay_time(time, f)
                time += f_delay_time
                time += self._calc_process_cycle(1, (is_last if fragments == fragments_list[-1] and f == fragments[-1] else False))
                total_delay_time += f_delay_time
                self.logger.debug('[ {} ] Predict Process: {} -> F delay {}'.format(time, f, f_delay_time))
            self.post_process(time, copy_dram, fragments, bound)

            if isinstance(op, ConvOp) and op.type != 'Depthwise':
                copy_dram.request_wmem(time)
                for i in range(1, self.filter_num):
                    req_idx = next(wcounter)
                    delay_time = copy_dram.get_wmem_delay_time(time, req_idx)
                    time += delay_time
                    total_delay_time += delay_time
                    self.logger.debug('[ {} ] Predict Process: {} -> W{} delay {}'.format(time, f, req_idx, delay_time))
                    for f in fragments:
                        time += self._calc_process_cycle(1, (is_last if fragments == fragments_list[-1] and f == fragments[-1] else False))
                    copy_dram.request_wmem(time)
                    self.post_process(time, copy_dram, fragments, bound)
        del wcounter
        copy_dram.delete()

        return total_delay_time

    def process(self, current_time, fragments, bound, is_last, dram):
        op = self.layer.main_op
        time = current_time
        self.logger.debug('============= [ {} ] Process: {} {} ============'.format(current_time, fragments, self.unit_time))
        self.logger.debug(dram.timeline)
        req_idx = next(self.wcounter)
        w_delay_time = dram.get_wmem_delay_time(time, req_idx)
        time += w_delay_time
        self.blocking_time += w_delay_time
        self.logger.debug('[ {} ] Process: 1st -> W delay {}'.format(time, w_delay_time))
        for f in fragments:
            f_delay_time = dram.get_fmem_delay_time(time, f)
            time += f_delay_time
            time += self._calc_process_cycle(1, (is_last if f == fragments[-1] else False))
            self.blocking_time += f_delay_time
            self.logger.debug('[ {} ] Process: {} -> F delay {}'.format(time, f, f_delay_time))
        self.post_process(time, dram, fragments, bound)
        if isinstance(op, ConvOp) and op.type != 'Depthwise':
            dram.request_wmem(time)
            for i in range(1, self.filter_num):
                req_idx = next(self.wcounter)
                delay_time = dram.get_wmem_delay_time(time, req_idx)
                time += delay_time
                self.blocking_time += delay_time
                self.logger.debug('[ {} ] Process: {} -> W delay {}'.format(time, f, delay_time))
                for f in fragments:
                    time += self._calc_process_cycle(1, (is_last if f == fragments[-1] else False))
                dram.request_wmem(time)
                self.post_process(time, dram, fragments, bound)
        return time


class TimeLine(object):
    def __init__(self, layer, prev):
        self.width = cfg.MIDAP.SYSTEM_WIDTH

        self.layer = layer
        self.current_time = 0

        if isinstance(layer.main_op, ArithmeticOp):
            self.cim = None
            self.dram = None
            return

        self.dram = DRAM(layer)
        self.logger = init_logger('TimeLine', logging.INFO)

        if all([isinstance(layer.main_op, ConvOp), prev]):
            prev_time = self._preload_calc(prev)
            self.dram.request_wmem(-prev_time)
            if layer.main_op.type != 'Depthwise':
                self.dram.request_wmem(0)
        elif isinstance(layer.main_op, ConvOp):
            self.dram.request_wmem(0)
            if layer.main_op.type != 'Depthwise':
                self.dram.request_wmem(0)
        self.cim = CIM(layer, self.dram)
        if self.cim.blocking_time != 0:
            self.current_time = self.cim.blocking_time
        if layer.main_op.type == 'Depthwise':
            self.dram.time = max(self.dram.time, 0)

    def __del__(self):
        del self.cim, self.dram

    def _preload_calc(self, prev):
        prev_cim = CIM(prev)
        last_process = prev.control_info.action[-1][1]
        second_process = None
        for a in reversed(prev.control_info.action[:-1]):
            if a[0] == 'PROCESS':
                second_process = a[1]
                break
        return prev_cim._calc_process_cycle(last_process - second_process, True) if second_process else prev_cim._calc_process_cycle(last_process, True)

    def limit_processing_fragments(self, fragments, bound, max_frag_num, is_last):
        max_frag_num = min(max_frag_num, len(fragments))
        op = self.layer.main_op
        if isinstance(op, ArithmeticOp) or (isinstance(op, PoolOp) and op.global_pooling):
            return max_frag_num
        elif isinstance(op, PoolOp):
            return 1
        # XXX
        if op.type == 'Depthwise':
            max_frag_num = 1

        frag_num = 0
        min_block_time = sys.maxsize
        for n in reversed(range(1, max_frag_num + 1)):
            last_flag = (is_last if n == max_frag_num else False)
            if is_last and n == max_frag_num - 1:
                block_time = self.cim.predict_block_time(self.current_time, [fragments[:n], [fragments[-1]]], bound, True, self.dram)
            else:
                block_time = self.cim.predict_block_time(self.current_time, [fragments[:n]], bound, last_flag, self.dram)
            if min_block_time > block_time:
                frag_num = n
                min_block_time = block_time

            if min_block_time == 0:
                break

        return frag_num

    def load(self, fragments, is_last=False):
        num_bank = len(fragments)
        if isinstance(self.layer.main_op, ArithmeticOp) or num_bank == 0:
            return

        self.current_time += 1
        self.dram.request_fmem(self.current_time, fragments, is_last)
        # print('Load', self.layer.name, fragments, self.time_cim, self.time_mem)

    def process(self, fragments, bound, is_last=False):
        if isinstance(self.layer.main_op, ArithmeticOp):
            return

        # print('Before Process', self.layer.name, fragments, self.current_time, self.cim.blocking_time, file=sys.stderr)
        self.current_time += 1
        self.current_time = self.cim.process(self.current_time, fragments, bound, is_last, self.dram)
        # print('Process', self.layer.name, fragments, self.current_time, self.cim.blocking_time, file=sys.stderr)

    @property
    def cim_blocking_time(self):
        return self.cim.blocking_time if self.cim else 0

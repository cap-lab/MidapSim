from __future__ import division, print_function

import logging
import math
from functools import reduce

from config import cfg
from generic_op import *
from logger import init_logger


class StaticEstimator(dict):
    def __init__(self):
        self.width = cfg.MIDAP.SYSTEM_WIDTH
        self.num_wmem = cfg.MIDAP.WMEM.NUM

        self.logger = init_logger('StaticEstimator', logging.INFO)

    def setup(self, model):
        self.model = model

    def calc_approximate_cycle(self):
        approximate_cycle = 0
        for _, v in self.model.items():
            op = v.main_op
            if isinstance(op, ConvOp) or isinstance(op, PoolOp):
                approximate_cycle += self.calc_layer(v)

        self.logger.debug("Model Approximate Cycle : {}".format(approximate_cycle))
        return self

    def calc_layer(self, v):
        op = v.main_op
        wh_out = reduce(lambda x, y: x * y, op.output_tensor.shape[:-1])
        if op.type == 'Conv' or op.type == 'FC':
            cycle = self._calc_general_conv_cycle(v)
        elif op.type == 'Gemm':
            # XXX
            in_chan, out_chan = self._get_chan_info(v)
            cycle = int(math.ceil(op.k_w * op.k_h * in_chan / self.width)) * int(math.ceil(out_chan / self.num_wmem)) * wh_out
            self.logger.debug("Name: {:16s} | Approximate Cycle: {:-10d}".format(v.name, cycle))
        elif op.type == 'Depthwise' or isinstance(op, PoolOp):
            pad_per_channel = self._calc_padding_size(v)
            _, out_chan = self._get_chan_info(v)
            cycle = op.k_w * op.k_h * int(math.ceil(out_chan / self.width)) * wh_out - int(math.ceil(out_chan / self.width)) * pad_per_channel
            self.logger.debug("Name: {:16s} | Pad: {:5d} | Approximate Cycle: {:-10d}".format(v.name, pad_per_channel, cycle))
        else:
            return 0
            raise NotImplementedError

        _tmp = [
            x.main_op.output_tensor.size if x.sub_op is None else x.sub_op.output_tensor.size for x in v.input]
        in_layer_size = reduce(lambda x, y: x + y, _tmp)
        static_conv_info = {
            'cycle': cycle,
            'in_size': in_layer_size,
            'out_size': op.output_tensor.size
        }
        self[v.name] = static_conv_info

        return cycle

    def _calc_padding_size(self, layer):
        op = layer.main_op

        pad_h = 0
        w = op.output_tensor.shape[0]
        h = op.output_tensor.shape[1]
        s = op.stride
        for dh in range(0, op.pad_t, s):
            pad_h += (dh + 1) * op.k_w * h
        for dh in range(0, op.pad_b, s):
            pad_h += (dh + 1) * op.k_w * h

        pad_w = 0
        for dw in range(0, op.pad_l, s):
            pad_w += (dw + 1) * op.k_h * w
        for dw in range(0, op.pad_r, s):
            pad_w += (dw + 1) * op.k_h * w

        pad_corner = 0
        for dw in range(0, op.pad_l, s):
            for dh in range(0, op.pad_t, s):
                pad_corner += (dh + 1) * (dw + 1)
            for dh in range(0, op.pad_b, s):
                pad_corner += (dh + 1) * (dw + 1)

        for dw in range(0, op.pad_r, s):
            for dh in range(0, op.pad_t, s):
                pad_corner += (dh + 1) * (dw + 1)
            for dh in range(0, op.pad_b, s):
                pad_corner += (dh + 1) * (dw + 1)

        return pad_w + pad_h - pad_corner

    def _calc_stall_cycle(self, op, in_chan):
        # TODO
        # This function do not consider in_height's effect.
        # (What if in_height is not a multiple of system width.)
        s = op.stride
        if s % 4 == 0:
            return 0.0 if op.pad_t % 4 == 0 else 1

        # XXX this code assume that system width(W) is 4x larger than the number of wmem(N)
        inchan_mod = (op.k_h * in_chan) % self.width
        if s % 2 == 1:
            if 0 < inchan_mod and inchan_mod <= self.num_wmem or (inchan_mod > self.num_wmem * 2 and inchan_mod <= self.num_wmem * 3):
                return 0.75
            elif 0 < inchan_mod and inchan_mod > self.num_wmem and inchan_mod <= self.num_wmem * 2:
                return 0.5
        else:
            if 0 < inchan_mod and inchan_mod <= self.num_wmem or (inchan_mod > self.num_wmem * 2 and inchan_mod <= self.num_wmem * 3):
                return 0.5 if op.pad_t % 2 == 0 else 1
            elif 0 < inchan_mod and inchan_mod > self.num_wmem and inchan_mod <= self.num_wmem * 2:
                return 0.0 if op.pad_t % 2 == 0 else 1
        return 0.0

    def _get_chan_info(self, layer):
        out_chan = layer.get_output_shape()[2]
        in_chan = layer.get_input_shape()[2]
        return in_chan, out_chan

    def _calc_general_conv_cycle(self, v):
        op = v.main_op
        in_chan, out_chan = self._get_chan_info(v)

        # XXX this overhead is related to hardware configuration
        overhead = 0.0
        if op.k_w != 1:
            overhead = self._calc_stall_cycle(op, in_chan)

        s = op.stride
        reduce_w = int(math.ceil(op.pad_l / s)) + int(math.ceil(op.pad_r / s))
        reduce_h = int(math.ceil(op.pad_t / s)) + int(math.ceil(op.pad_b / s))
        inner_wh_out = reduce(lambda x, y: (x - reduce_w)
                              * (y - reduce_h), op.output_tensor.shape[:-1])

        def calc_cycle(dw, dh, area):
            return int((op.k_w - dw) * (int(math.ceil(((op.k_h - dh) * in_chan) / self.width)) + overhead) * int(math.ceil(out_chan / self.num_wmem)) * area)

        cycle = calc_cycle(0, 0, inner_wh_out)

        w = op.output_tensor.shape[0]
        h = op.output_tensor.shape[1]
        for dw in range(op.pad_l, 0, -s):
            cycle += calc_cycle(dw, 0, (h - reduce_h))
        for dw in range(op.pad_r, 0, -s):
            cycle += calc_cycle(dw, 0, (h - reduce_h))

        for dh in range(op.pad_t, 0, -s):
            cycle += calc_cycle(0, dh, (w - reduce_w))
        for dh in range(op.pad_b, 0, -s):
            cycle += calc_cycle(0, dh, (w - reduce_w))

        for dw in range(op.pad_l, 0, -s):
            for dh in range(op.pad_t, 0, -s):
                cycle += calc_cycle(dw, dh, 1)
            for dh in range(op.pad_b, 0, -s):
                cycle += calc_cycle(dw, dh, 1)
        for dw in range(op.pad_r, 0, -s):
            for dh in range(op.pad_t, 0, -s):
                cycle += calc_cycle(dw, dh, 1)
            for dh in range(op.pad_b, 0, -s):
                cycle += calc_cycle(dw, dh, 1)

        self.logger.debug("Name: {:16s} | Overhead: {:.2f} | Approximate Cycle: {:-10d}".format(v.name, overhead, cycle))
        return cycle


estimator = StaticEstimator()

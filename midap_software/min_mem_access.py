from __future__ import print_function

from functools import reduce

from config import cfg
from midap_software.layer_compiler import LayerCompiler


class MinMemAccess(LayerCompiler):
    def _preprocess(self):
        from generic_op import ConvOp
        super(MinMemAccess, self)._preprocess()
        self.num_remain_banks = 1
        layer = self.layer
        op = layer.main_op
        if isinstance(op, ConvOp):
            self._set_outbank_num()

    def _calc_dram_access_by_weight(self):
        layer  = self.layer
        op     = layer.main_op
        action = layer.control_info.action

        process_num = 1 if layer.is_weight_in_wmem else reduce(lambda x, y: x + y, [0] + [1 if a[0] == 'PROCESS' else 0 for a in action])
        return (op.weight.size * process_num)

    def _calc_dram_access_by_outfeature(self):
        import numpy as np
        layer     = self.layer
        out_shape = layer.get_output_shape()
        mapping   = layer.control_info.output_mapping

        num_out_banks = len(mapping)
        reduced_width = layer.num_planes_per_fmem * num_out_banks

        return (max(out_shape[0] - reduced_width, 0)) * np.prod(out_shape[1:]) * 2

    def _flip(self, num_output, min_bank_num):
        layer = self.layer
        control_info = layer.control_info
        fmem_info = self.fmem_info

        num_available_bank = fmem_info.get_num_unreserved_bank()
        if num_output < layer.require_fmem and num_output < num_available_bank - min_bank_num:
            min_bank_num = num_available_bank - num_output
        self.num_remain_banks = min_bank_num
        # TODO clean code
        if control_info.output_stationary < 0:
            reverse_write = control_info.reverse_write = layer.require_fmem > num_available_bank - min_bank_num
            input_layer = layer.input[0]
            input_flip = control_info.input_flip = input_layer.control_info.flip
            control_info.flip = not input_flip if reverse_write else input_flip

    def _set_outbank_num(self):
        import sys
        layer = self.layer

        min_bank_num = 1
        min_access = sys.maxsize
        self.fmem_info.backup()
        layer.control_info.backup()
        num_output = layer.require_fmem
        control_info = layer.control_info
        for n in range(min_bank_num, cfg.MIDAP.FMEM.NUM - len(control_info.output_mapping)):
            end = False
            num_available_bank = self.fmem_info.get_num_unreserved_bank()
            self._flip(min(layer.require_fmem, num_available_bank - n), n)
            while not end:
                end = self._do_step()

            w = self._calc_dram_access_by_weight()
            of = self._calc_dram_access_by_outfeature()
            if w + of < min_access:
                min_access = w + of
                min_bank_num = n
                num_output = len(control_info.output_mapping)

            self.fmem_info.restore()
            layer.control_info.restore()

        self._flip(num_output, min_bank_num)

    def _do_operation(self):
        layer        = self.layer
        control_info = layer.control_info

        fragments = control_info.remain_inputs  # Remain input mappings
        if not control_info.fixed_output:
            output_fragments = layer.get_output_fragments(self.num_out_banks, self._next_layer)
            self._set_out_mappings(output_fragments)

            if control_info.num_output_mapping < len(output_fragments):
                fragments = control_info.limit_processing_fragments(output_fragments, fragments)
                if not fragments:
                    self._fix_exception()
            else:
                control_info.fixed_output = True  # no more change occurs

        # post process
        if not fragments:
            return

        self._generate_process_op(fragments)

    def _do_load(self, fragments):
        fmem_info = self.fmem_info
        control_info = self.layer.control_info

        if not control_info.fixed_output:
            num_available_banks = fmem_info.get_num_available_bank()
            num_unreserved_bank = fmem_info.get_num_unreserved_bank()
            num_output_bank = num_unreserved_bank - self.num_remain_banks
            fragments = control_info.limit_load_fragments(num_available_banks, num_output_bank, fragments, self._next_layer)

        if not fragments:
            return False

        self._generate_load_op(fragments)
        return True

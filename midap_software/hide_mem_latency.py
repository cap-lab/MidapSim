from __future__ import division, print_function

from config import cfg
from generic_op import ArithmeticOp

from .layer_compiler import LayerCompiler
from .time_line import TimeLine


class HideMemLatency(LayerCompiler):
    def setup_layer(self, path_info, next_layer=None):
        if self.layer and not self.layer.reduction:
            self.prev_layer = self.layer
        elif not self.layer:
            self.prev_layer = None

        super(HideMemLatency, self).setup_layer(path_info, next_layer)
        self.time_line = TimeLine(self.layer, self.prev_layer)

    def _preprocess(self):
        super(HideMemLatency, self)._preprocess()
        self.num_remain_banks = 1
        layer = self.layer
        op = layer.main_op
        if not any([layer.reduction, isinstance(op, ArithmeticOp)]):
            self._set_outbank_num()

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
        min_delay = sys.maxsize
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

            delay = self.time_line.cim_blocking_time
            # print(layer.name, delay, min_delay)
            if delay < min_delay:
                min_delay = delay
                min_bank_num = n
                num_output = len(control_info.output_mapping)

            self.fmem_info.restore()
            layer.control_info.restore()
            self.time_line = TimeLine(self.layer, self.prev_layer)
            if min_delay == 0:
                break

        self._flip(num_output, min_bank_num)
        # print('[SET]', layer.name, self.num_remain_banks, control_info.reverse_write, control_info.flip)

    def _do_operation(self):
        layer        = self.layer
        control_info = layer.control_info

        # all fragments which is loaded or loading
        restricted_fragments = processing_fragments = control_info.remain_inputs
        max_frag_num = len(processing_fragments)
        output_fragments = layer.get_output_fragments(self.num_out_banks, self._next_layer)
        if not control_info.fixed_output:
            self._set_out_mappings(output_fragments)

            if control_info.num_output_mapping < len(output_fragments):
                restricted_fragments = control_info.limit_processing_fragments(output_fragments, processing_fragments)
                if not restricted_fragments:
                    self._fix_exception()
            else:
                control_info.fixed_output = True  # no more change occurs

        last_flag = (self._is_load_end() and len(control_info.remain_inputs) == len(restricted_fragments))
        bound = control_info.get_out_x_in_fmem(output_fragments, 0) if output_fragments else -1
        process_num = self.time_line.limit_processing_fragments(restricted_fragments, bound, max(max_frag_num - (0 if self._is_load_end() else 1), 1), last_flag)
        processing_fragments = processing_fragments[:process_num]

        # post process
        if not processing_fragments:
            return

        processed_num = len(processing_fragments)
        last_flag = (self._is_load_end() and len(control_info.remain_inputs) == processed_num)
        self.time_line.process(processing_fragments, bound, is_last=last_flag)

        self._generate_process_op(processing_fragments)

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
        self.time_line.load(fragments, self._is_load_end())
        return True

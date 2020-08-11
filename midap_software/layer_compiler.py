import abc
import logging
from abc import ABC

from config import cfg
from logger import init_logger

from .fmem_info import FMEMInfo


class LayerCompiler(ABC):
    def __init__(self):
        self._logger     = init_logger(self.__class__.__name__, logging.INFO)
        self._force_init = False
        self._fmem_info  = FMEMInfo()
        self._layer      = None

    @property
    def layer(self):
        return self._layer

    @property
    def fmem_info(self):
        return self._fmem_info

    @property
    def num_out_banks(self):
        num_unreserved_bank = self.fmem_info.get_num_unreserved_bank()
        return max(num_unreserved_bank - (self.num_remain_banks if self.layer.input[0].require_fmem + self.layer.require_fmem > cfg.MIDAP.FMEM.NUM else 0), 0)

    @property
    def force_init(self):
        return self._force_init

    def force_setup_layer(self, num_init_banks):
        self._force_init = True
        self.force_init_banks = num_init_banks

    def _force_init_banks(self):
        fmem_info = self.fmem_info
        layer = self.layer
        fragments = layer.get_input_fragments(self.force_init_banks, 0)
        for _, f in enumerate(fragments):
            _ = fmem_info.save_data_to_empty_bank(layer.input[0], f)

    def setup_layer(self, layer, next_layer=None):
        self._layer = layer
        self._next_layer = next_layer

    def _fix_exception(self):
        fmem_info = self.fmem_info
        layer = self.layer
        control_info = layer.control_info
        if control_info.reverse_write:
            fmem_info.discard_data_by_layer(layer.name, True)

            control_info.clear_output_mapping()
            self.num_remain_banks += 1
            num_unreserved_bank = fmem_info.get_num_unreserved_bank()
            max_output = num_unreserved_bank - fmem_info.get_num_available_bank()
            if self.num_remain_banks >= max_output:
                control_info.fixed_output = True
                output_fragments = layer.get_output_fragments(num_unreserved_bank - max_output)
                self._set_out_mappings(output_fragments)
        else:
            control_info.fixed_output = True

    def compile_layer(self):
        self._preprocess()

        layer = self.layer
        control_info = layer.control_info
        if layer.reduction:
            if not control_info.fixed_output:
                output_fragments = layer.get_output_fragments(self.num_out_banks)
                self._set_out_mappings(output_fragments)
        else:
            end = False
            while not end:
                end = self._do_step()

        self._postprocess()

    def _is_load_end(self):
        layer = self.layer
        input_layer = layer.input[0]
        control_info = layer.control_info
        load_head = 0 if not control_info.input_mapping else control_info.input_mapping[-1][1][1]
        last_offset = (input_layer.output_tensor.shape[0] - (layer.x_offset[1] if layer.x_offset[1] != 0 else 0))
        return load_head == last_offset

    def _check_processable(self):
        from generic_op import ConvPoolOpBase
        layer        = self.layer
        op           = layer.main_op
        if isinstance(op, ConvPoolOpBase):
            control_info = layer.control_info
            fragments = control_info.remain_inputs
            in_w = layer.get_input_shape()[0]
            pad = (op.pad_l if fragments[0][1][0] == 0 else 0)
            width = (fragments[-1][1][1] - fragments[0][1][0]) + pad
            if fragments[-1][1][1] != in_w and width < op.k_w:
                self.num_remain_banks += 1
                return False
        return True

    def _do_step(self):
        layer         = self.layer
        control_info  = layer.control_info

        if self._is_load_end() or not self._do_load(self._get_all_unloaded_inputs()):
            if self._check_processable():
                self._do_operation()

        return all([self._is_load_end(), not control_info.remain_inputs])

    @abc.abstractmethod
    def _do_load():
        raise NotImplementedError

    @abc.abstractmethod
    def _do_operation():
        raise NotImplementedError

    def _get_all_unloaded_inputs(self):
        layer        = self.layer
        fmem_info    = self.fmem_info
        control_info = layer.control_info
        input_mapping = control_info.input_mapping

        overlap = (not control_info.remain_inputs) and (control_info.processed_input > 0)
        num_fragments = fmem_info.get_num_available_bank()
        load_head = 0 if not input_mapping else input_mapping[-1][1][1] - layer.x_offset[0]
        return layer.get_input_fragments(num_fragments, load_head, overlap=overlap)

    def _set_out_mappings(self, output_fragments):
        layer        = self.layer
        fmem_info    = self.fmem_info
        control_info = layer.control_info

        for o_frag in output_fragments[control_info.num_output_mapping:]:
            bank = fmem_info.save_data_to_empty_bank(layer, o_frag)
            if bank is None:  # there is no bank to use
                break
            control_info.add_output_mapping([(bank, o_frag)])

    def _update_state_by_load(self, fragments):
        fmem_info    = self.fmem_info
        input_layer  = self.layer.input[0]
        control_info = self.layer.control_info
        for fragment in fragments:
            bank = fmem_info.save_data_to_empty_bank(input_layer, fragment)
            if bank is None:  # there is no bank to use
                break
            mapping = fmem_info.get_fmem_mapping_info(input_layer.name)
            fmem_info.reserve_input_banks(mapping, control_info.input_stationary)
            control_info.set_mapping_load_flag(bank)
            control_info.add_input_mapping([[bank, fragment, False]])

    def _generate_load_op(self, fragments):
        control_info = self.layer.control_info
        control_info.action.append(('LOAD', len(control_info.input_mapping) + len(fragments)))
        self._update_state_by_load(fragments)

    def _update_state_by_process(self, fragments):
        fmem_info    = self.fmem_info
        control_info = self.layer.control_info
        for bank, _, _ in fragments:
            fmem_info.discard_data(bank)
        control_info.process_input(len(fragments))

    def _generate_process_op(self, fragments):
        control_info = self.layer.control_info
        num = len(fragments)
        control_info.action.append(('PROCESS', control_info.processed_input + num))
        self._update_state_by_process(fragments)

    def _preprocess(self):
        control_info = self.layer.control_info
        fmem_info = self.fmem_info

        if self.force_init:
            self._force_init = False
            self._force_init_banks()

        control_info.process_flip(fmem_info.get_num_unreserved_bank(), control_info.output_stationary)

        # Setting the initial MIDAP state
        self._preprocess_input_fmem()
        self._preprocess_output_fmem()

    def _preprocess_input_fmem(self):
        fmem_info    = self.fmem_info
        input_layer  = self.layer.input[0]
        control_info = self.layer.control_info

        mapping = fmem_info.get_fmem_mapping_info(input_layer.name)
        fmem_info.reserve_input_banks(mapping, control_info.input_stationary)
        if mapping:
            control_info.add_input_mapping(mapping, True)

    def _preprocess_output_fmem(self):
        control_info = self.layer.control_info
        if not self.layer.next:
            control_info['fixed_output'] = True
            return

        fmem_info         = self.fmem_info
        output_stationary = control_info.output_stationary

        control_info['fixed_output'] = output_stationary >= 0
        if output_stationary > 0:
            next_layer = self.layer.next[0]
            fmem_info.reserve_output_banks(next_layer, output_stationary)
            mapping = fmem_info.get_fmem_mapping_info(next_layer.name)
            control_info.add_output_mapping([(m[0], m[1]) for m in mapping])

    def _postprocess(self):
        layer = self.layer
        fmem_info = self.fmem_info
        control_info = layer.control_info

        if control_info.input_stationary <= 0:
            for in_layer in layer.input:
                fmem_info.discard_data_by_layer(in_layer.name)

        if layer.control_info.reverse_write:
            fmem_info.reverse_mapping(layer.name)

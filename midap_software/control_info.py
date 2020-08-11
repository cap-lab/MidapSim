from __future__ import print_function

from copy import copy, deepcopy

from acc_utils.attrdict import AttrDict


class Snapshot(object):
    def __init__(self):
        self.input_mapping      = None
        self.remain_inputs      = None
        self.output_mapping     = None
        self.num_output_mapping = None
        self.fixed_output       = None
        self.action             = None
        self.reverse_write      = None
        self.flip               = None

    def backup(self, control_info):
        self.input_mapping      = deepcopy(control_info.input_mapping)
        self.remain_inputs      = copy(control_info.remain_inputs)
        self.output_mapping     = copy(control_info.output_mapping)
        self.num_output_mapping = control_info.num_output_mapping
        self.fixed_output       = control_info.fixed_output
        self.action             = copy(control_info.action)
        self.reverse_write      = control_info.reverse_write
        self.flip               = control_info.flip
        self.processed_input    = control_info.processed_input

    def restore(self, control_info):
        control_info.input_mapping      = deepcopy(self.input_mapping)
        control_info.remain_inputs      = copy(self.remain_inputs)
        control_info.output_mapping     = copy(self.output_mapping)
        control_info.num_output_mapping = self.num_output_mapping
        control_info.fixed_output       = self.fixed_output
        control_info.action             = copy(self.action)
        control_info.reverse_write      = self.reverse_write
        control_info.flip               = self.flip
        control_info.processed_input    = self.processed_input


class ControlInfo(AttrDict):
    def __init__(self, layer):
        super(ControlInfo, self).__init__({
            'input_stationary': -1,
            'output_stationary': -1,
            'filter_group': [],
            'prepare': None,
            # [fmem_idx, (head, tail), load_new_fragment?]
            # Mapping
            'input_mapping': [],
            'output_mapping': [],
            'num_initial_fragments': 0,
            'action': [],
            # Write module
            'reverse_write': False,
            'flip': False,
            'input_flip': False,
        })

        # for compile
        self.layer = layer
        self.remain_inputs = []
        self.fixed_output = None
        self.num_output_mapping = 0
        self.processed_input = 0
        self._snapshot = Snapshot()

    def backup(self):
        self._snapshot.backup(self)

    def restore(self):
        self._snapshot.restore(self)

    def process_flip(self, num_available_bank, output_stationary):
        layer = self.layer
        # XXX assume that the first input layer remains its output feature map on FMEM
        self.input_layer = self.layer.input[0]
        input_flip = self.input_layer.control_info.flip

        if output_stationary < 0:
            reverse_write = layer.require_fmem >= num_available_bank
            flip = not input_flip if reverse_write else input_flip
        else:
            flip = False
            reverse_write = (flip != input_flip)

        self.flip = flip
        self.reverse_write = reverse_write
        self.input_flip = input_flip

        if input_flip:
            layer.main_op.flip_operation()
            if layer.sub_op is not None:
                layer.sub_op.flip_operation()

    # function about input/output mapping
    def add_input_mapping(self, inputs, is_initial=False):
        self.input_mapping.extend(inputs)
        self.remain_inputs.extend(inputs)
        if is_initial:
            self.num_initial_fragments += len(inputs)

    def process_input(self, num):
        del self.remain_inputs[:num]
        self.processed_input += num

    def add_output_mapping(self, outputs):
        self.output_mapping.extend(outputs)
        self.num_output_mapping += len(outputs)

    def clear_output_mapping(self):
        self.output_mapping = []
        self.num_output_mapping = 0

    def set_mapping_load_flag(self, fmem_id):
        for info in self.input_mapping:
            if info[0] == fmem_id:
                info[2] = True

    def get_out_x_in_fmem(self, output_fragments, len_output):
        return output_fragments[len_output][1] if self.reverse_write else output_fragments[len_output][0]

    # return input fragments to process, considering fmem size
    def limit_processing_fragments(self, output_fragments, processing_fragments):
        layer = self.layer
        len_output = self.num_output_mapping
        if len_output >= len(output_fragments):
            len_output = len(output_fragments) - 1
        criteria = self.get_out_x_in_fmem(output_fragments, len_output)
        # print('[ Limit Proc ] Out Frag. ', output_fragments, 'Out Criteria ', criteria)
        num_inputs_to_process = 0
        for _, fragment, _ in processing_fragments:
            output_x = layer.get_output_x(fragment, num_inputs_to_process + 1 < len(processing_fragments))
            # print('[ Limit proc ]', fragment, output_x, criteria)
            if (self.reverse_write and output_x < criteria) or (not self.reverse_write and output_x >= criteria):
                break
            num_inputs_to_process += 1
        return processing_fragments[:num_inputs_to_process]

    # return input fragments to load, considering fmem size
    def limit_load_fragments(self, num_available_banks, max_output_banks, input_fragments, next_layer):
        layer = self.layer

        output_fragments = layer.get_output_fragments(max_output_banks, next_layer)

        all_in_frags = [f[1] for f in self.remain_inputs] + input_fragments
        max_output_required = 0
        reduce_fmem = 0
        for in_frag in all_in_frags:
            output_x = layer.get_output_x(in_frag, in_frag != all_in_frags[-1])
            output_required = 0
            for head, tail in output_fragments:
                # print('[ Limit Load ]', output_x, (head, tail), self.reverse_write)
                if (output_x >= head and not self.reverse_write) or (output_x < tail and self.reverse_write):
                    output_required += 1
                else:
                    break
            if output_required != 0:
                max_output_required = max(max_output_required, output_required - reduce_fmem)
                del output_fragments[:output_required]
            else:
                reduce_fmem += 1

        max_output_required -= len(self.output_mapping)
        slots_for_output = num_available_banks - len(input_fragments)
        max_output_required -= slots_for_output

        if max_output_required > num_available_banks:
            input_fragments = []
        elif max_output_required > 0:
            input_fragments = input_fragments[:-max_output_required]
        return input_fragments

from collections import deque
from copy import copy

from acc_utils.errors import CompilerError
from config import cfg


class Snapshot(object):
    def __init__(self):
        self._reserved_banks  = None
        self._available_queue = None
        self._mapping_info    = None

    def backup(self, fmem_info):
        self._reserved_banks  = copy(fmem_info._reserved_banks)
        self._available_queue = copy(fmem_info._available_queue)
        self._mapping_info    = copy(fmem_info._mapping_info)

    def restore(self, fmem_info):
        fmem_info._reserved_banks  = copy(self._reserved_banks)
        fmem_info._available_queue = copy(self._available_queue)
        fmem_info._mapping_info    = copy(self._mapping_info)


class FMEMInfo(object):
    def __init__(self):
        self._reserved_banks  = set([])
        self._available_queue = deque([i for i in range(cfg.MIDAP.FMEM.NUM)])
        self._mapping_info    = []
        self._snapshot        = Snapshot()

    def backup(self):
        self._snapshot.backup(self)

    def restore(self):
        self._snapshot.restore(self)

    def discard_data_by_layer(self, name, reverse_order=False):
        discard_list = []
        for idx, data in enumerate(self._mapping_info):
            n = data[0]
            if n == name:
                discard_list.append(idx)

        # discard the data in the order used by CIMs. (available_queue)
        for idx in (reversed(discard_list) if reverse_order else discard_list):
            _, bank, _ = self._mapping_info[idx]
            if reverse_order:
                self._available_queue.appendleft(bank)
            else:
                self._available_queue.append(bank)
            self._reserved_banks.discard(bank)

        for idx in reversed(discard_list):
            del self._mapping_info[idx]

    def discard_data(self, bank):
        discard_idx = -1
        for idx, data in enumerate(self._mapping_info):
            b = data[1]
            if b == bank:
                discard_idx = idx
                break

        if bank not in self._reserved_banks:
            self._available_queue.append(bank)
            del self._mapping_info[discard_idx]

    def _pop_available_bank(self):
        if not self._available_queue:
            return None
        bank = self._available_queue.popleft()
        return bank

    def get_num_available_bank(self):
        return len(self._available_queue)

    def get_num_unreserved_bank(self):
        return cfg.MIDAP.FMEM.NUM - len(self._reserved_banks)

    def save_data_to_empty_bank(self, layer, data):
        name = layer.name

        # FIXME check that data is already in fmem.
        for n, b, d in self._mapping_info:
            if name == n and d == data:
                return None

        bank = self._pop_available_bank()
        if bank is not None:
            self._save_data(name, bank, data)
        return bank

    def _save_data(self, name, bank, data):
        self._mapping_info.append((name, bank, data))

        if len(self._mapping_info) > cfg.MIDAP.FMEM.NUM:
            raise CompilerError("Use more banks than exists: " + str(self._mapping_info) + self.__repr__())

    def reverse_mapping(self, name):
        mapping_list = []
        discard_list = []
        for idx, data in enumerate(self._mapping_info):
            n = data[0] if data else None
            if n == name:
                mapping_list.append(data)
                discard_list.append(idx)

        for idx in reversed(discard_list):
            del self._mapping_info[idx]
        self._mapping_info = self._mapping_info + list(reversed(mapping_list))

    def reserve_input_banks(self, mapping, input_stationary):
        if mapping and input_stationary >= 0:
            banks = [v[0] for v in mapping[:input_stationary]]
            self._reserved_banks = set(banks)

    def reserve_output_banks(self, next_layer, output_stationary):
        # FIXME
        # set output bank once per path(or branch)
        for data in self._mapping_info:
            n = data[0] if data else None
            if n == next_layer.name:
                return

        fragments = next_layer.get_output_fragments(cfg.MIDAP.FMEM.NUM - 1)
        for f in fragments[:output_stationary]:
            bank = self.save_data_to_empty_bank(next_layer, f)
            if bank is None:
                raise CompilerError("There is no bank to save output of " + next_layer.name + " " + self.__repr__())
            self._reserved_banks.add(bank)

    def get_fmem_mapping_info(self, name):
        mapping = []
        for data in self._mapping_info:
            n = data[0] if data else None
            if n == name:
                mapping.append([data[1], data[2], False])
        return mapping

    def __repr__(self):
        usage = ['O'] * cfg.MIDAP.FMEM.NUM
        for data in self._mapping_info:
            usage[data[1]] = data[0]
        return "FMEM Available Bank {}".format(usage)

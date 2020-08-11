from __future__ import print_function

import itertools
import logging
import sys
from collections import defaultdict
from functools import reduce

import numpy as np
from orderedset import OrderedSet

from acc_utils.errors import CompilerError
from acc_utils.model_utils import div_ceil
from config import cfg
from generic_op import ArithmeticOp, ConcatOp
from logger import init_logger
from midap_software.midap_layer import MidapLayer


class BlockBuilder():
    def __init__(self):
        self.layer2depth = {}
        self.logger = init_logger('BlockBuilder', logging.INFO)

    def make_block_path(self, input_layers):
        self._make_layer2depth(input_layers)
        return self._make_paths(input_layers)

    def _make_layer2depth(self, source_list):
        leafs = []
        next_leafs = source_list
        layer2depth = {}
        depth = 0
        checker = defaultdict(lambda: 1)

        def check_all_input_processed(layer):
            if layer.main_op.type in ['Concat', 'Sum', 'Mul']:
                num_input = len(layer.input)
                if checker[layer] != num_input:
                    checker[layer] += 1
                    return False
            return True

        while next_leafs:
            del leafs
            leafs = next_leafs
            next_leafs = []
            for leaf in leafs:
                if not check_all_input_processed(leaf):
                    continue
                layer2depth[leaf] = depth
                next_leafs.extend(leaf.next)
            depth += 1

        self.layer2depth = layer2depth

    def _make_paths(self, source_list):
        paths = OrderedSet(source_list)
        leaf = paths[0]
        while len(leaf.next) > 0:
            if len(leaf.next) == 1:
                if leaf.have_reduction_layer:
                    leaf.control_info.output_stationary = 0
                paths.add(leaf.next[0])
                leaf = leaf.next[0]
            elif len(leaf.next) > 1:
                blk_path = self._make_block_path(leaf)
                paths = paths | blk_path
                leaf = paths[-1]
        return paths

    def _extend_leafs(self, next_target, depth):
        from copy import copy
        layer2depth = self.layer2depth
        targets = copy(next_target)
        for idx, path in enumerate(targets):
            leaf = path[-1]
            if layer2depth[leaf] == depth:
                if len(leaf.next) > 1:
                    blk_path = self._make_block_path(leaf)
                    next_target[idx] = next_target[idx] | blk_path
                else:
                    next_target[idx].add(leaf.next[0])
            elif layer2depth[leaf] < depth:
                # TODO This leaf may be output path. Now, do not support output in the middle of the network.
                raise CompilerError("Leaf({})\'s depth is less than depth {}.".format(leaf.name, depth))
        del targets
        return leaf, next_target

    def _make_block_path(self, leaf):
        paths = OrderedSet()
        layer2depth = self.layer2depth
        next_target = [OrderedSet([leaf, l]) for l in leaf.next]
        depth = reduce(lambda x, y: min(x, layer2depth[y[-1]]), [sys.maxsize] + next_target)

        while len(next_target) > 1:
            leaf, next_target = self._extend_leafs(next_target, depth)
            depth += 1

            # merge
            merge_paths = self._find_block_path(next_target, depth)
            if merge_paths:
                next_target = self._merge_block_path(next_target, merge_paths)

        paths = paths | next_target[0]
        return paths

    def _find_block_path(self, blocks, depth):
        layer2depth = self.layer2depth
        merge_nodes = set()
        merge_paths = [[]]
        for idx, path1 in enumerate(blocks):
            if layer2depth[path1[-1]] > depth or path1[-1] in merge_nodes:
                continue

            for path2 in blocks[idx + 1:]:
                if path2[-1] == path1[-1]:
                    merge_paths[-1].append(path2)

            if merge_paths[-1]:
                merge_paths[-1].append(path1)
                merge_paths.append([])
                merge_nodes.add(path1[-1])
        del merge_paths[-1]
        return merge_paths

    def _merge_block_path(self, blocks, merge_paths):
        for pl in reversed(merge_paths):
            min_length = reduce(lambda x, y: min(x, len(y)), [sys.maxsize] + pl)
            if min_length == sys.maxsize:
                continue
            _pl = np.asarray([tmp[:min_length] for tmp in pl])
            source = None
            path_list = []
            path = OrderedSet()
            for idx in range(min_length):
                tmp = _pl[:, idx]
                if np.all(tmp == tmp[0]):
                    path.add(tmp[0])
                    source = tmp[0]
                else:
                    path_list = [p[idx:-1] for p in pl]
                    break
            lb = LayerBlock(source, pl[0][-1], path_list)
            path.update([source, lb, pl[0][-1]])
            blocks = [p for p in blocks if p not in pl]
            blocks.append(path)
        return blocks


class LayerBlock(object):
    counter = [itertools.count(1)]

    def __init__(self, source, sink, path_list, prefix="Block_", level=1):
        # for multi-level block
        if level > len(LayerBlock.counter):
            LayerBlock.counter.append(next(itertools.count(1)))
        elif level < len(LayerBlock.counter):
            del LayerBlock.counter[level:]
        self.level = level

        self.name = prefix + str(next(LayerBlock.counter[level - 1]))
        self.source = source
        self.sink = sink

        self.path_list = path_list
        self.input_stationary = []
        self.output_stationary = 0

        self.is_reduction_blk = True if source.have_reduction_layer else False
        self.num_outpaths = 0
        self.num_innerpaths = 0

        self.path_cost = []
        self.total_cost = 0
        self.max_require_fmem = 0
        self.available_fmem = cfg.MIDAP.FMEM.NUM
        self._determine_path_order(cfg.MIDAP.FMEM.NUM)

    @classmethod
    def init_counter(cls):
        del cls.counter
        cls.counter = [itertools.count(1)]

    def _determine_path_order(self, available_fmem, last_input_stationary=0):
        self.path_cost = []
        self.total_cost = 0
        self.available_fmem = available_fmem

        self._sort_path_list()
        self._set_stationary(last_input_stationary)

    def _set_stationary(self, last_input_stationary):
        # input stationary
        self._set_input_stationary(last_input_stationary)

        # output stationary
        max_require_fmem = 0
        max_path_idx = -1
        for idx, (stationary, path) in enumerate(zip(self.input_stationary, self.path_list)):
            if self.is_reduction_blk:
                max_require_fmem = self.available_fmem
                break

            max_require_fmem = max(max_require_fmem, stationary)
            for l in path[:-1]:
                max_path_idx = idx
                blk_require_fmem = l.require_fmem if isinstance(l, MidapLayer) else l.max_require_fmem
                max_require_fmem = max(max_require_fmem, stationary + blk_require_fmem)

        os = self._set_output_stationary(max_require_fmem)
        self.max_require_fmem = max_require_fmem + (os if max_path_idx != 0 else 0)
        self.output_stationary = os

    def _get_dram_access_size(self, path, stationary, init_banks):
        overhead = 0
        input_banks = 0
        for idx, v in enumerate(path):
            if isinstance(v, LayerBlock):
                last_stationary = (stationary if idx == 0 else 0)
                available_fmem = self.available_fmem - (stationary - last_stationary)
                v._determine_path_order(available_fmem, last_stationary)
                overhead += max(v.path_cost)
                input_banks = v.sink.require_fmem
            else:
                num_available_banks = self.available_fmem - stationary - (1 if self.source.require_fmem - init_banks > 0 else 0)
                if input_banks == 0:
                    input_banks = self.source.require_fmem
                # Output
                overhead += (2 * max(v.require_total_fsize - num_available_banks * v.require_fsize[0], 0) if idx != len(path) - 1 else 0)
                # Weight
                weight_load_num = 1 if v.is_weight_in_wmem else div_ceil(input_banks - (stationary if idx == 0 else 0), num_available_banks + 1)
                overhead += weight_load_num * v.get_weight_size()
                input_banks = v.require_fmem
        return overhead

    def _check_partitioned_block(self):
        from generic_op import Crop
        x_offset = None
        for p in self.path_list:
            if not p:
                continue

            if isinstance(p[0], LayerBlock):
                if p[0]._check_partitioned_block():
                    return True
            elif isinstance(p[0].main_op, Crop) or (x_offset and p[0].x_offset != x_offset):
                return True
            if not x_offset:
                x_offset = p[0].x_offset
        return False

    def _calc_input_stationary(self, last_input_stationary):
        num_input_fmem = self.source.require_fmem
        input_stationary = []

        def _get_minmax_stationary(path):
            if self._check_partitioned_block():
                max_stationary = 0
                min_stationary = 0
            else:
                # FIXME This can cause MAC delays.
                extra_fmem = 1
                initial_available_fmem = self.available_fmem - extra_fmem
                max_stationary = min(num_input_fmem, initial_available_fmem)
                min_stationary = 1
            return min_stationary, max_stationary

        # XXX Assume that all FMEM except one bank save input bank. But this is not all the cases.
        input_banks = min(cfg.MIDAP.FMEM.NUM - 1, num_input_fmem)
        for path in self.path_list[:-1]:
            min_overhead = (-1, sys.maxsize)  # stationary, overhead

            min_stationary, max_stationary = _get_minmax_stationary(path)
            for stationary in reversed(range(min_stationary, max_stationary + 1)):
                num_overhead_fmem = num_input_fmem - stationary
                input_overhead = max(num_overhead_fmem - 1, 0) * self.source.require_fsize[0] + (self.source.require_fsize[1] if num_overhead_fmem else 0)
                overhead = input_overhead + self._get_dram_access_size(path, stationary, input_banks)
                if overhead < min_overhead[1]:
                    min_overhead = (stationary, overhead)

            self.path_cost.append(min_overhead[1])
            self.total_cost += min_overhead[1]
            input_stationary.append(min_overhead[0])
            input_banks = min_overhead[0]

        # for last path
        overhead = self._get_dram_access_size(self.path_list[-1], last_input_stationary, input_banks)
        self.path_cost.append(overhead)
        self.total_cost += overhead
        input_stationary.append(last_input_stationary)

        return input_stationary

    def _set_input_stationary(self, last_input_stationary):
        self.input_stationary = self._calc_input_stationary(last_input_stationary)

        for stationary, path in zip(self.input_stationary[self.num_outpaths:], self.path_list[self.num_outpaths:]):
            if not path or isinstance(path[0], LayerBlock):
                continue
            path[0].control_info.input_stationary = stationary

    def _set_output_stationary(self, require_fmem):
        sink = self.sink
        # FIXME This can cause MAC delays.
        os = max(0, self.available_fmem - 1 - require_fmem)
        for p in self.path_list:
            if not p:
                os = 0
                break
        os = min(self.sink.require_fmem, os)

        if isinstance(sink.main_op, ConcatOp):
            if sink.main_op.axis != 2:
                os = 0
            for l in sink.input:
                l.control_info.output_stationary = os
        elif isinstance(sink.main_op, ArithmeticOp):
            if not sink.main_op.broadcast:  # FIXME tricky code..
                inner_paths = self.path_list[self.num_outpaths:]
                ordered_input = [path[-1] if len(path) > 0 else self.source for path in inner_paths]
                if len(ordered_input) == 2 and sink in ordered_input[0].next:
                    ordered_input[0].next.remove(sink)
                    ordered_input[0].write_on_dram = True
                    sink.input = list(reversed(ordered_input))

            os = 0
            if self.path_list[0]:
                self.path_list[0][-1].control_info.output_stationary = os
        return os

    def _sort_path_list(self):
        out_paths = []
        inner_paths = []
        for path in self.path_list:
            if not path:
                inner_paths.append(path)
                continue

            last_layer = path[-1] if isinstance(path[-1], MidapLayer) else path[-1].sink
            if last_layer != self.sink and not last_layer.next:
                out_paths.append(path)
            else:
                inner_paths.append(path)

        reduction_path = []
        if self.is_reduction_blk:
            reduction_path.append(inner_paths[0])
            del inner_paths[0]

        def _sort_func(v):
            if isinstance(v, MidapLayer):
                extra_fmem = 1
                return max(0, v.require_fmem - max(self.available_fmem - extra_fmem, 0))
            else:  # LayerBlock
                return max(0, v.total_cost)

        out_paths.sort(key=(lambda x: 100 * sum([_sort_func(v) for v in x[:-1]]) + len(x)))
        if not (isinstance(self.source, ConcatOp) and self.source.axis != 3):
            inner_paths.sort(key=(lambda x: 100 * sum([_sort_func(v) for v in x[:-1]]) + len(x)))
        self.path_list = out_paths + inner_paths + reduction_path
        self.num_outpaths = len(out_paths)
        self.num_innerpaths = len(inner_paths)

    def get_ordered_path(self, include_sink=False):
        po = [[v] if isinstance(v, MidapLayer) else v.get_ordered_path(False) for p in self.path_list for v in p]
        po = [i for v in po for i in v]
        if include_sink:
            po.append(self.sink)
        return po

    def get_print_string(self, simple=True):
        string = "| {:>8s}".format(self.name)
        if not simple:
            string += " ( {:>8} -> {:>8} )    |\n".format(self.source.name,
                                                          self.sink.name if self.sink else "None")
            for path in self.path_list:
                path_string = str([node.name if isinstance(
                    node, MidapLayer) else node.get_print_string() for node in path])
                string += "| " + path_string + " " * \
                    (37 - len(path_string)) + "|\n"
        return string

    def log_print(self, func=print):
        length = int((80 - len(self.name)) / 2) - 2
        mod = (80 - len(self.name)) % 2
        func("-" * length + "[ " + self.name + " ]" + "-" * (length + mod))
        for l in self.get_ordered_path():
            l.log_print(func)
        func("-" * 80)

    def _make_graphviz(self):
        from graphviz import Digraph
        g = Digraph(self.name)
        g.node(self.source.name)
        for p in self.path_list:
            prev = self.source
            for l in p:
                if isinstance(l, LayerBlock):
                    sg = l._make_graphviz()
                    g.subgraph(sg)
                elif not isinstance(prev, LayerBlock):
                    g.node(l.name)
                    g.edge(l.input[0].name, l.name)
                prev = l
        g.node(self.sink.name)
        for l in self.sink.input:
            g.edge(l.name, self.sink.name)
        return g

    def show_graph(self):
        g = self._make_graphviz()
        g.render(view=False)

    def __repr__(self):
        string = "-" * 40 + "\n"
        string += self.get_print_string(False)
        string += "-" * 40
        return string

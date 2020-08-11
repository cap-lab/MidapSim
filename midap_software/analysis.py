import abc
import sys
from abc import ABC
from collections import OrderedDict
from functools import reduce

import numpy as np

from config import cfg
from generic_op import ArithmeticOp, ConvPoolOpBase, Crop

from .subnet_compiler import SubNetCompiler


class AnalyzerFactory(object):
    analyzer_dict = {}
    # TODO how to divide item and analyzer? another factory?
    @classmethod
    def _init_factory(cls, analyze_all=False):
        if analyze_all:
            cls.items  = AnalysisItem.Cycle | AnalysisItem.DRAMAccess
            cls.policy = SimulationBaseAnalyzer
        elif cfg.MODEL.TILING_OBJECTIVE == "dram_access":
            cls.items  = AnalysisItem.DRAMAccess
            cls.policy = CompileBaseAnalyzer
        elif cfg.MODEL.TILING_OBJECTIVE == "cycle":
            cls.items  = AnalysisItem.Cycle
            cls.policy = SimulationBaseAnalyzer
        else:
            raise ValueError

    @classmethod
    def get_analyzer(cls, analyze_all=False):
        cls._init_factory(analyze_all)
        _id = (cls.items, cls.policy)
        if _id not in AnalyzerFactory.analyzer_dict:
            AnalyzerFactory.analyzer_dict[_id] = cls.policy(cls.items)
        return AnalyzerFactory.analyzer_dict[_id]


class AnalysisItem(object):
    DRAMAccess = 0b0001
    Cycle      = 0b0010
    NoItem     = 0b0100


class Analyzer(ABC):
    def __init__(self, items):
        assert items > 0 and items < AnalysisItem.NoItem
        self._analyze_items = items

        self._results = self.get_init_results()

    @abc.abstractmethod
    def analyze(self, model):
        raise NotImplementedError

    @staticmethod
    def _compile_and_get_path(model):
        subnet_compiler = SubNetCompiler()
        # subnet_compiler.force_setup(cfg.MIDAP.WMEM.NUM - 1)
        inputs, paths = subnet_compiler.compile(model)
        return inputs, paths

    def get_init_results(self):
        results = AnalysisResults()
        if AnalysisItem.DRAMAccess & self._analyze_items:
            results.append(DRAMAccess())
        if AnalysisItem.Cycle & self._analyze_items:
            results.append(Cycle())
        return results

    def get_worst_results(self):
        results = self.get_init_results()
        for r in results:
            r.set_worst_value()
        return results


class CompileBaseAnalyzer(Analyzer):
    def analyze(self, model):
        _, paths = self._compile_and_get_path(model)
        layers = paths

        # DRAMAccess
        for item in self._results:
            if isinstance(item, DRAMAccess):
                feature_size_dict = self._calc_feature_dram_access(layers)
                weight = self._calc_weight_dram_access(layers)
                feature = np.sum([v for l, v in feature_size_dict.items() if not isinstance(l.main_op, Crop)])
                item.set_value([feature, weight], feature_size_dict)
            elif isinstance(item, Cycle):
                raise NotImplementedError
            else:
                raise ValueError
        return self._results

    """ DRAMAccess """
    @staticmethod
    def _calc_input_feature_dram(layers, layer, feature_per_layer):
        control_info = layer.control_info
        op           = layer.main_op

        width = reduce(lambda x, y: x + (y[1][1] - y[1][0]), [0] + control_info.input_mapping)
        if isinstance(op, ArithmeticOp):  # XXX need an api to get maximal DRAM access size per layer.
            in_shape = layer.input[0].get_output_shape()
        else:
            in_shape = layer.get_input_shape()
        # FIXME cannot consider input stationary now
        num_in_banks  = len(layer.input[0].control_info.output_mapping) if layer.input[0].control_info else 0
        reduced_width = layer.input[0].num_planes_per_fmem * num_in_banks
        return max(width - reduced_width, 0) * np.prod(in_shape[1:])

    @staticmethod
    def _calc_output_feature_dram(layers, layer):
        out_shape = layer.get_output_shape()
        if layers[-1] != layer:
            num_out_banks = len(layer.control_info.output_mapping)
            reduced_width = layer.num_planes_per_fmem * num_out_banks
            if isinstance(layer.main_op, Crop):
                return 0
            else:
                return (max(out_shape[0] - reduced_width, 0)) * np.prod(out_shape[1:])
        else:
            return out_shape[0] * np.prod(out_shape[1:])

    @staticmethod
    def _calc_feature_dram_access(layers):
        feature_per_layer = OrderedDict()
        for l in layers:
            # Input
            in_feature = CompileBaseAnalyzer._calc_input_feature_dram(layers, l, feature_per_layer)
            feature_per_layer[l] = [in_feature]

            # Output
            out_feature = CompileBaseAnalyzer._calc_output_feature_dram(layers, l)
            feature_per_layer[l].append(out_feature)
            # print('[F]', l.name, feature_per_layer[l], file=sys.stderr)
        return feature_per_layer

    @staticmethod
    def _calc_weight_dram_access(layers):
        weight_access_size = 0
        for l in layers:
            op = l.main_op
            if op.type == 'Conv':
                process_num = 1 if l.is_weight_in_wmem else \
                    reduce(lambda x, y: x + y, [0] + [1 if a[0] == 'PROCESS' else 0 for a in l.control_info.action])
                weight_access_size += (op.orig_weight_size * process_num)
                weight_access_size += (op.bias.size)
                # print('[W]', l.name, (op.orig_weight_size * process_num), op.bias.size, op.bias_origin.size, file=sys.stderr)
            elif op.type == 'Sum':
                input_feature_size = [prev.get_output_size() for prev in l.input]
                weight_access_size += sum(input_feature_size[1:])
                # print('[W]', l.name, sum(input_feature_size[1:]), file=sys.stderr)
        return weight_access_size


class SimulationBaseAnalyzer(Analyzer):
    results_dict = {}

    def _simulate(self, model):
        from midap_simulator.midap_manager import MidapManager
        import midap_simulator.statistics as stats
        inputs, paths = self._compile_and_get_path(model)
        layers = paths

        init_layers = model.init_layer
        simulator = MidapManager()
        _, cycle, feature, weight = simulator.process_network_with_multiple_input(inputs, init_layers, layers)
        out_shape = layers[0].get_output_shape()
        stats.print_result(layers, out_shape)
        dram_delay = stats.get_dram_delay()
        return feature, weight, dram_delay, cycle - dram_delay

    def analyze(self, model):
        _id = get_model_id(model)
        if _id not in SimulationBaseAnalyzer.results_dict:
            SimulationBaseAnalyzer.results_dict[_id] = self._simulate(model)
        feature, weight, dram_delay, cycle = SimulationBaseAnalyzer.results_dict[_id]

        for item in self._results:
            if isinstance(item, Cycle):
                item.set_value([dram_delay, cycle])
            elif isinstance(item, DRAMAccess):
                item.set_value([feature, weight])
            else:
                raise ValueError
        return self._results


class AnalysisResults(object):
    def __init__(self, values=None):
        self._idx = 0
        self._values = list(values) if values else []

    def __add__(self, other):
        for idx, (a, b) in enumerate(zip(self, other)):
            self._values[idx] = a + b
        return self

    def append(self, value):
        self._values.append(value)

    def __getitem__(self, idx):
        return self._values[idx]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self._values):
            raise StopIteration
        ret = self._values[self._idx]
        self._idx += 1
        return ret


class AnalysisValue(ABC):
    num         = 0

    def __init__(self):
        self.set_init_value()

    @property
    def value(self):
        return self._value

    @classmethod
    def _create_value(cls, val):
        return tuple([val] * cls.num)

    @abc.abstractmethod
    def set_init_value(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_worst_value(self):
        raise NotImplementedError

    def __add__(self, other):
        self._value = tuple([a + b for a, b in zip(self.value, other.value)])
        return self

    @abc.abstractmethod
    def compare(self, b):
        raise NotImplementedError

    def set_value(self, value):
        assert len(value) == self.num
        self._value = tuple(value)


class AllMinimizeValue(AnalysisValue):
    def set_init_value(self):
        self._value = self._create_value(0)

    def set_worst_value(self):
        self._value = self._create_value(sys.maxsize)

    def compare(self, b):
        return sum(self.value) <= sum(b.value)


class DRAMAccess(AllMinimizeValue):
    """ (feature, weight) """
    num = 2

    def __init__(self):
        super(DRAMAccess, self).__init__()
        self._feature_per_layer = {}

    @property
    def feature_per_layer(self):
        return self._feature_per_layer

    def set_value(self, value, feature_per_layer=None):
        super(DRAMAccess, self).set_value(value)
        self._feature_per_layer = feature_per_layer


class Cycle(AllMinimizeValue):
    """ (delay, computation + delay) """
    num = 2


def get_layer_id(layer):
    op = layer.main_op
    if isinstance(op, ConvPoolOpBase):
        _id = (op.type, layer.get_output_shape(), (op.k_w, op.k_h), op.stride, (op.pad_l, op.pad_r))
    else:
        _id = (op.type, layer.get_output_shape())
    return _id


def get_model_id(model):
    _id = []
    for l in model.values():
        _id.append(get_layer_id(l))
    return tuple(_id)

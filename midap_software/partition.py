import abc
import logging
from abc import ABC

from generic_op import ConvPoolOpBase
from logger import init_logger

from .layer_block import BlockBuilder, LayerBlock
from .pyramid import Position, Pyramid


class PartitionAlgo(ABC):
    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def partition():
        raise NotImplementedError

    @abc.abstractmethod
    def add_front_model(self, mb):
        raise NotImplementedError

    @abc.abstractmethod
    def add_back_model(self, mb, in_tensors):
        raise NotImplementedError

    @staticmethod
    def _add_input(mb, layer):
        op = layer.main_op
        return mb.set_input_tensor(name='input', input_tensor=op.output_tensor, order='WHC')

    def _add_layer(self, mb, tensors, layer):
        op = layer.main_op
        tensor = tensors[0]

        if op.type == 'Conv' or op.type == 'Gemm':
            tensor, in_c = self._add_conv(mb, layer, tensor)
        elif op.type == 'FC':
            tensor, in_c = self._add_fc(mb, layer, tensor)
        elif 'GlobalPool' in op.type:
            tensor, in_c = mb.GlobalPool(tensor)
        elif 'Pool' in op.type:
            tensor, in_c = self._add_pool(mb, layer, tensor)
        elif op.type == 'Concat':
            tensor = mb.Concat(tensors)
        else:
            raise NotImplementedError("Name: {} Type: {}".format(layer.name, op.type))
        return [tensor]

    @staticmethod
    def _add_pool(mb, layer, tensor):
        op = layer.main_op
        in_c   = layer.get_input_shape()[2]
        tensor = mb.from_pool_op(tensor, op)
        return tensor, in_c

    @staticmethod
    def _add_fc(mb, layer, tensor):
        op     = layer.main_op
        out_c  = layer.get_output_shape()[2]
        tensor = mb.from_conv_op(tensor, op)
        return tensor, out_c

    @staticmethod
    def _add_conv(mb, layer, tensor):
        op     = layer.main_op
        out_c  = layer.get_output_shape()[2]
        tensor = mb.from_conv_op(tensor, op)
        return tensor, out_c


class SingleBlockPartitionAlgo(PartitionAlgo):
    """ Partition Algorithms """

    def partition(self):
        model           = self.model
        block_organizer = BlockBuilder()
        input_blob      = model.init_layer
        input_layer     = [model[x] for x in input_blob]

        path_list = block_organizer.make_block_path(input_layer)
        del block_organizer
        self._target = []
        self._others = []
        for v in path_list:
            if isinstance(v, LayerBlock):
                self._target.append(v)
            else:
                self._others.append(v)

        self._criteria = (self._target[0].source, self._target[-1].sink)
        return [SingleBlockPartition([b]) for b in self._target]

    def add_front_model(self, mb):
        tensors = [self._add_input(mb, self.model['input'])]

        for l in self._others[1:]:
            tensors = self._add_layer(mb, tensors, l)
            if self._criteria[0] == l:
                break
        return tensors

    def add_back_model(self, mb, in_tensors):
        tensors = in_tensors
        idx = self._others.index(self._criteria[1])
        for l in self._others[idx + 1:]:
            tensors = self._add_layer(mb, tensors, l)
        return tensors


class Partition(ABC):
    def __init__(self, layers):
        self.name    = layers[0].name
        self._layers = layers

        sink = layers[-1].sink if isinstance(layers[-1], LayerBlock) else layers[-1]
        self._out_w = sink.get_output_shape()[0]

        self.logger = init_logger(self.name, logging.INFO)

        # XXX the first value is objective. e.g. (DRAM, Cycle) -> Minimize DRAM.
        self._best      = None
        self._best_size = -1
        self._pyramids  = []

    @property
    def layers(self):
        return self._layers

    @property
    def pyramids(self):
        return self._pyramids

    @property
    def out_w(self):
        return self._out_w

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, b):
        self._best = b

    @property
    def best_size(self):
        return self._best_size

    @best_size.setter
    def best_size(self, bs):
        self._best_size = bs

    @property
    def crop_sizes(self):
        return self._crop_sizes

    @abc.abstractmethod
    def build_pyramid(self, out_w):
        raise NotImplementedError

    @abc.abstractmethod
    def make_model(self, mb, in_tensors):
        raise NotImplementedError

    @abc.abstractmethod
    def get_inner_feature_size(self):
        raise NotImplementedError


class SingleBlockPartition(Partition):
    def _calc_input_info(self, offset, out_w, pos):
        """ Calculate input size and crop sizes for each path """
        block  = self.layers[0]

        in_w = []
        in_offset = []
        for p in block.path_list:
            _out_w = out_w
            _offset = offset
            for l in reversed(p):
                _out_w = self._calc_input_size(l, _out_w, pos)
                _offset = self._calc_input_offset(l, _offset, pos)
            in_w.append(_out_w)
            in_offset.append(_offset)
        max_input = max(in_w)

        input_info = {}
        input_info['crop'] = self._calc_input_crop(in_w, max_input, in_offset, pos)
        input_info['in_size'] = max_input

        # Check Code
        self.logger.debug("[ Calculate Input Size and Crop Size ]")
        block.log_print(self.logger.debug)
        self.logger.debug("Output: {} Pos: {} Input Size: {} Crop Sizes: {}".format(out_w, pos, input_info['in_size'], input_info['crop']))
        return input_info

    def _calc_input_crop(self, in_w, max_input, in_offset, pos):
        _crops = [max_input - w for w in in_w]
        crops = []
        for o, c in zip(in_offset, _crops):
            if pos & Position.First and pos & Position.Last:
                assert o == 0 and max_input + o - c >= self.out_w
                crops.append((0, 0))
            elif pos & Position.First:
                assert o == 0
                crops.append((0, max_input - c))
            elif pos & Position.Last:
                assert max_input + o - c >= self.out_w
                crops.append((o, max_input + o - c))
            elif pos == Position.Mid:
                crops.append((o, max_input + o - c))
            else:
                raise ValueError("Unreachable Codes (Pos: {})".format(pos))
        return crops

    @staticmethod
    def _calc_input_offset(layer, offset, pos):
        op = layer.main_op
        if isinstance(op, ConvPoolOpBase):
            return op.stride * offset - op.k_w // 2 + (op.pad_l if pos & Position.First else 0)
        else:
            return op.stride * offset

    @staticmethod
    def _calc_input_size(layer, out_w, pos):
        op = layer.main_op
        if isinstance(op, ConvPoolOpBase):
            in_w = op.k_w + op.stride * (out_w - 1)
            if pos & Position.First:
                in_w -= op.pad_l
            if pos & Position.Last:
                in_w -= op.pad_r
            return in_w
        else:
            return out_w

    def _add_pyramid(self, pos, info):
        p = Pyramid(self, pos, info)
        self._pyramids.append(p)

    def build_pyramid(self, out_w):
        def pyramid_position(width):
            pos = 0b0
            if end_w == 0:
                pos |= Position.First
            if (end_w + width) == self.out_w:
                pos |= Position.Last
            if not (pos & Position.First or pos & Position.Last):
                pos = Position.Mid
            return pos

        end_w = 0
        self._pyramids = []
        while self.out_w > end_w:
            width = out_w if end_w + out_w <= self.out_w else self.out_w - end_w
            pos = pyramid_position(width)
            info = self._calc_input_info(end_w, width, pos)
            self._add_pyramid(pos, info)
            end_w += width

        # Check Code
        self.logger.debug("[ Build Pyramids ]")
        prev_pos = None
        for p in self._pyramids:
            if prev_pos != p.pos:
                self.logger.debug("Partition Name: {} / Pos: {}".format(self.name, p.pos))
                self.logger.debug(p)
                prev_pos = p.pos

    def make_model(self, mb, in_tensors):
        sink = self.layers[-1].sink if isinstance(self.layers[-1], LayerBlock) else self.layers[-1]
        tensors = []
        for p in self._pyramids:
            tensors += p.make_model(mb, in_tensors)
        if self.best_size != self.out_w:
            output_tensor = sink.output_tensor.transpose(2, 1, 0)
            output_tensor = output_tensor.reshape(1, *output_tensor.shape)
            tensor = mb.Concat(tensors, axis='w', output_tensor=output_tensor, name=self.name + '_X_Concat')
            tensors = [tensor]
        return tensors

    def simulate(self, analyzer):
        size = self.best_size
        self.build_pyramid(size)
        results = self.calc_objective(analyzer)
        ret = []
        for res in results:
            ret.extend(res.value)
        return tuple(ret)

    def calc_objective(self, analyzer):
        value = analyzer.get_init_results()
        for p in self._pyramids:
            value += p.calc_objective(analyzer)
        return value

    def get_inner_feature_size(self):
        import numpy as np

        def get_feature_size_from(p):
            return [v[0] for k, v in p.feature_size_dict.items() if k.input[0].main_op.type != "HEAD"]

        in_feature = None
        for p in self.pyramids:
            if in_feature is not None:
                in_feature = np.add(in_feature, np.array(get_feature_size_from(p)))
            else:
                in_feature = np.array(get_feature_size_from(p))
        return np.sum(in_feature)

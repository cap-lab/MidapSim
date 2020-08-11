import itertools

from generic_op import GenericConvertor
from models.model_builder import ModelBuilder

from .layer_block import LayerBlock
from .midap_model import MidapModel


class Position(object):
    First = 0b0001
    Mid   = 0b0010
    Last  = 0b0100


class PyramidModelBuilder(object):
    counter = itertools.count(1)
    op_counter = {}

    def __init__(self, partition, pos, input_info):
        self._layers     = partition.layers
        self._in_w       = input_info['in_size']
        self._crop_sizes = input_info['crop']
        self._pos        = pos
        self._name = partition.name + "_Pyramid_{}".format(next(PyramidModelBuilder.counter))

        self._mb        = ModelBuilder(self._name)
        self._in_tensor = self._make_input()
        self._model     = self._make_model()

        if pos & Position.Last:
            PyramidModelBuilder.counter = itertools.count(1)
            PyramidModelBuilder.op_counter = {}

    @property
    def mb(self):
        return self._mb

    @property
    def model(self):
        return self._model

    @property
    def layers(self):
        return self._layers

    @property
    def pos(self):
        return self._pos

    def _build_model(self):
        odict = self.mb.get_operator_dict()
        convertor = GenericConvertor()
        convertor.operator_dict = odict
        convertor.post_process()
        model = MidapModel()
        model.from_generic_op_dict(odict)
        return model

    def _make_input(self):
        source = self.layers[0].source if isinstance(self.layers[0], LayerBlock) else self.layers[0]
        shape = source.get_output_shape()
        shape = (1, shape[2], shape[1], shape[0])
        return self.mb.set_input_tensor(name='input', tensor_shape=shape)

    def _add_layer(self, tensors, layer):
        op = layer.main_op
        tensor = tensors[0]

        if op.type not in PyramidModelBuilder.op_counter:
            PyramidModelBuilder.op_counter[op.type] = itertools.count(1)
        counter = PyramidModelBuilder.op_counter[op.type]

        if op.type == 'Conv':
            tensor, in_c = self._add_conv(layer, tensor, counter)
        elif 'Pool' in op.type:
            tensor, in_c = self._add_pool(layer, tensor, counter)
        elif op.type == 'Sum':
            tensor = self.mb.Sum(*tensors, name=self._name + '_Sum_' + str(next(counter)))
        elif op.type == 'Concat':
            tensor = self.mb.Concat(tensors, name=self._name + '_Concat_' + str(next(counter)))
        else:
            raise NotImplementedError("Name: {} Type: {}".format(layer.name, op.type))
        return [tensor]

    def _add_pool(self, layer, tensor, counter):
        op = layer.main_op
        pad    = (op.pad_t, op.pad_b,
                  op.pad_l if self.pos & Position.First else 0,
                  op.pad_r if self.pos & Position.Last else 0)
        in_c   = layer.get_input_shape()[2]
        tensor = self.mb.from_pool_op(tensor, op, pad, name=self._name + '_Pool_' + str(next(counter)))
        return tensor, in_c

    def _add_conv(self, layer, tensor, counter):
        op = layer.main_op
        pad    = (op.pad_t, op.pad_b,
                  op.pad_l if self.pos & Position.First else 0,
                  op.pad_r if self.pos & Position.Last else 0)
        out_c  = layer.get_output_shape()[2]
        tensor = self.mb.from_conv_op(tensor, op, pad, name=self._name + '_Conv_' + str(next(counter)))
        return tensor, out_c

    def _add_crop(self, tensors, crop):
        if 'Crop' not in PyramidModelBuilder.op_counter:
            PyramidModelBuilder.op_counter['Crop'] = itertools.count(1)
        counter = PyramidModelBuilder.op_counter['Crop']
        tensor = self.mb.Crop(tensors[0], crop_x=crop, name=self._name + '_Crop_' + str(next(counter)))
        return [tensor]

    def _add_path(self, path, tensors):
        for l in path:
            tensors = self._add_layer(tensors, l)
        return tensors

    def _add_block(self, block, tensors):
        first_tensors = tensors

        outputs = []
        for p, c in zip(block.path_list, self._crop_sizes):
            tensors = first_tensors
            if c != (0, 0):
                tensors = self._add_crop(tensors, c)
            tensors = self._add_path(p, tensors)
            outputs.append(tensors[0])

        tensors = self._add_layer(outputs, block.sink)

        return tensors

    def _make_model(self):
        self.setup_model_builder()
        return self._build_model()

    def setup_model_builder(self, mb=None, in_tensors=None):
        if mb:
            self._mb = mb
        tensors = [self._in_tensor] if not in_tensors else in_tensors

        for l in self.layers:
            if isinstance(l, LayerBlock):
                tensors = self._add_block(l, tensors)
            else:
                raise NotImplementedError
        return tensors


class Pyramid(object):
    def __init__(self, partition, pos, input_info):
        self._builder = PyramidModelBuilder(partition, pos, input_info)
        self._model = self._builder.model
        self._pos   = pos

        self._feature_per_layer = None

    @property
    def model(self):
        return self._model

    @property
    def pos(self):
        return self._pos

    @property
    def feature_size_dict(self):
        return self._feature_per_layer

    def calc_objective(self, analyzer):
        from .analysis import DRAMAccess
        results = analyzer.analyze(self.model)
        for item in results:
            if isinstance(item, DRAMAccess):
                self._feature_per_layer = item.feature_per_layer
                break
        return results

    def make_model(self, mb, in_tensors):
        return self._builder.setup_model_builder(mb, in_tensors)

    def log_print(self, func=print):
        raise NotImplementedError

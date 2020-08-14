from __future__ import print_function

import abc
import logging
from abc import ABC

from config import cfg
from logger import init_logger
from models.model_builder import ModelBuilder

from .analysis import AnalyzerFactory
from .partition import SingleBlockPartitionAlgo


class NetFuser(object):
    algo = None

    def __init__(self):
        self.logger = init_logger('NetFuser', logging.INFO)
        self._partition_algo = None
        if cfg.MODEL.TILING_METHOD == 'all':
            self.pyramid_search_algo = AllSearch
        elif cfg.MODEL.TILING_METHOD == 'zero':
            self.pyramid_search_algo = ZeroFeatureAccessSearch
        elif cfg.MODEL.TILING_METHOD == 'manual':
            self.pyramid_search_algo = ManualSearch
        else:
            raise ValueError

    def fusing_network(self, model):
        # Partition
        self._partition_algo = SingleBlockPartitionAlgo(model)
        self._partitions = self._partition_algo.partition()

        mb = ModelBuilder('FusedModel')
        tensors = self._partition_algo.add_front_model(mb)

        # Determine pyramid size
        all_analyzer = AnalyzerFactory.get_analyzer(True)
        analyzer = AnalyzerFactory.get_analyzer()
        for p in self._partitions:
            self.pyramid_search_algo.search(p, analyzer)
            feature, weight, delay, cycle = p.simulate(all_analyzer)

            value_str = ""
            for v in p.best.value:
                value_str += "{:,}\t".format(v)
            self.logger.info("{}\t{}/{}\t{}{:,}\t{:,}\t{:,}\t{:,}".format(p.name, p.best_size, p.out_w, value_str, delay, cycle, feature, weight))

            tensors = p.make_model(mb, tensors)

        self._partition_algo.add_back_model(mb, tensors)

        from generic_op import GenericConvertor
        from midap_software import MidapModel, LayerBlock
        odict = mb.get_operator_dict()
        convertor = GenericConvertor()
        convertor.operator_dict = odict
        convertor.post_process()
        fused_model = MidapModel()
        fused_model.from_generic_op_dict(odict)

        LayerBlock.init_counter()
        return fused_model


class PyramidSearchAlgo(ABC):
    @classmethod
    def search(cls, partition, analyzer):
        partition.best = analyzer.get_worst_results()[0]
        out_w = partition.out_w
        for size in cls._get_next_size(out_w):
            partition.build_pyramid(size)
            cls.set_best(partition, size, analyzer)

    @classmethod
    def _get_next_size(cls, out_w):
        raise NotImplementedError

    @abc.abstractclassmethod
    def set_best(cls, partition, size):
        raise NotImplementedError


class AllSearch(PyramidSearchAlgo):
    @classmethod
    def _get_next_size(cls, out_w):
        for w in range(1, out_w + 1):
            yield w

    @classmethod
    def set_best(cls, partition, size, analyzer):
        value = partition.calc_objective(analyzer)[0]
        if value.compare(partition.best):
            partition.best = value
            partition.best_size = size


class ManualSearch(AllSearch):
    # AllSerach Cycle Results (ResNet50 / 512x512)
    pyramid_size = [ 117, 128, 128, 53, 64, 64, 64, 21, 32, 32, 32, 32, 32, 8, 16, 16, ]

    @classmethod
    def search(cls, partition, analyzer):
        partition.best = analyzer.get_worst_results()[0]

        tmp = partition.name.split("_")
        size = cls.pyramid_size[int(tmp[-1]) - 1]
        partition.build_pyramid(size)
        cls.set_best(partition, size, analyzer)

    @classmethod
    def _get_next_size(cls, out_w):
        pass


class ZeroFeatureAccessSearch(AllSearch):
    @classmethod
    def set_best(cls, partition, size, analyzer):
        value = partition.calc_objective(analyzer)[0]
        if partition.get_inner_feature_size() == 0 and \
                value.compare(partition.best):
            partition.best = value
            partition.best_size = size

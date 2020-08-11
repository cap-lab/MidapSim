import logging

import pytest

from config import cfg
from generic_op import GenericConvertor
from logger import init_logger
from midap_software.midap_model import MidapModel
from midap_software.net_fuser import NetFuser
from midap_software.partition import SingleBlockPartitionAlgo
from models import resnet as resnet

cfg.MODEL.USE_TILING       = True


class ModelExecutor(object):
    @staticmethod
    def make_model(mb):
        odict = mb.get_operator_dict()
        convertor = GenericConvertor()
        convertor.operator_dict = odict
        convertor.post_process()
        model = MidapModel()
        model.from_generic_op_dict(odict)
        return model

    @staticmethod
    def blk_fusing(model, blk_idx, diff_check):
        from midap_software.analysis import AnalyzerFactory
        fuser = NetFuser()
        logger = init_logger('FusionTest', logging.INFO)
        partition_algo = SingleBlockPartitionAlgo(model)
        partitions = partition_algo.partition()
        p = partitions[blk_idx]
        analyzer = AnalyzerFactory.get_analyzer()
        fuser.pyramid_search_algo.search(p, analyzer)

        value_str = ""
        for v in p.best.value:
            value_str += "{:,}\t".format(v)
        logger.info("{}\t{}/{}\t{}".format(p.name, p.best_size, p.out_w, value_str))
        if diff_check:
            all_analyzer = AnalyzerFactory.get_analyzer(True)
            feature, weight, delay, cycle = p.simulate(all_analyzer)
            assert (feature == p.best.value[0])

        return p.best.value


def run_fusing(input_size, blk_idx, diff_check=False):
    network = (lambda x: resnet.resnet50(input_size=x[-1]) if x is not None else resnet.resnet50())
    input_shape = None if input_size == 0 else (1, 3, input_size, input_size)
    mb = network(input_shape)
    model = ModelExecutor.make_model(mb)
    ModelExecutor.blk_fusing(model, blk_idx, diff_check)


@pytest.mark.parametrize("blk_idx", [0, 1, 3, 4, 7, 8, 13, 14])
@pytest.mark.parametrize("input_size", [128, 256, 512])
# @pytest.mark.parametrize("input_size", [520])
@pytest.mark.parametrize("method", ['zero', 'all'])
def test_dram_resnet50(input_size, blk_idx, method):
    cfg.MODEL.TILING_OBJECTIVE                = 'dram_access'
    cfg.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER = 'MIN_DRAM_ACCESS'
    cfg.MODEL.TILING_METHOD                   = method

    run_fusing(input_size, blk_idx)


@pytest.mark.parametrize("blk_idx", [0, 1, 3, 4, 7, 8, 13, 14])
@pytest.mark.parametrize("input_size", [128, 256, 512])
# @pytest.mark.parametrize("input_size", [520])
def test_cycle_resnet50(input_size, blk_idx):
    cfg.MODEL.TILING_OBJECTIVE                = 'cycle'
    cfg.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER = 'HIDE_DRAM_LATENCY'
    cfg.MODEL.TILING_METHOD                   = 'all'

    run_fusing(input_size, blk_idx)

import pytest

import models.efficientnet.efficientnet as efficientnet
import models.examples as ex
import models.inception as inception
import models.mobilenet as mobilenet
import models.resnet as resnet
import models.se_resnet as se_resnet
from config import cfg
from test_wrapper import TestWrapper

custom_examples = {
    'dcgan'            : (lambda x: ex.dcgan(nz=x[-1]) if x is not None else ex.dcgan()),
    'discogan'         : (lambda x: ex.discogan(input_shape=x) if x is not None else ex.discogan()),
    'resnet50'         : (lambda x: resnet.resnet50(input_size=x[-1]) if x is not None else resnet.resnet50()),
    'resnet101'        : (lambda x: resnet.resnet101(input_size=x[-1]) if x is not None else resnet.resnet101()),
    'resnet152'        : (lambda x: resnet.resnet152(input_size=x[-1]) if x is not None else resnet.resnet152()),
    'inception_v3'     : (lambda x: inception.inception_v3(input_size=x[-1]) if x is not None else inception.inception_v3()),
    'se_resnet50'      : (lambda x: se_resnet.se_resnet50(input_size=x[-1]) if x is not None else se_resnet.se_resnet50()),
    'se_resnet101'     : (lambda x: se_resnet.se_resnet101(input_size=x[-1]) if x is not None else se_resnet.se_resnet101()),
    'se_resnet152'     : (lambda x: se_resnet.se_resnet152(input_size=x[-1]) if x is not None else se_resnet.se_resnet152()),
    'mobilenet'        : (lambda x: mobilenet.mobilenet(input_size=x[-1]) if x is not None else mobilenet.mobilenet()),
    'mobilenet_v2'     : (lambda x: mobilenet.mobilenet_v2(input_size=x[-1]) if x is not None else mobilenet.mobilenet_v2()),
    'wide_resnet101_2' : (lambda x: resnet.wide_resnet101_2(input_size=x[-1]) if x is not None else resnet.wide_resnet101_2()),
    'wide_resnet50_2'  : (lambda x: resnet.wide_resnet50_2(input_size=x[-1]) if x is not None else resnet.wide_resnet50_2()),
    'efficientnet-b0'  : (lambda x: efficientnet.efficientnet('efficientnet-b0', input_channels=x[-1] if x is not None else None)),
    'efficientnet-b5'  : (lambda x: efficientnet.efficientnet('efficientnet-b5', input_channels=x[-1] if x is not None else None)),
    'efficientnet-b6'  : (lambda x: efficientnet.efficientnet('efficientnet-b6', input_channels=x[-1] if x is not None else None)),
    'efficientnet-b7'  : (lambda x: efficientnet.efficientnet('efficientnet-b7', input_channels=x[-1] if x is not None else None)),
    'efficientnet-b8'  : (lambda x: efficientnet.efficientnet('efficientnet-b8', input_channels=x[-1] if x is not None else None)),
    'efficientnet-l2'  : (lambda x: efficientnet.efficientnet('efficientnet-l2', input_channels=x[-1] if x is not None else None)),
}

expected_stats = {
    'wide_resnet50_2': {384:
                        {4: {'HIDE_DRAM_LATENCY': (105249056, 825806), 'MIN_DRAM_ACCESS': (98802976, 971002)},
                         8: {'HIDE_DRAM_LATENCY': (101776672, 742624), 'MIN_DRAM_ACCESS': (94915872, 805252)}},
                        512:
                        {4: {'HIDE_DRAM_LATENCY': (93801760, 709554), 'MIN_DRAM_ACCESS': (90400032, 927204)},
                         8: {'HIDE_DRAM_LATENCY': (90686752, 676520), 'MIN_DRAM_ACCESS': (86877472, 739490)}},
                        1024:
                        {4: {'HIDE_DRAM_LATENCY': (80985376, 644720), 'MIN_DRAM_ACCESS': (79408416, 744076)},
                         8: {'HIDE_DRAM_LATENCY': (78949664, 629262), 'MIN_DRAM_ACCESS': (78445856, 657196)}},
                        },
    'resnet50': {384:
                 {4: {'HIDE_DRAM_LATENCY': (49035168, 660046), 'MIN_DRAM_ACCESS': (47076256, 796324)},
                  8: {'HIDE_DRAM_LATENCY': (45370272, 600592), 'MIN_DRAM_ACCESS': (44789664, 663213)}},
                 512:
                 {4: {'HIDE_DRAM_LATENCY': (43028384, 580692), 'MIN_DRAM_ACCESS': (41528224, 694900)},
                  8: {'HIDE_DRAM_LATENCY': (41822112, 559632), 'MIN_DRAM_ACCESS': (41248672, 614838)}},
                 1024:
                 {4: {'HIDE_DRAM_LATENCY': (35930016, 540320), 'MIN_DRAM_ACCESS': (35428256, 621836)},
                  8: {'HIDE_DRAM_LATENCY': (35041184, 519760), 'MIN_DRAM_ACCESS': (34983840, 544876)}},
                 },
    'inception_v3': {384:
                     {4: {'HIDE_DRAM_LATENCY': (44538832, 287766), 'MIN_DRAM_ACCESS': (41773232, 509548)},
                      8: {'HIDE_DRAM_LATENCY': (43463232, 331416), 'MIN_DRAM_ACCESS': (40988416, 417488)}},
                     512:
                     {4: {'HIDE_DRAM_LATENCY': (38588256, 252230), 'MIN_DRAM_ACCESS': (36955104, 408200)},
                      8: {'HIDE_DRAM_LATENCY': (36057216, 173064), 'MIN_DRAM_ACCESS': (35686864, 245562)}},
                     1024:
                     {4: {'HIDE_DRAM_LATENCY': (29509760, 176216), 'MIN_DRAM_ACCESS': (29141792, 236856)},
                      8: {'HIDE_DRAM_LATENCY': (28749632, 134310), 'MIN_DRAM_ACCESS': (28749632, 150716)}},
                     }    ,
    'mobilenet_v2': {384:
                     {4: {'HIDE_DRAM_LATENCY': (12764208, 278348), 'MIN_DRAM_ACCESS': (11896880, 359136)},
                      8: {'HIDE_DRAM_LATENCY': (11423792, 239964), 'MIN_DRAM_ACCESS': (11176496, 306802)}},
                     512:
                     {4: {'HIDE_DRAM_LATENCY': (11172912, 247682), 'MIN_DRAM_ACCESS': (9886256, 343048)},
                      8: {'HIDE_DRAM_LATENCY': (9803824, 235354), 'MIN_DRAM_ACCESS': (9499184, 273158)}},
                     1024:
                     {4: {'HIDE_DRAM_LATENCY': (8334384, 256398), 'MIN_DRAM_ACCESS': (6700080, 263592)},
                      8: {'HIDE_DRAM_LATENCY': (6517296, 230814), 'MIN_DRAM_ACCESS': (6201904, 235170)}},
                     }
}

cfg.SYSTEM.BANDWIDTH = (cfg.DRAM.CHANNEL_SIZE * cfg.DRAM.FREQUENCY * cfg.DRAM.NUM_CHANNELS * 2) // cfg.SYSTEM.DATA_SIZE


@pytest.mark.parametrize("network", ['wide_resnet50_2', 'resnet50', 'inception_v3', 'mobilenet_v2'])
@pytest.mark.parametrize("total_fmem_size", [384, 512, 1024])
@pytest.mark.parametrize("num_fmem_banks", [4, 8])
@pytest.mark.parametrize("layer_compiler", ['HIDE_DRAM_LATENCY', 'MIN_DRAM_ACCESS'])
def test_all_cases(network, total_fmem_size, num_fmem_banks, layer_compiler):
    cfg.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER = layer_compiler

    cfg.MIDAP.FMEM.NUM_ENTRIES = total_fmem_size // num_fmem_banks * 1024
    cfg.MIDAP.FMEM.NUM         = num_fmem_banks
    cfg.MIDAP.WMEM.NUM_ENTRIES = 16 * 1024
    cfg.MIDAP.WMEM.NUM         = 16
    cfg.MODEL.USE_TILING       = False

    shape = [1, 3, 224, 224] if network != 'inception_v3' else [1, 3, 299, 299]

    tr = TestWrapper()
    mb = custom_examples[network](shape)
    _, dram_access = tr.run_all(mb)
    delay = tr.midap_simulator.stats.global_stats.MEM_LATENCY
    assert delay == expected_stats[network][total_fmem_size][num_fmem_banks][layer_compiler][1]
    assert dram_access == expected_stats[network][total_fmem_size][num_fmem_banks][layer_compiler][0]

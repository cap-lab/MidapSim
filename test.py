from __future__ import print_function

import argparse
import logging

import models.efficientnet.efficientnet as efficientnet
import models.examples as ex
import models.inception as inception
import models.mobilenet as mobilenet
import models.resnet as resnet
import models.se_resnet as se_resnet
from config import cfg
from test_wrapper import TestWrapper


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str, choices=list(custom_examples.keys()) + ["all", "test"], required=True)
    parser.add_argument('-i', '--input_size', type=int, default=0)
    parser.add_argument('-l', '--layer_compiler', type=str, choices=['MIN_DRAM_ACCESS', 'HIDE_DRAM_LATENCY', 'DOUBLE_BUFFER'], default='HIDE_DRAM_LATENCY')
    parser.add_argument('-b', '--bus_policy', type=str, choices=['WMEM_FIRST', 'FIFO'], default='WMEM_FIRST')
    parser.add_argument('-W', '--system_width', type=int, default=64)
    parser.add_argument('-N', '--num_cims', type=int, default=16)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-da', '--disable_abstract_layer', action="store_true", default=False)
    parser.add_argument('-dr', '--disable_reduction_layer', action="store_true", default=False)
    parser.add_argument('-f', '--fmem_entries', type=int, default=256)
    parser.add_argument('-w', '--wmem_entries', type=int, default=16)
    parser.add_argument('-t', '--tiling_method', type=str, choices=['no', 'all', 'zero', 'manual'], default=None)
    parser.add_argument('-to', '--tiling_objective', type=str, choices=['dram_access', 'cycle'], default=None)
    parser.add_argument('-nb', '--num_banks', type=int, default=4)
    parser.add_argument('-df', '--dram_freq', type=float, default=1.6)
    parser.add_argument('--load', type=str, choices=['ONE', 'MAXIMUM'], default='ONE')
    parser.add_argument('-or', '--output_enable_read_stats', action="store_true", default=False)
    parser.add_argument('-ow', '--output_enable_write_stats', action="store_true", default=False)
    parser.add_argument('-oc', '--output_enable_components_stats', action="store_true", default=False)
    parser.add_argument('-oe', '--output_enable_etc_stats', action="store_true", default=False)
    parser.add_argument('-oa', '--output_enable_all_stats', action="store_true", default=False)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--debug', action="store_true", default = False)
    return parser.parse_args()


custom_examples = {
    'dcgan'            : (lambda x: ex.dcgan(nz=x[-1]) if x is not None else ex.dcgan()),
    'discogan'         : (lambda x: ex.discogan(input_shape=x) if x is not None else ex.discogan()),
    'unet'             : (lambda x: ex.unet(input_shape=x, decompose=not args.disable_abstract_layer) if x is not None else ex.unet(decompose=not args.disable_abstract_layer)),
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
#    'efficientnet-b0'  : (lambda x: efficientnet.efficientnet('efficientnet-b0', input_channels=x[-1] if x is not None else None)),
#    'efficientnet-b1'  : (lambda x: efficientnet.efficientnet('efficientnet-b1', input_channels=x[-1] if x is not None else None)),
#    'efficientnet-b2'  : (lambda x: efficientnet.efficientnet('efficientnet-b2', input_channels=x[-1] if x is not None else None)),
    'efficientnet-b3'  : (lambda x: efficientnet.efficientnet('efficientnet-b3', input_channels=x[-1] if x is not None else None)),
#    'efficientnet-b4'  : (lambda x: efficientnet.efficientnet('efficientnet-b4', input_channels=x[-1] if x is not None else None)),
#    'efficientnet-b5'  : (lambda x: efficientnet.efficientnet('efficientnet-b5', input_channels=x[-1] if x is not None else None)),
#    'efficientnet-b6'  : (lambda x: efficientnet.efficientnet('efficientnet-b6', input_channels=x[-1] if x is not None else None)),
#    'efficientnet-b7'  : (lambda x: efficientnet.efficientnet('efficientnet-b7', input_channels=x[-1] if x is not None else None)),
#    'efficientnet-b8'  : (lambda x: efficientnet.efficientnet('efficientnet-b8', input_channels=x[-1] if x is not None else None)),
#    'efficientnet-l2'  : (lambda x: efficientnet.efficientnet('efficientnet-l2', input_channels=x[-1] if x is not None else None)),
}

test_set = ['discogan', 'resnet50', 'wide_resnet50_2', 'inception_v3', 'se_resnet50', 'mobilenet_v2',
        'efficientnet-b3']

expected_stats = {  # Expected (Latency, Dram Access) of each testset
    'inception_v3_m'     :  (6779606,  29541104),
    'mobilenet_v2_m'     :   (943431,   7446000),
    'resnet101_m'        :  (8309377,  65020912),
    'resnet152_m'        : (12134411,  88488944),
    'resnet50_m'         :  (4530107,  39013360),
    'se_resnet101_m'     :  (8682840,  71852528),
    'se_resnet152_m'     : (12670019,  97810928),
    'se_resnet50_m'      :  (4759578,  43855856),
    'squeezenet_m'       :   (571549,   2203200),
    'wide_resnet101_2_m' : (22166196, 148249584),
    'wide_resnet50_2_m'  : (11455550,  83248112),

    'inception_v3'     :  (6741155,  29952080),
    'mobilenet_v2'     :   (937441,   8281584),
    'resnet152'        : (12071839,  88990704),
    'resnet101'        :  (8246805,  65522672),
    'resnet50'         :  (4467535,  39515120),
    'se_resnet152'     : (12607447,  98312688),
    'se_resnet101'     :  (8620268,  72354288),
    'se_resnet50'      :  (4696549,  44327408),
    'squeezenet'       :   (571255,   2696384),
    'wide_resnet101_2' : (22092751, 149187568),
    'wide_resnet50_2'  : (11382105,  84186096)
}

args = parse()

cfg.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER = args.layer_compiler
cfg.MIDAP.BUS_POLICY                      = args.bus_policy
cfg.MODEL.USE_TILING                      = True if args.tiling_method else False
cfg.MODEL.TILING_METHOD                   = args.tiling_method
cfg.MODEL.TILING_OBJECTIVE                = args.tiling_objective
cfg.MIDAP.CONTROL_STRATEGY.FILTER_LOAD    = args.load
cfg.MODEL.ALLOW_ABSTRACT_DATA             = not args.disable_abstract_layer
cfg.MODEL.REDUCTION_LOGIC                 = not args.disable_reduction_layer

# Configuration
cfg.MIDAP.SYSTEM_WIDTH     = args.system_width
cfg.MIDAP.FMEM.NUM_ENTRIES = args.fmem_entries * 1024
cfg.MIDAP.FMEM.NUM         = args.num_banks
cfg.MIDAP.WMEM.NUM_ENTRIES = args.wmem_entries * 1024
cfg.MIDAP.WMEM.NUM         = args.num_cims

cfg.DRAM.FREQUENCY   = args.dram_freq
cfg.SYSTEM.BANDWIDTH = (cfg.DRAM.CHANNEL_SIZE * cfg.DRAM.FREQUENCY * cfg.DRAM.NUM_CHANNELS * 2) // cfg.SYSTEM.DATA_SIZE

if args.debug:
    cfg.LOGGING_CONFIG_DICT['root']['level'] = 'DEBUG'
    cfg.LOGGING_CONFIG_DICT['root']['handlers'] = ['console', 'file']
    cfg.LOGGING_CONFIG_DICT['loggers']['debug']['level'] = 'DEBUG'

input_shape = None if args.input_size == 0 else (1, 3, args.input_size, args.input_size)

output_dir = args.output_dir
if args.output_enable_all_stats:
    output_option = (True, True, True, True)
else:
    output_option = (args.output_enable_read_stats, args.output_enable_write_stats,
                     args.output_enable_components_stats, args.output_enable_etc_stats)

tr = TestWrapper(args.level)

if args.network == 'all':
    for model in custom_examples:
        mb = custom_examples[model](input_shape)
        tr.run_all(mb)

elif args.network == 'test':
    for network in test_set:
        mb = custom_examples[network](input_shape)
        latency, dram_access = tr.run_all(mb)

elif args.network in custom_examples:
    if input_shape is None:
        raise ValueError("you must specify image_size w/ --image_size or -i $IMG_SIZE to test a specific model")
    mb = custom_examples[args.network](input_shape)
    tr.run_all(mb, output_dir=output_dir, output_option=output_option)

else:
    raise ValueError("{} is not supported model.. please check custom_examples list.".format(args.network))

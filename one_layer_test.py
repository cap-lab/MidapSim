from __future__ import print_function

import argparse
import os.path

import models.examples as ex
from config import cfg
from generic_op import *
from midap_simulator import *
from midap_software import Compiler, MidapModel


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_shape', nargs='+', type=int, required=True)
    parser.add_argument('-oc', '--out_chan', type=int, required=True)
    parser.add_argument('-k', '--kern_info', nargs='+', type=int, required=True)
    parser.add_argument('-l', '--layer_compiler', type=str, choices=['MIN_DRAM_ACCESS', 'HIDE_DRAM_LATENCY'], default='HIDE_DRAM_LATENCY')
    parser.add_argument('-ib', '--init_banks', type=int, default=0)

    parser.add_argument('-b', '--bus_policy', type=str, choices=['WMEM_FIRST', 'FIFO'], default='WMEM_FIRST')
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-da', '--disable_abstract_layer', action="store_true", default=False)

    parser.add_argument('-f', '--fmem_entries', type=int, default=256)
    parser.add_argument('-nb', '--num_banks', type=int, default=4)
    parser.add_argument('--latency', type=int, default=100)
    parser.add_argument('--bandwidth', type=int, default=32)
    return parser.parse_args()


class TestWrapper(object):
    def __init__(self):
        self.cv = GenericConvertor()
        self.midap_model = MidapModel()
        self.cm = Compiler()
        self.midap_simulator = MidapManager()
        self.step_checker = [0, 0, 0]

    def setup_from_builder(self, builder):
        odict = builder.get_operator_dict()
        self.cv.operator_dict = odict
        self.cv.post_process()
        self.midap_model.from_generic_op_dict(odict)
        self.step_checker[0] = 1
        if self.step_checker[1] > 0:
            del self.cm
            self.cm = Compiler()
            self.step_checker[1] = 0

    def compile(self, num_init_banks):
        if self.step_checker[0] == 0:
            print("Please setup the model first")
            return
        self.cm.force_setup(num_init_banks)
        static_info = self.cm.compile(self.midap_model)
        self.step_checker[1] = 1
        if self.step_checker[2] > 0:
            del self.midap_simulator
            self.midap_simulator = MidapManager()
            self.step_checker[2] = 0
        return static_info

    def simulate(self):
        if self.step_checker[0] == 0:
            print("Please setup the model first")
            return
        elif self.step_checker[1] == 0:
            print("Please run compile")
            return
        input_tensor_list, path_info = self.cm.control_info
        init_layer_list = self.midap_model.init_layer
        _ = self.midap_simulator.process_network_with_multiple_input(input_tensor_list, init_layer_list, path_info)
        self.step_checker[2] = 1
        return path_info

    def run_all(self, model, output_dir=None, output_option=(True, False, False, False)):
        self.__init__()
        self.setup_from_builder(model)
        model = model.name
        self.logger.info("[ {} ]".format(model))
        _ = self.compile()
        sim_instruction, stat = self.simulate()
        diff, latency, feature_dram, weight_dram = stat
        # print("check stat(Checking info) of network {}: {}".format(model ,stat), file=sys.stderr)
        if diff > 0:
            self.logger.error(
                "Network Result Diff > 0: Functional Problem may occur, network {}".format(model))
        self.midap_simulator.stats.print_result(sim_instruction.processing_order, model)


args = parse()

cfg.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER = args.layer_compiler
cfg.MIDAP.BUS_POLICY                      = args.bus_policy
cfg.MODEL.ALLOW_ABSTRACT_DATA             = not args.disable_abstract_layer
cfg.MODEL.REDUCTION_LOGIC                 = True

# Configuration
cfg.MIDAP.SYSTEM_WIDTH     = 64
# cfg.MIDAP.FMEM.SIZE = 256 * 1024
cfg.MIDAP.FMEM.NUM_ENTRIES = args.fmem_entries * 1024
cfg.MIDAP.FMEM.NUM         = args.num_banks

cfg.SYSTEM.BANDWIDTH  = args.bandwidth  # GB ( * 10^9 byte) / s
cfg.LATENCY.DRAM_READ = args.latency

output_dir = args.output_dir

tr = TestWrapper()

mb = ex.one_layer_example(args.input_shape, args.out_chan, args.kern_info)
tr.run_all("custom", mb, args.init_banks, output_dir=output_dir)

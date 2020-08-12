from __future__ import print_function

import logging
import os.path

from config import cfg
from generic_op import *
from midap_simulator import *
from data_structure import SimulatorInstructionV1
from midap_software import Compiler, MidapModel

class TestWrapper(object):
    def __init__(self, simulation_level = 0):
        self.cv = None
        self.midap_model = None
        self.cm = None
        self.midap_simulator = None
        self.simulation_level = simulation_level

    def setup_from_builder(self, builder):
        if self.cv is not None:
            del self.cv
        odict = builder.get_operator_dict()
        self.cv = GenericConvertor()
        self.cv.operator_dict = odict
        self.cv.post_process()
        if self.midap_model is not None:
            del self.midap_model
        self.midap_model = MidapModel()
        self.midap_model.from_generic_op_dict(odict)
        if self.cm is not None:
            del self.cm
        self.cm = Compiler()
        self.cm.set_op_dict(odict)

    def compile(self):
        if self.cm is None:
            raise RuntimeError("Setup the wrapper first")
        static_info = self.cm.compile(self.midap_model)
        return static_info

    def simulate(self):
        if self.cm is None:
            raise RuntimeError("Setup the wrapper first")
        input_tensor_list, path_info = self.cm.control_info
        init_layer_list = self.midap_model.init_layer
        sim_instruction = SimulatorInstructionV1()
        sim_instruction.from_compiler_input(input_tensor_list, init_layer_list, path_info)
        if self.midap_simulator is not None:
            del self.midap_simulator
        self.midap_simulator = MidapManager(self.simulation_level)
        stats = self.midap_simulator.simulate(sim_instruction)
        return sim_instruction, stats

    def run_all(self, model, output_dir=None, output_option=(True, False, False, False)):
        self.setup_from_builder(model)
        model = model.name
        _ = self.compile()
        sim_instruction, stat = self.simulate()
        diff, latency, feature_dram, weight_dram = stat
        self.midap_simulator.stats.print_result(sim_instruction.processing_order, model)
        return latency, (feature_dram + weight_dram)


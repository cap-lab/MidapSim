from __future__ import absolute_import, division, print_function, unicode_literals

from .main_stage import FMainStage, VMainStage
from .reduction_stage import FReductionStage, VReductionStage
from .write_stage import FWriteStage, VWriteStage
from .dataflow import generate_dataflow_info 

from config import cfg

import numpy as np

class Pipeline(list):
    def __init__(self, manager, **kwargs):
        super().__init__(**kwargs)
        self.system_width = cfg.MIDAP.SYSTEM_WIDTH
        self.num_wmem = cfg.MIDAP.WMEM.NUM
        self.num_fmem = cfg.MIDAP.FMEM.NUM
        self.manager = manager
        self.initialize()
        for idx in range(len(self)-1):
            self[idx+1].set_input_buf(self[idx].output_buf)
        self.dataflow_info_buf = [generate_dataflow_info() for _ in range(len(self))]

    def __del__(self):
        del self.dataflow_info_buf

    def initialize(self):
        pass

    def setup(self, modules):
        for stage in self:
            stage.setup(modules)

    def run(self, dataflow_info):
        out_info = self[-1].run(self.dataflow_info_buf[-1])
        self.dataflow_info_buf[0] = dataflow_info
        for idx in reversed(range(len(self) - 1)):
            self.dataflow_info_buf[idx + 1] = self[idx].run(self.dataflow_info_buf[idx])
        return out_info

    def get_skipped_pipeline_stages(self):
        stages = 0
        for stage in self:
            stages += stage.skipped_pipeline_stages
        return stages

class FPipeline(Pipeline):
    def initialize(self):
        self.append(FMainStage(self.manager))
        self.append(FReductionStage(self.manager))
        self.append(FWriteStage(self.manager))

class VPipeline(Pipeline):
    def initialize(self):
        self.append(VMainStage(self.manager))
        self.append(VReductionStage(self.manager))
        self.append(VWriteStage(self.manager))

    def run(self, dataflow_info):
        # No Pipeline
        for stage in self:
            stage.run(dataflow_info)
        return dataflow_info

def get_pipeline(manager, simulation_level):
    if simulation_level == 0:
        return FPipeline(manager)
    if simulation_level in [1, 2]:
        return VPipeline(manager)
    raise ValueError("Not supported simulation level : " + str(simulation_level))



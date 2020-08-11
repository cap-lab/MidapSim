from config import cfg

class Stage():
    def __init__(self, manager):
        self.system_width = cfg.MIDAP.SYSTEM_WIDTH
        self.num_wmem = cfg.MIDAP.WMEM.NUM
        self.num_fmem = cfg.MIDAP.FMEM.NUM
        self.manager = manager
        self.memory_controller = manager.memory_controller
        self.skipped_pipeline_stages = 0
        self.initialize()
    
    def initialize(self):
        self.output_buf = None
        self.input_buf = None

    def set_input_buf(self, input_buf):
        self.input_buf = input_buf
    
    def setup(self, modules):
        pass

    def run(self, dataflow_info):
        pass

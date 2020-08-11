from __future__ import absolute_import, division, print_function, unicode_literals

from config import cfg
from midap_software.net_fuser import NetFuser
from midap_software.static_estimator import estimator
from midap_software.subnet_compiler import SubNetCompiler


class Compiler(dict):
    def __init__(self, *args, **kwargs):
        super(Compiler, self).__init__(*args, **kwargs)
        self.model = None
        self.op_dict = None

        self.subnet_compiler = SubNetCompiler()
        if cfg.MODEL.USE_TILING:
            self.network_fuser = NetFuser()

    def set_op_dict(self, op_dict):
        self.op_dict = op_dict

    def compile(self, model):
        self.model = model

        # static performance estimator
        estimator.setup(model)
        static_info = estimator.calc_approximate_cycle()

        if cfg.MODEL.USE_TILING:
            model = self.network_fuser.fusing_network(model)
        self._control_info = self.subnet_compiler.compile(model)

        return static_info

    def force_setup(self, num_init_banks):
        self.subnet_compiler.force_setup(num_init_banks)

    @property
    def control_info(self):
        return self._control_info

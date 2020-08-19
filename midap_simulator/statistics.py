from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time

from acc_utils.attrdict import AttrDict
from acc_utils.errors import *
from acc_utils.model_utils import *
from config import cfg
from generic_op import *


class Stats():
    def __init__(self):
        self.global_stats = None
        self.local_stats = None
        self.layer_stats = None
        self.description = self.init_description()
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger()

    def init(self):
        if self.global_stats is not None:
            del self.global_stats
        if self.layer_stats is not None:
            for layer in self.layer_stats:
                del self.layer_stats[layer]
            del self.layer_stats
        self.global_stats = self.create_branch()
        self.local_stats = self.create_branch()
        self.layer_stats = {}

    def create_branch(self):
        branch = AttrDict()
        branch.SIM_TIME = time.time()
        branch.CLOCK = 0
        branch.WRITE = AttrDict()
        branch.WRITE.DRAM = 0
        branch.RUN = AttrDict()
        branch.READ = AttrDict()
        branch.READ.DRAM = 0
        branch.READ.DRAM2FMEM = 0
        branch.READ.DRAM2WMEM = 0
        branch.READ.FMEM = 0
        branch.READ.WMEM = 0
        branch.MACs = 0
        branch.DELAY = AttrDict()
        branch.DELAY.DRAM = 0
        branch.DELAY.READ_WMEM = 0
        branch.DELAY.READ_FMEM = 0
        branch.DELAY.WRITE_DRAM = 0
        branch.DELAY.WRITE_FMEM = 0
        branch.DRAM = AttrDict()
        branch.DRAM.BUS_BUSY_TIME = 0
        return branch

    def init_description(self):
        __DESCRIPTION = self.create_branch()
        __DESCRIPTION.SIM_TIME = "Elapsed time"
        __DESCRIPTION.CLOCK = "Simulated Cycle"
        __DESCRIPTION.WRITE.DRAM = "# of DRAM write"
        __DESCRIPTION.READ.DRAM = "# of DRAM read"
        __DESCRIPTION.READ.DRAM2FMEM = "data size of DRAM -> FMEM"
        __DESCRIPTION.READ.DRAM2WMEM = "data size of DRAM -> WMEM"
        __DESCRIPTION.READ.FMEM = "# of FMEM read"
        __DESCRIPTION.READ.WMEM = "# of WMEM read"
        __DESCRIPTION.MACs = "MACs"
        __DESCRIPTION.DELAY.DRAM = "Total memory delay"
        __DESCRIPTION.DELAY.READ_WMEM = "WMEM Read delay"
        __DESCRIPTION.DELAY.READ_FMEM = "FMEM Read delay"
        __DESCRIPTION.DELAY.WRITE_DRAM = "DRAM Write Delay"
        __DESCRIPTION.DELAY.WRITE_FMEM = "LATENCY due to the write operation bottleneck"
        __DESCRIPTION.DRAM.BUS_BUSY_TIME = "DRAM busy time"
        return __DESCRIPTION

    def update(self, layer_name, print_stats=True):
        for attr in self.global_stats:
            if attr == 'SIM_TIME':
                continue
            if isinstance(self.global_stats[attr], dict):
                for key in self.global_stats[attr]:
                    self.global_stats[attr][key] += self.local_stats[attr][key]
            else:
                self.global_stats[attr] += self.local_stats[attr]

        self.local_stats.SIM_TIME = time.time() - self.local_stats.SIM_TIME
        stat = self.local_stats
        if print_stats:
            self.logger.info(" LAYER {} - Elapsed Time(s): {}, Simulated cycles: {}, MACs: {}.".format(
                layer_name,
                stat.SIM_TIME,
                stat.CLOCK,
                stat.MACs)
            )

        self.layer_stats[layer_name] = self.local_stats
        self.local_stats = self.create_branch()

    def end_simulation(self):
        self.global_stats.SIM_TIME = time.time() - self.global_stats.SIM_TIME
        stat = self.global_stats
        self.logger.info(""" SIMULATION FINISHED. SUMMARY:
                SIMULATION TIME(s): {}
                MACs: {}
                SIMULATED CYCLE(cylce): {}
                DRAM Delay, Write Delay: {}, {}
                DRAM_READ, WRITE: {}, {}
                DRAM2FMEM, DRAM2WMEM: {}, {}
                FMEM_READ: {}
                WMEM_READ: {}
                """.format(stat.SIM_TIME,
                           stat.MACs,
                           stat.CLOCK,
                           stat.DELAY.DRAM, stat.DELAY.WRITE_FMEM,
                           stat.READ.DRAM, stat.WRITE.DRAM,
                           stat.READ.DRAM2FMEM, stat.READ.DRAM2WMEM,
                           stat.READ.FMEM,
                           stat.READ.WMEM)
                         )

    def dram_read(self, cnt=1):
        self.local_stats.READ.DRAM += cnt

    def read_dram2fmem(self, cnt=1):
        self.local_stats.READ.DRAM2FMEM += cnt
        self.dram_read(cnt)

    def read_dram2wmem(self, cnt=1):
        self.local_stats.READ.DRAM2WMEM += cnt
        self.dram_read(cnt)

    def fmem_read(self, cnt=1):
        self.local_stats.READ.FMEM += cnt

    def wmem_read(self, cnt=1):
        self.local_stats.READ.WMEM += cnt

    def dram_write(self, cnt=1):
        self.local_stats.WRITE.DRAM += cnt

    def total_cycle(self):
        return self.local_stats.CLOCK + self.global_stats.CLOCK

    def current_cycle(self):
        return self.local_stats.CLOCK

    def increase_cycle(self, t=1):
        self.local_stats.CLOCK += t

    def memory_latency(self):
        return self.local_stats.DELAY.DRAM

    def wait_memory(self, t=1):
        self.local_stats.DELAY.DRAM += t
        self.local_stats.CLOCK += t

    def wait_dram2fmem(self, t=1):
        self.local_stats.DELAY.READ_FMEM += t
        self.wait_memory(t)

    def wait_dram2wmem(self, t=1):
        self.local_stats.DELAY.READ_WMEM += t
        self.wait_memory(t)

    def wait_write_dram(self, t=1):
        self.local_stats.DELAY.WRITE_DRAM += t

    def wait_writing(self, t=1):
        self.local_stats.DELAY.WRITE_FMEM += t
        self.local_stats.CLOCK += t

    def use_dram_bus(self, t):
        self.local_stats.DRAM.BUS_BUSY_TIME += t

    def set_macs(self, size):
        self.local_stats.MACs = size

    def diff_static_and_simulate(path_info, static_info):
        for layer in path_info:
            if isinstance(layer.main_op, ConvOp) or isinstance(layer.main_op, PoolOp):
                stats_layer = self.layer_stats[layer.name]
                profile_value = stats_layer['CLOCK'] - stats_layer['DELAY.DRAM']
                static_cycle = static_info[layer.name]['cycle']
                static_in_size = static_info[layer.name]['in_size']
                static_out_size = static_info[layer.name]['out_size']
                print("Layer: {:>16s} {:^12s}\tKern: {:1d}x{:1d}\tSimulated: {:>8d}\tStatic Calc: {:>8d}\tDiff. Rate: {:>.2f}\tDRAM: {:>8d}\tFMEM: {:>8d} ({:>5d}, {:>5d})\tTensor: {:>8d} -> {:>8d}".format(
                    layer.name, "(" + layer.main_op.type + ")", layer.main_op.k_w, layer.main_op.k_h, profile_value, static_cycle,
                    (static_cycle - profile_value) / profile_value * 100, stats_layer['DELAY.DRAM'], stats_layer['WRITE']['FMEM'], stats_layer['DELAY.READ_FMEM'], stats_layer['DELAY.WRITE_DRAM'], static_in_size, static_out_size))
            elif layer.name in self.layer_stats:
                stats_layer = self.layer_stats[layer.name]
                print("Layer: {:>16s} {:^12s}\tDRAM: {:>8d}\tFMEM: {:>8d} ({:>5d}, {:>5d})".format(
                    layer.name, "(" + layer.main_op.type + ")", stats_layer['DELAY.DRAM'], stats_layer['WRITE']['FMEM'], stats_layer['DELAY.READ_FMEM'], stats_layer['DELAY.WRITE_DRAM']))

    def print_result(self, path_info, model):
        import math
        name = []
        dram = []
        fmem_dram = []
        wmem_dram = []
        total_delay = []
        fmem_delay = []
        wmem_delay = []
        cycle = []
        mac = []
        fps = []
        utilization = []
        dram_busy_time = []
        dram_utilization = []
        dram_is = []
        dram_ws = []

        # for conv
        conv_clock = 0
        conv_mac = 0
        conv_delay = 0

        # for fc
        fc_delay = 0

        # for residual
        residual_delay = 0

        num_mac_units = cfg.MIDAP.SYSTEM_WIDTH * cfg.MIDAP.WMEM.NUM
        cps = cfg.SYSTEM.FREQUENCY * 1.0e6
        for layer in path_info:
            stats_layer = self.layer_stats[layer.name]
            it = layer.input[0]
            input_size = it.shape[0] * it.shape[1] * it.shape[2]
            weight_size = 0
            name.append(layer.name)
            dram.append(stats_layer['READ']['DRAM'] + stats_layer['WRITE']['DRAM'])
            wmem_dram.append(stats_layer['READ']['DRAM2WMEM'])
            fmem_dram.append(stats_layer['READ']['DRAM2FMEM'])

            total_delay.append(stats_layer['DELAY']['DRAM'])
            fmem_delay.append(stats_layer['DELAY']['READ_FMEM'])
            wmem_delay.append(stats_layer['DELAY']['READ_WMEM'])

            mac.append(stats_layer['MACs'])
            cycle.append(stats_layer['CLOCK'])
            dram_busy_time.append(stats_layer['DRAM']['BUS_BUSY_TIME'])
            dram_utilization.append(dram_busy_time[-1]/cycle[-1])
            fps.append(cps / stats_layer['CLOCK'])
            utilization.append(stats_layer['MACs'] / (stats_layer['CLOCK'] * num_mac_units))
            main_op = layer.modules[0].op
            if isinstance(main_op, ConvOp):
                conv_clock += stats_layer['CLOCK']
                conv_mac += stats_layer['MACs']
                conv_delay += stats_layer['DELAY']['READ_WMEM']
                weight_size = main_op.orig_weight_size
            elif isinstance(main_op, ConvOp) and main_op.type == 'FC':
                fc_delay += stats_layer['DELAY']['READ_WMEM']
                weight_size = main_op.orig_weight_size
            elif isinstance(main_op, ArithmeticOp):
                residual_delay += stats_layer['DELAY']['READ_WMEM']
            dram_is.append(input_size + int(math.ceil(input_size / (cfg.MIDAP.FMEM.NUM_ENTRIES * cfg.MIDAP.FMEM.NUM * cfg.SYSTEM.DATA_SIZE))) * weight_size)
            dram_ws.append(weight_size + max(int(math.ceil(weight_size / (cfg.MIDAP.WMEM.NUM_ENTRIES * cfg.MIDAP.WMEM.NUM * cfg.SYSTEM.DATA_SIZE))), 1) * input_size)

        print("{}\tDRAM_Access\tDRAM_Delay\tDRAM_Access(FMEM)\tDRAM_Access(WMEM)\tDRAM_Delay(FMEM)\t\
            DRAM_Delay(WMEM)\tMACs\tCYCLE\tDRAM_BUSY_TIME\tDRAM_Utilization\tFPS\tUtilization\tDRAM_Access(IS)\tDRAM_Access(WS)\tUtil.(Conv)\tResidual_Delay(WMEM)\t\
            FC_Delay(WMEM)\tConv_Delay(WMEM)\tDRAM_Dealy_Ratio\tDRAM_Delay_Ratio(FMEM)\tDRAM_Delay_Ratio(WMEM)\tDRAM_Delay_Ratio(WMEM, Conv)".format(model))
        for v in zip(name, dram, total_delay, fmem_dram, wmem_dram, fmem_delay, wmem_delay, mac, cycle, dram_busy_time, dram_utilization, fps, utilization, dram_is, dram_ws):
            print("{}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:.4f}\t{:.0f}\t{:.4f}\t{}\t{}".format(*v))

        # dram, fmem_dram, wmem_dram, total_delay, fmem_delay, wmem_delay, mac, cycle, fps, utilization
        # conv util, residual delay, fc delay, conv delay, dram delay ratio, fmem ratio, wmem ratio, wmem ratio (only conv)
        global_stat = self.global_stats
        print("Total\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}\t\
                \t{:.4f}\t{:.0f}\t{:.4f}\t{}\t{}\t{:.4f}\t{}\t{}\t\
                {}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(global_stat.READ.DRAM + global_stat.WRITE.DRAM, global_stat.DELAY.DRAM,
                                                           global_stat.READ.DRAM2FMEM,
                                                           global_stat.READ.DRAM2WMEM,
                                                           global_stat.DELAY.READ_FMEM, global_stat.DELAY.READ_WMEM,
                                                           global_stat.MACs, global_stat.CLOCK, global_stat.DRAM.BUS_BUSY_TIME,
                                                           global_stat.DRAM.BUS_BUSY_TIME/global_stat.CLOCK, cps / global_stat.CLOCK,
                                                           global_stat.MACs / (global_stat.CLOCK * num_mac_units), sum(dram_is), sum(dram_ws),
                                                           conv_mac / (conv_clock * num_mac_units), residual_delay, fc_delay, conv_delay,
                                                           global_stat.DELAY.DRAM / global_stat.CLOCK,
                                                           global_stat.DELAY.READ_FMEM / global_stat.CLOCK,
                                                           global_stat.DELAY.READ_WMEM / global_stat.CLOCK,
                                                           conv_delay / global_stat.CLOCK))

    def get_dram_delay(self):
        return self.global_stats.DELAY.DRAM

from __future__ import print_function

import logging

from config import cfg
from logger import init_logger

from .pyramid import SingleBlockPartition
from .tile_search_algorithm import *

class NetDivider(object):
    algo = None

    def __init__(self):
        self.partitioner = SingleBlockPartition()

        if cfg.MODEL.TILING_METHOD == 'zero':
            self.algo = ZeroAccessInMidLayerAlgo
        elif cfg.MODEL.TILING_METHOD == 'all':
            self.algo = AllSearchAlgo
        elif cfg.MODEL.TILING_METHOD == 'no':
            self.algo = NoTilingAlgo
        elif cfg.MODEL.TILING_METHOD == 'manual':
            self.algo = ManualAlgo

        self.snapshots = []

        self.logger = init_logger('NetDivider', logging.INFO)

    def devide_network(self, model):
        import sys
        pyramids = self.partitioner.partition(model)

        for p in pyramids:
            tile, dram_breakdown = self.algo.search(p)
            delay, cycle, dram = p.simulate(tile)
            self.logger.info("{}\t{}/{}\t{:,}\t{:,}\t{:,}\t{:,}\t{:,}".format(p.name, tile, p.orig_size, *dram_breakdown, delay, cycle, dram))

        sys.exit()

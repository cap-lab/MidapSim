from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from collections import deque

from config import cfg
from .memory_manager import MemoryManager
from .packet_manager import PacketManager

class RequestInfo(object): # Memorymanager ---(QUEUE)---> DMA
    request_id = 0
    def __init__(
            self,
            request_type = 0, # 0: Read, 1: Write
            request_size = 0, # R/W Size
            buf = None, # memory object (numpy array)
            offset = 0, # memory (buffer) R/W start offset
            dram_address = 0, # target dram address
            ):
        self.id = RequestInfo.request_id
        RequestInfo.request_id += 1
        self.type = request_type
        self.size = request_size
        # In fact, on-chip info should be specified as 'Integrated Address'
        # But, for simple simulation, on-chip data info is specified with [Memory Object & Offset] Format
        # Otherwise, memory object can be replaced with memory id
        self.buf = buf # Memory object
        self.offset = offset
        # DRAM Info
        self.dram_address = dram_address

class QueueElement(list):
    def __init__(self, mem_id, request_time, request):
        super().__init__([mem_id, request_time, request])
        self.mem_id = mem_id
        self.request_time = request_time
        self.request = request

class DMemoryManager(MemoryManager):
    # DMA-based manage 
    # Interface: Donghyun
    # Implementation: Keonjoo
    def __init__(self, manager):
        super().__init__(manager)
        self.lp_rqueue = deque() # Low priority request queue
        self.hp_rqueue = deque() # High priority request queue
        self.wqueue = deque() # Write queue
        self.wait_request_info = [-1 for _ in range(2 + 2 + self.num_fmem)]
        self.dram_address_dict = {}
        # 0 , 1 : WMEM id
        # 2 , 3 : BMMEM id
        # 4 ~ N + 3 : FMEM id

        self.time = 0 #DMA Time
        self.last_time = 0

        # Communicate with HSIM
        if cfg.MIDAP.CORE_ID >= 0:
            self.dram_data = np.fromfile(cfg.DRAM.DUMP_FILE, dtype=np.float32)
            self.packet_manager = PacketManager("../../shared/.args.shmem.dat_" + str(cfg.MIDAP.CORE_ID))

    def __del__(self):
        if cfg.MIDAP.CORE_ID >= 0:
            self.packet_manager.terminatedRequest()
            del self.packet_manager

    def set_dram_info(self, dram_data, dram_dict):
        self.dram_address_dict = dram_dict

    def reset_wmem(self):
        self.wmem_in_use = -1
        for i in range(2):
            if self.wait_request_info[i] != -1:
                self.logger.error("WMEM Request information must be empty before reset. WMEM {}: {}".format(i, self.wait_request_info[i]))
                # Or, it must be a wrong reset timing 
                raise RuntimeError()
    
    ########################### TODO: If the data request size is too huge, you may split the request into small packets (ex: 2KiB)
    def add_request(
            self,
            mem_id = -1,
            request_type = 0, # 0: Read, 1: Write
            request_size = 0, # R/W Size
            buf = None, # memory object (numpy array)
            offset = 0, # memory (buffer) R/W start offset
            dram_address = 0, # target dram address
            ):
        if buf is None:
            self.logger.error("Memory(Buffer) object must be specified!")
            raise RuntimeError()
        request_time = self.manager.stats.total_cycle()
        request_id = -1

        # Split the request into small packets and return the last packet id.
        while request_size > 0:
            size = cfg.MIDAP.PACKET_SIZE
            if request_size < cfg.MIDAP.PACKET_SIZE:
                size = request_size
            request = RequestInfo(request_type, size, buf, offset, dram_address)
            qe = QueueElement(mem_id, request_time, request)

            request_size -= size
            dram_address += size
            offset +=  size

            if request_type == 1:
                self.wqueue.append(qe)
            elif mem_id < 2:
                self.hp_rqueue.append(qe)
            else:
                self.lp_rqueue.append(qe)

            request_id = request.id

        return request_id
    ###########################

    def get_dram_address(self, name, offset):
        return self.dram_address_dict[name] + offset

    def load_wmem(
            self,
            wmem_idx,
            filter_name,
            filter_size = 0,
            wmem_offset = 0,
            dram_offset = 0,
            continuous_request = False):
        # TODO: determine the data written checking (when the wmem data is the result of previous layers)
        wmem_not_in_use = (self.wmem_in_use + 1) % 2
        self.logger.debug("Load data [{}] - addr {}, size {} to WMEM {}, offset {}".format(filter_name, dram_offset, filter_size, (wmem_not_in_use, wmem_idx), wmem_offset))
        dram_address = self.get_dram_address(filter_name, dram_offset) 
        request_id = self.add_request(
                wmem_not_in_use,
                0,
                filter_size,
                self.wmem[wmem_not_in_use][wmem_idx],
                wmem_offset,
                dram_address)
        if not continuous_request:
            if self.wait_request_info[wmem_not_in_use] != -1:
                self.logger.error("WMEM Request information must be empty before the new request. WMEM {}: {}".format(wmem_not_in_use, self.wait_request_info[wmem_not_in_use]))
            self.wait_request_info[wmem_not_in_use] = request_id
            self.update_queue()
    
    def load_fmem(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0):
        # TODO: determine the data written checking (when the fmem data is the result of previous layers)
        self.logger.debug("Load data [{}] - addr {}, size {} to FMEM {}".format(data_name, dram_offset, data_size, fmem_idx))
        dram_address = self.get_dram_address(data_name, dram_offset)
        mem_id = fmem_idx + 4
        request_id = self.add_request(
                mem_id,
                0,
                data_size,
                self.fmem[fmem_idx],
                fmem_offset,
                dram_address)
        if self.wait_request_info[mem_id] != -1:
            self.logger.error("FMEM Request information must be empty before the new request. FMEM {}: {}".format(fmem_idx, self.wait_request_info[mem_id]))
        self.wait_request_info[mem_id] = request_id
        self.update_queue() # Not necessary

    def load_bmmem(self, bias_name, bias_size):
        bmmem_not_in_use = (self.bmmem_in_use + 1) % 2
        dram_address = self.get_dram_address(bias_name, 0)
        mem_id = bmmem_not_in_use + 2
        request_id = self.add_request(
                mem_id,
                0,
                bias_size,
                self.bmmem[bmmem_not_in_use],
                0,
                dram_address)
        if self.wait_request_info[mem_id] != -1:
            self.logger.error("BMMEM Request information must be empty before the new request. BMMEM {}: {}".format(bmmem_not_in_use, self.wait_request_info[mem_id]))
        self.wait_request_info[mem_id] = request_id
        self.update_queue() # Not necessary 
    
    def write_dram(self, data_name, offset, data):
        buf = data.copy()
        dram_address = self.get_dram_address(data_name, offset)
        self.add_request(
                -1,
                1,
                data.size,
                buf,
                0,
                dram_address)
        return 0

    def read_wmem(self, buf, extended_cim, address):
        # Wait for the Request
        if not extended_cim and address + self.system_width > self.wmem_size:
            self.logger.error("WMEM Size: {} vs Requested Address: {}".format(self.wmem_size, address + self.system_width))
            raise ValueError("Wrong Address")
        if address + self.system_width > self.extended_wmem_size:
            self.logger.error("Extended WMEM Size: {} vs Requested Address: {}".format(self.extended_wmem_size, address + self.system_width))
            raise ValueError("Wrong Address")
        time_gap = 0
        if self.wait_request_info[self.wmem_in_use] >= 0:
            time_gap = self.wait_for_end_request(self.wmem_in_use)
            if time_gap > 0:
                self.logger.debug("WMEM {} load delay occured & time_gap = {}".format(self.wmem_in_use, time_gap))
        data_set_size = 1 if extended_cim else self.num_wmem
        buf[:data_set_size,:self.system_width] = \
                self.wmem[self.wmem_in_use, :data_set_size, address:address+self.system_width]
        return time_gap

    def read_fmem(self, buf, bank_idx, address):
        time_gap = 0
        mem_id = bank_idx + 4
        if self.wait_request_info[mem_id] >= 0:
            time_gap = self.wait_for_end_request(mem_id)
            if time_gap > 0:
                self.logger.debug("FMEM {} load delay occured & time_gap = {}".format(bank_idx, time_gap))
        buf[:self.system_width] = self.fmem[bank_idx, address:address+self.system_width]
        return time_gap

    def read_bmmem(self, buf, extended_cim, address):
        time_gap = 0
        mem_id = self.bmmem_in_use + 2
        if self.wait_request_info[mem_id] >= 0:
            time_gap = self.wait_for_end_request(mem_id)
            if time_gap > 0:
                self.logger.debug("BMMEM {} load delay occured & time_gap = {}".format(self.bmmem_in_use, time_gap))
        size = self.system_width if extended_cim else self.num_wmem
        buf[:size] = self.bmmem[self.bmmem_in_use, address:address + size]
        return time_gap
    
    def write_fmem(self, bank_idx, address, data):
        mem_id = bank_idx + 4
        if self.wait_request_info[mem_id] >= 0:
            self.logger.debug("FMEM {} must not be touched... DRAM2FMEM READ & FMEM Write simultaenously occur".format(bank_idx))
        self.fmem[bank_idx, address:address+data.size] = data[:]
        return 0

    ###### TODO: DMA-aware Implementation
    def update_queue(self, mem_id = -1, sync = False):
        current_time = self.manager.stats.total_cycle()
        # mem_id < 1: Update DMA - HSIM Timer until t <= current_time
        # When mem_id >= 0: Run DMA until the request of id = target_request_id is finished
        # time = 0 #DMA Time

        if mem_id == -1:
            check = lambda t, md: t <= current_time
        elif sync:
            check = lambda t, md: len(self.wqueue) > 0
        else:
            check = lambda t, md: self.wait_request_info[md] >= 0
        while check(self.time, mem_id):
            # do sth // junk code // not only the priority of the request but timing must be considered together
            if len(self.wqueue) > 0:
                mid, req_time, request = self.wqueue.popleft()
            elif len(self.hp_rqueue) > 0:
                mid, req_time, request = self.hp_rqueue.popleft()
            elif len(self.lp_rqueue) > 0:
                mid, req_time, request = self.lp_rqueue.popleft()
            else:
                self.time = current_time
                self.packet_manager.elapsedRequest(current_time)
                return self.time
            #else:
            #    if len(self.lp_rqueue) == 0:
            #        raise RuntimeError("Cannot find the request for mem id {}, wait_request_info".format(mem_id, self.wait_request_info))
            #    mid, req_time, request = self.lp_rqueue.popleft()
            if request.type == 1:
                self.time = max(self.time, req_time)
                self.packet_manager.writeRequest(request.dram_address, request.size, request.buf[request.offset:request.offset+request.size], self.time)
                self.time += 1 # Junk code
                # Write the request.buf on the DRAM
                del request.buf
            else:
                self.time = max(self.time, req_time)
                data, work_time = self.packet_manager.readRequest(request.dram_address, request.size, self.time)
                if self.time < work_time:
                    self.time = work_time
                #self.time += work_time # Junk code ... the time that DMA receives HSIM respond
                if self.wait_request_info[mid] == request.id:
                    self.wait_request_info[mid] = -1 # Processed
                request.buf[int(request.offset):int(request.offset+request.size)] = data[0:int(request.size)] # READ DRAM DATA.. dram[request.dram_address:request.dram_address + request.size
        return int(self.time) # DMA Time
    ###### TODO End


    def wait_for_end_request(self, mem_id):
        current_time = self.manager.stats.total_cycle()
        end_time = self.update_queue(mem_id = mem_id)
        if end_time < current_time:
            self.update_queue()
        return max(0, end_time - current_time)

    def sync(self): # Called when each layer processing is finished
        time = self.update_queue(mem_id = 0, sync = True)
        return time

class TDMemoryManager(DMemoryManager): # Test dump memory file & DMA logic works
    def __init__(self, manager):
        super().__init__(manager)
    
    def set_dram_info(self, dram_data, dram_dict):
        self.dram_address_dict = dram_dict
        self.dram_data = dram_data

    def update_queue(self, mem_id = -1, sync = False):
        current_time = self.manager.stats.total_cycle()
        # mem_id < 1: Update DMA - HSIM Timer until t <= current_time
        # When mem_id >= 0: Run DMA until the request of id = target_request_id is finished
        time = 0 #DMA Time
        if mem_id == -1:
            check = lambda t, md: t <= current_time
        elif sync:
            check = lambda t, md: len(self.wqueue) > 0
        else:
            check = lambda t, md: self.wait_request_info[md] >= 0
        while check(time, mem_id):
            # do sth // junk code // not only the priority of the request but timing must be considered together
            # 1) req_time < dma_time : highest priority
            # 2) wqueue > hp_rqueue > lp_rqueue
            if len(self.wqueue) > 0:
                mid, req_time, request = self.wqueue.popleft()
            elif len(self.hp_rqueue) > 0:
                mid, req_time, request = self.hp_rqueue.popleft()
            else:
                if len(self.lp_rqueue) == 0:
                    if mem_id < 0:
                        break
                    raise RuntimeError("Cannot find the request for mem id {}, wait_request_info".format(mem_id, self.wait_request_info))
                mid, req_time, request = self.lp_rqueue.popleft()
            if request.type == 1:
                time = max(time, req_time) + 1 # Junk code, update timer
                self.dram_data[request.dram_address: request.dram_address + request.size] = request.buf[request.offset:request.offset + request.size]
                # Write the request.buf on the DRAM
                del request.buf
            else:
                time = max(time, req_time) + 10 # Junk code ... the time that DMA receives HSIM respond
                if self.wait_request_info[mid] == request.id:
                    self.wait_request_info[mid] = -1 # Processed
                    self.logger.debug("REQUEST ID {} FOR MEM {} is finished".format(self.wait_request_info[mid],mid))
                request.buf[request.offset:request.offset+request.size] = \
                        self.dram_data[request.dram_address:request.dram_address + request.size]
                # READ DRAM DATA.. dram[request.dram_address:request.dram_address + request.size
        return time # DMA Time

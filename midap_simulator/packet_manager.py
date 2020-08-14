import mmap
import numpy as np

from time import sleep

import os

class PacketManager(object):
    buf_size = 0x1000
    packet_size = 2072

    #typedef struct _Packet{
    #   PacketType type;
    #   uint32_t size;
    #   uint64_t cycle;
    #   uint32_t address;
    #   uint8_t data[8];
    #   uint32_t flags;
    #} Packet;
    data_type = np.dtype([('type', 'u4'), ('size', 'u4'), ('cycle', 'u8'), ('address', 'u4'), ('data', 'f4', (512)), ('flags', 'u4')])
    #typedef struct {
    #   volatile int start;			/* index of oldest element              */
    #   volatile int end;			/* index at which to write new element  */
    #   int capacity;
    #   int size;
    #   Packet elems[PKT_BUFFER_SIZE+1];		/* vector of elements                   */
    #} PacketBuffer;
    data_info_type = np.dtype([('start', 'u4'), ('end', 'u4'), ('capacity', 'u4'), ('size', 'u4')])

    def __init__(self, path):
        self._infoPath = path

        self._lastCycle = 0

        self._pType = self.enum('read', 'write', 'elapsed', 'terminated')
        self._pFlag = self.enum('none', 'flush')

        f = open(path, 'r')
        name = f.readline()
        ib_name = f.readline()
        bi_name = f.readline()
        f.close()

        ibFile = open('/dev/shm' + ib_name.rstrip('\n'), 'r+')
        self._sendBuffer = mmap.mmap(ibFile.fileno(), 0, mmap.PROT_READ | mmap.PROT_WRITE)
        ibFile.close()

        biFile = open('/dev/shm' + bi_name.rstrip('\n'), 'r+')
        self._receiveBuffer = mmap.mmap(biFile.fileno(), 0, mmap.PROT_READ | mmap.PROT_WRITE)
        biFile.close()

        # Check if the connection is established.
        self.writeRequest(0x0, 4, 0, 0)

    def enum(self, *sequential, **named):
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)

    def isEmpty(self, buffer):
        start, end, _, _ = self.readBufInfo(buffer)
        return start == end

    def isFull(self, buffer):
        start, end, _, _ = self.readBufInfo(buffer)
        return (end + 1) % self.buf_size == start;

    def readBufInfo(self, buffer):
        buffer.seek(0)
        data_info = np.array(np.frombuffer(buffer.read(16), dtype=self.data_info_type), dtype=self.data_info_type)

        return data_info['start'], data_info['end'], data_info['capacity'], data_info['size']

    def readPacket(self):
        buffer = self._receiveBuffer
        while self.isEmpty(buffer) == True:
            sleep(0.000000001)

        start, end, capacity, size = self.readBufInfo(self._receiveBuffer)

        buffer.seek(16 + int(start) * self.packet_size)
        data = np.array(np.frombuffer(buffer.read(self.packet_size), dtype=self.data_type), dtype=self.data_type)

        # Increase the read index (start)
        start = (start + 1) % self.buf_size
        buffer.seek(0)
        buffer.write(start.tobytes())

        return data

    def writePacket(self, packet):
        buffer = self._sendBuffer
        while self.isFull(buffer) == True:
            sleep(0.000000001)

        start, end, capacity, size = self.readBufInfo(buffer)

        data = np.array(packet, dtype=self.data_type)
        buffer.seek(16 + int(end) * self.packet_size)
        buffer.write(data.tobytes())

        # Increase the write index (end)
        end = (end + 1) % self.buf_size
        buffer.seek(4)
        buffer.write(end.tobytes())
        buffer.flush()

    def readRequest(self, addr, size, cycle, flush = False):
        delta_cycle = 0
        if cycle > self._lastCycle:
            delta_cycle = cycle - self._lastCycle

        #packet = np.array((self._pType.read, size * 4, delta_cycle, addr * 4, 0, 0), dtype=self.data_type)
        packet = np.array((self._pType.read, size, cycle, addr * 4, 0, 0), dtype=self.data_type)
        if flush == True:
            packet['flags'] = self._pFlag.flush

        self.writePacket(packet)
        packet = self.readPacket()
        data = packet['data']
        data = np.resize(data, int(size))
        self._lastCycle = cycle

        return data, packet['cycle']

    def writeRequest(self, addr, size, data, cycle):
        delta_cycle = 0
        if cycle > self._lastCycle:
            delta_cycle = cycle - self._lastCycle

        #packet = np.array((self._pType.write, size * 4, delta_cycle, addr * 4, np.resize(data, 512), 0), dtype=self.data_type)
        packet = np.array((self._pType.write, size, cycle, addr * 4, np.resize(data, 512), 0), dtype=self.data_type)
        self.writePacket(packet)

        self._lastCycle = cycle

    def elapsedRequest(self, cycle):
        delta_cycle = 0
        if cycle > self._lastCycle + 100:
            delta_cycle = cycle - self._lastCycle

        if delta_cycle > 0:
            packet = np.array((self._pType.elapsed, 0, int(cycle), 0, 0, 0), dtype=self.data_type)
            self.writePacket(packet)
            self._lastCycle = cycle

    def terminatedRequest(self):
        packet = np.array((self._pType.terminated, 0, 0, 0, 0, 0), dtype=self.data_type)
        self.writePacket(packet)


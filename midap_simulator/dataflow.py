def generate_dataflow_info(
        phase = 0,
        loc = None,
        filter_idx = None,
        out_x = 0,
        out_y = 0,
        out_z = 0,
        fmem_idx = 0,
        fmem_row = -1,
        wmem_row = -1,
        broadcast_offset = 0,
        delete_foffset = 0,
        delete_boffset = 0,
        reset = True,
        last = False,
        junk = False,
        ):
    if filter_idx is not None:
        out_z = filter_idx
    if loc is not None:
        if len(loc) == 2:
            out_x, out_y = loc
        elif len(loc) == 3:
            out_x, out_y, out_z = loc

    return Dataflow(
            phase,
            out_x,
            out_y,
            out_z,
            fmem_idx,
            fmem_row,
            wmem_row,
            broadcast_offset,
            delete_foffset,
            delete_boffset,
            reset,
            last,
            junk
            )

class Dataflow():
    def __init__(
            self,
            phase,
            out_x,
            out_y,
            out_z,
            fmem_idx,
            fmem_row,
            wmem_row,
            broadcast_offset,
            delete_foffset,
            delete_boffset,
            reset,
            last,
            junk
            ):

        self.phase = phase
        self.out_x = out_x
        self.out_y = out_y
        self.out_z = out_z
        self.fmem_idx = fmem_idx
        self.fmem_row = fmem_row
        self.wmem_row = wmem_row
        self.broadcast_offset = broadcast_offset
        self.delete_foffset = delete_foffset
        self.delete_boffset = delete_boffset
        self.reset = reset
        self.last = last
        self.junk = junk

    def __repr__(self):
        phase = None
        p = self.phase
        if p == 0:
           phase = 'None'
        elif p == 1:
           phase = 'Main'
        elif p == 2:
           phase = 'Reduction'
        elif p == 3:
           phase = 'End'
        else:
           raise ValueError("Unknown Phase")
        loc = [self.out_x, self.out_y, self.out_z]
        fmem_addr = [self.fmem_idx, self.fmem_row]
        wmem_addr = self.wmem_row
        alignment_info = [self.broadcast_offset, self.delete_foffset, self.delete_boffset]
        etc = [self.reset, self.last, self.junk]
        out_str = '[Phase: ' + str(phase)
        if p in [0, 3]:
           return out_str + ']'
        out_str += ', Loc: {}, [reset, last, junk] = {}'.format(loc, etc)
        if p == 2:
           return out_str + ']'
        out_str += ', FMEM Address: {}, WMEM Address: {}, Alignment Info: {}]'.format(fmem_addr, wmem_addr, alignment_info)
        return out_str

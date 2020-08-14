import numpy as np

class VirtualTensor(object):
    mapping_algorithms = ['default', 'linear']
    def __init__(self, **kwargs):
        self.name = None
        self.orig_shape = None
        self.shape = None
        self.scale = [1, 1, 1]
        self.offset = [0, 0, 0]
        self.mapping_type = 'default'
        
    def set_tensor(self, name, shape, orig_shape = None, mapping_type = 'default', offset = None, scale = None, **kwargs):
        self.name = name
        self.shape = shape
        if orig_shape is None:
            orig_shape = shape
        self.orig_shape = orig_shape
        if mapping_type.lower() not in self.mapping_algorithms:
            raise ValueError("mapping_type should be one of {}".format(self.mapping_algorithms))
        self.mapping_type = mapping_type.lower()
        if offset is not None:
            self.offset = offset
        if scale is not None:
            self.scale = scale
        if not isinstance(self.offset, list) or len(self.offset) != 3:
            raise ValueError("STensor offset should be None or 3-dim list")

    def get_loc(self, loc): # return relative address / invalid: return -1
        pass
     
    def get_address(self, loc):
        w, h, c = self.orig_shape
        out_x, out_y, out_z = loc
        address = out_x * h * c + out_y * c + out_z
        return address

    def calculate_linear_address(self, loc):
        pass

    def __repr__(self):
        return self.str_name() + self.str_algo() + self.str_shape() + '\n'
    
    def str_name(self):
        return "Tensor name: " + str(self.name)
    
    def str_algo(self):
        if self.mapping_type == 'default':
            return ''
        algo = "\nMapping algorithm: " + self.mapping_type
        algo_constant = "\nScale: " + str(self.scale) + " || Offset: " + str(self.offset)
        info = "\nOriginal shape: " + str(self.orig_shape)
        return algo + algo_constant + info

    def str_shape(self):
        return " || (Virtualized) Tensor shape: " + str(self.shape)

class VInputTensor(VirtualTensor):
    mapping_algorithms = ['default', 'linear', 'valid']
    def __init__(self, flip_x = False, **kwargs):
        super(VInputTensor, self).__init__(**kwargs)
        self.flip_x = flip_x

    def valid(self, x, y):
        if self.mapping_type == 'valid':
            scale_x, scale_y, scale_z = self.scale
            if self.flip_x:
                x = self.shape[0] - x - 1
            return x % scale_x == 0 and y % scale_y == 0
        return True
    
    def get_loc(self, loc):
        return self.calculate_linear_loc(loc)

    def calculate_linear_loc(self, loc):
        in_x, in_y, in_z = loc
        scale_x, scale_y, scale_z = self.scale
        offset_x, offset_y, offset_z = self.offset
        w, h, c, = self.orig_shape
        out_x = in_x // scale_x + offset_x
        out_y = in_y // scale_y + offset_y
        out_z = in_z // scale_z + offset_z
        return (out_x, out_y, out_z)

class VOutputTensor(VirtualTensor):
    mapping_algorithms = ['default', 'linear', 'shuffle']
    def __init__(self, reverse_write = False, write_on_dram = False, virtual = False, **kwargs):
        super(VOutputTensor, self).__init__(**kwargs)
        self.write_on_dram = write_on_dram
        self.reverse_write = reverse_write
        self.virtual = virtual

    def get_loc(self, loc):
        locs = []
        if self.mapping_type == 'default':
            locs = [loc]
        if self.mapping_type == 'linear':
            scale_x, scale_y, scale_z = self.scale
            pivot_x, pivot_y, pivot_z = self.calculate_linear_loc(loc)
            locs = [(pivot_x+ i,pivot_y+j,pivot_z+k) for i in range(scale_x) for j in range(scale_y) for k in range(scale_z)]
        if self.mapping_type == 'shuffle':
            locs = [self.calculate_shuffle_loc(loc)]
        return locs 

    def get_output_loc(self, loc):
        locs = self.get_loc(loc)
        return list(map(self.calculate_output_loc, locs))

    def calculate_output_loc(self, loc):
        in_x, in_y, in_z = loc
        offset_x, offset_y, offset_z = self.offset
        if self.reverse_write:
            in_x = self.shape[0] - in_x - 1
        return (in_x + offset_x, in_y + offset_y, in_z + offset_z)

    def calculate_linear_loc(self, loc):
        in_x, in_y, in_z = loc
        scale_x, scale_y, scale_z = self.scale
        out_x = scale_x * in_x
        out_y = scale_y * in_y
        out_z = scale_z * in_z
        return (out_x, out_y, out_z)

    def calculate_shuffle_loc(self, loc):
        r = self.scale[0]
        w, h, c = self.shape
        c = c / (r*r)
        in_x, in_y, in_z = loc
        out_x = in_x * r + math.ceil(in_z / (r * c)) 
        out_y = in_y * r + math.ceil(in_z / c) % r
        out_z = in_z % c
        return (out_x, out_y, out_z)

    def __repr__(self):
        s = super().__repr__()
        if self.reverse_write:
            r = "Reverse Write: ON\n"
        else:
            r = "Reverse Write: OFF\n"
        return r+s


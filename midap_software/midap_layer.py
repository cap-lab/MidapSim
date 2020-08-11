from __future__ import absolute_import, division, print_function, unicode_literals

from functools import reduce

from acc_utils.model_utils import *
from config import cfg
from generic_op import *
from midap_software.control_info import ControlInfo


class MidapLayer(object):
    def __init__(self, main_op, mapping_op=None, sub_op=None, reduction=False):
        self.input = main_op.input_layers if mapping_op is None else mapping_op.input_layers
        # v1.3.0 memory mapping func added
        # mapping func format
        self.mapping_func = None
        self.valid_func = None
        # 1) mapping op is not none
        self.scale_w = 1
        self.scale_h = 1
        self.x_offset = (0, 0)
        self.y_offset = (0, 0)
        self.dilation = main_op.dilation if isinstance(main_op, ConvOp) else 1
        self.mapping_op = mapping_op
        if isinstance(mapping_op, UpsampleOp):
            # padding is not allowed for upsample op
            scale_h, scale_w = mapping_op.k_h, mapping_op.k_w
            self.mapping_func = lambda x, y: (x // scale_w, y // scale_h)
            if mapping_op.algorithm == 'Zero':
                self.valid_func = lambda x, y: (x % scale_w) == 0 and (y % scale_h) == 0
            else:
                self.valid_func = lambda x, y: True
            self.scale_w, self.scale_h = scale_w, scale_h
        elif isinstance(mapping_op, Crop):
            x1, x2 = mapping_op.crop_x if mapping_op.crop_x else (0, 0)
            y1, y2 = mapping_op.crop_y if mapping_op.crop_y else (0, 0)
            self.mapping_func = lambda x, y: (x + x1, y + y1)
            self.valid_func = lambda x, y: all([x + x1 >= 0, y + y1 >= 0, x < mapping_op.output_tensor.shape[0], y < mapping_op.output_tensor.shape[1]])
            self.x_offset = (x1, -x2)
            self.y_offset = (y1, -y2)
        #
        self.output_tensor = main_op.output_tensor if sub_op is None else sub_op.output_tensor
        self.output_name = None
        self.name = main_op.name if sub_op is None else sub_op.name
        self.main_op = main_op
        self.sub_op = sub_op  # Will be deprecated
        self.next = []
        self.offset_axis = 2
        self.offset = (0, 0)
        self.q_param = None
        self.write_on_dram = False

        self.load_filter_once = False

        self.control_info = ControlInfo(self)

        self.extra_output_info = []  # Main op
        self.reduction = reduction
        self.have_reduction_layer = False

        self.parallel_type = None
        if main_op.type in ['Add', 'Sum', 'Mul', 'Depthwise'] or isinstance(main_op, PoolOp):
            self.parallel_type = 'X'

    def __del__(self):
        del self.main_op, self.sub_op
        del self.input, self.next
        del self.output_tensor
        del self.control_info
        del self.extra_output_info

    def set_sub_op(self, sub_op):
        # input_check = isinstance(self.main_op, ConvOp) and cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER == 'GEMM'
        # if input_check:
        #    input_check = 'data' in self.input[0]
        if self.main_op.type == 'HEAD' or len(self.main_op.next) > 1:
            return False
        elif self.sub_op is None and any([  # [isinstance(self.main_op, ConvOp) and isinstance(sub_op, PoolOp),
                isinstance(sub_op, UpsampleOp), isinstance(sub_op, Crop)]):
            self.sub_op = sub_op
            self.output_tensor = sub_op.output_tensor
            self.name = sub_op.name
            return True
        else:
            return False

    def set_reduction_op(self, reduction_op):
        if self.have_reduction_layer or self.reduction or isinstance(self.main_op, ConcatOp):
            return None
        self.have_reduction_layer = True
        reduction_layer = MidapLayer(reduction_op, reduction=True)
        reduction_layer.parallel_type = self.parallel_type
        self.next.append(reduction_layer)
        return reduction_layer

    def __repr__(self):
        return '<< name: {}, next: {}, output_shape: {}\n[[control info: {}]]\n[main_op]{}\n'.format(
            self.name, [layer.name for layer in self.next], self.output_tensor.shape, self.control_info, self.main_op)

    def log_print(self, func=print):
        func("|{:^20s}|{:^15s}|{:^20s}|{:^20s}|".format(self.name, self.main_op.type,
                                                        str(self.get_input_shape()), str(list(self.get_output_shape()))))

    @property
    def is_weight_in_wmem(self):
        if isinstance(self.main_op, ConvOp):
            return self.load_filter_once or self.main_op.type == 'Depthwise'
        return True

    def setup(self):
        self.yz_plane_size = np.prod(self.output_tensor.shape[1:])
        self._num_planes_per_fmem = cfg.MIDAP.FMEM.NUM_ENTRIES // self.yz_plane_size
        self.require_fmem = div_ceil(self.output_tensor.shape[0], self.num_planes_per_fmem)
        if isinstance(self.main_op, Crop):
            self.require_fmem = 0

        # XXX ESWeek20
        self.require_total_fsize = np.prod(self.output_tensor.shape)
        self.require_fsize = (self.num_planes_per_fmem * self.yz_plane_size, (self.output_tensor.shape[0] % self.num_planes_per_fmem) * self.yz_plane_size)

        if len(self.next) == 1 and self.have_reduction_layer:
            self.require_fmem = 0

        # Check if the model run on MIDAP.
        input_layer = self.input[0] if self.input else None
        # FIXME
        if isinstance(self.main_op, ConvOp) and input_layer and input_layer.num_planes_per_fmem < self.main_op.k_w - max(self.main_op.pad_l, self.main_op.pad_r):
            raise MIDAPError("Can not process because kernel width is too large."
                             "(Layer Name: {} Num Plain: {} Kernel W: {}".format(self.name, input_layer.num_planes_per_fmem, self.main_op.k_w))
        # elif isinstance(self.main_op, PoolOp) and not self.reduction and input_layer and input_layer.num_planes_per_fmem < self.main_op.k_w - 1:
        #     raise MIDAPError("Can not process because kernel width is too large."
        #                      "(Layer Name: {} Input Tensor: {}".format(self.name, self.get_input_shape()))

    def get_input_fragments(
        self,
        num_fragments,
        head,
        overlap=False
    ):
        input_layer = self.input[0]
        fragments = input_layer.get_fragments(
            num_fragments,
            head=head,
            x_offset=self.x_offset,
            layer=self,
            overlap=overlap,
        )
        return fragments

    def get_output_fragments(self, num_bank, next_layer=None):
        control_info = self.control_info
        if next_layer and isinstance(next_layer.main_op, Crop):
            x_offset = (next_layer.main_op.crop_x[0], -next_layer.main_op.crop_x[1])
        elif next_layer:
            x_offset = next_layer.x_offset
        else:
            x_offset = (0, 0)
        output_fragments = self.get_fragments(num_bank, 0, x_offset=x_offset)
        output_fragments = list(reversed(output_fragments)) if control_info.reverse_write else output_fragments
        return output_fragments

    # FIXME x_offset..?
    def get_fragments(
            self,
            num_fragments,
            head=0,
            x_offset=(0, 0),
            layer=None,
            overlap=False):
        if not layer:
            layer = self
            overlap = False
        main_op = layer.main_op
        head += x_offset[0]
        input_shape = self.output_tensor.shape
        overlap_load = overlap and isinstance(main_op, ConvPoolOpBase) and main_op.type != 'Gemm'
        fixed_plane_size = self.num_planes_per_fmem
        if overlap_load and main_op.k_w > 1:
            head = layer.scale_w * head
            head = max(head - (main_op.k_w - 1) * layer.dilation, 0)
            head = div_ceil(head - main_op.pad_l, main_op.stride) * main_op.stride + main_op.pad_l
            if layer.mapping_func is not None:
                head, _ = layer.mapping_func(head, 0)
        fragments = []
        for i in range(num_fragments):
            if head >= input_shape[0] - x_offset[1]:
                break
            tail = min(head + fixed_plane_size, input_shape[0] - x_offset[1])
            fragments.append([head, tail])
            head = tail
        return fragments

    def get_output_x(self, fragment, next_available=False):
        reverse_write = self.control_info.reverse_write
        input_shape = self.input[0].output_tensor.shape
        input_x = self.scale_w * fragment[1] - 1 - self.x_offset[0]
        main_x = input_x
        main_op = self.main_op
        if input_x >= self.scale_w * input_shape[0] - self.x_offset[1] - self.x_offset[0] - 1:
            return 0 if reverse_write else self.output_tensor.shape[0] - 1
        elif main_op.type == 'Gemm':
            main_x = main_x // self.output_tensor.shape[1]
        elif isinstance(main_op, PoolOp) and main_op.global_pooling == 1:
            return 0
        elif isinstance(main_op, ConvPoolOpBase):
            main_x += main_op.pad_l
            if not next_available:
                main_x = main_x - main_op.k_w + 1
            main_x = main_x // main_op.stride
        if isinstance(self.sub_op, PoolOp):
            pool_op = self.sub_op
            if pool_op.global_pooling == 1:
                return 0
            pool_x = main_x
            pool_x += pool_op.pad_l - pool_op.k_w + 1
            pool_x = pool_x // pool_op.stride
        elif isinstance(self.sub_op, UpsampleOp):
            pool_x = min(main_x * self.sub_op.k_w,
                         self.output_tensor.shape[0] - 1)
        elif isinstance(self.sub_op, Crop):
            pool_x = main_x - self.sub_op.crop_x[0]
        else:
            pool_x = main_x
        pool_x = min(pool_x, self.output_tensor.shape[0] - 1)
        return self.output_tensor.shape[0] - pool_x - 1 if reverse_write else pool_x

    @property
    def num_planes_per_fmem(self):
        op = self.main_op
        out_shape = op.output_tensor.shape
        num_planes = self._num_planes_per_fmem
        return (num_planes if op.type == 'HEAD' or out_shape[0] >= num_planes else out_shape[0])

    def get_fsize_per_bank(self):
        if self.main_op.type == 'Gemm':
            op = self.main_op
            aligned_data_unit = div_ceil(op.k_h * op.k_w * self.gemm_input_shape[2], cfg.MIDAP.SYSTEM_WIDTH) * cfg.MIDAP.SYSTEM_WIDTH
            num_gemm_input = int(cfg.MIDAP.FMEM.NUM_ENTRIES / aligned_data_unit)
            return num_gemm_input * aligned_data_unit

        yz_plane_size = np.prod(self.input[0].main_op.output_tensor.shape[1:])
        return yz_plane_size * self.input[0].num_planes_per_fmem

    def get_rest_fsize(self):
        if self.main_op.type == 'Gemm':
            op = self.main_op
            aligned_data_unit = div_ceil(op.k_h * op.k_w * self.gemm_input_shape[2], cfg.MIDAP.SYSTEM_WIDTH) * cfg.MIDAP.SYSTEM_WIDTH
            num_gemm_input = int(cfg.MIDAP.FMEM.NUM_ENTRIES / aligned_data_unit)
            return aligned_data_unit * (np.prod(self.output_tensor.shape[:2]) % num_gemm_input)

        yz_plane_size = np.prod(self.input[0].main_op.output_tensor.shape[1:])
        return yz_plane_size * (self.main_op.output_tensor.shape[0] % self.input[0].num_planes_per_fmem)

    def get_input_size(self):  # This func might be affected
        if self.main_op.type == 'Gemm':
            op = self.main_op
            aligned_data_unit = div_ceil(op.k_h * op.k_w * self.gemm_input_shape[2], cfg.MIDAP.SYSTEM_WIDTH) * cfg.MIDAP.SYSTEM_WIDTH
            return aligned_data_unit * np.prod(self.output_tensor.shape[:2])
        return reduce(lambda x, y: x + y.main_op.output_tensor.size, [0] + self.input)

    def get_input_shape(self):  # This func might be affected
        if not self.input:
            return None
        if self.main_op.type == 'Gemm':
            return self.gemm_input_shape
        shape = list(self.input[0].output_tensor.shape[:2])
        chan = reduce(lambda x, y: x + y.main_op.output_tensor.shape[2], [0] + self.input)
        return shape + [chan]

    def get_output_size(self):
        return self.main_op.output_tensor.size

    def get_output_shape(self):
        return self.main_op.output_tensor.shape

    def get_filter_size(self):
        if isinstance(self.main_op, ConvOp):
            weight_size = self.main_op.weight.size
            return weight_size / self.main_op.output_tensor.shape[2]
        else:
            return 0

    def get_weight_size(self):
        return self.main_op.weight.size if isinstance(self.main_op, ConvOp) else 0

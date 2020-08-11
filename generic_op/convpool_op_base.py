from past.builtins import basestring

from acc_utils.errors import _assert

from .operator_base import OperatorBase


class ConvPoolOpBase(OperatorBase):
    def __init__(
            self,
            kernel=1,
            stride=1,
            pad=0,
            **kwargs
    ):
        super(ConvPoolOpBase, self).__init__(**kwargs)
        if isinstance(kernel, list) or isinstance(kernel, tuple):
            _assert(len(kernel) == 2,
                    "kernel with int value or (int, int) format is only supported")
            self.k_h, self.k_w = kernel
        else:
            self.k_h, self.k_w = kernel, kernel
        self.stride = stride
        if isinstance(pad, list) or isinstance(pad, tuple):
            _assert(len(pad) == 4 or len(
                pad) == 2, "pad with int value or (pad_h, pad_w) = int, int or (pad_t, pad_b, pad_l, pad_r) - int, int ,int ,int format is only supported")
            if len(pad) == 2:
                pad = [pad[0], pad[0], pad[1], pad[1]]
        elif isinstance(pad, basestring):
            if pad == 'VALID':
                pad = [0, 0, 0, 0]
            else:
                _assert(False, 'not supported type for padding')
        else:
            pad = [pad for _ in range(4)]
        self.pad_t, self.pad_b, self.pad_l, self.pad_r = pad

    def flip_operation(self):
        self.pad_r, self.pad_l = self.pad_l, self.pad_r

    def __repr__(self):
        options = "kernel(h,w): {}\tstride: {}\tpad: {}\t".format(
            [self.k_h, self.k_w], self.stride, [self.pad_t, self.pad_b, self.pad_l, self.pad_r])
        return super(ConvPoolOpBase, self).__repr__() + options

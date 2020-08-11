from acc_utils.errors import _assert

from .operator_base import OperatorBase


class ConcatOp(OperatorBase):
    def __init__(self, op_type='Concat', axis = 1, concat_info=None, **kwargs):
        super(ConcatOp, self).__init__(op_type=op_type, **kwargs)
        _assert(isinstance(concat_info, list),
                'concat_info must be given as a size list')
        self.axis = axis
        self.size_info = [sum(concat_info[:i])
                          for i in range(len(concat_info))]

    def __repr__(self):
        return super(ConcatOp, self).__repr__() + "offset information: {}\n".format(self.size_info)

    def tensor_to_midap_tensor(self):
        if self.order == 'NCHW':
            super(ConcatOp, self).tensor_to_midap_tensor()
            axis_translate_arr = [-1, 2, 1, 0]
            self.axis = axis_translate_arr[self.axis]


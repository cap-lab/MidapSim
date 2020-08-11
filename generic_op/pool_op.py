from .convpool_op_base import ConvPoolOpBase


class PoolOp(ConvPoolOpBase):
    def __init__(
            self,
            op_type='Pool',
            pool_type=None,
            global_pooling=False,
            **kwargs
    ):
        super(PoolOp, self).__init__(op_type=op_type, **kwargs)
        self.global_pooling = global_pooling
        if pool_type is not None:
            self.type = pool_type

    def flip_operation(self):
        self.pad_r, self.pad_l = self.pad_l, self.pad_r

from .arithmetic_op import ArithmeticOp


class MulOp(ArithmeticOp):
    def __init__(self, **kwargs):
        # TODO: add other arguments for the upsampling operation
        # Upsampling ratio, ...
        super(MulOp, self).__init__(op_type='Mul', **kwargs)

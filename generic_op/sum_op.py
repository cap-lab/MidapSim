from .arithmetic_op import ArithmeticOp


class SumOp(ArithmeticOp):
    def __init__(self, **kwargs):
        # TODO: add other arguments for the upsampling operation
        # Upsampling ratio, ...
        super(SumOp, self).__init__(op_type='Sum', **kwargs)

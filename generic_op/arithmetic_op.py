from .operator_base import OperatorBase


class ArithmeticOp(OperatorBase):
    def __init__(self, broadcast=False, activation=None, **kwargs):
        # TODO: add other arguments for the upsampling operation
        # Upsampling ratio, ...
        super(ArithmeticOp, self).__init__(**kwargs)
        self.broadcast = broadcast
        self.activation = activation

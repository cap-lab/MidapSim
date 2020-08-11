from .convpool_op_base import ConvPoolOpBase


class UpsampleOp(ConvPoolOpBase):
    def __init__(self, op_type='Upsample', algorithm='NearestNeighbor', **kwargs):
        # TODO: add other arguments for the upsampling operation
        # Upsampling ratio, ...
        super(UpsampleOp, self).__init__(op_type=op_type, **kwargs)
        self.algorithm = algorithm
        # Not yet supported
        if algorithm not in ['NearestNeighbor', 'NN', 'Zero']:
            raise ValueError("[NN, Zero Padding] upsampling is only supported")

import numpy as np

from acc_utils.errors import _assert

from .convpool_op_base import ConvPoolOpBase


class ConvOp(ConvPoolOpBase):
    def __init__(
            self,
            weight=None,
            bias=None,
            bn=None,
            dilation=1,
            group=1,
            op_type='Conv',
            **kwargs
    ):
        super(ConvOp, self).__init__(op_type=op_type, **kwargs)
        _assert(weight is not None, 'weight tensor(np.array) must be set')
        self.weight = weight  # NCHW --> NWHC
        self.weight_origin = weight
        self.orig_weight_size = weight.size
        self.bias = bias
        self.bias_origin = bias
        # bn = [gamma, beta, mu, sigma] [Warning: bn[3] (sigma) should be merged with epsilon, like as sqrt(sigma^2 + epsilon)
        self.bn = bn
        self.dilation = dilation
        _assert(group == 1 or group ==
                weight.shape[0], 'Normal Conv or Depthwise Conv is only supported.')
        if group > 1:
            self.type = 'Depthwise'

    def __del__(self):
        super(ConvOp, self).__del__()
        del self.weight, self.bias, self.bn

    def add_bias(self, bias):
        if self.bn is None:
            self.bias = bias if self.bias is None else np.add(self.bias, bias)
        else:
            self.bn[1] = np.add(bias, self.bn[1])

    def mul_scale(self, scale):
        if self.bn is None:
            self.bn = np.array([
                scale,
                np.zeros(scale.shape, dtype=np.float32),
                np.zeros(scale.shape, dtype=np.float32),
                np.ones(scale.shape, dtype=np.float32)])
        else:
            self.bn[0, :] = np.multiply(self.bn[0, :], scale)

    def merge_normalization(self):
        if self.bn is not None:
            channelwise_scale = np.divide(self.bn[0, :], self.bn[3, :])
            channelwise_bias = np.subtract(self.bn[1, :], np.multiply(
                self.bn[0, :], np.divide(self.bn[2, :], self.bn[3, :])))
            self.bias = channelwise_bias
            self.weight = channelwise_scale[:, np.newaxis, np.newaxis, np.newaxis] * self.weight
            # self.bn = None

    def tensor_to_midap_tensor(self):
        if self.order == 'NCHW':
            super(ConvOp, self).tensor_to_midap_tensor()
            # Weight -> NCHW -> NWHC
            self.weight = self.weight.transpose(0, 3, 2, 1)
            self.weight_origin = self.weight
            self.bias_origin = self.bias

    def get_macs(self):  # Overrided
        return self.output_tensor[:, :, 0].size * self.orig_weight_size

    def flip_operation(self):
        super(ConvOp, self).flip_operation()
        self.weight = np.flip(self.weight, axis=1)  # NW[HC]

    def __repr__(self):
        return super(ConvOp, self).__repr__() + "kernel shape: {}\n".format(self.weight.shape)

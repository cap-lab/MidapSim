from . import ConvOp


class FC(ConvOp):
    def __init__(
            self,
            **kwargs
    ):
        super(FC, self).__init__(**kwargs)
        self.type = 'FC'

    def tensor_to_midap_tensor(self):
        self.weight = self.weight.reshape(self.weight.shape[0], 1, 1, -1)
        super(ConvOp, self).tensor_to_midap_tensor()

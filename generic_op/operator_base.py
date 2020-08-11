from acc_utils.errors import _assert


class OperatorBase(object):
    # All tensor-like features should be np.array type
    def __init__(
            self,
            name=None,
            op_type=None,
            order='NCHW',
            input_layers=[],
            output_tensor=None,
            activation=None,
            **kwargs):
        _assert(name is not None, 'name: operator name must be defined')
        _assert(op_type is not None, 'op_type: operator type must be defined')
        # _assert(output_tensor is not None, 'output_tensor must be set')
        self.name = name  # Operator name
        self.type = op_type  # Operator type
        # Shape - Default : NCHW, tensor_to_midap_tensor converts order to 'NWHC'
        self.order = order if isinstance(order, str) else 'NCHW'
        self.input_layers = input_layers  # Operator input
        self.output_tensor = output_tensor  # Operator output tensor - for verification
        self.activation = activation
        self.next = []
        if len(kwargs) > 0:
            print("Warning: operator {}: parameter {} is not used".format(name, kwargs))

    def __del__(self):
        del self.input_layers, self.output_tensor, self.activation

    def tensor_to_midap_tensor(self):
        # NCHW --> NWHC
        if self.order == 'NCHW':
            if self.output_tensor.shape[0] > 1 and len(self.output_tensor.shape) == 4:
                raise ValueError("batch size > 1 is not yet supported")
            if len(self.output_tensor.shape) == 4:
                self.output_tensor = self.output_tensor.reshape(
                    self.output_tensor.shape[1:])
            if len(self.output_tensor.shape) == 3:
                self.output_tensor = self.output_tensor.transpose(2, 1, 0)
            elif len(self.output_tensor.shape) == 2:
                self.output_tensor = self.output_tensor.reshape(
                    1, 1, self.output_tensor.size)
            self.order = 'NWHC'

    def get_macs(self):
        return 0

    def flip_operation(self):
        pass

    def __repr__(self):
        base_repr = "\nOperator name: {}\ttype: {}\tinput_layers: {}\toutput_shape: {}\n".format(
            self.name, self.type, self.input_layers, self.output_tensor.shape)
        return base_repr

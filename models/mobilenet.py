from . import ModelBuilder


def inverted_residual(mb, x, inp, oup, stride, expand_ratio):
    hidden_dim = int(round(inp * expand_ratio))
    use_res_connect = stride == 1 and inp == oup
    in_x = x
    if expand_ratio != 1:
        x = mb.Conv(x, inp, hidden_dim, 1)
    x = mb.DWConv(x, hidden_dim, 3, stride, 'same')
    x = mb.Conv(x, hidden_dim, oup, 1, 1, 0, activation='Linear')
    if use_res_connect:
        x = mb.Sum(in_x, x)
    return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mobilenet_v2(
        input_size=224,
        num_classess=1000,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        softmax=False):
    input_channel = 32
    last_channel = 1280
    if inverted_residual_setting is None:
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
        raise ValueError(
            "inverted_residual_setting should be non-empty or a 4-element listm got {}".format(inverted_residual_setting))

    last_channel = _make_divisible(
        last_channel * max(1.0, width_mult), round_nearest)
    mb = ModelBuilder("MobilenetV2_{}".format(input_size))
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    x = mb.Conv(x, 3, input_channel, 3, 2, 1)
    for t, c, n, s in inverted_residual_setting:
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            x = inverted_residual(mb, x, input_channel,
                                  output_channel, stride, expand_ratio=t)
            input_channel = output_channel
    x = mb.Conv(x, input_channel, last_channel, 1, 1, 0)
    x = mb.GlobalPool(x)
    x = mb.FC(x, last_channel, num_classess)
    if softmax:
        x = mb.Softmax(x)

    return mb


def conv_dw(mb, x, inp, oup, stride):
    x = mb.DWConv(x, inp, 3, stride, 1)
    x = mb.Conv(x, inp, oup, 1, 1, 0)
    return x


def mobilenet(
        input_size=224,
        num_classess=1000,
        softmax=False):

    mb = ModelBuilder("MobilenetV1_{}".format(input_size))
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    x = mb.Conv(x, 3, 32, 3, 2, 1)
    x = conv_dw(mb, x, 32, 64, 1)
    x = conv_dw(mb, x, 64, 128, 2)
    x = conv_dw(mb, x, 128, 128, 1)
    x = conv_dw(mb, x, 128, 256, 2)
    x = conv_dw(mb, x, 256, 256, 1)
    x = conv_dw(mb, x, 256, 512, 2)
    x = conv_dw(mb, x, 512, 512, 1)
    x = conv_dw(mb, x, 512, 512, 1)
    x = conv_dw(mb, x, 512, 512, 1)
    x = conv_dw(mb, x, 512, 512, 1)
    x = conv_dw(mb, x, 512, 1024, 2)
    x = conv_dw(mb, x, 1024, 1024, 1)
    x = mb.GlobalPool(x)
    x = mb.FC(x, 1024, 1000)
    if softmax:
        x = mb.Softmax(x)
    return mb

import math

from models import ModelBuilder

from .block_utils import BlockArgsDecoder, GlobalParams


def efficientnet(net='efficietnet-b0', input_channels=None):
    decoded_effnet_args, global_params = get_base_args_params(net)
    scale_blocks_args(decoded_effnet_args, global_params)
    blocks_args = decoded_effnet_args

    width_mult, depth_mult, default_input_channels, _ = params_dict[net]
    if input_channels is None:
        input_channels = default_input_channels
    # COnv 1
    mb = ModelBuilder()
    input_shape = (1, 3, input_channels, input_channels)
    x = mb.set_input_tensor(tensor_shape=input_shape)

    scaled_first_filters = round_filters(32, global_params)
    res = mb.Conv(x, input_shape[1], scaled_first_filters, 3, 2, 'same')

    for block_args in blocks_args:
        assert block_args.num_repeat > 0

        res = add_mbconv_block(mb, res, block_args)
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
        for _ in range(block_args.num_repeat - 1):
            res = add_mbconv_block(mb, res, block_args)

    scaled_last_fileters = round_filters(1280, global_params)
    res = mb.Conv(res, blocks_args[-1].output_filters, scaled_last_fileters, 1, 1, 'same')

    res = mb.GlobalPool(res)

    res = mb.FC(res, scaled_last_fileters, 1000, activation='Sigmoid')

    return mb


def add_mbconv_block(mb, input, block_args):
    in_C = block_args.input_filters
    out_C = block_args.output_filters
    expand_ratio = int(block_args.expand_ratio)
    k_size = block_args.kernel_size
    stride = block_args.strides[0]
    se_ratio = block_args.se_ratio

    mid_C = in_C * expand_ratio
    if expand_ratio != 1:
        expand = mb.Conv(input, in_C, mid_C, 1, 1, 'same')
    else:
        expand = input

    dwise = mb.Conv(expand, mid_C, mid_C, k_size, stride, 'same', groups=mid_C)

    if se_ratio:
        se_input = dwise
        reduced_C = round(in_C * se_ratio)
        avg = mb.GlobalPool(se_input)
        fc1 = mb.FC(avg, mid_C, reduced_C, bias=True, activation='Relu')
        fc2 = mb.FC(fc1, reduced_C, mid_C, bias=True, activation='Sigmoid')
        out = mb.Mul(se_input, fc2)

        dwise = out

    out = mb.Conv(dwise, mid_C, out_C, 1, 1, 'same')

    if in_C == out_C and stride == 1:
        out = mb.Sum(input, out)
    return out


params_dict = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
}


def get_base_args_params(net):
    effnet_b0_blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = effnet_b0_blocks_args

    decoder = BlockArgsDecoder()
    decoded_blocks_args = decoder.decode(blocks_args)

    width_mult, depth_mult, _, _ = params_dict[net]
    global_params = GlobalParams(depth_coefficient=depth_mult,
                                 width_coefficient=width_mult,
                                 depth_divisor=8,
                                 min_depth=None)
    return decoded_blocks_args, global_params


def scale_blocks_args(blocks_args, global_params):
    for i, block_args in enumerate(blocks_args):
        in_c = round_filters(block_args.input_filters, global_params)
        out_c = round_filters(block_args.output_filters, global_params)
        num_rep = round_repeats(block_args.num_repeat, global_params)
        blocks_args[i] = block_args._replace(input_filters=in_c, output_filters=out_c, num_repeat=num_rep)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

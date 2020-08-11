from . import ModelBuilder


def inception_v3(input_size=299, num_classes=1000, softmax=False):
    mb = ModelBuilder("InceptionV3_{}".format(input_size))
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    Conv2d_1a_3x3 = mb.Conv(x, 3, 32, k=3, stride=2)
    Conv2d_2a_3x3 = mb.Conv(Conv2d_1a_3x3, 32, 32, k=3)
    Conv2d_2b_3x3 = mb.Conv(Conv2d_2a_3x3, 32, 64, k=3, pad=1)
    MaxPool_1 = mb.MaxPool(Conv2d_2b_3x3, k=3, stride=2)
    Conv2d_3b_1x1 = mb.Conv(MaxPool_1, 64, 80, k=1)
    Conv2d_4a_3x3 = mb.Conv(Conv2d_3b_1x1, 80, 192, k=3)
    MaxPool_2 = mb.MaxPool(Conv2d_4a_3x3, k=3, stride=2)
    Mixed_5b = inception_a(mb, MaxPool_2, 192, pool_features=32)
    Mixed_5c = inception_a(mb, Mixed_5b, 256, pool_features=64)
    Mixed_5d = inception_a(mb, Mixed_5c, 288, pool_features=64)
    Mixed_6a = inception_b(mb, Mixed_5d, 288)
    Mixed_6b = inception_c(mb, Mixed_6a, 768, channels_7x7=128)
    Mixed_6c = inception_c(mb, Mixed_6b, 768, channels_7x7=160)
    Mixed_6d = inception_c(mb, Mixed_6c, 768, channels_7x7=160)
    Mixed_6e = inception_c(mb, Mixed_6d, 768, channels_7x7=192)
    Mixed_7a = inception_d(mb, Mixed_6e, 768)
    Mixed_7b = inception_e(mb, Mixed_7a, 1280)
    Mixed_7c = inception_e(mb, Mixed_7b, 2048)
    GlobalPool_1 = mb.GlobalPool(Mixed_7c)
    fc = mb.FC(GlobalPool_1, 2048, num_classes)
    if softmax:
        _ = mb.Softmax(fc)
    return mb


def inception_a(mb, x, in_channels, pool_features):
    branch1x1 = mb.Conv(x, in_channels, 64, k=1)

    branch5x5 = mb.Conv(x, in_channels, 48, k=1)
    branch5x5 = mb.Conv(branch5x5, 48, 64, k=5, pad=2)

    branch3x3dbl = mb.Conv(x, in_channels, 64, k=1)
    branch3x3dbl = mb.Conv(branch3x3dbl, 64, 96, k=3, pad=1)
    branch3x3dbl = mb.Conv(branch3x3dbl, 96, 96, k=3, pad=1)

    branch_pool = mb.AvgPool(x, k=3, stride=1, pad=1)
    branch_pool = mb.Conv(branch_pool, in_channels, pool_features, k=1)

    output = mb.Concat([branch1x1, branch5x5, branch3x3dbl, branch_pool])
    return output


def inception_b(mb, x, in_channels):
    branch3x3 = mb.Conv(x, in_channels, 384, k=3, stride=2)

    branch3x3dbl = mb.Conv(x, in_channels, 64, k=1)
    branch3x3dbl = mb.Conv(branch3x3dbl, 64, 96, k=3, pad=1)
    branch3x3dbl = mb.Conv(branch3x3dbl, 96, 96, k=3, stride=2)

    branch_pool = mb.MaxPool(x, k=3, stride=2)

    output = mb.Concat([branch3x3, branch3x3dbl, branch_pool])
    return output


def inception_c(mb, x, in_channels, channels_7x7):
    branch1x1 = mb.Conv(x, in_channels, 192, k=1)

    c7 = channels_7x7
    branch7x7 = mb.Conv(x, in_channels, c7, k=1)
    branch7x7 = mb.Conv(branch7x7, c7, c7, k=(1, 7), pad=(0, 3))
    branch7x7 = mb.Conv(branch7x7, c7, 192, k=(7, 1), pad=(3, 0))

    branch7x7dbl = mb.Conv(x, in_channels, c7, k=1)
    branch7x7dbl = mb.Conv(branch7x7dbl, c7, c7, k=(7, 1), pad=(3, 0))
    branch7x7dbl = mb.Conv(branch7x7dbl, c7, c7, k=(1, 7), pad=(0, 3))
    branch7x7dbl = mb.Conv(branch7x7dbl, c7, c7, k=(7, 1), pad=(3, 0))
    branch7x7dbl = mb.Conv(branch7x7dbl, c7, 192, k=(1, 7), pad=(0, 3))

    branch_pool = mb.AvgPool(x, k=3, stride=1, pad=1)
    branch_pool = mb.Conv(branch_pool, in_channels, 192, k=1)

    output = mb.Concat([branch1x1, branch7x7, branch7x7dbl, branch_pool])
    return output


def inception_d(mb, x, in_channels):
    branch3x3 = mb.Conv(x, in_channels, 192, k=1)
    branch3x3 = mb.Conv(branch3x3, 192, 320, k=3, stride=2)

    branch7x7x3 = mb.Conv(x, in_channels, 192, k=1)
    branch7x7x3 = mb.Conv(branch7x7x3, 192, 192, k=(1, 7), pad=(0, 3))
    branch7x7x3 = mb.Conv(branch7x7x3, 192, 192, k=(7, 1), pad=(3, 0))
    branch7x7x3 = mb.Conv(branch7x7x3, 192, 192, k=3, stride=2)

    branch_pool = mb.MaxPool(x, k=3, stride=2)

    output = mb.Concat([branch3x3, branch7x7x3, branch_pool])
    return output


def inception_e(mb, x, in_channels):
    branch1x1 = mb.Conv(x, in_channels, 320, k=1)

    branch3x3 = mb.Conv(x, in_channels, 384, k=1)
    branch3x3a = mb.Conv(branch3x3, 384, 384, k=(1, 3), pad=(0, 1))
    branch3x3b = mb.Conv(branch3x3, 384, 384, k=(3, 1), pad=(1, 0))
    # branch3x3 = mb.Concat([branch3x3a, branch3x3b])

    branch3x3dbl = mb.Conv(x, in_channels, 448, k=1)
    branch3x3dbl = mb.Conv(branch3x3dbl, 448, 384, k=3, pad=1)
    branch3x3dbl_a = mb.Conv(branch3x3dbl, 384, 384, k=(1, 3), pad=(0, 1))
    branch3x3dbl_b = mb.Conv(branch3x3dbl, 384, 384, k=(3, 1), pad=(1, 0))
    # branch3x3dbl = mb.Concat([branch3x3dbl_a, branch3x3dbl_b])

    branch_pool = mb.AvgPool(x, k=3, stride=1, pad=1)
    branch_pool = mb.Conv(branch_pool, in_channels, 192, k=1)

    output = mb.Concat([branch1x1, branch3x3a, branch3x3b,
                        branch3x3dbl_a, branch3x3dbl_b, branch_pool])
    return output

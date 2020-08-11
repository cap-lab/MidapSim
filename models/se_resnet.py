from .resnet import *


def se_module(mb, name, channel, reduction=16):
    avg = mb.GlobalPool(name)
    fc1 = mb.FC(avg, channel, channel // reduction,
                bias=True, activation='Relu')
    fc2 = mb.FC(fc1, channel // reduction, channel,
                bias=True, activation='Sigmoid')
    mul = mb.Mul(name, fc2)
    return mul


class SEBottleneck:
    def __init__(self):
        self.expansion = 4
        self.__constants__ = ['downsample']

    def __call__(self, mb, input, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, reduction=16):
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        identity = input
        out = conv1x1(mb, input, inplanes, width)
        out = conv3x3(mb, out, width, width, stride, groups, dilation)
        out = conv1x1(mb, out, width, planes * self.expansion)
        out = se_module(mb, out, planes * self.expansion, reduction)

        if downsample is not None:
            identity = downsample(input)

        out = mb.Sum(out, identity)
        return out


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(arch, block, layers, **kwargs)
    return model.mb


def se_resnet50(**kwargs):
    return _resnet('se_resnet50', SEBottleneck(), [3, 4, 6, 3], **kwargs)


def se_resnet101(**kwargs):
    return _resnet('se_resnet101', SEBottleneck(), [3, 4, 23, 3], **kwargs)


def se_resnet152(**kwargs):
    return _resnet('se_resnet152', SEBottleneck(), [3, 8, 36, 3], **kwargs)

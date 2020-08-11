from . import ModelBuilder


def conv3x3(mb, input, in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with pad"""
    return mb.Conv(input, in_planes, out_planes, k=3, stride=stride,
                   pad=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(mb, input, in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return mb.Conv(input, in_planes, out_planes, k=1, stride=stride, bias=True)


class Bottleneck:
    def __init__(self):
        self.expansion = 4
        self.__constants__ = ['downsample']

    def __call__(self, mb, input, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        identity = input
        out = conv1x1(mb, input, inplanes, width)
        out = conv3x3(mb, out, width, width, stride, groups, dilation)
        out = conv1x1(mb, out, width, planes * self.expansion)

        if downsample is not None:
            identity = downsample(input)

        out = mb.Sum(out, identity)
        return out


class ResNet:
    def __init__(self, name, block, layers, input_size=224, num_classes=1000,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, softmax=False):
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.mb = ModelBuilder("{}_{}".format(name, input_size))
        input_shape = (1, 3, input_size, input_size)
        x = self.mb.set_input_tensor(tensor_shape=input_shape)

        self.conv1 = self.mb.Conv(x, input_shape[1], self.inplanes, 7, 2, 'same')
        self.maxpool = self.mb.MaxPool(self.conv1, k=3, stride=2, pad=1)

        self.layer1 = self._make_layer(self.maxpool, block, 64, layers[0])
        self.layer2 = self._make_layer(self.layer1, block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.layer2, block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.layer3, block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = self.mb.GlobalPool(self.layer4)

        self.fc = self.mb.FC(self.avgpool, 512 * block.expansion, num_classes)
        if softmax:
            self.softmax = self.mb.Softmax(self.fc)

    def _make_layer(self, input, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            def downsample(input):
                return conv1x1(self.mb, input, self.inplanes, planes * block.expansion, stride)

        out = block(self.mb, input, self.inplanes, planes, stride, downsample, self.groups,
                    self.base_width, previous_dilation)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            out = block(self.mb, out, self.inplanes, planes, groups=self.groups,
                        base_width=self.base_width, dilation=self.dilation)
        return out


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(arch, block, layers, **kwargs)
    return model.mb


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck(), [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck(), [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck(), [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck(), [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck(), [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck(), [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck(), [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

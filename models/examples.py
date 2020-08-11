from .model_builder import ModelBuilder


def dcgan(nz=100, ngf=64):
    # KAIST, Deep Convolutional GAN
    mb = ModelBuilder("dcGAN_in_{}_out_64x64".format(nz))
    x = mb.set_input_tensor(tensor_shape=(1, nz, 1, 1))
    x = mb.ConvTranspose(x, nz, ngf * 8, 4, 1, 0)
    x = mb.ConvTranspose(x, ngf * 8, ngf * 4, 4, 2, 1)
    x = mb.ConvTranspose(x, ngf * 4, ngf * 2, 4, 2, 1)
    x = mb.ConvTranspose(x, ngf * 2, ngf, 4, 2, 1)
    x = mb.ConvTranspose(x, ngf, 3, 4, 2, 1, activation='LeakyRelu')
    return mb


def discogan(input_shape=(1, 3, 64, 64), channel_config=[64, 128, 256, 512, 256, 128, 64, 3]):
    # 2016 SKT disco GAN
    mb = ModelBuilder("discoGAN_{}x{}".format(input_shape[2], input_shape[3]))
    i = 0
    x = mb.set_input_tensor(tensor_shape=input_shape)
    h1 = mb.Conv(x, input_shape[1], channel_config[i],
                 4, 2, 1, activation='LeakyRelu')
    h2 = mb.Conv(
        h1, channel_config[i], channel_config[i + 1], 4, 2, 1, activation='LeakyRelu')
    i += 1
    h3 = mb.Conv(
        h2, channel_config[i], channel_config[i + 1], 4, 2, 1, activation='LeakyRelu')
    i += 1
    h4 = mb.Conv(
        h3, channel_config[i], channel_config[i + 1], 4, 2, 1, activation='LeakyRelu')
    i += 1
    h5 = mb.ConvTranspose(
        h4, channel_config[i], channel_config[i + 1], 4, 2, 1)
    i += 1
    h6 = mb.ConvTranspose(
        h5, channel_config[i], channel_config[i + 1], 4, 2, 1)
    i += 1
    h7 = mb.ConvTranspose(
        h6, channel_config[i], channel_config[i + 1], 4, 2, 1)
    i += 1
    _ = mb.ConvTranspose(
        h7, channel_config[i], channel_config[i + 1], 4, 2, 1, activation='Sigmoid')
    return mb


def unet(input_shape=(1, 1, 128, 128), decompose=True):
    # Based on Official Tensorflow U-Net code (Vokeh)
    mb = ModelBuilder("unet_{}x{}".format(input_shape[2], input_shape[3]))
    x = mb.set_input_tensor(tensor_shape=input_shape)
    conv1 = mb.Conv(x, input_shape[1], 64, 3, 1, 'same')
    conv1 = mb.Conv(conv1, 64, 64, 3, 1, 'same')
    pool1 = mb.MaxPool(conv1, 2, 2)
    conv2 = mb.Conv(pool1, 64, 128, 3, 1, 'same')
    conv2 = mb.Conv(conv2, 128, 128, 3, 1, 'same')
    pool2 = mb.MaxPool(conv2, 2, 2)
    conv3 = mb.Conv(pool2, 128, 256, 3, 1, 'same')
    conv3 = mb.Conv(conv3, 256, 256, 3, 1, 'same')
    pool3 = mb.MaxPool(conv3, 2, 2)
    conv4 = mb.Conv(pool3, 256, 512, 3, 1, 'same')
    conv4 = mb.Conv(conv4, 512, 512, 3, 1, 'same')
    pool4 = mb.MaxPool(conv4, 2, 2)

    conv5 = mb.Conv(pool4, 512, 1024, 3, 1, 'same')
    conv5 = mb.Conv(conv5, 1024, 1024, 3, 1, 'same')

    up6 = mb.Conv(conv5, 1024, 512, 2, 1, 'same')
    up6 = mb.Upsample(up6, 2)
    if decompose:
        conv4 = mb.Conv(conv4, 512, 512, 3, 1, 'same', activation='Linear')
        conv6 = mb.Conv(up6, 512, 512, 3, 1, 'same', activation='Linear')
        conv6 = mb.Sum(conv4, conv6, activation='Relu')
    else:
        merge6 = mb.Concat([conv4, up6])
        conv6 = mb.Conv(merge6, 1024, 512, 3, 1, 'same')
    conv6 = mb.Conv(conv6, 512, 512, 3, 1, 'same')

    up7 = mb.Conv(conv6, 512, 256, 2, 1, 'same')
    up7 = mb.Upsample(up7, 2)
    if decompose:
        conv3 = mb.Conv(conv3, 256, 256, 3, 1, 'same', activation='Linear')
        conv7 = mb.Conv(up7, 256, 256, 3, 1, 'same', activation='Linear')
        conv7 = mb.Sum(conv3, conv7, activation='Relu')
    else:
        merge7 = mb.Concat([conv3, up7])
        conv7 = mb.Conv(merge7, 512, 256, 3, 1, 'same')
    conv7 = mb.Conv(conv7, 256, 256, 3, 1, 'same')

    up8 = mb.Conv(conv7, 256, 128, 2, 1, 'same')
    up8 = mb.Upsample(up8, 2)
    if decompose:
        conv2 = mb.Conv(conv2, 128, 128, 3, 1, 'same', activation='Linear')
        conv8 = mb.Conv(up8, 128, 128, 3, 1, 'same', activation='Linear')
        conv8 = mb.Sum(conv2, conv8, activation='Relu')
    else:
        merge8 = mb.Concat([conv2, up8])
        conv8 = mb.Conv(merge8, 256, 128, 3, 1, 'same')
    conv8 = mb.Conv(conv8, 128, 128, 3, 1, 'same')

    up9 = mb.Conv(conv8, 128, 64, 2, 1, 'same')
    up9 = mb.Upsample(up9, 2)
    if decompose:
        conv1 = mb.Conv(conv1, 64, 64, 3, 1, 'same', activation='Linear')
        conv9 = mb.Conv(up9, 64, 64, 3, 1, 'same', activation='Linear')
        conv9 = mb.Sum(conv1, conv9, activation='Relu')
    else:
        merge9 = mb.Concat([conv1, up9])
        conv9 = mb.Conv(merge9, 128, 64, 3, 1, 'same')
    conv9 = mb.Conv(conv9, 64, 64, 3, 1, 'same')
    conv9 = mb.Conv(conv9, 64, 2, 3, 1, 'same')
    _ = mb.Conv(conv9, 2, 1, 1, 1, 0, activation='Sigmoid')

    return mb


# Conditional GAN for image to image .... it cannot be compiled yet (not supported)
def im2im_cdgan(input_shape=(1, 3, 64, 64), nf=64):
    mb = ModelBuilder("cdgan")
    top = mb.set_input_tensor(tensor_shape=input_shape)

    def generator(mb, x, input_shape, nf):
        in_c = input_shape[1]
        img_size = input_shape[2]
        x = mb.Conv(x, in_c, nf, 4, 2, 'same', activation='LeakyRelu')
        x = mb.Conv(x, nf, nf * 2, 4, 2, 'same', activation='LeakyRelu')
        x = mb.Conv(x, nf * 2, nf * 4, 4, 2, 'same', activation='LeakyRelu')
        i = mb.Conv(x, nf * 4, nf * 8, 4, 2, 'same', activation='Linear')
        s = mb.Conv(x, nf * 4, nf * 8, img_size //
                    8, 1, 0, activation='LeakyRelu')
        s = mb.FC(s, nf * 8, nf * 8, activation='Linear')
        return i, s

    def decoder(mb, x, out_c, nf):
        x = mb.ConvTranspose(x, nf * 8, nf * 4, 4, 2, 'same')
        x = mb.ConvTranspose(x, nf * 4, nf * 2, 4, 2, 'same')
        x = mb.ConvTranspose(x, nf * 2, nf, 4, 2, 'same')
        x = mb.ConvTranspose(x, nf, out_c, 4, 2, 1, activation='Sigmoid')
        return x

    in_A, sp_A = generator(mb, top, input_shape, nf)
    in_B, sp_B = generator(mb, top, input_shape, nf)
    input_A = mb.Sum(in_B, sp_A)
    _ = decoder(mb, input_A, input_shape[1], nf)
    input_B = mb.Sum(in_A, sp_B)
    _ = decoder(mb, input_B, input_shape[1], nf)
    return mb


def one_layer_example(input_shape, out_chan, kern_info):
    mb = ModelBuilder("{}_{}_{}".format(input_shape, out_chan, kern_info))
    x = mb.set_input_tensor(tensor_shape=input_shape)
    f1 = mb.Conv(x, input_shape[1], out_chan, kern_info[0], kern_info[1], 'same')
    # f1 = mb.DWConv(x, input_shape[1], kern_info[0], kern_info[1], 'same')
    # pool_size = (input_shape[1] + 2 ^ kern_info[0] - 1) // 2 ^ kern_info[0]
    # mb.Conv(f1, out_chan, 24, 1, 1, 'same')
    mb.MaxPool(f1, 2, 2)
    return mb

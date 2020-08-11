from __future__ import absolute_import, division, print_function, unicode_literals

import os.path
import re

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
from PIL import Image

from config import cfg

from .errors import *


def to_tensor(PILimage):
    # MODEL_TRANSLATOR : CAFFE2 MODEL TO MIDAP MODEL
    pic = PILimage
    if pic.mode == 'I':
        img = np.array(pic, np.int32, copy=False)
    elif pic.mode == 'I;16':
        img = np.array(pic, np.int16, copy=False)
    elif pic.mode == 'F':
        img = np.array(pic, np.float32, copy=False)
    elif pic.mode == '1':
        img = 255 * np.array(pic, np.uint8, copy=False)
    else:
        img = np.frombuffer(pic.tobytes(), dtype=np.uint8)

    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    print('Open the image with {} channel mode'.format(nchannel))

    img = img.reshape(1, pic.size[1], pic.size[0], nchannel)  # (N, W, H, C)
    img = np.transpose(img, (0, 3, 1, 2)).copy()  # (N, C, H, W) for the contiguous format

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    else:
        img = img.astype(np.float32)

    return img


def get_abs_path(path):
    if cfg.SYSTEM.ROOT in os.path.abspath('.'):
        root_dir = cfg.SYSTEM.ROOT
    else:
        print("Warning: you may use incorrect root directory. Please check your configuration file!")
        root_dir = os.path.abspath('.')
    return os.path.join(root_dir, path)


def _GetLegacyDims(net, net_params, dummy_input, legacy_pad_ops):
    dim_map = {}
    # Get dimensions with legacy pad
    for i in range(len(net.op)):
        op_def = net.op[i]
        if i in legacy_pad_ops:
            output = op_def.output[0]
            blob_legacy = workspace.FetchBlob(output)
            dim_map[i] = blob_legacy.shape
    return dim_map


def _GetLegacyPadArgs(op_def, arg_map):
    pads = {}
    keys = ['pad_l', 'pad_t', 'pad_r', 'pad_b']
    is_pad = 'pad' in arg_map
    if is_pad:
        for k in keys:
            pads[k] = arg_map['pad'].i
    else:
        pads = {x: arg_map[x].i for x in keys}
    return pads


def _AdjustDims(op_def, arg_map, pads, dim1, dim2):
    n1, c1, h1, w1 = dim1
    n2, c2, h2, w2 = dim2
    assert(n1 == n2)
    assert(c1 == c2)
    is_pad = 'pad' in arg_map
    if h1 != h2 or w1 != w2:
        if h1 == h2 + 1:
            pads['pad_b'] += 1
        elif h1 != h2:
            raise Exception("Unexpected dimensions for height:", h1, h2)
        if w1 == w2 + 1:
            pads['pad_r'] += 1
        elif w1 != w2:
            raise Exception("Unexpected dimensions for width:", w1, w2)
        if is_pad:
            op_def.arg.remove(arg_map['pad'])
            args = []
            for name in pads.keys():
                arg = caffe2_pb2.Argument()
                arg.name = name
                arg.i = pads[name]
                args.append(arg)
            op_def.arg.extend(args)
        else:
            for name in pads.keys():
                arg_map[name].i = pads[name]


def _RemoveLegacyPad(net, net_params, input_dims):
    legacy_pad_ops = []
    for i in range(len(net.op)):
        op_def = net.op[i]
        if re.match(r'^(Conv|ConvTranspose|MaxPool|AveragePool)(\dD)?$',
                    op_def.type):
            for arg in op_def.arg:
                if arg.name == 'legacy_pad':
                    legacy_pad_ops.append(i)
                    break
    if legacy_pad_ops:
        n, c, h, w = input_dims
        dummy_input = np.random.randn(n, c, h, w).astype(np.float32)
        dim_map = _GetLegacyDims(net, net_params, dummy_input, legacy_pad_ops)

        # Running with the legacy pad argument removed
        # compare the dimensions and adjust pad argument when necessary
        for i in range(len(net.op)):
            op_def = net.op[i]
            if i in legacy_pad_ops:
                arg_map = {}
                for arg in op_def.arg:
                    arg_map[arg.name] = arg
                pads = _GetLegacyPadArgs(op_def, arg_map)
                # remove legacy pad arg
                for j in range(len(op_def.arg)):
                    arg = op_def.arg[j]
                    if arg.name == 'legacy_pad':
                        del op_def.arg[j]
                        break
                output = op_def.output[0]
                # use a new name to avoid the interference with inplace
                nonlegacy_output = output + '_nonlegacy'
                op_def.output[0] = nonlegacy_output
                workspace.RunOperatorOnce(op_def)
                blob_nonlegacy = workspace.FetchBlob(nonlegacy_output)
                # reset output name
                op_def.output[0] = output

                dim1 = dim_map[i]
                dim2 = blob_nonlegacy.shape
                _AdjustDims(op_def, arg_map, pads, dim1, dim2)

            workspace.RunOperatorOnce(op_def)
    return net


def get_input_tensor(img_path=None, img_shape=None):
    if img_path is None:
        img_path = get_abs_path(cfg.SYSTEM.INPUT)
    if img_shape is None:
        img_shape = cfg.SYSTEM.INPUT_SHAPE
    w, h, c = 1, 1, 3
    if isinstance(img_shape, int):
        w, h = img_shape, img_shape, 3
    elif len(img_shape) == 2:
        w, h = img_shape
    else:
        w, h, c = img_shape
    image = Image.open(img_path)
    if c == 1:
        image = image.convert('L')
    image = image.resize((w, h), Image.BILINEAR)
    input_tensor = to_tensor(image)
    return input_tensor


def load_network():
    init_def = caffe2_pb2.NetDef()
    net_def = caffe2_pb2.NetDef()

    print("Load '{}', '{}' as the input network and its weight file".format(cfg.SYSTEM.NETWORK, cfg.SYSTEM.WEIGHTS))
    with open(get_abs_path(cfg.SYSTEM.WEIGHTS), 'r') as f:
        init_def.ParseFromString(f.read())
    with open(get_abs_path(cfg.SYSTEM.NETWORK), 'r') as f:
        net_def.ParseFromString(f.read())

    if net_def.op[0].input[0] != 'data':
        raise ModelBuildError('Incorrect input format - the name of the network input blob should be "data"')

    image = Image.open(get_abs_path(cfg.SYSTEM.INPUT))
    w, h = cfg.SYSTEM.INPUT_SIZE
    resized_img = image.resize((w, h), Image.BILINEAR)

    input_tensor = to_tensor(resized_img)
    print(input_tensor.shape)
    workspace.RunNetOnce(init_def)
    workspace.FeedBlob("data", input_tensor)
    workspace.CreateNet(net_def, overwrite=True)
    workspace.RunNetOnce(net_def)
    net_def = _RemoveLegacyPad(net_def, init_def, input_tensor.shape)
    workspace.CreateNet(net_def, overwrite=True)

    workspace.RunNetOnce(net_def)

    return net_def


def im2col(input_data, main_op):
    W, H, C = input_data.shape
    input_data = input_data.transpose(2, 0, 1)  # WHC --> # C W H
    filter_w, filter_h = main_op.k_w, main_op.k_h
    stride = main_op.stride
    pad_t, pad_b, pad_l, pad_r = main_op.pad_t, max(0, main_op.pad_b), main_op.pad_l, max(0, main_op.pad_r)
    out_w = (W + pad_l + pad_r - filter_w) // stride + 1
    out_h = (H + pad_t + pad_b - filter_h) // stride + 1

    img = np.pad(input_data, [(0, 0), (pad_l, pad_r), (pad_t, pad_b)], 'constant')
    if main_op.pad_r < 0:
        img = img[:, :main_op.pad_r, :]
    if main_op.pad_b < 0:
        img = img[:, :, :main_op.pad_b]
    col = np.zeros((C, filter_w, filter_h, out_w, out_h))
    # C*W*H, outw*outh
    for x in range(filter_w):
        x_max = x + stride * out_w
        for y in range(filter_h):
            y_max = y + stride * out_h
            col[:, x, y, :, :] = img[:, x:x_max:stride, y:y_max:stride]

    col = col.reshape(-1, 1, out_w * out_h)  # CHW
    return col


def div_ceil(a, b):
    ret = a // b
    ret += 0 if a % b == 0 else 1
    return ret


def calc_avgpool_size(op, out_loc, output_shape):
    pad_t, pad_b, pad_l, pad_r = op.pad_t, op.pad_b, op.pad_l, op.pad_r
    k_x, k_y = op.k_w, op.k_h
    stride = op.stride
    o_x, o_y = out_loc
    output_x, output_y = output_shape[0], output_shape[1]
    in_max_x = output_x * stride + k_x - 1 - pad_l - pad_r - 1
    in_max_y = output_y * stride + k_y - 1 - pad_t - pad_b - 1
    in_first_x = o_x * stride - pad_l
    in_first_y = o_y * stride - pad_r
    in_last_x = in_first_x + k_x - 1
    in_last_y = in_first_y + k_y - 1
    in_first_x = max(0, in_first_x)
    in_first_y = max(0, in_first_y)
    in_last_x = min(in_max_x, in_last_x)
    in_last_y = min(in_max_y, in_last_y)
    return (in_last_x - in_first_x + 1) * (in_last_y - in_first_y + 1)

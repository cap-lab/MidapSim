import collections
import re
from copy import deepcopy

from acc_utils.attrdict import AttrDict


class BlockArgsDecoder(object):
    """A class of decoder to get model configuration."""

    def _decode_blocks_string(self, block_string):
        """Gets a block through a string notation of arguments.

        E.g. r2_k3_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
        k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
        o - output filters, se - squeeze/excitation ratio

        Args:
          block_string: a string, a string representation of block arguments.

        Returns:
          A BlockArgs instance.
        Raises:
          ValueError: if the strides option is not correctly specified.
        """
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        options['e'] = int(options['e'])
        return [TupleStyleDict(
            BlockArgs(
                kernel_size=int(options['k']),
                num_repeat=int(options['r']),
                input_filters=int(options['i']),
                output_filters=int(options['o']),
                expand_ratio=int(options['e']),
                id_skip=('noskip' not in block_string),
                se_ratio=float(options['se']) if 'se' in options else None,
                strides=[int(options['s'][0]), int(options['s'][1])],
                conv_type=int(options['c']) if 'c' in options else 0
            )._asdict()
        )]

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.

        Args:
          string_list: a list of strings, each string is a notation of a block.

        Returns:
          A list of namedtuples to represent MnasNet-based blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.extend(self._decode_blocks_string(block_string))
        return blocks_args

    def span_blocks_args(self, blocks_args, every_num_repeat_is_one=True):
        """
        Remove the 'repeat' term of blocks_args. All the block_args will become repeat=1
        :param blocks_args: list of block_args.
        :param every_num_repeat_is_one if False, we only split the one block from the (num_repeat)s blocks.
        This is efficient if every blocks are repeated by some number,
        but the expression is too compact because it contains reduction block at the very beginning.
        :return:
        """
        res_args = []
        for block_args in blocks_args:
            assert block_args.num_repeat > 0
            # TODO: what if the first block is not reduction block???!! This splits every blocks one block.
            res_args.append(block_args._replace(num_repeat=1))
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            if every_num_repeat_is_one:
                for _ in range(block_args.num_repeat - 1):
                    res_args.append(block_args._replace(num_repeat=1))
            else:
                if block_args.num_repeat > 1:
                    res_args.append(block_args._replace(num_repeat=block_args.num_repeat - 1))
        return res_args

    def decode_to_cells_args(self, string_list, every_num_repeat_is_one=True):
        """Decodes a list of string notations to specify blocks inside the network.
        :param every_num_repeat_is_one if False, we only split the one reduction block from the (num_repeat)s blocks
        """
        assert isinstance(string_list, list)
        cells_args = []
        for blocks_string in string_list:
            blocks_args = self._decode_blocks_string(blocks_string)
            blocks_args = self.span_blocks_args(blocks_args, every_num_repeat_is_one)
            cell_args = TupleStyleDict({
                'cls': 'BasicCell',
                'blocks_args': blocks_args
            })
            cells_args.append(cell_args)

        return cells_args

    def cells_args_to_blocks_args(self, cells_args):
        blocks_args = []
        for cell_args in cells_args:
            for block_args in cell_args['blocks_args']:
                blocks_args.append(self._encode_block_string(block_args))

        return blocks_args

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        from graph.cell import _get_conv_cls
        conv_type = block.conv_type
        if not isinstance(conv_type, int):
            conv_type = _get_conv_cls(conv_type, cls_str_to_num=True)

        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'c%d' % conv_type
        ]
        se_ratio = block.se_ratio
        if se_ratio is not None and (se_ratio > 0 and se_ratio <= 1):
            args.append('se%s' % se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)


GlobalParams = collections.namedtuple('GlobalParams', [
    'depth_coefficient', 'width_coefficient', 'depth_divisor', 'min_depth',
])
# TODO(hongkuny): Consider rewrite an argument class with encoding/decoding.
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'conv_type',
    'kernel_size', 'input_filters', 'output_filters',
    'expand_ratio', 'strides', 'id_skip', 'se_ratio',
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = \
    BlockArgs(
        num_repeat=1,
        conv_type=0,
        kernel_size=None,
        input_filters=None,
        output_filters=None,
        expand_ratio=None,
        strides=None,
        id_skip=None,
        se_ratio=None,
    )


class TupleStyleDict(AttrDict):
    def __init__(self, *args, **kwargs):
        super(TupleStyleDict, self).__init__(*args, **kwargs)

    def _replace(self, **kwargs):
        res = deepcopy(self)
        res.update(**kwargs)
        return res

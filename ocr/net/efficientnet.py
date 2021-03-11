import torch

from torch import nn
from efficientnet_pytorch import EfficientNet as EffNet
from efficientnet_pytorch.utils import Swish
from torch.nn import functional as F


class MBExpandConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block : origin effnet block
        drop_connect_rate

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block, drop_connect_rate):
        super().__init__()
        self._bn_mom = block._bn_mom
        self._bn_eps = block._bn_eps
        self.id_skip = block.id_skip  # skip connection and drop connect
        self.drop_connect_rate = drop_connect_rate
        self.input_filters = block._block_args.input_filters
        self.output_filters = block._block_args.output_filters
        self.stride = block._block_args.stride

        # Expansion phase
        self._expand_conv = block._expand_conv
        self._bn0 = block._bn0

        # Depthwise convolution phase
        self._depthwise_conv = block._depthwise_conv
        self._bn1 = block._bn1

        # Squeeze and Excitation layer, if desired
        self._se_reduce = block._se_reduce
        self._se_expand = block._se_expand

        # Output phase
        self._project_conv = block._project_conv
        self._bn2 = block._bn2
        self._swish = Swish()

    def forward(self, inputs):

        x = inputs
        x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
        x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = self.input_filters, self.output_filters
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            if self.training:
                x = self.drop_connect(x)
            x = x + inputs
        return x

    def drop_connect(self, inputs):
        """ Drop connect. """
        p = self.drop_connect_rate
        batch_size = inputs.shape[0]
        keep_prob = 1 - p
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block : origin effnet block
        drop_connect_rate

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block, drop_connect_rate):
        super().__init__()
        self._bn_mom = block._bn_mom
        self._bn_eps = block._bn_eps
        self.id_skip = block.id_skip  # skip connection and drop connect
        self.drop_connect_rate = drop_connect_rate
        self.input_filters = block._block_args.input_filters
        self.output_filters = block._block_args.output_filters
        self.stride = block._block_args.stride

        # Depthwise convolution phase
        self._depthwise_conv = block._depthwise_conv
        self._bn1 = block._bn1

        # Squeeze and Excitation layer, if desired
        self._se_reduce = block._se_reduce
        self._se_expand = block._se_expand

        # Output phase
        self._project_conv = block._project_conv
        self._bn2 = block._bn2
        self._swish = Swish()

    def forward(self, inputs):

        x = inputs
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
        x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = self.input_filters, self.output_filters
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            if self.training:
                x = self.drop_connect(x)
            x = x + inputs
        return x

    def drop_connect(self, inputs):
        """ Drop connect. """
        p = self.drop_connect_rate
        batch_size = inputs.shape[0]
        keep_prob = 1 - p
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output


class ModifiedEfficientNet(nn.Module):
    def __init__(self, compound_coef, load_weights=True, in_channels=3):
        super(ModifiedEfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights, in_channels=in_channels)
        model.set_swish(False)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc

        self._conv_stem = model._conv_stem
        self._bn0 = model._bn0
        self._swish = model._swish
        self._blocks = nn.ModuleList([])
        self.num_blocks = len(model._blocks)
        for i, block in enumerate(model._blocks):
            drop_connect_rate = model._global_params.drop_connect_rate
            drop_connect_rate *= float(i) / len(model._blocks)
            if block._block_args.expand_ratio != 1:
                self._blocks.append(MBExpandConvBlock(block, drop_connect_rate))
            else:
                self._blocks.append(MBConvBlock(block, drop_connect_rate))

    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)
        feature_maps = []

        for idx, block in enumerate(self._blocks):
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(x)
                x = block(x)
            elif idx == self.num_blocks - 1:
                x = block(x)
                feature_maps.append(x)
            else:
                x = block(x)
            """
            x = block(x)
            if idx == self.num_blocks - 1:
                feature_maps.append(x)
            elif self._blocks[idx+1]._depthwise_conv.stride == [2, 2]:
                feature_maps.append(x)
            """
        return feature_maps

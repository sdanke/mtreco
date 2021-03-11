#!/usr/bin/env python
import os
import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair


# torch.ops.load_library(f'{os.path.dirname(__file__)}/deform_conv_v2_torch_171.dll')
torch.ops.load_library(f'{os.path.dirname(__file__)}/deform_conv_v2.dll')


class DeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=64,
        bias=True,
    ):
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError(
                "in_channels {} must be divisible by groups {}".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            raise ValueError(
                "out_channels {} must be divisible by groups {}".format(
                    out_channels, groups
                )
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        # n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, offset):
        assert (
            2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == offset.shape[1]
        )
        return torch.ops.dcn_v2_ops.deform_conv_forward(
            x,
            self.weight,
            self.bias,
            offset,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
            self.deformable_groups,
            self.im2col_step
        )


class DeformConvPack(DeformConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=64,
        bias=True,
        lr_mult=0.1,
    ):
        super(DeformConvPack, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            deformable_groups,
            im2col_step,
            bias,
        )

        out_channels = (
            self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        )
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return torch.ops.dcn_v2_ops.deform_conv_forward(
            x,
            self.weight,
            self.bias,
            offset,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
            self.deformable_groups,
            self.im2col_step
        )

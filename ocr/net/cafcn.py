import torch
import os

from torch import nn
from torch.nn import init
from ocr.net.dcn import DeformConvPack
from ocr.net.efficientnet import ModifiedEfficientNet
from ocr.utils.io import load_checkpoint


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class AttnModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, conv_module=nn.Conv2d):
        super(AttnModule, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            conv_module(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        attn_map = self.attn(x)
        output = self.conv(x)
        output = (attn_map + 1) * output
        return output, attn_map


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class CharAttnFCN(nn.Module):
    def __init__(self, backbone, num_classes=1000, planes=[32, 48, 136, 384]):
        super(CharAttnFCN, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone

        self.attn2 = AttnModule(planes[0], planes[0], planes[0])
        self.attn3 = AttnModule(planes[1], planes[1], planes[1])
        self.attn4 = AttnModule(planes[2], planes[2], planes[2], DeformConvPack)
        self.attn5 = AttnModule(planes[3], planes[3], planes[3], DeformConvPack)

        self.upconv6 = UpConv(planes[3], planes[3], planes[3])
        self.upconv7 = UpConv(planes[2] + planes[3], planes[2] + planes[3], planes[2] + planes[3])
        self.upconv8 = UpConv(sum(planes) - planes[0], sum(planes) - planes[0], sum(planes) - planes[0])
        self.upconv9 = UpConv(sum(planes), sum(planes), sum(planes))

        self.classifier = nn.Conv2d(sum(planes), self.num_classes, kernel_size=1)

        init_weights(self.attn2.modules())
        init_weights(self.attn3.modules())
        init_weights(self.attn4.modules())
        init_weights(self.attn5.modules())
        init_weights(self.upconv6.modules())
        init_weights(self.upconv7.modules())
        init_weights(self.upconv8.modules())
        init_weights(self.upconv9.modules())
        init_weights(self.classifier.modules())

    def forward(self, x):
        _, stage2, stage3, stage4, stage5 = self.backbone(x)

        y5, attn_map5 = self.attn5(stage5)
        y4, attn_map4 = self.attn4(stage4)
        y3, attn_map3 = self.attn3(stage3)
        y2, attn_map2 = self.attn2(stage2)
        y6 = self.upconv6(y5)

        output = torch.cat([y6, y4], dim=1)
        output = self.upconv7(output)    # batch, 512, h/8, w/8

        output = torch.cat([output, y3], dim=1)    # batch, 1204, h/8, w/8
        output = self.upconv8(output)    # batch, 256, h/4, w/4

        output = torch.cat([output, y2], dim=1)    # batch, 512, h/4, w/4
        output = self.upconv9(output)    # batch, 256, h/2, w/2

        output = self.classifier(output)
        return output, attn_map2, attn_map3, attn_map4, attn_map5


def get_cafcn(num_classes=0, weights_file=None, in_channels=1):
    backbone = ModifiedEfficientNet(3, False, in_channels)
    model = CharAttnFCN(backbone, num_classes=num_classes)
    if weights_file is not None and os.path.exists(weights_file):
        cp = torch.load(weights_file)
        load_checkpoint(model, cp)
        print(f'Pretrained model:{weights_file} loaded')
    return model

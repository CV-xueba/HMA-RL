import torch
from torch import nn


def exists(x):
    return x is not None


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, bias=True, bn=False, res_scale=0.2):
        super(ResBlock, self).__init__()
        if bn:
            self.block1 = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, bias=bias, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size, bias=bias, padding=1),
                nn.BatchNorm2d(out_dim),
            )
        else:
            self.block1 = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, bias=bias, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size, bias=bias, padding=1),
            )

        self.CALayer = CALayer(out_dim)
        self.res_conv = nn.Conv2d(
            in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.res_scale = res_scale

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.CALayer(h)
        res = h * self.res_scale + self.res_conv(x)
        return res


class EDSRUNetSkipConnection(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, image_size=256, bn=False, inner_channel=64, res_scale=0.1, n_resblocks=32):
        super(EDSRUNetSkipConnection, self).__init__()
        # Define parameters
        res_blocks = 4
        channel_mults = [1, 2, 4, 4]
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]

        self.conv_first = nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=(3, 3), padding=1)]

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(pre_channel, channel_mult, kernel_size=3, bias=True, bn=bn, res_scale=res_scale))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [ResBlock(pre_channel, pre_channel, kernel_size=3, bn=bn, res_scale=res_scale) for _ in
             range(n_resblocks)])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResBlock(pre_channel + feat_channels.pop(), channel_mult, kernel_size=3, bias=True, bn=bn,
                                    res_scale=res_scale))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))

        self.ups = nn.ModuleList(ups)
        self.conv_hr = nn.Conv2d(pre_channel, pre_channel, kernel_size=(3, 3), padding=1)
        self.final_conv = nn.Conv2d(pre_channel, out_channel, kernel_size=(3, 3), padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        h = self.conv_first(x)
        feats = []
        for layer in self.downs:
            x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        out = self.final_conv(self.lrelu(self.conv_hr(h+x)))
        # out = self.add_mean(out)
        return out

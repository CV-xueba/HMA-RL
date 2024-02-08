import torch.nn as nn
import torch
import torch.nn.functional as F
import math


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

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
    def __init__(self, in_dim, out_dim, kernel_size, bias=True, res_scale=0.2):
        super(ResBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, bias=bias, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size, bias=bias, padding=1)
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


class PPO(nn.Module):
    def __init__(self, state_num, action_num, n_feats=64, res_scale=0.1):
        super(PPO, self).__init__()
        agent_feat = 64
        # Define parameters
        res_blocks = 4
        n_resblocks = 32
        channel_mults = [1, 2, 4, 4]
        num_mults = len(channel_mults)
        pre_channel = n_feats
        feat_channels = [pre_channel]

        # RGB mean for DIV2K
        self.y_mean = 119.5359

        downs = [nn.Conv2d(state_num, n_feats, kernel_size=(3, 3), padding=1)]

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = n_feats * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(pre_channel, channel_mult, kernel_size=3, bias=True, res_scale=res_scale))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [ResBlock(pre_channel, pre_channel, kernel_size=3, res_scale=res_scale) for _ in range(n_resblocks)])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = n_feats * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResBlock(pre_channel + feat_channels.pop(), channel_mult, kernel_size=3, bias=True,
                                    res_scale=res_scale))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))

        self.ups = nn.ModuleList(ups)

        self.final_conv = nn.Conv2d(pre_channel, agent_feat, kernel_size=(3, 3), padding=1)

        # net_list
        modules_policy_action = [
            ResBlock(agent_feat, agent_feat, 3, res_scale=1) for _ in range(4)
        ]

        # sigma
        self.policy_action = nn.Sequential(*modules_policy_action)

        self.policy_alpha_action = nn.Sequential(
            nn.Conv2d(in_channels=agent_feat, out_channels=agent_feat, kernel_size=3, stride=1, padding=(1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=agent_feat, out_channels=action_num,
                      kernel_size=3, stride=1, padding=(1, 1), bias=True)
        )

        self.policy_beta_action = nn.Sequential(
            nn.Conv2d(in_channels=agent_feat, out_channels=agent_feat, kernel_size=3, stride=1, padding=(1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=agent_feat, out_channels=action_num,
                      kernel_size=3, stride=1, padding=(1, 1), bias=True)
        )

        modules_value = [
            ResBlock(agent_feat, agent_feat, 3, res_scale=1) for _ in range(4)
        ]

        self.value_conv = nn.Sequential(*modules_value)

        self.value = nn.Sequential(
            nn.Conv2d(in_channels=agent_feat, out_channels=agent_feat, kernel_size=3, stride=1, padding=(1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=agent_feat, out_channels=1, kernel_size=3, stride=1, padding=(1, 1), bias=True),
        )

    def forward(self, x):

        # Beta distribution
        x = x - self.y_mean

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

        # Continuous action
        p = self.policy_action(x)
        policy_alpha = F.softplus(self.policy_alpha_action(p)) + 1.0
        policy_beta = F.softplus(self.policy_beta_action(p)) + 1.0

        # Value function
        v = self.value_conv(x)
        value = self.value(v)

        return policy_alpha, policy_beta, value

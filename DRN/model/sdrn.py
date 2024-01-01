import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import common


def make_model(opt, parent=False):
    return SDRN(opt)


class SDRN(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(SDRN, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        assert opt.configs_path is not None, "configs file should be provided."
        self.channel_configs = np.loadtxt(opt.configs_path, dtype=int)
        
        channel_counter = 0 
        kernel_size = 3

        act = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)  # or resblocks

        self.down = []
        for p in range(self.phase):
            # the inner channels of the down blocks is defined by configs
            inner_chs = self.channel_configs[channel_counter]
            channel_counter += 1
            self.down.append(common.DownBlock(
                opt, 2, inner_chs, n_feats * pow(2, p), n_feats * pow(2, p + 1))) 

        self.down = nn.ModuleList(self.down)

        # init the first blocks before upsample the feature
        up_body_blocks = [[]]
        for _ in range(n_resblocks):
            inner_chs = self.channel_configs[channel_counter]
            channel_counter += 1
            up_body_blocks[0].append(
                common.RCAB(conv, n_feats * pow(2, self.phase), 
                    kernel_size, act=act, pruned_feats=inner_chs))
            
        # the first upsamler after the blocks
        inner_chs = self.channel_configs[channel_counter]
        channel_counter += 1
        up = [[common.Upsampler(conv, 2, 
                    n_feats * pow(2, self.phase), pruned_feats=inner_chs, act=False),
            conv(inner_chs, n_feats * pow(2, self.phase - 1), kernel_size=1)]]

        for p in range(self.phase, 1, -1):
            # init up_body_block 
            up_body_block = []
            for _ in range(n_resblocks):
                inner_chs = self.channel_configs[channel_counter]
                channel_counter += 1
                up_body_block.append(common.RCAB(conv, 
                    n_feats * pow(2, p), kernel_size, act=act, pruned_feats=inner_chs))
            up_body_blocks.append(up_body_block)
            
            # add upsampling operation
            inner_chs = self.channel_configs[channel_counter]
            channel_counter += 1
            up.append([common.Upsampler(conv, 2, n_feats * pow(2, p), 
                    pruned_feats=inner_chs, act=False),
                conv(inner_chs, n_feats * pow(2, p - 2), kernel_size=1)])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail blocks output sr imgs from intermedia features
        self.tail = [conv(
            n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            self.tail.append(
                conv(n_feats * pow(2, p), opt.n_colors, kernel_size))
        self.tail = nn.ModuleList(self.tail)

        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        # upsample x to target sr size
        x = self.upsample(x)

        # preprocess
        x = self.sub_mean(x)
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        sr = self.tail[0](x)
        sr = self.add_mean(sr)
        results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # output sr imgs
            sr = self.tail[idx + 1](x)
            sr = self.add_mean(sr)

            results.append(sr)

        return results
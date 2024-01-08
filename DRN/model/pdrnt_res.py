import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
import math

def make_model(opt, parent=False):
    return DRNTRes(opt)


class DRNTRes(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(DRNTRes, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        pruned_feats = opt.n_feats * (1 - opt.pruning_rate)
        kernel_size = 3

        act = nn.LeakyReLU(0.2, True) # nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bilinear', align_corners=False)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            nn.Sequential(
                nn.Conv2d(n_feats * pow(2, p), n_feats * pow(2, p + 1), 
                        kernel_size, stride=2, padding=kernel_size//2, bias=True),
                act
            ) for p in range(self.phase)
        ]  # the channels of the down blocks is n_feats * pow(2, p)

        self.down = nn.ModuleList(self.down)

        # the learning resblocks before upsample the feature
        assert n_resblocks >= 2, "Too small number of blocks"
        up_body_blocks = [[
            common.ResBlock(
                conv, n_feats * pow(2, p), kernel_size, act=act,
                pruned_feats=math.ceil(pruned_feats * pow(2, p))
            ) for _ in range(n_resblocks//2)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
          common.ResBlock(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act,
                pruned_feats=math.ceil(pruned_feats * pow(2, self.phase))
            ) for _ in range(n_resblocks+n_resblocks//2)
        ])

        # Upsample blocks, using conv and pixel shuffle, after upsample, \
        # compresee_units reduce the channel to be suitable for the networks.
        # The first up don't concat the replicated features and LR inputs features
        up = [[
            nn.PixelShuffle(2),
            conv(n_feats * pow(2, self.phase) // 4, 
                 n_feats * pow(2, self.phase - 1), kernel_size=1)
            # nn.ConvTranspose2d(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), 
            #         kernel_size=4, stride=2, padding=1,  output_padding=0)
        ]]

        # The rest of up blocks concat the replicated features and LR inputs features
        for p in range(self.phase - 1, 0, -1):
            up.append([
                nn.PixelShuffle(2),
                conv(2 * n_feats * pow(2, p) // 4, n_feats * pow(2, p - 1), kernel_size=1)
                # nn.ConvTranspose2d(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), 
                #     kernel_size=4, stride=2, padding=1,  output_padding=0)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail blocks output sr imgs from intermedia features
        self.tail = [conv(
            n_feats * pow(2, self.phase), opt.n_colors, kernel_size
        )]
        for p in range(self.phase, 0, -1):
            self.tail.append(
                conv(
                    n_feats * pow(2, p), opt.n_colors, kernel_size
                )
            )
        self.tail = nn.ModuleList(self.tail)


    def forward(self, x):
        # upsample x to target sr size
        x = self.upsample(x)

        # preprocess
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        results = []
        if self.train:
            sr = self.tail[0](x)
            results.append(sr)
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # output sr imgs
            # if idx == self.phase - 1:
            sr = self.tail[idx + 1](x)
            results.append(sr)

        return results # sr
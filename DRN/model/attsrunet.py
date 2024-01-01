import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


def make_model(opt, parent=False):
    return ATTSRUNET(opt)


class ATTSRUNET(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(ATTSRUNET, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size = 3

        act = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)  # or resblocks

        self.down = [
            common.DownBlock(
                opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]  # the channels of the down blocks is n_feats * pow(2, p)

        self.down = nn.ModuleList(self.down)

        # the learning resblocks before upsample the feature
        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_resblocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_resblocks)
        ])

        # Upsample blocks, using conv and pixel shuffle, after upsample, \
        # compresee_units reduce the channel to be suitable for the networks.
        # The first up don't concat the replicated features and LR inputs features
        up = [[
            common.Upsampler(
                conv, 2, n_feats * pow(2, self.phase), act=False
            ),
            conv(
                n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1
            )
        ]]

        # The rest of up blocks concat the replicated features and LR inputs features
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(
                    conv, 2, 2 * n_feats * pow(2, p), act=False
                ),
                conv(
                    2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1
                )
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail blocks output sr imgs from intermedia features
        # self.tail = nn.ModuleList([
        #     conv(
        #         n_feats * pow(2, p), opt.n_colors, kernel_size
        #     ) for p in range(self.phase, 0, -1)
        # ])
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
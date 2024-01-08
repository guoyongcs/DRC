from model import common
import math
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opt, gan_type='GAN'):
        super(Discriminator, self).__init__()
        self.opt = opt
        in_channels = 3
        out_channels = 64
        depth = 7
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        m_features = [
            common.BasicBlock(opt.n_colors, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(common.BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        self.patch_size = opt.patch_size // (2**((depth + 1) // 2))
        
        scale = [1]
        scale += opt.scale
        self.linear = nn.ModuleDict([
            [str(s), nn.Linear(out_channels * math.ceil(self.patch_size/s)**2 , 1024)] for s in scale
        ])

        m_classifier = [
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        scale = self.opt.patch_size // x.size(-1)
        features = self.linear[str(scale)](features.view(features.size(0), -1))
        output = self.classifier(features)
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common

def make_model(opt, parent=False):
    return USRN(opt)

class UBlock(nn.Module):
    def __init__(self, opt, n_feats, kernel_size=3, act=nn.ReLU(True)):
        super(UBlock, self).__init__()
        self.down1 = nn.Conv2d(n_feats, n_feats, kernel_size, stride=2, padding=1, bias=True)
        self.down2 = nn.Conv2d(n_feats, n_feats, kernel_size, stride=2, padding=1, bias=True)
        self.up1 = nn.Conv2d(n_feats, 4 * n_feats, kernel_size, padding=1, bias=True)
        self.up2 = nn.Conv2d(n_feats, 4 * n_feats, kernel_size, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.act = act

    def forward(self, x):
        x_down1 = self.act(self.down1(x))
        x_down2 = self.act(self.down2(x_down1))

        x_up1 = self.act(self.up1(x_down2))
        x_up1 = self.pixel_shuffle(x_up1) + x_down1
        x_up2 = self.act(self.up2(x_up1))
        x_up2 = self.pixel_shuffle(x_up2) + x
        return x_up2

class USRN(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(USRN, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats # 48 
        s_feats = n_feats // 4
        kernel_size = 3

        act = nn.ReLU(True) # nn.LeakyReLU(0.2, True) # 
        
        self.head = conv(opt.n_colors, n_feats, kernel_size)
        
        self.shrink = nn.Sequential(
            nn.Conv2d(n_feats, s_feats, kernel_size=1, stride=1, padding=0), nn.ReLU(True))
        

        bodys = []
        blocks = [UBlock(opt, s_feats, kernel_size, act=act) for i in range(n_resblocks)]
        bodys.append(nn.Sequential(*blocks))
        blocks = [UBlock(opt, s_feats, kernel_size, act=act) for i in range(n_resblocks)]
        bodys.append(nn.Sequential(*blocks))
        self.bodys = nn.ModuleList(bodys)
        
        # tail blocks for output sr imgs
        tail = []
        tail.append(nn.Sequential(
            conv(s_feats, 4 * opt.n_colors, kernel_size), 
            nn.PixelShuffle(2)
        ))
        tail.append(nn.Sequential(
            conv(s_feats, 16 * opt.n_colors, kernel_size), 
            nn.PixelShuffle(4)
        ))
        
        self.tail = nn.ModuleList(tail)


    def forward(self, x):
        # preprocess
        x = self.head(x)
        x = self.shrink(x)
        
        results = []
        # non-linear mapping
        x = self.bodys[0](x)
        sr = self.tail[0](x)
        results.append(sr)
        
        x = self.bodys[1](x)
        sr = self.tail[1](x)
        results.append(sr)

        # return sr 
        return results
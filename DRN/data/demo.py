import os
import glob
from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            img_paths = sorted(glob.glob(os.path.join(args.dir_demo, f, '*')))
            self.filelist.extend(img_paths)

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        video = os.path.basename(os.path.dirname(self.filelist[idx]))
        lr = imageio.imread(self.filelist[idx])
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

        return lr_t, -1, [filename, video]

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale


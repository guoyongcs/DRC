import os
import glob

from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale.copy()
        self.scale.reverse()
        
        self._set_filesystem(args.data_dir)
        self._get_imgs_path(args)
        self._set_dataset_length()

    def _get_imgs_path(self, args):
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or self.benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(
                    args.ext, h, b, verbose=True
                )

            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(
                        args.ext, l, b,  verbose=True
                    )
    
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    # Below functions as used to prepare images
    def _set_dataset_length(self):
        if self.train:
            self.dataset_length = self.args.test_every * self.args.batch_size
            repeat = max(self.dataset_length // len(self.images_hr), 1)
            self.random_border = len(self.images_hr) * repeat
        else:
            self.dataset_length = len(self.images_hr)

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)# only load one lr image with current scale

        if self.train:
            th, tw = hr.shape[:2]
            tmp_idx = idx
            while th < self.args.patch_size or tw < self.args.patch_size or len(hr.shape) < 3:
                tmp_idx += 1
                tmp_idx = tmp_idx % self.dataset_length
                lr, hr, filename = self._load_file(tmp_idx)  # only load one lr image with current scale
                th, tw = hr.shape[:2]

        lr, hr = self.get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        
        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )

        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return self.dataset_length

    def _get_index(self, idx):
        if self.train:
            if idx < self.random_border:
                return idx % len(self.images_hr)
            else:
                return np.random.randint(len(self.images_hr))
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = [self.images_lr[idx_scale][idx] for idx_scale in range(len(self.images_lr))]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = [imageio.imread(f_lr[idx_scale]) for idx_scale in range(len(self.images_lr))]
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f: 
                hr = pickle.load(_f)
            lr = []
            for f in f_lr:
                with open(f, 'rb') as _f: 
                    lr.append(pickle.load(_f))
        

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            if isinstance(lr, list):
                ih, iw = lr[0].shape[:2]
            else:
                ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale[0], 0:iw * scale[0]]
            
        return lr, hr


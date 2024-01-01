import os
from data import srdata

class ImageNet3K(srdata.SRData):
    def __init__(self, args, name='ImageNet3K', train=True, benchmark=False):
        super(ImageNet3K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(ImageNet3K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'ImageNet3K_HR')
        self.dir_lr = os.path.join(self.apath, 'ImageNet3K_LR_bicubic')
        self.ext = ('.png', '.png')
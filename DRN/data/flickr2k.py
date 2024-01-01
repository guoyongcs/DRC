import os
from data import srdata

class Flickr2K(srdata.SRData):
    def __init__(self, args, name='Flickr2K', train=True, benchmark=False):
        super(Flickr2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(Flickr2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'Flickr2K_HR')
        self.dir_lr = os.path.join(self.apath, 'Flickr2K_LR_bicubic')


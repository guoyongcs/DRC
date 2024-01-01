import os
from data import srdata

class DIV2KS(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        super(DIV2KS, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        self._set_data_range()
        names_hr, names_lr = super(DIV2KS, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2KS, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')

    def _set_data_range(self):
        data_range = [r.split('-') for r in self.args.data_range.split('/')]
        if self.train:
            data_range = data_range[0]
        else:
            if self.args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
    
    # Below functions as used to prepare images
    def _set_dataset_length(self):
        if self.train:
            self.dataset_length = self.args.test_every * self.args.batch_size
            self.dataset_length = min(self.dataset_length, len(self.images_hr))
            self.images_hr = self.images_hr[:self.dataset_length]
            self.images_lr = [images_lr[:self.dataset_length] for images_lr in self.images_lr]
        else:
            self.dataset_length = len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx


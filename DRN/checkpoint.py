import os
import torch
import math
import datetime
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Checkpoint():
    def __init__(self, opt):
        self.opt = opt
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if opt.load == '.':
            if opt.save == '.':
                opt.save = '../experiment/EXP/' + now
            self.dir = opt.save
        else:
            self.dir = opt.load
            if not os.path.exists(self.dir):
                opt.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if opt.reset:
            os.system('rm -rf ' + self.dir)
            opt.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(opt):
                f.write('{}: {}\n'.format(arg, getattr(opt, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )
        if self.opt.dual:
            dual_optimizers = {}
            for i in range(len(trainer.dual_optimizers)):
                dual_optimizers[i] = trainer.dual_optimizers[i]
            torch.save(
                dual_optimizers,
                os.path.join(self.dir, 'dual_optimizers.pt')
            )

            
    def load(self, apath):
        pass

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.opt.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate([self.opt.scale[0]]):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.opt.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        temp = filename.split('_')
        if len(temp) == 2:
            video, filename = temp
            apath = '{}/results/{}/{}'.format(self.dir, self.opt.data_test, video)
        else:
            filename = temp[0]
            apath = '{}/results/{}'.format(self.dir, self.opt.data_test)
        if not os.path.exists(apath):
            os.makedirs(apath)
        filename = os.path.join(apath, filename)

        v = save_list[0]
        normalized = v[0].data.mul(255 / self.opt.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        imageio.imwrite('{}.png'.format(filename), ndarr)
    
    def save_results_nopostfix(self, filename, save_list, scale):
        apath = '{}/results/{}/X{}'.format(self.dir, self.opt.data_test, scale)
        if not os.path.exists(apath):
            os.makedirs(apath)
        filename = os.path.join(apath, filename)

        v = save_list[0]
        normalized = v[0].data.mul(255 / self.opt.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        imageio.imwrite('{}.png'.format(filename), ndarr)
import os
import math
import time
import torch
import utility
from decimal import Decimal
from tqdm import tqdm
from importlib import import_module
import torch.nn.functional as F


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        # self.loss_hook = self.loss.register_backward_hook(backward_hook)
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        if self.opt.dual:
            self.dual_models = self.model.dual_models
            self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
            self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        
        self.error_last = 1e8
        # to be compatible with torch when version >= 1.1
        self.torch_version = float(torch.__version__[0:3])

        if self.opt.load != '.':
            print("load optimizer")
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            if self.opt.dual:
                print("load dual_optimizer")
                dual_optimizers = torch.load(
                    os.path.join(ckp.dir, 'dual_optimizers.pt')
                )

                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i] = dual_optimizers[i]
            
            # update the step of the optimizers
            if self.torch_version < 1.1:
                for _ in range(len(ckp.log)): 
                    self.scheduler.step()
                    if self.opt.dual:
                        for i in range(len(self.dual_scheduler)):
                            self.dual_scheduler[i].step()
            else:
                self.scheduler.last_epoch = len(ckp.log)
                if self.opt.dual:
                    for i in range(len(self.dual_scheduler)):
                        self.dual_scheduler[i].last_epoch == len(ckp.log)


    def train(self):
        if self.torch_version < 1.1:
            self.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            # caculate the loss
            if self.opt.dual:
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].zero_grad()
                
                # forward
                sr = self.model(lr[0])
                sr2lr = []
                assert len(self.dual_models) == len(self.scale), \
                    'the length of dual_models is worng.'
                for i in range(len(self.dual_models)):
                    sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                    sr2lr.append(sr2lr_i)

                # compute primary loss
                loss_primary = self.loss(sr[-1], hr) 
                for i in range(1, len(sr)):
                    loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

                # compute dual loss
                loss_dual = self.loss(sr2lr[0], lr[0])
                for i in range(1, len(self.scale)):
                    loss_dual += self.loss(sr2lr[i - len(self.scale)], lr[i - len(self.scale)])

                # print('loss_primary', loss_primary.item(), 'loss_dual', loss_dual.item())
                # compute total loss
                loss = loss_primary + self.opt.dual_weight * loss_dual
            else:
                sr = self.model(lr[0])
                if not isinstance(sr, list): sr = [sr]
                loss = self.loss(sr[-1], hr)
                for i in range(1, len(sr)):
                    loss += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])
            
            if loss.item() < self.opt.skip_threshold * self.error_last:
                # update the model
                loss.backward()                
                self.optimizer.step()
                if self.opt.dual:
                    for i in range(len(self.dual_optimizers)):
                        self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        # update scheduler
        if self.torch_version >= 1.1:
            self.step()

    def test(self):
        if self.torch_version < 1.1:
            epoch = self.scheduler.last_epoch + 1
        else:
            epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        # self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                eval_ssim = 0
                # self.loader_test.dataset.set_scale(si)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)
                    if isinstance(lr, list): lr = lr[0]
                    
                    if self.opt.bicubic:
                        sr = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
                    else:
                        sr = self.model(lr)

                    if isinstance(sr, list):
                        sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)
                    save_list = [sr]

                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        if self.opt.calc_ssim:
                            eval_ssim += utility.calc_ssim(
                                sr, hr, s, self.opt.rgb_range,
                                benchmark=self.loader_test.dataset.benchmark
                            )

                    # save test results
                    if self.opt.save_results:
                        # self.ckp.save_results_nopostfix(filename, save_list, s)
                        self.ckp.save_results(filename, save_list, s)

                    # save intermediate images during training 
                    if self.opt.save_train_sr and epoch % self.opt.save_every == 0:
                        self.ckp.save_results_nopostfix(filename + "_" + str(epoch), save_list, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.opt.data_test,
                        s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )
                if self.opt.calc_ssim:
                    ssim = eval_ssim / len(self.loader_test)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR/SSIM {:.2f}/{:.4f}'.format(
                            self.opt.data_test,
                            s,
                            self.ckp.log[-1, si],
                            ssim
                        )
                    )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        if self.opt.dual:
            for i in range(len(self.dual_scheduler)):
                self.dual_scheduler[i].step()
        self.loss.step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')
        def _prepare(tensor):
            if self.opt.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        if isinstance(args[0], list):
            return [_prepare(a) for a in args[0]], _prepare(args[1])
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            if self.torch_version < 1.1:
                epoch = self.scheduler.last_epoch + 1
            else:
                epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
    



""" Search cell """
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from decimal import Decimal
from torch.optim import Adam
from torch.optim import lr_scheduler

from utils import pruned_utils
from utils.pruned_utils import get_logger, timer, tensor2uint

from .search_controller import SearchController
from .architect_update import ArchitectUpdate
from .search_checkpoint import SearchCheckpoint
from models.modules.loss import CorrectionLoss


class Search():
    def __init__(self, args, model, train_loader, test_loader, prepro):
        self.args = args
        opt = args.opt['search']
        # initalize logger
        self.logger = get_logger(args.save_dir, "search", log_file='search.log')
        # self.tensorboard_logger = TensorboardLogger(osp.join(args.save_dir, 'search_tb'))
        self.tensorboard_logger = None
        self.logger.info("|===>Result will be saved at {}".format(args.save_dir))
        
        self.train_loader, self.valid_loader = self.split_data(train_loader)
        self.test_loader = test_loader
        self.prepro = prepro

        self.model = model.netG
        self.criterion = nn.L1Loss().cuda()
        self.cri_kernel = CorrectionLoss(scale=args.scale, eps=1e-20).cuda()

        self.controller = SearchController(args, self.model, self.criterion, self.logger).cuda()
        self.architect = ArchitectUpdate(self.controller, opt['w_momentum'], opt['w_weight_decay'])
        
        self.create_optimizer()
        self.checkpoint = SearchCheckpoint(args)

        self.start_epoch = 1
        self.best_psnr = 0
        self.best_epoch = 0
        self.logger_counter = 0

        if opt['resume_search'] is not None:
            self.checkpoint.resume(self, opt['resume_search'])
    
    def split_data(self, loader_train):
        # split data to train/validation
        n_train = len(loader_train.dataset)
        split = n_train // 2
        indices = list(range(n_train))
        batch_size = self.args.opt['datasets']['train']['batch_size']
        num_workers = self.args.opt['datasets']['train']['n_workers']
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        train_loader = torch.utils.data.DataLoader(loader_train.dataset,
                                                batch_size=batch_size,
                                                sampler=train_sampler,
                                                num_workers=num_workers,
                                                pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(loader_train.dataset,
                                                batch_size=batch_size,
                                                sampler=valid_sampler,
                                                num_workers=num_workers,
                                                pin_memory=True)
        return train_loader, valid_loader


    def create_optimizer(self):
        opt_train = self.args.opt['train']
        opt_search = self.args.opt['search']
        self.w_optimizer = self.make_optimizer(opt_train, tuple(self.controller.weights()))
        self.w_scheduler = self.make_scheduler(opt_train, self.w_optimizer)
        self.alpha_optimizer = self.make_alpha_optimizer(opt_search, tuple(self.controller.alphas()))
        self.alpha_scheduler = self.make_scheduler(opt_train, self.alpha_optimizer)
    
    def make_optimizer(self, opt_train, parameters):
        betas = (opt_train['beta1'], opt_train['beta2'])
        wd_G = opt_train["weight_decay_G"] if opt_train["weight_decay_G"] else 0
        optimizer = Adam(parameters, lr=opt_train['lr_G'],
                        betas=betas, weight_decay=wd_G)
        
        return optimizer
    
    def make_alpha_optimizer(self, opt_search, parameters):
        betas = (0.9, 0.999)
        optimizer = Adam(parameters, lr=opt_search['alpha_lr'],
                        betas=betas, weight_decay=opt_search['alpha_weight_decay'])
        
        return optimizer

    def make_scheduler(slef, opt_train, optimizer):
        if opt_train['lr_scheme'] == 'MultiStepLR':
            scheduler =  lr_scheduler.MultiStepLR(optimizer,
                                    opt_train['lr_steps'],
                                    opt_train['lr_gamma']
                        )
        else:
            raise NotImplementedError

        return scheduler

    def search(self):
        timer_epoch = timer()
        total_epochs = self.args.opt['search']['search_epochs']
        for epoch in range(self.start_epoch, total_epochs + 1):           
            self.train(epoch)
            self.validate(epoch)
            
            if epoch % 10 == 0:
                self.controller.print_alphas()
            is_best = epoch == self.best_epoch
            self.checkpoint.save_config(self.controller, is_best)
            
            # print log about remain search time  
            epoch_time = int(timer_epoch.toc())
            log = pruned_utils.remain_time(epoch_time, epoch, total_epochs)
            self.logger.info(log)
            timer_epoch.tic()

    def prepare_data(self, batch):
        hr = batch["GT"]
        
        lr, ker_map, kernels, lr_blured_t, lr_t = \
            self.prepro(hr, True, return_blur=True)
        lr = (lr * 255).round() / 255
        
        lr, hr = lr.cuda(), hr.cuda()
        lr_blured_t, lr_t = lr_blured_t.cuda(), lr_t.cuda()
        # do not need to use ker_map and kernels for computing loss

        return lr, hr, ker_map, kernels, lr_blured_t, lr_t

    def train(self, epoch):
        w_lr = self.w_scheduler.get_last_lr()[-1]
        alpha_lr = self.alpha_scheduler.get_last_lr()[-1]
        self.controller.train() 
        timer_data, timer_model = timer(), timer()
        for step, (train_batch, val_batch) in \
                        enumerate(zip(self.train_loader, self.valid_loader)):
            
            # prepare data
            trn_lr, trn_hr, _, _, trn_lr_blured_t, trn_lr_t = self.prepare_data(train_batch)
            val_lr, val_hr, _, _, val_lr_blured_t, val_lr_t = self.prepare_data(val_batch)

            timer_data.hold()
            timer_model.tic()
            # phase 2. architect step (alpha)
            self.alpha_optimizer.zero_grad()
            # fix bugs, change alpha lr to w_lr (Done!)
            val_loss = self.architect.unrolled_backward(
                trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t,
                val_lr, val_hr, val_lr_blured_t, val_lr_t,
                w_lr, self.w_optimizer)
            # clip grad for stability
            torch.nn.utils.clip_grad_norm_(self.controller.alphas(), .1)
            torch.nn.utils.clip_grad_value_(self.controller.alphas(), .1)

            self.alpha_optimizer.step()

            # phase 1. child network step (w)
            self.w_optimizer.zero_grad()
            sr, est_kernel, sr2lr = self.controller(trn_lr)
            trn_loss = self.compute_loss(
                trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t,
                sr, est_kernel, sr2lr)
            trn_loss.backward()
            # clip grad for stability
            torch.nn.utils.clip_grad_norm_(self.controller.weights(), .1)
            torch.nn.utils.clip_grad_value_(self.controller.weights(), .1)

            self.w_optimizer.step()

            timer_model.hold()

            if (step + 1) % self.args.opt['search']['print_every'] == 0:
                # print log
                # self.tensorboard_logger.scalar_summary(
                #     tag="Train_loss", value=trn_loss, step=self.logger_counter)
                # self.tensorboard_logger.scalar_summary(
                #     tag="Valid_loss", value=val_loss, step=self.logger_counter)
                self.logger_counter += 1
                
                batch_size = self.args.opt['datasets']['train']['batch_size']
                log = "Epoch: {:0>4d} [{:0>5d}/{:d}] ".format(epoch, 
                        (step + 1) * batch_size, len(self.train_loader.dataset)//2)
                log += "w_lr: {:.2e}  alpha_lr: {:.2e} ".format(Decimal(w_lr), Decimal(alpha_lr))
                log += 'Train Loss: {:.4e} Valid Loss: {:.4e} '.format(trn_loss, val_loss)
                log += '{:.1f}+{:.1f}s'.format(timer_model.release(), timer_data.release())
                self.logger.info(log)

            timer_data.tic()
            
        self.w_scheduler.step()
        self.alpha_scheduler.step()

    def validate(self, epoch):
        self.controller.eval() # repalce model with search controller
        psnr = 0
        with torch.no_grad():
            for step, test_batch in enumerate(self.test_loader):
                lr, hr = test_batch['LQ'].cuda(), test_batch['GT'].cuda()
                sr, _, _ = self.controller(lr) # repalce model with search controller
                sr = tensor2uint(sr)
                hr = tensor2uint(hr)
                psnr += pruned_utils.calc_psnr(sr, hr, self.args.scale)

            psnr = psnr / len(self.test_loader)
    
            if psnr >= self.best_psnr:
                self.best_psnr = psnr
                self.best_epoch = epoch

            search_log = '[{} x{}]  '.format(
                self.args.opt['datasets']['val']['name'], self.args.scale)
            search_log += 'PSNR:  {:.3f} '.format(psnr)
            search_log += '(Best: {:.3f} @epoch {})'.format(
                            self.best_psnr, self.best_epoch)
            self.logger.info(search_log)

        self.checkpoint.save(self, epoch, self.args.save_dir)

    def compute_loss(self, lr, hr, lr_blured, lr_t, sr, est_kernel, sr2lr=None):
        # compute primal regression loss
        loss = self.criterion(sr, hr)
        
        # compute dual regression loss
        if sr2lr is not None:
            loss_dual = self.criterion(sr2lr, lr)
            loss += self.args.opt['dual_weight'] * loss_dual
        
        # compute kernel regression loss
        loss_k, _ = self.cri_kernel(est_kernel, lr_blured, lr_t)
        loss += loss_k

        return loss
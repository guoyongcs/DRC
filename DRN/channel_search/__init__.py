""" Search cell """
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from decimal import Decimal
import utility
from utils import get_logger, TensorboardLogger
from .search_controller import SearchController
from .architect_update import ArchitectUpdate
from .search_checkpoint import SearchCheckpoint


class Search():
    def __init__(self, args, model, loader):
        self.args = args
        # initalize logger
        self.logger = get_logger(args.save, "search", log_file='search.log')
        self.tensorboard_logger = TensorboardLogger(osp.join(args.save, 'search_tb'))
        self.logger.info("|===>Result will be saved at {}".format(args.save))
        
        self.train_loader, self.valid_loader = self.split_data(loader.loader_train)
        self.test_loader = loader.loader_test

        self.model = model
        self.criterion_L1 = nn.L1Loss().cuda()
        self.controller = SearchController(args, model, self.criterion_L1, self.logger).cuda()
        self.architect = ArchitectUpdate(self.controller, args.w_momentum, args.w_weight_decay)
        
        self.create_optimizer()
        self.checkpoint = SearchCheckpoint(args)

        self.start_epoch = 1
        self.best_psnr = 0
        self.best_epoch = 0
        self.logger_counter = 0

        if args.resume_search is not None:
            self.checkpoint.resume(self, args.resume_search)
    
    def split_data(self, loader_train):
        # split data to train/validation
        n_train = len(loader_train.dataset)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        train_loader = torch.utils.data.DataLoader(loader_train.dataset,
                                                batch_size=self.args.batch_size,
                                                sampler=train_sampler,
                                                num_workers=self.args.n_threads,
                                                pin_memory=not self.args.cpu)
        valid_loader = torch.utils.data.DataLoader(loader_train.dataset,
                                                batch_size=self.args.batch_size,
                                                sampler=valid_sampler,
                                                num_workers=self.args.n_threads,
                                                pin_memory=not self.args.cpu)
        return train_loader, valid_loader


    def create_optimizer(self):
        self.w_optimizer = utility.make_drn_optimizer(self.args, self.model)
        self.w_scheduler = utility.make_scheduler(self.args, self.w_optimizer)
        self.alpha_optimizer = utility.make_alpha_optimizer(self.args, tuple(self.controller.alphas()))
        self.alpha_scheduler = utility.make_scheduler(self.args, self.alpha_optimizer)

    def search(self):
        timer_epoch = utility.timer()
        for epoch in range(self.start_epoch, self.args.epochs + 1):           
            self.train(epoch)
            self.validate(epoch)
            
            if epoch % 10 == 0:
                self.controller.print_alphas()
            is_best = epoch == self.best_epoch
            self.checkpoint.save_config(self.controller, epoch, is_best)
            
            # print log about remain search time  
            epoch_time = int(timer_epoch.toc())
            log = utility.remain_time(epoch_time, epoch, self.args.epochs)
            self.logger.info(log)
            timer_epoch.tic()

    def train(self, epoch):
        w_lr = self.w_scheduler.get_last_lr()[-1]
        alpha_lr = self.alpha_scheduler.get_last_lr()[-1]
        self.controller.train() # Fix, change original model to controller 
        timer_data, timer_model = utility.timer(), utility.timer()
        for step, ((trn_lr, trn_hr, trn_fn), (val_lr, val_hr, val_fn)) in \
                        enumerate(zip(self.train_loader, self.valid_loader)):
            if not self.args.cpu:
                trn_lr = [lr.cuda() for lr in trn_lr]
                trn_hr = trn_hr.cuda()
                val_lr = [lr.cuda() for lr in val_lr]
                val_hr = val_hr.cuda()
            timer_data.hold()
            timer_model.tic()
            # phase 2. architect step (alpha)
            self.alpha_optimizer.zero_grad()
            # fix bugs, change alpha lr to w_lr (Done!)
            val_loss = self.architect.unrolled_backward(
                trn_lr, trn_hr, val_lr, val_hr, w_lr, self.w_optimizer)
            self.alpha_optimizer.step()

            # phase 1. child network step (w)
            self.w_optimizer.zero_grad()
            sr, sr2lr = self.controller(trn_lr)
            trn_loss = self.compute_loss(trn_lr, trn_hr, sr, sr2lr)
            trn_loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(self.controller.weights(), self.args.w_grad_clip)
            self.w_optimizer.step()

            timer_model.hold()

            if (step + 1) % self.args.print_every == 0:
                # print log
                self.tensorboard_logger.scalar_summary(
                    tag="Train_loss", value=trn_loss, step=self.logger_counter)
                self.tensorboard_logger.scalar_summary(
                    tag="Valid_loss", value=val_loss, step=self.logger_counter)
                self.logger_counter += 1
                log = "Epoch: {:0>4d} [{:0>5d}/{:d}] ".format(epoch, 
                    (step + 1) * self.args.batch_size, len(self.train_loader.dataset)//2)
                log += "w_lr: {:.2e}  alpha_lr: {:.2e} ".format(Decimal(w_lr), Decimal(alpha_lr))
                log += 'Train Loss: {:.4e} Valid Loss: {:.4e} '.format(trn_loss, val_loss)
                log += '{:.1f}+{:.1f}s'.format(
                    timer_model.release(), timer_data.release())
                self.logger.info(log)

            timer_data.tic()
            
        self.w_scheduler.step()
        self.alpha_scheduler.step()

    def validate(self, epoch):
        self.controller.eval() # repalce model with search controller
        psnr = 0
        with torch.no_grad():
            for step, (lr, hr, _) in enumerate(self.test_loader):
                if isinstance(lr, list): lr = lr[0]
                lr, hr = lr.cuda(), hr.cuda()
                sr, _ = self.controller(lr) # repalce model with search controller
                if isinstance(sr, list): sr = sr[-1]
                sr = utility.quantize(sr)
                psnr += utility.calc_psnr(sr, hr, max(self.args.scale))

            psnr = psnr / len(self.test_loader)
    
            if psnr >= self.best_psnr:
                self.best_psnr = psnr
                self.best_epoch = epoch

            search_log = '[{} x{}]  '.format(
                self.args.data_test, max(self.args.scale))
            search_log += 'PSNR:  {:.3f} '.format(psnr)
            search_log += '(Best: {:.3f} @epoch {})'.format(
                            self.best_psnr, self.best_epoch)
            self.logger.info(search_log)

        self.checkpoint.save(self, epoch, self.args.save)

    def compute_loss(self, lr, hr, sr, sr2lr=None):
        total_loss = 0
        # compute primal regression loss
        if not isinstance(sr, list): sr = [sr]
        loss_primary = self.criterion_L1(sr[-1], hr)
        for i in range(1, len(sr)):
            loss_primary += self.criterion_L1(
                sr[i - 1 - len(sr)], lr[i - len(sr)])
        
        total_loss += loss_primary
        
        # compute dual regression loss
        if self.args.dual:
            loss_dual = self.criterion_L1(sr2lr[0], lr[0])
            for i in range(1, len(self.args.scale)):
                loss_dual += self.criterion_L1(
                    sr2lr[i - len(self.args.scale)], lr[i - len(self.args.scale)])
            
            total_loss += self.args.dual_weight * loss_dual

        return total_loss
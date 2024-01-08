import torch
import torch.nn as nn
import os.path as osp
from decimal import Decimal
import utility
from utils import get_logger, concat_gpu_datalist
from .finetune_checkpoint import FinetuneCheckpoints


class Finetuner():
    def __init__(self, args, original_model, pruned_model, loader):
        self.args = args
        self.scale = args.scale
        self.original_model = original_model
        self.pruned_model = pruned_model
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        
        self.pruned_optimizer = utility.make_drn_optimizer(args, self.pruned_model)
        self.pruned_scheduler = utility.make_scheduler(args, self.pruned_optimizer)

        self.cache_original_feature = {}
        self.cache_pruned_feature = {}

        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_L1 = nn.L1Loss().cuda()

        self.logger = get_logger(args.save, "finetune", log_file='finetune.txt')
        self.tensorboard_logger = None
        self.logger.info("|===>Result will be saved at {}".format(args.save))
        
        self.start_epoch = 1
        
        self.pruned_best_psnr, self.pruned_best_epoch = 0, 0

        self.logger_counter = 0
        
        self.fix_original_model()
        self.register_hooks()
        self.checkpoint = FinetuneCheckpoints(args)
        
        if args.resume_finetune is not None:
            self.checkpoint.resume(self, args.resume_finetune)
    
    def fix_original_model(self):
        self.original_model.eval()
        for p in self.original_model.parameters():
            p.required_grad = False

    def finetune(self):
        timer_epoch = utility.timer()
        self.test(0)
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            
            epoch_time = int(timer_epoch.toc())
            log = utility.remain_time(epoch_time, epoch, self.args.epochs)
            self.logger.info(log)
            timer_epoch.tic()
        self.remove_layer_hook()

    def train(self, epoch):
        lr = self.pruned_scheduler.get_lr()[0]

        self.pruned_model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lrs, hr, _) in enumerate(self.loader_train):
            if not self.args.cpu:
                lrs = [lr.cuda().detach() for lr in lrs]
                hr = hr.cuda().detach()
            timer_data.hold()
            timer_model.tic()

            self.cache_original_feature = {}
            self.cache_pruned_feature = {}
            
            pruned_sr = self.pruned_model(lrs[0])
            pruned_sr2lr = self.pruned_model.dual_forward(pruned_sr) if self.args.dual else None
            
            ## obtain the intermedia features of the original model
            with torch.no_grad():
                self.original_model(lrs[0])
            
            kd_loss, rec_loss, total_loss = \
                self.compute_loss(lrs, hr, pruned_sr, pruned_sr2lr)
            
            self.pruned_optimizer.zero_grad()
            total_loss.backward()
            self.pruned_optimizer.step()
            
            timer_model.hold()
            
            if (batch + 1) % self.args.print_every == 0:
                self.logger_counter += 1
                log = 'Epoch: {:0>4d} lr: {:.2e} '.format(epoch, Decimal(lr))
                log += "[{:0>5d}/{:d}]  ".format(
                                (batch + 1) * self.args.batch_size, 
                                len(self.loader_train.dataset))
                log += 'KD_loss:{:.4e}  Rec_loss:{:.4e}  Total_loss:{:.4e}  '\
                    .format(kd_loss, rec_loss, total_loss)
                log += '{:.1f}+{:.1f}s'.format(
                            timer_model.release(), timer_data.release())
                self.logger.info(log)
            
            timer_data.tic()        

        self.pruned_scheduler.step()
    
    def test(self, epoch):
        
        self.original_model.eval()
        self.pruned_model.eval()
        
        pruned_psnr = 0
        scale = max(self.scale)
        with torch.no_grad():
            for _, (lr, hr, filename) in enumerate(self.loader_test):
                filename = filename[0]
                if isinstance(lr, list): lr = lr[0]

                no_eval = (hr.nelement() == 1)
                if not no_eval and not self.args.cpu:
                    lr, hr = lr.cuda(), hr.cuda()
                elif not self.args.cpu:
                    lr = lr.cuda()

                sr_pruned = self.pruned_model(lr)
                if isinstance(sr_pruned, list):
                    sr_pruned = sr_pruned[-1]

                sr_pruned = utility.quantize(sr_pruned, self.args.rgb_range)

                if not no_eval:
                    pruned_psnr += utility.calc_psnr(
                        sr_pruned, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )

            pruned_psnr = pruned_psnr / len(self.loader_test)
            
            if pruned_psnr >= self.pruned_best_psnr:
                self.pruned_best_psnr = pruned_psnr
                self.pruned_best_epoch = epoch
            pruned_log = '[{} x{}]  '.format(self.args.data_test, scale)
            pruned_log += 'PSNR:  {:.3f} '.format(pruned_psnr)
            pruned_log += '(Best: {:.3f} @epoch {})'.format(
                            self.pruned_best_psnr, self.pruned_best_epoch)
            self.logger.info(pruned_log)
        
        self.checkpoint.save(self, epoch, self.args.save)

    def register_hooks(self):
        if self.args.n_GPUs > 1:
            original_model = self.original_model.model.module
            pruned_model = self.pruned_model.model.module
        else:
            original_model = self.original_model.model
            pruned_model = self.pruned_model.model
        
        self.original_hooks = [] 
        self.pruned_hooks = []

        for i in range(len(self.scale)):
            # kd_nums: the number of kd losses for each level
            # if kd_nums is zero, only one kd loss for the last block
            if self.args.kd_nums == 2:
                mid_idx = len(original_model.up_blocks[i]) // 2 - 2
                self.original_hooks.append(
                    original_model.up_blocks[i][mid_idx].\
                    register_forward_hook(self._hook_origin_feature))
            if self.args.kd_nums >= 1 or i == len(self.scale) - 1:
                self.original_hooks.append(
                    original_model.tail[i + 1].\
                    register_forward_hook(self._hook_origin_feature))
            
            if self.args.kd_nums == 2:
                mid_idx = len(pruned_model.up_blocks[i]) // 2 - 2
                self.pruned_hooks.append(
                    pruned_model.up_blocks[i][mid_idx].\
                    register_forward_hook(self._hook_pruned_feature))
            if self.args.kd_nums >= 1 or i == len(self.scale) - 1:
                self.pruned_hooks.append(
                    pruned_model.tail[i + 1].\
                    register_forward_hook(self._hook_pruned_feature))
        self.logger.info(
            'The total number of KD loss: {}'.format(len(self.pruned_hooks)))
    
    def _hook_origin_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        if gpu_id not in self.cache_original_feature.keys():
            self.cache_original_feature[gpu_id] = []
        if isinstance(module, nn.Conv2d): # the tail conv of drn
            self.cache_original_feature[gpu_id].append(
                            input[0].mean(dim=1, keepdim=True))
        else:
            self.cache_original_feature[gpu_id].append(
                            output.mean(dim=1, keepdim=True))

    def _hook_pruned_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        if gpu_id not in self.cache_pruned_feature.keys():
            self.cache_pruned_feature[gpu_id] = []
        if isinstance(module, nn.Conv2d): # the tail conv of drn
            self.cache_pruned_feature[gpu_id].append(
                            input[0].mean(dim=1, keepdim=True))
        else:
            self.cache_pruned_feature[gpu_id].append(
                            output.mean(dim=1, keepdim=True))
    
    def remove_layer_hook(self):
        for i in range(len(self.original_hooks)):
            self.original_hooks[i].remove()
            self.pruned_hooks[i].remove()
        self.cache_original_feature = {}
        self.cache_pruned_feature = {}
        self.logger.info("|===>remove hook")

    def compute_loss(self, lr, hr, sr, sr2lr=None):
        # compute features reconstruction loss
        kd_loss = 0
        
        # note that cache features have been averaged across the channel dimension
        # with the shape of (B, 1, H, W)
        original_features = concat_gpu_datalist(self.cache_original_feature)
        pruned_features = concat_gpu_datalist(self.cache_pruned_feature)
        assert pruned_features[0].size(1) == 1, \
            "did not average the feature across channels"
        for j in range(len(original_features)):
            kd_loss += self.criterion_mse(pruned_features[j],
                                    original_features[j].detach())

        # compute primal regression loss
        loss_primary = self.criterion_L1(sr[-1], hr)
        for i in range(1, len(sr)):
            loss_primary += self.criterion_L1(
                sr[i - 1 - len(sr)], lr[i - len(sr)])
        
        # compute dual regression loss
        if self.args.dual:
            loss_dual = self.criterion_L1(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.criterion_L1(
                    sr2lr[i - len(self.scale)], lr[i - len(self.scale)])
            
            # compute total loss
            rec_loss = loss_primary + self.args.dual_weight * loss_dual
        else:
            rec_loss = loss_primary
        total_loss = rec_loss + self.args.kd_weight * kd_loss
        return kd_loss, rec_loss, total_loss
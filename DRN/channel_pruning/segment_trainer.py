import math
import time

import torch.autograd
import torch.nn as nn

import utils as utils
import utility
from model.common import RCAB, ResBlock, DownBlock, Upsampler, MeanShift
from .segement import DRNSegment


class SegmentWiseTrainer(object):
    """
        Segment-wise trainer for channel selection
    """

    def __init__(self, opt, model, original_model, pruned_model, train_loader, val_loader, logger, tensorboard_logger, run_count=0):
        self.opt = opt
        self.model = model
        self.original_model = original_model
        self.pruned_model = pruned_model
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger

        self.lr = self.opt.lr
        self.scale = self.opt.scale
        self.original_modules = None
        self.pruned_modules = None
        self.run_count = run_count
        
        self.model_segment = DRNSegment(opt, original_model, pruned_model)
        
        # collect the layers to prune
        self.create_segments()
        # parallel setting for segments
        self.model_parallelism()

    def create_segments(self):
        net_origin = None
        net_pruned = None

        block_count = 0

        for ori_module, pruned_module in \
            zip(self.original_model.modules(), self.pruned_model.modules()):
            if isinstance(ori_module, (DownBlock, RCAB, ResBlock, nn.Conv2d)): # (RCAB)
                if isinstance(ori_module, nn.Conv2d):
                    # pruning the last conv2d in up_blocks
                    if self.opt.model.lower().find('drnt_res')>=0: 
                        continue
                    k_size = ori_module.kernel_size
                    in_chans = ori_module.in_channels
                    out_chans = ori_module.out_channels
                    if k_size != (1, 1): continue
                    if in_chans / out_chans not in [2., 4.]: continue
                    
                
                self.logger.info("enter block: {}".format(type(ori_module)))
                if net_origin is not None:
                    net_origin.add_module(str(len(net_origin)), ori_module)
                else:
                    net_origin = nn.Sequential(ori_module)

                if net_pruned is not None:
                    net_pruned.add_module(str(len(net_pruned)), pruned_module)
                else:
                    net_pruned = nn.Sequential(pruned_module)
                block_count += 1

        self.final_block_count = block_count
        self.original_modules = net_origin
        self.pruned_modules = net_pruned
        self.logger.info('block_count: {}'.format(block_count))
        self.logger.info(net_pruned)

    def model_parallelism(self):
        self.model_segment = utils.data_parallel(
            model=self.model_segment, n_gpus=self.opt.n_GPUs)

        # turn gradient off
        # avoid computing the gradient
        for params in self.original_model.parameters():
            params.requires_grad = False
        for params in self.pruned_model.parameters():
            params.requires_grad = False
        if self.opt.dual:
            for i in range(len(self.scale)):
                for params in self.model.dual_models[i].parameters():
                    params.requires_grad = False
                self.model.dual_models[i].eval()
        
        # freeze the Batch Normalization
        self.original_model.eval()
        self.pruned_model.eval()


    @staticmethod
    def _correct_nan(grad):
        """
        Fix nan
        :param grad: gradient input
        """

        grad.masked_fill_(grad.ne(grad), 0)
        return grad

    def val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """
        
        psnr = utils.AverageMeter()

        self.pruned_model.eval()
        
        iters = len(self.val_loader)

        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (lr, hr, filename) in enumerate(self.val_loader):
                start_time = time.time()
                data_time = start_time - end_time
                if isinstance(lr, list): lr = lr[0]
                if self.opt.n_GPUs >= 1:
                    lr = lr.cuda()
                hr = hr.cuda()

                sr = self.pruned_model(lr)
                if isinstance(sr, list): sr = sr[-1]
                sr = utility.quantize(sr, self.opt.rgb_range)

                # compute psnr
                single_psnr = utility.calc_psnr(sr, hr, max(self.scale), 
                                self.opt.rgb_range, benchmark=True)
                psnr.update(single_psnr, hr.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

                utils.print_result(epoch, 1, i + 1, iters, self.lr, 
                                    data_time, iter_time, psnr.avg,
                                    mode="Validation",
                                    logger=self.logger)

        if self.logger is not None:
            self.tensorboard_logger.scalar_summary(
                "segment_wise_val_psnr", psnr.avg, self.run_count)
        self.run_count += 1

        self.logger.info("|===>Validation PSNR: {:4f} ".format(psnr.avg))
        return psnr.avg
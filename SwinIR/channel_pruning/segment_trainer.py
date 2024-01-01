import math
import time

import torch.autograd
import torch.nn as nn

from utils import pruned_utils
# from model.common import RCAB, ResBlock, DownBlock, Upsampler, MeanShift
from .segement import SwinIRSegment
from utils import utils_image as util
from models.network_swinir_pruned import Mlp, WindowAttention


class SegmentWiseTrainer(object):
    """
        Segment-wise trainer for channel selection
    """

    def __init__(self, args, original_model, pruned_model, train_loader, val_loader, logger, tensorboard_logger, run_count=0):
        self.args = args
        self.original_model = original_model
        self.pruned_model = pruned_model
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger

        self.learning_rate = args.opt['train']['G_optimizer_lr']
        self.scale = args.scale
        self.original_modules = None
        self.pruned_modules = None
        self.run_count = run_count
        
        self.model_segment = SwinIRSegment(args, original_model, pruned_model)
        
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
            if isinstance(ori_module, (Mlp, WindowAttention)):                  
                
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
        self.model_segment = pruned_utils.data_parallel(
            model=self.model_segment, n_gpus=self.args.opt['gpu_ids'])

        # turn gradient off
        # avoid computing the gradient
        for params in self.original_model.parameters():
            params.requires_grad = False
        for params in self.pruned_model.parameters():
            params.requires_grad = False
        
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

    def val(self, epoch=0):
        """
        Validation
        :param epoch: index of epoch
        """
        
        psnr = pruned_utils.AverageMeter()

        self.pruned_model.eval()
        
        iters = len(self.val_loader)

        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, test_data in enumerate(self.val_loader):
                start_time = time.time()
                data_time = start_time - end_time
                
                if len(self.args.opt['gpu_ids']) >= 1:
                    lr = test_data['L'].cuda()
                    hr = test_data['H'].cuda()

                sr, _ = self.pruned_model(lr)
                # round to uint8
                sr = util.tensor2uint(sr) 
                hr = util.tensor2uint(hr)

                # compute psnr
                single_psnr = pruned_utils.calc_psnr(sr, hr, self.scale, benchmark=True)
                psnr.update(single_psnr, 1)

                end_time = time.time()
                iter_time = end_time - start_time

                pruned_utils.print_result(epoch, 1, i + 1, iters, self.learning_rate, 
                                    data_time, iter_time, psnr.avg,
                                    mode="Validation",
                                    logger=self.logger)

        if self.logger is not None:
            self.tensorboard_logger.scalar_summary(
                "segment_wise_val_psnr", psnr.avg, self.run_count)
        self.run_count += 1

        self.logger.info("|===>Validation PSNR: {:4f} ".format(psnr.avg))
        return psnr.avg
import datetime
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn

import utils as utils
from model.common import MaskConv2d
from utils import write_log, concat_gpu_data
from .thinet_prune import ThinetFilterPrune
from .lasso_prune import LassoFilterPrune



class LayerChannelSelection(object):
    """
    Dual Discrimination-aware channel selection
    """

    def __init__(self, opt, model, trainer, train_loader, val_loader, checkpoint, logger, tensorboard_logger):
        self.model = model
        self.segment_wise_trainer = trainer
        self.pruned_model = trainer.pruned_model
        self.pruned_modules = trainer.pruned_modules
        self.model_segment = trainer.model_segment
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.scale = opt.scale
        self.checkpoint = checkpoint
        
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger

        self.cache_original_output = {}
        self.cache_pruned_input = {}
        self.cache_pruned_output = {}

        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_L1 = nn.L1Loss().cuda()

        self.logger_counter = 0

        self.record_time = utils.AverageMeter()
        self.record_selection_loss = utils.AverageMeter()
        self.record_sub_problem_loss = utils.AverageMeter()
        self.channel_configs = None
        if opt.configs_path is not None:
            self.channel_configs = np.loadtxt(opt.configs_path, dtype=int)

    def split_segments(self, block_count):
        """
        Split the segment into three parts:
            segment_before_pruned_module, pruned_module, segment_after_pruned_module.
        In this way, we can store the input of the pruned module.
        """
        if isinstance(self.model_segment, nn.DataParallel):
            self.model_segment.module.split_segment(block_count)
        else:
            self.model_segment.split_segment(block_count)


    def replace_layer_with_mask_conv(self, module, layer_name, block_count):
        """
        Replace the pruned layer with mask convolution
        """

        if layer_name == "body2":
            layer = module.body[2]
        elif layer_name == "module1":
            layer = module.dual_module[1]
        elif layer_name == "conv1":
            layer = module
        else:
            assert False, "unsupport layer: {}".format(layer_name)

        if not isinstance(layer, MaskConv2d):
            temp_conv = MaskConv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=(layer.bias is not None))
            temp_conv.weight.data.copy_(layer.weight.data)

            if layer.bias is not None:
                temp_conv.bias.data.copy_(layer.bias.data)

            if self.opt.prune_type.lower() != 'dcp': 
                temp_conv.init_beta()
                temp_conv.beta.data.fill_(1)
                temp_conv.d.fill_(1)
                temp_conv.pruned_weight.data.copy_(layer.weight.data)
            else:
                temp_conv.d.fill_(0)
                temp_conv.pruned_weight.data.fill_(0)
            
            device = layer.weight.device
            temp_conv = temp_conv.to(device)
            
            if layer_name == "body2":
                module.body[2] = temp_conv
            elif layer_name == "module1":
                module.dual_module[1] = temp_conv
            elif layer_name == "conv1":
                for i in range(len(self.scale)):
                    conv1 = self.pruned_model.up_blocks[i][-1]
                    if conv1.out_channels == temp_conv.out_channels:
                        self.pruned_model.up_blocks[i][-1] = temp_conv
                        self.pruned_modules[block_count - 1] = temp_conv
                        module = temp_conv
            
            layer = temp_conv
        return layer, module

    def register_layer_hook(self, origin_module, pruned_module, layer_name):
        """
        In order to get the input and the output of the intermediate layer, we register
        the forward hook for the pruned layer
        """

        if layer_name == "body2":
            self.hook_origin = origin_module.body[2].register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_module.body[2].register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "module1":
            self.hook_origin = origin_module.dual_module[1].register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_module.dual_module[1].register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "conv1":
            self.hook_origin = origin_module.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_module.register_forward_hook(self._hook_pruned_feature)
    
    def _hook_origin_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        if self.opt.prune_type != 'thinet':
            self.cache_original_output[gpu_id] = output

    def _hook_pruned_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        if self.opt.prune_type != 'dcp':
            self.cache_pruned_input[gpu_id] = input[0]
        if self.opt.prune_type != 'cp':
            self.cache_pruned_output[gpu_id] = output
    
    def remove_layer_hook(self):
        self.hook_origin.remove()
        self.hook_pruned.remove()
        self.cache_original_output = {}
        self.cache_pruned_input = {}
        self.cache_pruned_output = {}
        self.logger.info("|===>remove hook")

    def reset_average_meter(self):
        self.record_time.reset()
        self.record_selection_loss.reset()
        self.record_sub_problem_loss.reset()

    def prepare_channel_selection(self, origin_module, 
                            pruned_module, layer_name, block_count):
        """
        Prepare for channel selection
        1. Split the segment into three parts.
        2. Replace the pruned layer with mask convolution.
        3. Store the input feature map of the pruned layer in advance to accelerate channel selection.
        """

        self.split_segments(block_count)
        if layer_name == 'conv1': print(type(pruned_module))
        layer, pruned_module = self.replace_layer_with_mask_conv(
                    pruned_module, layer_name, block_count)
        if layer_name == 'conv1': print(type(pruned_module))
        self.register_layer_hook(origin_module, pruned_module, layer_name)

        self.num_batch = len(self.train_loader)

        # turn on the gradient with respect to the pruned layer
        layer.pruned_weight.requires_grad = True

        self.logger_counter = 0
        return layer

    def get_batch_data(self, train_dataloader_iter):
        lrs, hr, _ = train_dataloader_iter.next()
        lrs = [lr.cuda() for lr in lrs]
        hr = hr.cuda()
        return lrs, hr
    
    def prepare_data(self, batch):
        lrs, hr, _ = batch
        lrs = [lr.cuda() for lr in lrs]
        hr = hr.cuda()
        return lrs, hr

    def find_maximum_grad_fnorm(self, grad_fnorm, layer):
        """
        Find the channel index with maximum gradient frobenius norm,
        and initialize the pruned_weight w.r.t. the selected channel.
        """

        grad_fnorm.data.mul_(1 - layer.d).sub_(layer.d)
        _, max_index = torch.topk(grad_fnorm, 1)
        layer.d[max_index] = 1
        # warm-started from the pre-trained model
        if self.opt.warm_start:
            layer.pruned_weight.data[:, max_index, :, :] = layer.weight[:, max_index, :, :].data.clone()

    def find_most_violated(self, layer):
        """
        Find the channel with maximum gradient frobenius norm.
        :param layer: the layer to be pruned
        """

        layer.pruned_weight.grad = None
        train_dataloader_iter = iter(self.train_loader)

        for j, batch in enumerate(self.train_loader):
            # get data
            # lr, hr = self.get_batch_data(train_dataloader_iter)
            lr, hr = self.prepare_data(batch)
            
            pruned_output = self.model_segment(lr[0], final_output=True)
            dual_outputs = self.model.dual_forward(pruned_output) if self.opt.dual else None
            
            loss = self.compute_loss_error(lr, hr, pruned_output, dual_outputs)

            loss.backward()

            self.record_selection_loss.update(loss.item(), hr.size(0))

        cum_grad = layer.pruned_weight.grad.data.clone()
        layer.pruned_weight.grad = None

        # calculate F norm of gradient
        grad_fnorm = cum_grad.mul(cum_grad).sum((2, 3)).sqrt().sum(0)

        # find grad_fnorm with maximum absolute gradient
        self.find_maximum_grad_fnorm(grad_fnorm, layer)

    def set_layer_wise_optimizer(self, layer):
        params_list = []
        params_list.append({"params": layer.pruned_weight, "lr": self.opt.layer_wise_lr})
        if layer.bias is not None:
            layer.bias.requires_grad = True
            params_list.append({"params": layer.bias, "lr": self.opt.layer_wise_lr})

        optimizer = torch.optim.SGD(params=params_list,
                                    weight_decay=self.opt.weight_decay,
                                    momentum=self.opt.momentum,
                                    nesterov=True)

        return optimizer
    
    def compute_loss_error(self, lr, hr, sr, sr2lr=None):
        # compute features reconstruction loss
        loss = 0
        if self.opt.prune_type != 'thinet':
            for i in range(len(self.cache_original_output)):
                device = self.cache_original_output[str(i)].get_device()
                criterion_mse = self.criterion_mse.to(device)
                loss += criterion_mse(
                            self.cache_original_output[str(i)],
                            self.cache_pruned_output[str(i)])

        # compute primal regression loss
        loss_primary = self.criterion_L1(sr[-1], hr)
        for i in range(1, len(sr)):
            loss_primary += self.criterion_L1(
                sr[i - 1 - len(sr)], lr[i - len(sr)])
        
        # compute dual regression loss
        if self.opt.dual:
            loss_dual = self.criterion_L1(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.criterion_L1(
                    sr2lr[i - len(self.scale)], lr[i - len(self.scale)])
            
            # compute total loss
            loss += loss_primary + self.opt.dual_weight * loss_dual
        else:
            loss += loss_primary
        return loss

    def finetune_params(self, layer, epoch=1, original_forward=True):
        """
        We optimize W w.r.t. the selected channels by minimizing the loss
        :param layer: the layer to be pruned
        """

        optimizer = self.set_layer_wise_optimizer(layer)
        train_dataloader_iter = iter(self.train_loader)

        for e in range(epoch):
            for j, batch in enumerate(self.train_loader):
                # get data
                lr, hr = self.prepare_data(batch)
                pruned_output = self.model_segment(lr[0], 
                    final_output=True, original_forward=original_forward)
                dual_outputs = self.model.dual_forward(pruned_output) if self.opt.dual else None
                
                loss = self.compute_loss_error(lr, hr, pruned_output, dual_outputs)
                optimizer.zero_grad()
                # compute gradient
                loss.backward()
                # we only optimize W with respect to the selected channel
                if self.opt.prune_type == 'thinet':
                    if isinstance(layer, MaskConv2d):
                        layer.beta.grad.data.mul_(layer.d)
                        
                layer.pruned_weight.grad.data.mul_(
                    layer.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))
                optimizer.step()

            # update record info
            self.record_sub_problem_loss.update(loss.item(), hr.size(0))

        layer.pruned_weight.grad = None
        if layer.bias is not None:
            layer.bias.grad = None
        if layer.bias is not None:
            layer.bias.requires_grad = False

    def write_log(self, layer, block_count, layer_name):
        self.write_tensorboard_log(block_count, layer_name)
        self.write_log2file(layer, block_count, layer_name)

    def write_tensorboard_log(self, block_count, layer_name):
        self.tensorboard_logger.scalar_summary(
            tag="Selection-block-{}_{}_Loss".format(block_count, layer_name),
            value=self.record_selection_loss.avg,
            step=self.logger_counter)

        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_Loss".format(block_count, layer_name),
            value=self.record_sub_problem_loss.avg,
            step=self.logger_counter)
        
        self.logger_counter += 1

    def write_log2file(self, layer, block_count, layer_name):
        write_log(
            dir_name=os.path.join(self.opt.save, "log"),
            file_name="log_block-{:0>2d}_{}.txt".format(block_count, layer_name),
            log_str="{:d}\t{:f}\t{:f}\t\n".format(
                int(layer.d.sum()),
                self.record_selection_loss.avg,
                self.record_sub_problem_loss.avg))
        log_str = "Block-{:0>2d}-{}  #channels: [{:0>4d}|{:0>4d}]  ".format(
            block_count, layer_name,
            int(layer.d.sum()), layer.d.size(0))
        log_str += "[selection]loss: {:4f}".format(self.record_selection_loss.avg)
        log_str += "[subproblem]loss: {:4f}".format(self.record_sub_problem_loss.avg)
        self.logger.info(log_str)

    def channel_selection_for_one_layer(self, origin_module, 
                pruned_module, block_count, layer_name="body2"):
        """
        Conduct channel selection for one layer in a module
        :param origin_module: original module that corresponding to pruned_module
        :param pruned_module: the module need to be pruned
        :param block_count: current block index
        :param layer_name: the name of layer need to be pruned
        """

        # layer-wise channel selection
        self.logger.info("|===>layer-wise channel selection: block-{}-{}".format(block_count, layer_name))
        layer = self.prepare_channel_selection(origin_module, 
                                    pruned_module, layer_name, block_count)

        if self.opt.prune_type.lower() == 'dcp':
            # find the channel with the maximum gradient norm
            self.dcp_selection(layer, block_count, layer_name)
        elif self.opt.prune_type.lower() == 'cp':
            self.lasso_selection(layer, block_count, layer_name)
        elif self.opt.prune_type.lower() == 'thinet':
            self.thinet_selection(layer, block_count, layer_name)
        else:
            assert False, "unsupport prune type: {}".format(self.opt.prune_type)
        
        self.tensorboard_logger.scalar_summary(
            tag="Channel_num",
            value=layer.d.eq(1).sum(),
            step=block_count)
        log_str = "|===>Select channel from block-{:d}_{}: time_total:{} time_avg: {}".format(
            block_count, layer_name,
            str(datetime.timedelta(seconds=self.record_time.sum)),
            str(datetime.timedelta(seconds=self.record_time.avg)))
        self.logger.info(log_str)
        
        self.remove_layer_hook()
        # test pruned model
        self.segment_wise_trainer.val(0)
        return pruned_module
        
    def dcp_selection(self, layer, block_count, layer_name):
        pruning_num = math.floor(layer.in_channels * self.opt.pruning_rate)
        if self.channel_configs is not None:
            pruning_num = layer.in_channels - self.channel_configs[block_count - 1]
            assert block_count - 1 >= 0, "The range of block_count should start from 1."
        for channel in range(layer.in_channels):
            if layer.d.eq(0).sum() <= pruning_num:
                break

            self.reset_average_meter()

            time_start = time.time()
            # find the channel with the maximum gradient norm
            self.find_most_violated(layer)
            # finetune parameters of the selected channels
            self.finetune_params(layer)
            time_interval = time.time() - time_start

            self.write_log(layer, block_count, layer_name)
            self.record_time.update(time_interval)
        
    def lasso_selection(self, layer, block_count, layer_name):
        filter_prune = LassoFilterPrune(self.opt, prune_ratio=self.opt.pruning_rate)
        time_start = time.time()
        for i, batch in enumerate(self.train_loader):
            # get data
            lr, _ = self.prepare_data(batch)
            self.model_segment(lr[0], final_output=False)
            input_fea = concat_gpu_data(self.cache_pruned_input)
            output_fea = concat_gpu_data(self.cache_original_output)
            filter_prune.feature_extract(
                input_fea, output_fea, layer)
            self.cache_pruned_input = {}
            self.cache_original_output = {}

        beta, d = filter_prune.channel_select(layer, self.opt.lasso_alpha)
        layer.beta.data.copy_(beta)
        layer.d.copy_(d)

        time_interval = time.time() - time_start
        self.write_log(layer, block_count, layer_name)
        self.record_time.update(time_interval)

    def thinet_selection(self, layer, block_count, layer_name):
        filter_prune = ThinetFilterPrune(prune_ratio=self.opt.pruning_rate)
        time_start = time.time()
        for i, batch in enumerate(self.train_loader):
            # get data
            lr, hr = self.prepare_data(batch)
            self.model_segment(lr[0], final_output=False, original_forward=False)
            input_fea = concat_gpu_data(self.cache_pruned_input)
            output_fea = concat_gpu_data(self.cache_pruned_output)
            filter_prune.feature_extract(
                input_fea, output_fea, layer)
            self.cache_pruned_input = {}
            self.cache_pruned_output = {}
        
        beta, d = filter_prune.channel_select(layer)
        layer.beta.data.copy_(beta)
        layer.d.copy_(d)
        # set the unselected params to zeros
        layer.pruned_weight.data.mul_(
            layer.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))

        # finetune parameters of all selected channels
        self.finetune_params(layer, epoch=self.opt.finetune_epochs, original_forward=False)
        
        time_interval = time.time() - time_start
        self.write_log(layer, block_count, layer_name)
        self.record_time.update(time_interval)
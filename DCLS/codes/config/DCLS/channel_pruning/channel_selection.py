import datetime
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn

from .common_module import MaskConv2d, MaskModuleList
from models.modules.dcls_arch import DPCAB, Estimator
from models.modules.module_util import ResidualBlock_noBN
from models.modules.loss import CorrectionLoss
from utils.pruned_utils import write_log, AverageMeter


class LayerChannelSelection(object):
    """
    Dual Discrimination-aware channel selection
    """

    def __init__(self, args, model, trainer, train_loader, val_loader, prepro, checkpoint, logger, tensorboard_logger):
        self.model = model
        self.segment_wise_trainer = trainer
        self.pruned_model = trainer.pruned_model
        self.pruned_modules = trainer.pruned_modules
        self.model_segment = trainer.model_segment
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.prepro = prepro

        self.args = args
        self.scale = args.scale
        self.checkpoint = checkpoint
        
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger

        self.cache_original_output = {}
        self.cache_pruned_input = {}
        self.cache_pruned_output = {}

        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_L1 = nn.L1Loss().cuda()
        # reformulate kernel and compute L1 loss
        self.cri_kernel = CorrectionLoss(scale=self.scale, eps=1e-20).cuda()

        self.logger_counter = 0

        self.record_time = AverageMeter()
        self.record_selection_loss = AverageMeter()
        self.record_sub_problem_loss = AverageMeter()
        self.channel_configs = None
        if args.opt['pruning']['configs_path'] is not None:
            self.channel_configs = np.loadtxt(args.opt['pruning']['configs_path'], dtype=int)

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
        Replace the pruned layer with mask conv2d
        """

        if layer_name == "body1_2":
            layer = module.body1[2]
        elif layer_name == "conv2":
            layer = module.conv2
        elif layer_name == "module_list":
            layer = module.dec
        else:
            assert False, "unsupport layer: {}".format(layer_name)

        if not isinstance(layer, (MaskConv2d, MaskModuleList)):
            
            if layer_name == "module_list":
                # prune the model list module
                temp_layer = MaskModuleList(layer[:])
                
                temp_layer.init_pruning_params()
                
                # put into the same device
                device = layer[0].weight.device
                temp_layer = temp_layer.to(device)

            else:
                temp_layer = MaskConv2d(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                            bias=(layer.bias is not None))
                temp_layer.weight.data.copy_(layer.weight.data)

                if layer.bias is not None:
                    temp_layer.bias.data.copy_(layer.bias.data)

                if self.args.opt['pruning']['prune_type'].lower() != 'dcp': 
                    temp_layer.init_beta()
                    temp_layer.beta.data.fill_(1)
                    temp_layer.d.fill_(1)
                    temp_layer.pruned_weight.data.copy_(layer.weight.data)
                else:
                    temp_layer.d.fill_(0)
                    temp_layer.pruned_weight.data.fill_(0)
                
                device = layer.weight.device
                temp_layer = temp_layer.to(device)

            if layer_name == "body1_2":
                module.body1[2] = temp_layer
            elif layer_name == "conv2":
                module.conv2 = temp_layer
            elif layer_name == "module_list":
                module.dec = temp_layer
            
            layer = temp_layer
        return layer, module

    def register_layer_hook(self, origin_module, pruned_module, layer_name):
        """
        In order to get the input and the output of the intermediate layer, we register
        the forward hook for the pruned layer
        """

        if layer_name == "body1_2":
            self.hook_origin = origin_module.body1[2].register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_module.body1[2].register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "conv2":
            self.hook_origin = origin_module.conv2.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_module.conv2.register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "module_list":
            self.hook_origin = origin_module.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_module.register_forward_hook(self._hook_pruned_feature)
    
    def _hook_origin_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        if self.args.opt['pruning']['prune_type'] != 'thinet':
            self.cache_original_output[gpu_id] = output

    def _hook_pruned_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        if self.args.opt['pruning']['prune_type'] != 'dcp':
            self.cache_pruned_input[gpu_id] = input[0]
        if self.args.opt['pruning']['prune_type'] != 'cp':
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
        print(f'replace layer in {type(pruned_module)} module')
        layer, pruned_module = self.replace_layer_with_mask_conv(
                    pruned_module, layer_name, block_count)
        print(f'replace layer to {type(layer)}')
        self.register_layer_hook(origin_module, pruned_module, layer_name)

        self.num_batch = len(self.train_loader)

        # turn on the gradient with respect to the pruned layer
        if isinstance(layer, MaskModuleList):
            layer.set_train_pruning_params()
        else:
            layer.pruned_weight.requires_grad = True

        self.logger_counter = 0
        return layer, pruned_module
    
    def prepare_data(self, batch):
        hr = batch["GT"]
        
        lr, ker_map, kernels, lr_blured_t, lr_t = \
            self.prepro(hr, True, return_blur=True)
        lr = (lr * 255).round() / 255
        
        lr, hr = lr.cuda(), hr.cuda()
        lr_blured_t, lr_t = lr_blured_t.cuda(), lr_t.cuda()

        return lr, hr, ker_map, kernels, lr_blured_t, lr_t

    def find_maximum_grad_fnorm(self, grad_fnorm, pruned_module, layer, layer_name):
        """
        Find the channel index with maximum gradient frobenius norm,
        and initialize the pruned_weight w.r.t. the selected channel.
        """
        grad_fnorm.data.mul_(1 - layer.d).sub_(layer.d)
        _, max_index = torch.topk(grad_fnorm, 1)

        if isinstance(pruned_module, Estimator):
            layer.select_channels(max_index)
            layer.warm_start(max_index)
        else:
            layer.d[max_index] = 1
            # warm-started from the pre-trained model
            layer.pruned_weight.data[:, max_index, ...] = \
                layer.weight[:, max_index, ...].data.clone()
        
        return max_index

    def find_most_violated(self, pruned_module, layer, layer_name):
        """
        Find the channel with maximum gradient frobenius norm.
        :param layer: the layer to be pruned
        dim: the dimention of features to be pruned
        """
        if isinstance(layer, MaskModuleList):
            for m in layer:
                m.pruned_weight.grad = None
        else:
            layer.pruned_weight.grad = None

        for j, batch in enumerate(self.train_loader):
            # get data
            lr, hr, _, _, lr_blured_t, lr_t = self.prepare_data(batch)
            
            sr, est_kernel, sr2lr = self.model_segment(lr, final_output=True)
            
            loss = self.compute_loss_error(lr, hr, lr_blured_t, lr_t, sr, est_kernel, sr2lr)

            loss.backward()

            # clip grad for stability
            torch.nn.utils.clip_grad_norm_(self.pruned_model.parameters(), .1)
            torch.nn.utils.clip_grad_value_(self.pruned_model.parameters(), .1)

            self.record_selection_loss.update(loss.item(), hr.size(0))
        
        if isinstance(layer, MaskModuleList):
            cum_grad = layer[-1].pruned_weight.grad.data.clone()
            for m in layer:
                m.pruned_weight.grad = None
            
            grad_fnorm = cum_grad.abs().sum(0)
        else:
            cum_grad = layer.pruned_weight.grad.data.clone()
            layer.pruned_weight.grad = None

            grad_fnorm = cum_grad.mul(cum_grad).sum((2, 3)).sqrt().sum(0)

        # find grad_fnorm with maximum absolute gradient
        self.find_maximum_grad_fnorm(
            grad_fnorm, pruned_module, layer, layer_name)


    def set_layer_wise_optimizer(self, layer):
        params_list = []
        if isinstance(layer, MaskModuleList):
            for m in layer:
                params_list.append({"params": m.pruned_weight,
                                   "lr": self.args.opt['pruning']['layer_wise_lr']})
                if m.bias is not None:
                    m.bias.requires_grad = True
                    params_list.append({"params": m.bias, 
                                    "lr": self.args.opt['pruning']['layer_wise_lr']})
        else: 
            params_list.append({"params": layer.pruned_weight, 
                            "lr": self.args.opt['pruning']['layer_wise_lr']})
        
            if layer.bias is not None:
                layer.bias.requires_grad = True
                params_list.append({"params": layer.bias, 
                                "lr": self.args.opt['pruning']['layer_wise_lr']})

        optimizer = torch.optim.SGD(params=params_list,
                                    weight_decay=self.args.opt['pruning']['weight_decay'],
                                    momentum=self.args.opt['pruning']['momentum'],
                                    nesterov=True)

        return optimizer
    
    def compute_loss_error(self, lr, hr, lr_blured, lr_t, sr, est_kernel, sr2lr=None):
        # compute features reconstruction loss
        loss = torch.zeros(1).cuda()
        if self.args.opt['pruning']['prune_type'] != 'thinet':
            for i in range(len(self.cache_original_output)):
                device = self.cache_original_output[str(i)].get_device()
                criterion_mse = self.criterion_mse.to(device)
                loss_mse = criterion_mse(
                            self.cache_original_output[str(i)],
                            self.cache_pruned_output[str(i)])
                loss += loss_mse.to(loss.device)

        # compute primal regression loss
        loss_primary = self.criterion_L1(sr, hr)

        # compute dual regression loss
        if sr2lr is not None:
            loss_dual = self.criterion_L1(sr2lr, lr)
            
            # compute total loss
            loss += loss_primary + self.args.opt['dual_weight'] * loss_dual
        else:
            loss += loss_primary
        
        loss_k, _ = self.cri_kernel(est_kernel, lr_blured, lr_t)

        loss += loss_k
        
        return loss

    def finetune_params(self, pruned_module, layer, epoch=1, original_forward=True):
        """
        We optimize W w.r.t. the selected channels by minimizing the loss
        :param layer: the layer to be pruned
        """

        optimizer = self.set_layer_wise_optimizer(layer)

        for e in range(epoch):
            for j, batch in enumerate(self.train_loader):
                # get data
                lr, hr, _, _, lr_blured_t, lr_t = self.prepare_data(batch)
                sr, est_kernel, sr2lr = self.model_segment(lr, 
                    final_output=True, original_forward=original_forward)
                
                loss = self.compute_loss_error(lr, hr, lr_blured_t, lr_t, sr, est_kernel, sr2lr)
                optimizer.zero_grad()
                # compute gradient
                loss.backward()
                
                # clip grad for stability
                torch.nn.utils.clip_grad_norm_(self.pruned_model.parameters(), .1)
                torch.nn.utils.clip_grad_value_(self.pruned_model.parameters(), .1)
                
                # we only optimize W with respect to the selected channel
                if self.args.opt['pruning']['prune_type'] == 'thinet':
                    if isinstance(layer, MaskConv2d):
                        layer.beta.grad.data.mul_(layer.d)
                
                # ignore finetune the pruned weight
                if isinstance(layer, MaskModuleList):
                    for m in layer:
                        m.pruned_weight.grad.data.mul_(
                        m.d.unsqueeze(0).expand_as(m.pruned_weight))
                else:
                    layer.pruned_weight.grad.data.mul_(
                    layer.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))
                
                optimizer.step()

            # update record info
            self.record_sub_problem_loss.update(loss.item(), hr.size(0))

        if isinstance(layer, MaskModuleList):
            for m in layer:
                m.pruned_weight.grad = None
                if m.bias is not None:
                    m.bias.grad = None
                if m.bias is not None:
                    m.bias.requires_grad = False
        else:
            layer.pruned_weight.grad = None
            if layer.bias is not None:
                layer.bias.grad = None
            if layer.bias is not None:
                layer.bias.requires_grad = False

    def write_log(self, layer, block_count, layer_name):
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
            dir_name=os.path.join(self.args.save_dir, "log"),
            file_name="log_block-{:0>2d}_{}.txt".format(block_count, layer_name),
            log_str="{:d}\t{:f}\t{:f}\t\n".format(
                int(layer.d.sum()),
                self.record_selection_loss.avg,
                self.record_sub_problem_loss.avg))
        log_str = "Block-{:0>2d}-{}  #channels: [{:0>4d}|{:0>4d}]  ".format(
            block_count, layer_name,
            int(layer.d.sum()), layer.d.size(0))
        log_str += "[selection] loss: {:4f} ".format(self.record_selection_loss.avg)
        log_str += "[subproblem] loss: {:4f}".format(self.record_sub_problem_loss.avg)
        self.logger.info(log_str)

    def channel_selection_for_one_layer(self, origin_module, 
                pruned_module, block_count, layer_name):
        """
        Conduct channel selection for one layer in a module
        :param origin_module: original module that corresponding to pruned_module
        :param pruned_module: the module need to be pruned
        :param block_count: current block index
        :param layer_name: the name of layer need to be pruned
        """

        # layer-wise channel selection
        self.logger.info("|===>layer-wise channel selection: block-{}-{}".format(block_count, layer_name))
        layer, pruned_module = self.prepare_channel_selection(origin_module, 
                                    pruned_module, layer_name, block_count)

        if self.args.opt['pruning']['prune_type'].lower() == 'dcp':
            # find the channel with the maximum gradient norm
            self.dcp_selection(pruned_module, layer, block_count, layer_name)
        else:
            assert False, "unsupport prune type: {}".format(self.args.opt['pruning']['prune_type'])
        
        log_str = "|===>Select channel from block-{:d}_{}: time_total:{} time_avg: {}".format(
            block_count, layer_name,
            str(datetime.timedelta(seconds=self.record_time.sum)),
            str(datetime.timedelta(seconds=self.record_time.avg)))
        self.logger.info(log_str)
        
        self.remove_layer_hook()
        # test pruned model
        self.segment_wise_trainer.val(0)
        return pruned_module
        
    def dcp_selection(self, pruned_module, layer, block_count, layer_name):
        # search the multihead channels simultaneously
        if isinstance(layer, MaskModuleList):
            num_channels = layer[-1].in_features
        else:
            num_channels = layer.in_channels
        pruning_num = math.floor(num_channels * self.args.opt['pruning']['pruning_rate'])
        
        if self.channel_configs is not None:
            pruning_num = num_channels - self.channel_configs[block_count - 1]
            assert block_count - 1 >= 0, "The range of block_count should start from 1."
        
        for _ in range(num_channels):
            if layer.d.eq(0).sum() <= pruning_num:
                break

            self.reset_average_meter()

            time_start = time.time()
            # find the channel with the maximum gradient norm
            self.find_most_violated(pruned_module, layer, layer_name)
            # finetune parameters of the selected channels
            self.finetune_params(pruned_module, layer)
            time_interval = time.time() - time_start

            self.write_log(layer, block_count, layer_name)
            self.record_time.update(time_interval)
        
import datetime
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn

from .common_module import MaskLinear, MaskOutLinear
from models.network_swinir_pruned import WindowAttention
from utils.pruned_utils import write_log, AverageMeter
# from .thinet_prune import ThinetFilterPrune
# from .lasso_prune import LassoFilterPrune


class LayerChannelSelection(object):
    """
    Dual Discrimination-aware channel selection
    """

    def __init__(self, args, model, trainer, train_loader, val_loader, checkpoint, logger, tensorboard_logger):
        self.model = model
        self.segment_wise_trainer = trainer
        self.pruned_model = trainer.pruned_model
        self.pruned_modules = trainer.pruned_modules
        self.model_segment = trainer.model_segment
        self.train_loader = train_loader
        self.val_loader = val_loader
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


    def replace_layer_with_mask_linear(self, module, layer_name, block_count):
        """
        Replace the pruned layer with mask linear
        """

        if layer_name == "proj":
            layer = module.proj
            layer_qkv = module.qkv
        elif layer_name == "fc2":
            layer = module.fc2
        else:
            assert False, "unsupport layer: {}".format(layer_name)

        if not isinstance(layer, (MaskLinear)):
            temp_layer = MaskLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
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

            if isinstance(module, WindowAttention):
                # replace qkv into MaskOutLinear
                temp_qkv = MaskOutLinear(
                    in_features=layer_qkv.in_features,
                    out_features=layer_qkv.out_features,
                    bias=(layer_qkv.bias is not None))
                temp_qkv.weight.data.copy_(layer_qkv.weight.data)

                if layer_qkv.bias is not None:
                    temp_qkv.bias.data.copy_(layer_qkv.bias.data)
                
                if self.args.opt['pruning']['prune_type'].lower() != 'dcp': 
                    temp_qkv.init_beta()
                    temp_qkv.beta.data.fill_(1)
                    temp_qkv.d.fill_(1)
                    temp_qkv.pruned_weight.data.copy_(layer.weight.data)
                else:
                    temp_qkv.d.fill_(0)
                    temp_qkv.pruned_weight.data.fill_(0)

                temp_qkv = temp_qkv.to(device)
            
            if layer_name == "proj":
                module.proj = temp_layer
                module.qkv = temp_qkv
            elif layer_name == "fc2":
                module.fc2 = temp_layer
            
            layer = temp_layer
        return layer, module

    def register_layer_hook(self, origin_module, pruned_module, layer_name):
        """
        In order to get the input and the output of the intermediate layer, we register
        the forward hook for the pruned layer
        """

        if layer_name == "proj":
            self.hook_origin = origin_module.proj.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_module.proj.register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "fc2":
            self.hook_origin = origin_module.fc2.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_module.fc2.register_forward_hook(self._hook_pruned_feature)
    
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
        layer, pruned_module = self.replace_layer_with_mask_linear(
                    pruned_module, layer_name, block_count)
        print(f'replace layer to {type(layer)}')
        self.register_layer_hook(origin_module, pruned_module, layer_name)

        self.num_batch = len(self.train_loader)

        # turn on the gradient with respect to the pruned layer
        layer.pruned_weight.requires_grad = True

        self.logger_counter = 0
        return layer, pruned_module
    
    def prepare_data(self, batch):
        lr, hr = batch['L'], batch['H']
        lr = lr.cuda()
        hr = hr.cuda()
        return lr, hr

    def find_maximum_grad_fnorm(self, grad_fnorm, pruned_module, layer, layer_name):
        """
        Find the channel index with maximum gradient frobenius norm,
        and initialize the pruned_weight w.r.t. the selected channel.
        """
        if isinstance(pruned_module, WindowAttention):
            # simultaneously pruned the weight of multihead
            dim_mha = pruned_module.dim // pruned_module.num_heads
            temp_d = layer.d[:dim_mha]
            grad_fnorm.data.mul_(1 - temp_d).sub_(temp_d)
        else:
            grad_fnorm.data.mul_(1 - layer.d).sub_(layer.d)
        _, max_index = torch.topk(grad_fnorm, 1)
        
        if isinstance(pruned_module, WindowAttention):
            # simultaneously pruned the weight of multihead
            layer.d[max_index::dim_mha] = 1
            # warm-started from the pre-trained model
            layer.pruned_weight.data[:, max_index::dim_mha, ...] = \
                layer.weight[:, max_index::dim_mha, ...].data.clone()
        

            # select the corresponding weights of qkv
            dim_e   = pruned_module.dim
            dim_mha = pruned_module.dim // pruned_module.num_heads
            qkv_layer = pruned_module.qkv
            
            q_feature_indices = np.arange(
                0, dim_e, dtype=int)[max_index :dim_e :dim_mha]
            k_feature_indices = np.arange(
                dim_e, dim_e*2, dtype=int)[max_index :dim_e :dim_mha]
            v_feature_indices = np.arange(
                dim_e*2, dim_e*3, dtype=int)[max_index :dim_e :dim_mha]
            
            # select weights in q (the first dim_e weights)
            qkv_layer.d[q_feature_indices] = 1
            # select weights in k (the middle dim_e weights)
            qkv_layer.d[k_feature_indices] = 1
            # select weights in v (the last dim_e weights)
            qkv_layer.d[v_feature_indices] = 1

            # copy the weights of out_features from the pre-trained model
            qkv_layer.pruned_weight.data[q_feature_indices, :] = \
                qkv_layer.weight[q_feature_indices, :].data.clone()
            qkv_layer.pruned_weight.data[k_feature_indices] = \
                qkv_layer.weight[k_feature_indices, :].data.clone()
            qkv_layer.pruned_weight.data[v_feature_indices, :] = \
                qkv_layer.weight[v_feature_indices, :].data.clone()

            # copy the weights of bias from the pre-trained model
            if qkv_layer.bias is not None:
                qkv_layer.pruned_bias.data[q_feature_indices] = \
                    qkv_layer.bias[q_feature_indices].data.clone()
                qkv_layer.pruned_bias.data[k_feature_indices] = \
                    qkv_layer.bias[k_feature_indices].data.clone()
                qkv_layer.pruned_bias.data[v_feature_indices] = \
                    qkv_layer.bias[v_feature_indices].data.clone()
        
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

        layer.pruned_weight.grad = None

        for j, batch in enumerate(self.train_loader):
            # get data
            lr, hr = self.prepare_data(batch)
            
            sr, sr2lr = self.model_segment(lr, final_output=True)
            
            loss = self.compute_loss_error(lr, hr, sr, sr2lr)

            loss.backward()

            self.record_selection_loss.update(loss.item(), hr.size(0))

        cum_grad = layer.pruned_weight.grad.data.clone()
        layer.pruned_weight.grad = None

        # reshape cum_grad into the multihead format
        if isinstance(pruned_module, WindowAttention):
            n_dim = pruned_module.dim
            n_heads = pruned_module.num_heads
            cum_grad = cum_grad.reshape((-1, n_heads, n_dim // n_heads))

            # calculate L1 norm of gradient, TODO checking
            # (1 - pruned_dim) the dimention of the preserved features
            # grad_fnorm = cum_grad.mul(cum_grad).sum((2, 3)).sqrt().sum(0)
            grad_fnorm = cum_grad.abs().sum((0, 1))
        else:
            grad_fnorm = cum_grad.abs().sum(0)

        # find grad_fnorm with maximum absolute gradient
        self.find_maximum_grad_fnorm(
            grad_fnorm, pruned_module, layer, layer_name)


    def set_layer_wise_optimizer(self, layer):
        params_list = []
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
    
    def compute_loss_error(self, lr, hr, sr, sr2lr=None):
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
            loss += loss_primary + self.args.opt['train']['dual_weight'] * loss_dual
        else:
            loss += loss_primary
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
                lr, hr = self.prepare_data(batch)
                sr, sr2lr = self.model_segment(lr, 
                    final_output=True, original_forward=original_forward)
                
                loss = self.compute_loss_error(lr, hr, sr, sr2lr)
                optimizer.zero_grad()
                # compute gradient
                loss.backward()
                # we only optimize W with respect to the selected channel
                if self.args.opt['pruning']['prune_type'] == 'thinet':
                    if isinstance(layer, MaskLinear):
                        layer.beta.grad.data.mul_(layer.d)
                
                # ignore finetune the pruned weight
                layer.pruned_weight.grad.data.mul_(
                    layer.d.unsqueeze(0).expand_as(layer.pruned_weight))
                
                # ### qkv updating
                # if isinstance(pruned_module, WindowAttention):
                #     qkv = pruned_module.qkv
                #     qkv.pruned_weight.grad.data.mul_(
                #         qkv.d.unsqueeze(1).expand_as(qkv.pruned_weight))
                #     if qkv.bias is not None:
                #         qkv.pruned_bias.grad.data.mul_(
                #             qkv.d.expand_as(qkv.pruned_bias))
                
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
                pruned_module, block_count, layer_name="proj"):
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
        # elif self.args.opt['pruning']['prune_type'].lower() == 'cp':
        #     self.lasso_selection(layer, block_count, layer_name)
        # elif self.args.opt['pruning']['prune_type'].lower() == 'thinet':
        #     self.thinet_selection(layer, block_count, layer_name)
        else:
            assert False, "unsupport prune type: {}".format(self.args.opt['pruning']['prune_type'])
        
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
        
    def dcp_selection(self, pruned_module, layer, block_count, layer_name):
        # search the multihead channels simultaneously
        pruned_feature = layer.in_features
        pruning_num = math.floor(pruned_feature * self.args.opt['pruning']['pruning_rate'])
        
        if self.channel_configs is not None:
            pruning_num = pruned_feature - self.channel_configs[block_count - 1]
            assert block_count - 1 >= 0, "The range of block_count should start from 1."
        
        for _ in range(pruned_feature):
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
        
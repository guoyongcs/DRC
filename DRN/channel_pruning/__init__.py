import time
import datetime
import copy
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data
from importlib import import_module
from .channel_selection import LayerChannelSelection
from .segment_trainer import SegmentWiseTrainer
from .pruning_checkpoint import PruningCheckPoint
from .pruning import DRNModelPrune
from utils import get_logger, TensorboardLogger, model2list, ModelAnalyse
from model.common import DownBlock, RCAB, ResBlock, MaskConv2d


class ChannelPruning():
    """
    Run Channel Pruning with pre-defined pipeline
    """
    def __init__(self, opt, model, loader=None):
        self.opt = opt
        self.scale = opt.scale
        self.checkpoint = None
        
        self.model = model
        self.original_model = model.get_model()
        self.pruned_model = copy.deepcopy(self.original_model)
        
        self.train_loader = loader.loader_train
        self.val_loader = loader.loader_test
        
        self.epoch = 0
        self.current_block_count = 0

        self.logger = get_logger(self.opt.save, "pruning")
        self.tensorboard_logger = TensorboardLogger(osp.join(self.opt.save, 'tb'))
        self.logger.info("|===>Result will be saved at {}".format(self.opt.save))
        self.prepare()
    
    def prepare(self):
        """
        Preparing experiments
        """
        
        self._set_trainier()
        self._set_checkpoint()
        self._set_channel_selection()
        torch.set_num_threads(4)

    def _set_channel_selection(self):
        self.layer_channel_selection = LayerChannelSelection(self.opt, 
                                                self.model,
                                                self.segment_wise_trainer, 
                                                self.train_loader,
                                                self.val_loader,  
                                                self.checkpoint, 
                                                self.logger, 
                                                self.tensorboard_logger)

    def _set_trainier(self):
        """
        Initialize segment-wise trainer
        """

        # initialize segment-wise trainer
        self.segment_wise_trainer = SegmentWiseTrainer(self.opt, 
                                                self.model,
                                                self.original_model,
                                                self.pruned_model,
                                                self.train_loader,
                                                self.val_loader,
                                                self.logger,
                                                self.tensorboard_logger)

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.original_model is not None and self.pruned_model is not None, "please create model first"

        self.checkpoint = PruningCheckPoint(self.opt.save, self.logger)
        self._load_resume()

    def _load_resume(self):
        """
        Load resume checkpoint
        """

        if self.opt.resume_pruning is not None:
            check_point_params = torch.load(self.opt.resume_pruning)
            pruned_model_state = check_point_params["pruned_model"]
            self.current_block_count = check_point_params["block_num"]

            if self.current_block_count > 0:
                self.replace_layer_mask_conv()
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)
            self.logger.info("|===>load resume file: {}".format(self.opt.resume_pruning))

    def replace_layer_mask_conv(self):
        """
        Replace the convolutional layer with mask convolutional layer
        """

        block_count = 0
        if self.opt.model.lower() in ["attsrunet", "drn"] or \
                        self.opt.model.lower().find('drnt_res') >= 0:
            pruned_modules = self.segment_wise_trainer.pruned_modules
            for idx, module in enumerate(pruned_modules):
                if isinstance(module, (RCAB, ResBlock)):
                    block_count += 1
                    layer = module.body[2]
                elif isinstance(module, DownBlock):
                    block_count += 1
                    layer = module.dual_module[1]
                elif isinstance(module, nn.Conv2d):
                    block_count += 1
                    layer = module
                self.logger.info(type(layer))
                if block_count <= self.current_block_count and not isinstance(layer, MaskConv2d):
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
                    
                    device = layer.weight.device
                    temp_conv = temp_conv.to(device)
                    
                    if isinstance(module, (RCAB, ResBlock)):
                        module.body[2] = temp_conv
                    elif isinstance(module, DownBlock):
                        module.dual_module[1] = temp_conv
                    elif isinstance(module, nn.Conv2d):
                        for i in range(len(self.scale)):
                            conv1 = self.pruned_model.up_blocks[i][-1]
                            if conv1.out_channels == temp_conv.out_channels:
                                self.pruned_model.up_blocks[i][-1] = temp_conv
                                pruned_modules[idx] = temp_conv

    def channel_selecton(self):
        """
        Conduct channel selection
        """

        # get testing error
        self.segment_wise_trainer.val(0)
        time_start = time.time()

        # conduct channel selection
        original_modules = model2list(self.segment_wise_trainer.original_modules)
        pruned_modules = model2list(self.segment_wise_trainer.pruned_modules)

        # original_seq = nn.Sequential(*original_modules)
        pruned_seq = nn.Sequential(*pruned_modules)
        
        self.logger.info(pruned_seq)
        
        net_pruned = self.channel_selection_for_segment(
                            original_modules, pruned_modules)

        self.logger.info(self.segment_wise_trainer.pruned_modules)
        self.logger.info(net_pruned)
        self.logger.info(self.original_model)
        self.logger.info(self.pruned_model)
        self.segment_wise_trainer.val(0)
        
        block_count = len(pruned_modules)
        # self.checkpoint.save_models(self.pruned_model, block_count)
        
        self.pruning()

        self.checkpoint.save_models(self.pruned_model, block_count, pruned=True)

        time_interval = time.time() - time_start
        log_str = "cost time: {}".format(
            str(datetime.timedelta(seconds=time_interval)))
        self.logger.info(log_str)
    
    def channel_selection_for_segment(self, original_modules, pruned_modules):
        """
        Conduct channel selection for one segment
        """

        block_count = 0
        for ori_module, pruned_module in zip(original_modules, pruned_modules):
            if isinstance(pruned_module, (RCAB, ResBlock)):
                block_count += 1
                # We will not prune the pruned blocks again
                if not isinstance(pruned_module.body[2], MaskConv2d):
                    self.layer_channel_selection.\
                        channel_selection_for_one_layer(
                            ori_module, pruned_module, block_count, "body2")
                    self.logger.info("|===>checking layer type: {}".format(type(pruned_module.body[2])))
            elif isinstance(pruned_module, DownBlock):
                block_count += 1
                # We will not prune the pruned blocks again
                if not isinstance(pruned_module.dual_module[1], MaskConv2d):
                    self.layer_channel_selection.\
                        channel_selection_for_one_layer(
                            ori_module, pruned_module, block_count, "module1")
                    self.logger.info("|===>checking layer type: {}".format(type(pruned_module.dual_module[1])))
            elif isinstance(pruned_module, nn.Conv2d):
                block_count += 1
                # We will not prune the pruned blocks again
                if not isinstance(pruned_module, MaskConv2d):
                    pruned_module = self.layer_channel_selection.\
                        channel_selection_for_one_layer(
                            ori_module, pruned_module, block_count, "conv1")
                    self.logger.info("|===>checking layer type: {}".format(type(pruned_module)))
            
            if block_count > self.current_block_count:
                self.checkpoint.save_checkpoints(self.pruned_model, block_count)


    def pruning(self):
        """
        Prune channels
        """
        self.logger.info("Before pruning:")
        self.logger.info(self.pruned_model)
        self.segment_wise_trainer.val(0)
        model_analyse = ModelAnalyse(self.pruned_model, self.logger)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))
        if max(self.scale) == 4:
            test_input = torch.randn(1, 3, 180, 320).cuda()
        else:
            test_input = torch.randn(1, 3, 90, 160).cuda()
        model_analyse.madds_compute(test_input)

        if self.opt.model.lower() in ["attsrunet", "drn"] or \
                        self.opt.model.lower().find('drnt_res') >= 0:
            model_prune = DRNModelPrune(opt=self.opt, 
                                        model=self.pruned_model,
                                        net_type=self.opt.model)
        else:
            assert False, "unsupport model: {}".format(self.opt.model)

        model_prune.run()

        self.logger.info("After pruning:")
        self.logger.info(self.pruned_model)
        self.segment_wise_trainer.val(0)
        model_analyse = ModelAnalyse(self.pruned_model, self.logger)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))
        model_analyse.madds_compute(test_input)
import time
import datetime
import copy
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data
from .channel_selection import LayerChannelSelection
from .segment_trainer import SegmentWiseTrainer
from .pruning_checkpoint import PruningCheckPoint
from .pruning import SwinIRPrune
from utils.pruned_utils import get_logger, TensorboardLogger, model2list, ModelAnalyse
from .common_module import MaskLinear
from models.network_swinir_pruned import Mlp, WindowAttention


class ChannelPruning():
    """
    Run Channel Pruning with pre-defined pipeline
    """
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.scale = args.scale
        self.checkpoint = None
        
        self.model = model
        self.original_model = model.netG
        self.pruned_model = copy.deepcopy(self.original_model)
        
        self.train_loader = train_loader
        self.val_loader = test_loader
        
        self.epoch = 0
        self.current_block_count = 0

        self.logger = get_logger(self.args.save_dir, "pruning")
        self.tensorboard_logger = TensorboardLogger(osp.join(self.args.save_dir, 'tb'))
        self.logger.info("|===>Result will be saved at {}".format(self.args.save_dir))
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
        self.layer_channel_selection = LayerChannelSelection(self.args, 
                                                self.model,
                                                self.segment_wise_trainer, 
                                                self.train_loader,
                                                self.val_loader,  
                                                self.checkpoint, 
                                                self.logger, 
                                                self.tensorboard_logger)

    def _set_trainier(self):
        """
        Initialize segment-wise trainer trainer
        """

        # initialize segment-wise trainer
        self.segment_wise_trainer = SegmentWiseTrainer(self.args,
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

        self.checkpoint = PruningCheckPoint(self.args.save_dir, self.logger)
        self._load_resume()

    def _load_resume(self):
        """
        Load resume checkpoint
        """

        if self.args.opt['pruning']['resume_pruning'] is not None:
            check_point_params = torch.load(self.args.opt['pruning']['resume_pruning'])
            pruned_model_state = check_point_params["pruned_model"]
            self.current_block_count = check_point_params["block_num"]

            if self.current_block_count > 0:
                self.replace_layer_mask_linear()
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)
            self.logger.info("|===>load resume file: {}".format(self.args.opt['pruning']['resume_pruning']))

    def replace_layer_mask_linear(self):
        """
        Replace the convolutional layer with mask convolutional layer
        """
        block_count = 0
        pruned_modules = self.segment_wise_trainer.pruned_modules
        for idx, module in enumerate(pruned_modules):
            if isinstance(module, (WindowAttention)):
                block_count += 1
                layer = module.proj
            elif isinstance(module, Mlp):
                block_count += 1
                layer = module.fc2
            self.logger.info(type(layer))

            if block_count <= self.current_block_count and not isinstance(layer, MaskLinear):
                temp_layer = MaskLinear(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=(layer.bias is not None))
                temp_layer.weight.data.copy_(layer.weight.data)
                
                if layer.bias is not None:
                    temp_layer.bias.data.copy_(layer.bias.data)
                
                if self.args.opt['pruning']['prune_type'].lower() != 'dcp': 
                    temp_layer.init_beta()
                
                device = layer.weight.device
                temp_layer = temp_layer.to(device)
                
                if isinstance(module, (WindowAttention)):
                    module.proj = temp_layer
                elif isinstance(module, Mlp):
                    module.fc2  = temp_layer

    def channel_selecton(self):
        """
        Conduct channel selection
        """

        # get testing error
        self.segment_wise_trainer.val()
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
        self.segment_wise_trainer.val()
        
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
            if isinstance(pruned_module, WindowAttention):
                block_count += 1
                # We will not prune the pruned blocks again
                if not isinstance(pruned_module.proj, MaskLinear):
                    self.layer_channel_selection.\
                        channel_selection_for_one_layer(
                            ori_module, pruned_module, block_count, "proj")
                    self.logger.info("|===>checking layer type: {}".format(type(pruned_module.proj)))
            elif isinstance(pruned_module, Mlp):
                block_count += 1
                # We will not prune the pruned blocks again
                if not isinstance(pruned_module.fc2, MaskLinear):
                    self.layer_channel_selection.\
                        channel_selection_for_one_layer(
                            ori_module, pruned_module, block_count, "fc2")
                    self.logger.info("|===>checking layer type: {}".format(type(pruned_module.fc2)))
            
            if block_count > self.current_block_count:
                self.checkpoint.save_checkpoints(self.pruned_model, block_count)


    def pruning(self):
        """
        Prune channels
        """
        self.logger.info("Before pruning:")
        self.logger.info(self.pruned_model)
        self.segment_wise_trainer.val()
        model_analyse = ModelAnalyse(self.pruned_model, self.logger)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))
        test_input = torch.randn(1, 3, 96, 96).cuda()
        model_analyse.madds_compute(test_input)

        model_prune = SwinIRPrune(args=self.args, 
                                    model=self.pruned_model)

        model_prune.run()

        self.logger.info("After pruning:")
        self.logger.info(self.pruned_model)
        self.segment_wise_trainer.val()
        model_analyse = ModelAnalyse(self.pruned_model, self.logger)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))
        model_analyse.madds_compute(test_input)
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.dcls_arch import DPCAB
from models.modules.module_util import ResidualBlock_noBN

class DCLSSegment(nn.Module):
    """
    Early Stopping for DCLS networks during pruning
    """
    def __init__(self, args, original_model, pruned_model):
        super(DCLSSegment, self).__init__()
        self.original_model = original_model
        self.pruned_model = pruned_model

        self.num_blocks_each_stage = self.count_layers(original_model)
        self.block_index = 0
        self.stage_index = 1
        self.stop_index = 1

    
    def count_layers(self, model):
        """Count the number block in each RSTB module"""
        if isinstance(model, nn.DataParallel):
            model = model.module
        num_blocks_each_stage = []
        # resblock and the linear for computing kernel are pruned
        num_blocks_each_stage.append(len(model.Estimator.body) + 1)
        num_blocks_each_stage.append(len(model.Restorer.feature_block))
        num_blocks_each_stage.append(len(model.Restorer.body))
        return num_blocks_each_stage
    
    def split_segment(self, block_index):
        # index of the pruned module, windowAtt or mlp layer
        self.block_index = block_index
        self.stage_index = 1 # start from 1
        for num_blocks in self.num_blocks_each_stage:
            # a swin block contain one windowAtt module and one mlp module
            if block_index > num_blocks:
                # do not use in-plance operators
                block_index = block_index - num_blocks
                self.stage_index += 1
            else:
                # if block is the last mlp layer (or the layer before)
                self.stop_index = block_index
                break

    def forward(self, inputs, final_output=False, original_forward=True):
        if original_forward:
            self.segemnt_forward(self.original_model, inputs)
        if not final_output:
            pruned_output = self.segemnt_forward(self.pruned_model, inputs)
        else:
            if isinstance(self.pruned_model, nn.DataParallel):
                # the data have been distributed before forward function
                pruned_output = self.pruned_model.module(inputs)
            else:
                pruned_output = self.pruned_model(inputs)
        return pruned_output
    

    def segemnt_forward(self, model, x):
        if isinstance(model, nn.DataParallel):
            model = model.module

        if self.stage_index <= 1:
            if self.stop_index <= len(model.Estimator.body):
                f1 = model.Estimator.head(x)
                x = model.Estimator.body[:self.stop_index](f1)
            else:
                x = model.Estimator(x)

        else:
            f = model.Restorer.conv_first(x)
            if self.stage_index == 2:
                x = model.Restorer.feature_block(f)

            else: # stage_index == 3 
                kernel = model.Estimator(x)
                feature = model.Restorer.feature_block(f)
                f1 = model.Restorer.head1(feature)
                f2 = model.Restorer.head2(feature, kernel)
                inputs = [f2, f1]
                x = model.Restorer.body[:self.stop_index](inputs)# x: [f2, f1]

        return x


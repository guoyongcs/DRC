import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network_swinir_pruned import RSTB

__all__ = ['SwinIRSegment']

class SwinIRSegment(nn.Module):
    """
    Early Stopping for SwinIR networks during pruning
    """
    def __init__(self, args, original_model, pruned_model):
        super(SwinIRSegment, self).__init__()
        self.original_model = original_model
        self.pruned_model = pruned_model

        # addtional 1 for conv2d after upsampler
        self.num_swin_blocks_each_RSTB = self.count_layers(original_model)
        self.block_index = 0
        self.layer_index = 1
        self.swin_index = 1
    
    def count_layers(self, model):
        """Count the number block in each RSTB module"""
        num_swin_blocks_each_RSTB = []
        for module in model.modules():
            if isinstance(module, RSTB):
                num_swin_blocks_each_RSTB.append(
                    module.residual_group.depth) 
        
        return num_swin_blocks_each_RSTB
    
    def split_segment(self, block_index):
        # index of the pruned module, windowAtt or mlp layer
        self.block_index = block_index
        self.layer_index = 1 # start from 1
        for num_swin_blocks in self.num_swin_blocks_each_RSTB:
            # a swin block contain one windowAtt module and one mlp module
            if block_index > num_swin_blocks * 2:
                # do not use in-plance operators
                block_index = block_index - num_swin_blocks * 2
                self.layer_index += 1
            else:
                # if block is the last mlp layer (or the layer before)
                self.swin_index = torch.ceil(torch.tensor(block_index / 2)).byte().item()
                break

    def forward(self, inputs, final_output=False, original_forward=True):
        if original_forward:
            self.segemnt_forward(self.original_model, inputs)
        if not final_output:
            pruned_output = self.segemnt_forward(self.pruned_model, inputs)
        else:
            if isinstance(self.pruned_model, nn.DataParallel):
                # the data have been distributed before forward function of SwinIRSegment
                pruned_output = self.pruned_model.module(inputs)
            else:
                pruned_output = self.pruned_model(inputs)
        return pruned_output
    
    def forward_features(self, x, model):
        # Swin IR forward
        x_size = (x.shape[2], x.shape[3])
        x = model.patch_embed(x)
        if model.ape:
            x = x + model.absolute_pos_embed
        x = model.pos_drop(x)

        assert self.layer_index <= len(model.layers) + 1, "out of boundary"

        for idx, layer in enumerate(model.layers[:self.layer_index]):
            if idx != self.layer_index - 1:
                x = layer(x, x_size)
            else:
                for blk in layer.residual_group.blocks[:self.swin_index]:
                    x = blk(x, x_size)

        return x

    def segemnt_forward(self, model, x):
        if isinstance(model, nn.DataParallel):
            model = model.module

        H, W = x.shape[2:]
        
        x = model.check_image_size(x)
        
        model.mean = model.mean.type_as(x)
        x = (x - model.mean) * model.img_range
        
        # for classical and lightweight SR
        x = model.conv_first(x)
        x = self.forward_features(x, model)

        return x


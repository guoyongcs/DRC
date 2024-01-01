import torch
import torch.nn as nn
__all__ = ['DRNSegment']

class DRNSegment(nn.Module):
    """
    Prune drn networks
    """
    def __init__(self, opt, original_model, pruned_model):
        super(DRNSegment, self).__init__()
        self.original_model = original_model
        self.pruned_model = pruned_model
        self.phase = len(opt.scale)
        # addtional 1 for conv2d after upsampler
        self.num_blocks_each_phase = opt.n_resblocks + 1
        self.block_index = 0
    
    def split_segment(self, block_index):
        if block_index <= self.phase:
            self.stage = 0
            self.block_index = block_index
        else:
            block_index -= self.phase
            self.stage = (block_index - 1) // self.num_blocks_each_phase + 1
            self.block_index = (block_index - 1) % self.num_blocks_each_phase + 1
            # If the block is the last one, 
            # forward the conv1 module after upsampler
            if self.block_index == self.num_blocks_each_phase:
                # one addition upsampler layer that does not be pruned
                self.block_index = self.block_index + 1
            
    
    def forward(self, inputs, final_output=False, original_forward=True):
        if original_forward:
            self.segemnt_forward(self.original_model, inputs)
        if not final_output:
            pruned_output = self.segemnt_forward(self.pruned_model, inputs)
        else:
            pruned_output = self.pruned_model(inputs)
        return pruned_output

    def segemnt_forward(self, model, inputs):
        # upsample x to target sr size
        x = model.upsample(inputs)

        # preprocess
        if hasattr(model, 'sub_mean'):
            x = model.sub_mean(x)
        x = model.head(x)

        # down phases
        copies = []
        phase = self.phase if self.stage != 0 else self.block_index
        for idx in range(phase):
            if self.stage > 0:
                copies.append(x)
            x = model.down[idx](x)

        # up phases
        for idx in range(self.stage):
            if idx != self.stage - 1:
                # upsample to SR features
                x = model.up_blocks[idx](x)
                # concat down features and upsample features
                x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            else:
                x = model.up_blocks[idx][:self.block_index](x)

        return x
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class MaskConv2d(nn.Conv2d):
    """
    Custom convolutional layers for channel pruning.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super(MaskConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        self.beta = None
        self.register_buffer("d", torch.ones(self.weight.data.size(1)))
        self.pruned_weight = Parameter(self.weight.clone())

    def init_beta(self):
        """beta: the scalar or the mask of weights"""
        self.beta = Parameter(torch.ones(self.weight.data.size(1)))

    def forward(self, input):
        if self.beta is not None:
            new_weight = self.pruned_weight * self.beta.unsqueeze(0).\
                unsqueeze(2).unsqueeze(3).expand_as(self.weight)
        else:
            new_weight = self.pruned_weight
        return F.conv2d(input, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class MaskLinear(nn.Linear):
    """
    Custom Linear layers for in_features (channel) pruning.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.beta = None # use for pruning methods except dcp
        # pruning the input channels of the second layer
        self.register_buffer("d", torch.ones(self.weight.data.size(1)))
        self.pruned_weight = Parameter(self.weight.clone())

    def init_beta(self):
        """beta: the scalar or the mask of weights"""
        self.beta = Parameter(torch.ones(self.weight.data.size(1)))

    def forward(self, input):
        if self.beta is not None:
            new_weight = self.pruned_weight * \
                self.beta.unsqueeze(0).expand_as(self.weight)
        else:
            new_weight = self.pruned_weight
        return F.linear(input, new_weight, self.bias)


class MaskModuleList(nn.ModuleList):
    """
    Custom ModuleList module for channel pruning.
    All layers should have the same number of input channels.
    """
    def __init__(self, modules):
        super(MaskModuleList, self).__init__(modules)        
        
        self.register_buffer("d", torch.ones(modules[-1].weight.data.size(1)))
        
        self.init_pruning()
    
    def init_pruning(self):
        for idx, m in enumerate(self):
            if not isinstance(m, (MaskLinear)):
                temp_layer = MaskLinear(
                    in_features=m.in_features,
                    out_features=m.out_features,
                    bias=(m.bias is not None))
                temp_layer.weight.data.copy_(m.weight.data)

                if m.bias is not None:
                    temp_layer.bias.data.copy_(m.bias.data)

                # put into the same device
                device = m.weight.device
                temp_layer = temp_layer.to(device)
        
            self[idx] = temp_layer
        
        self.in_channels = self[-1].in_features
                
    def init_pruning_params(self):
        for m in self:
            m.d.fill_(0)
            m.pruned_weight.data.fill_(0)
        
        self.d = self[-1].d.clone()

    def set_train_pruning_params(self):
        for m in self:
            m.pruned_weight.requires_grad = True

    def select_channels(self, index=0, value=1):
        for m in self:
            m.d[index] = value
        self.d = self[-1].d.clone()
                
    def warm_start(self, index):
        """copy weights from the original module"""
        for m in self:
            m.pruned_weight.data[:, index, ...] = \
                m.weight[:, index, ...].data.clone()

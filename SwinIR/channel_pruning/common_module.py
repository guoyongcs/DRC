import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


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

class MaskOutLinear(nn.Linear):
    """
    Custom Linear layers for out_features (channel) pruning.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MaskOutLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.beta = None # use for pruning methods except dcp
        self.register_buffer("d", torch.ones(self.weight.data.size(0)))
        self.pruned_weight = Parameter(self.weight.clone())
        if self.bias is not None:
            self.pruned_bias = Parameter(self.bias.clone())

    def init_beta(self):
        """beta: the scalar or the mask of weights"""
        self.beta = Parameter(torch.ones(self.weight.data.size(0)))

    def forward(self, input):
        new_bias = None
        if self.beta is not None:
            new_weight = self.pruned_weight * \
                self.beta.unsqueeze(1).expand_as(self.weight)
            if self.bias is not None:
                new_bias = self.pruned_bias * self.beta
        else:
            new_weight = self.pruned_weight
            if self.bias is not None:
                new_bias = self.pruned_bias
        return F.linear(input, new_weight, new_bias)
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class SearchConv2d(nn.Conv2d):
    """
    Custom convolutional layers for channel search.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super(SearchConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("channel_weights", torch.ones(self.weight.data.size(1))) # do not update during optimizer step

    def set_channel_weights(self, channel_weights):
        self.channel_weights = channel_weights

    def forward(self, input):
        if self.channel_weights is not None:
            new_weight = self.weight * self.channel_weights.unsqueeze(0).\
                unsqueeze(2).unsqueeze(3).expand_as(self.weight)
        else:
            assert False, "the weights should be rescaled"
        return F.conv2d(input, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SearchLinear(nn.Linear):
    """
    Custom convolutional layers for channel search.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SearchLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias)
        # do not update during optimizer step
        self.register_buffer("channel_weights", torch.ones(self.weight.data.size(1))) 

    def set_channel_weights(self, channel_weights):
        device = self.weight.device
        self.channel_weights = channel_weights.to(device)

    def forward(self, input):
        if self.channel_weights is not None:
            new_weight = self.weight * self.channel_weights.\
                            unsqueeze(0).expand_as(self.weight)
        else:
            assert False, "the weights should be rescaled"
        return F.linear(input, new_weight, self.bias)


class SearchModuleList(nn.ModuleList):
    """
    Custom ModuleList layers for channel search.
    """

    def __init__(self, modules):
        super(SearchModuleList, self).__init__(modules)
        self.init_search()

    def init_search(self):
        for idx, m in enumerate(self):
            if not isinstance(m, (SearchLinear)):
                temp_layer = SearchLinear(
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

    def set_channel_weights(self, channel_weights):
        for module in self:
            module.set_channel_weights(channel_weights)

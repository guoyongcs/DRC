import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.network_swinir_pruned import WindowAttention


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


class SearchOutLinear(nn.Linear):
    """
    Custom convolutional layers for channel search.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SearchOutLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias)
        # do not update during optimizer step
        self.register_buffer("channel_weights", torch.ones(self.weight.data.size(0))) 

    def set_channel_weights(self, channel_weights):
        device = self.weight.device
        self.channel_weights = channel_weights.to(device)

    def forward(self, input):
        new_bias = None
        if self.channel_weights is not None:
            new_weight = self.weight * self.channel_weights.\
                            unsqueeze(1).expand_as(self.weight)
            if self.bias is not None:
                new_bias = self.bias * self.channel_weights
        else:
            assert False, "the weights should be rescaled"
        return F.linear(input, new_weight, new_bias)


class SearchWindowAttention(nn.Module):
    def __init__(self, args, win_att, alpha=None):
        super(SearchWindowAttention, self).__init__()
        self.args = args
        self.win_att = win_att
        self.alpha = alpha
        self.device = win_att.qkv.weight.device
        self.num_heads = win_att.num_heads
        self.dim = win_att.dim

    def set_alpha(self, alpha):
        self.alpha = alpha
       
    def set_channel_options(self, channel_options):
        self.channel_options = channel_options
    
    def forward(self, x, mask=None):
        assert self.alpha is not None, "alpha should be initialized."

        weight_indices = self.get_weight_indices()
        outputs = None
        for idx in range(len(self.alpha)):
            # select the channels of qkv and proj
            self.set_channel_weights(idx, weight_indices)
            
            if outputs is None:
                outputs = self.alpha[idx] * self.win_att(x, mask)
            else:
                outputs += self.alpha[idx] * self.win_att(x, mask)
            
        return outputs

    def get_weight_indices(self):
        # sort weights of the multihead simultanously
        weight = self.win_att.proj.weight.reshape(self.dim, 
            self.num_heads, self.dim // self.num_heads)

        if not self.args.opt['search']['unordered_channels']:
            # sort the channels based on L1-norm
            with torch.no_grad():
                _, indices = torch.sort(
                    weight.data.abs().sum((0, 1)), descending=True)
                indices = indices.cpu().numpy()
        else:
            # do not sort the channels
            indices = torch.arange(self.dim // self.num_heads).numpy()
        return indices

    def set_channel_weights(self, num_index, weight_indices):
        num_channels = self.channel_options[num_index]
        # obtain the indexs
        weight_indices = weight_indices[:num_channels]
        weight_indices = torch.tensor(weight_indices).to(self.device)
        weight_indices = F.one_hot(weight_indices, 
                            self.dim//self.num_heads).sum(0).bool().float()

        # repeat for the multihead
        # (self.dim//self.num_heads) -> (self.num_heads, self.dim//self.num_heads)
        weight_indices = weight_indices.reshape(
            (1, weight_indices.size(0))).repeat(self.num_heads, 1)
        proj_weights = weight_indices.reshape(-1)
        
        # (self.num_heads, self.dim//self.num_heads) -> \
        # (3, self.num_heads, self.dim//self.num_heads)
        qkv_weight_indices = weight_indices.reshape(
            (1, self.num_heads, self.dim//self.num_heads)).repeat(3, 1, 1)
        qkv_weights = qkv_weight_indices.reshape(-1)
        
        self.win_att.proj.set_channel_weights(proj_weights.detach())
        self.win_att.qkv.set_channel_weights(qkv_weights.detach())

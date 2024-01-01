import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Tuple, Iterator
from torch.autograd import Variable
try:
    from utils.quant_op import *
except:
    pass

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, pruned_feats=None,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        
        if pruned_feats is None: pruned_feats = n_feats
        
        m = []
        m.append(conv(n_feats, pruned_feats, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(pruned_feats))
        m.append(act)
        m.append(conv(pruned_feats, n_feats, kernel_size, bias=bias))
        
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, pruned_feats=None, bn=False, act=False, bias=True):
        if pruned_feats is None: pruned_feats = n_feats
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * pruned_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class DownBlock(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = opt.negval

        if nFeat is None:
            nFeat = opt.n_feats
        
        if in_channels is None:
            in_channels = opt.n_colors
        
        if out_channels is None:
            out_channels = opt.n_colors

        
        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, pruned_feats=None, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        if pruned_feats is None:
            pruned_feats = n_feats
        modules_body = []
        
        modules_body.append(conv(n_feats, pruned_feats, kernel_size, bias=bias))
        if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(act)
        modules_body.append(conv(pruned_feats, n_feats, kernel_size, bias=bias))
        modules_body.append(CALayer(n_feats, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


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

class Conv2dInt8(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """
    def __init__(self, in_channels, out_channels, 
                kernel_size, stride=1, padding=0, 
                dilation=1, groups=1, bias=True, activation_value=0.0):
        super(Conv2dInt8, self).__init__(in_channels,  out_channels, 
                    kernel_size, stride, padding, dilation, groups, bias)
        self.activation_value = activation_value

    def forward(self, input):
        quantized_input = \
            quantization_on_input_fix_scale(input, self.activation_value)
        quantized_weight = quantization_on_weights(self.weight)
        if self.bias is not None:   # not quantization self.bias
            quantized_bias = self.bias
        else:
            quantized_bias = None
        return F.conv2d(quantized_input, quantized_weight, quantized_bias, 
                        self.stride, self.padding, self.dilation, self.groups)

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

class SearchMultiConv2d(nn.Module):
    def __init__(self, args, conv_layer, alpha=None):
        super(SearchMultiConv2d, self).__init__()
        self.args = args
        self.alpha = alpha
        self.device = conv_layer.weight.device
        
        self.conv_layers = nn.ModuleList()
        if self.args.unordered_channels:
            self.channel_indices = torch.arange(conv_layer.in_channels).to(self.device)
        else:
            _, indices = torch.sort(
                    conv_layer.weight.data.detach().abs().sum(dim=(0,2,3)), descending=True)
            self.channel_indices = indices
        
        self.selected_channels_sets = []

        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.k_size      = conv_layer.kernel_size
        self.stride      = conv_layer.stride
        self.padding     = conv_layer.padding
        self.bias        = (conv_layer.bias is not None)
        self.original_weight = conv_layer.weight.data.detach().clone() # do not load as parameter
        if self.bias:
            self.original_bias = conv_layer.bias.data.detach().clone()
        
    def init_multi_branchs(self, channel_options, alpha=None):
        if self.alpha is None:
            self.alpha = alpha
        
        conv_layers = []
        for c_option in channel_options:
            temp_conv = nn.Conv2d(in_channels=c_option, out_channels=self.out_channels,
                kernel_size=self.k_size, stride=self.stride, padding=self.padding, bias=self.bias)
            temp_conv = temp_conv.to(self.device)
            if self.args.unordered_channels:
                # copy the first k channels
                weight_data = self.original_weight[:,:c_option,:,:].clone()
                select_channels = torch.arange(c_option).to(self.device)
            else:
                # copy the k channels with the largest L1-norm value
                weight_data = torch.zeros_like(self.original_weight)
                for i in range(c_option):
                    weight_data[:, self.channel_indices[i], :, :] = self.original_weight[:, self.channel_indices[i], :, :]
                select_channels = weight_data.ne(0).sum(dim=(0,2,3)).ne(0)
                select_channels = torch.nonzero(select_channels, as_tuple=False).flatten()
                weight_data = weight_data.index_select(dim=1, index=select_channels)
            
            self.selected_channels_sets.append(select_channels)

            temp_conv.weight.data.copy_(weight_data)
            if self.bias:
                temp_conv.bias.data.copy_(self.original_bias)

            conv_layers.append(temp_conv)

        self.conv_layers = nn.ModuleList(conv_layers)
        
    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, inputs):
        assert len(self.conv_layers) > 0, 'wrong init search module'

        outputs = None
        for i in range(len(self.conv_layers)):
            if outputs is None:
                # select channels of input 
                input_temp = inputs.index_select(
                    dim=1, index=self.selected_channels_sets[i])
                outputs = self.alpha[i] * self.conv_layers[i](input_temp)
            else:
                # select channels of input 
                input_temp = inputs.index_select(
                    dim=1, index=self.selected_channels_sets[i])
                outputs += self.alpha[i] * self.conv_layers[i](input_temp)

        return outputs
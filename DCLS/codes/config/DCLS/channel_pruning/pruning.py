import logging
import torch
import torch.nn as nn
from .common_module import MaskConv2d, MaskLinear, MaskModuleList
from models.modules.dcls_arch import DPCAB, Estimator
from models.modules.module_util import ResidualBlock_noBN


logger = logging.getLogger('channel_selection')


def get_select_channels(d):
    """
    Get select channels
    """

    select_channels = (d > 0).nonzero().squeeze()
    return select_channels


def get_thin_params(layer, select_channels, dim=0):
    """
    Get params from layers after pruning
    """

    if isinstance(layer, (nn.Conv2d, MaskConv2d)):
        if hasattr(layer, 'beta') and layer.beta is not None:
            layer.pruned_weight.data.mul_(layer.beta.unsqueeze(0).\
                unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))

        if isinstance(layer, MaskConv2d):
            layer.weight.data = layer.pruned_weight.clone().data
        
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    elif isinstance(layer, (nn.Linear, MaskLinear)):
        if hasattr(layer, 'beta') and layer.beta is not None:
            layer.pruned_weight.data.mul_(
                layer.beta.unsqueeze(0).expand_as(layer.pruned_weight))

        if isinstance(layer, (MaskLinear)):
            layer.weight.data = layer.pruned_weight.clone().data
            if layer.bias is not None and dim == 0:
                layer.bias.data = layer.pruned_bias.clone().data
        
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    elif isinstance(layer, nn.PReLU):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_bias = None
    
    else:
        raise NotImplementedError
    
    return thin_weight, thin_bias


def replace_layer(old_layer, init_weight, init_bias=None, keeping=False):
    """
    Replace specific layer of model
    :params layer: original layer
    :params init_weight: thin_weight
    :params init_bias: thin_bias
    :params keeping: whether to keep MaskConv2d
    """

    if hasattr(old_layer, "bias") and old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False
    if isinstance(old_layer, MaskConv2d) and keeping:
        new_layer = MaskConv2d(
            init_weight.size(1),
            init_weight.size(0),
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=bias_flag)

        new_layer.pruned_weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)
        new_layer.d.copy_(old_layer.d)
        if old_layer.beta is not None:
            new_layer.beta.copy_(old_layer.beta)
        new_layer.float_weight.data.copy_(old_layer.d)

    elif isinstance(old_layer, (nn.Conv2d, MaskConv2d)):
        if old_layer.groups != 1:
            new_groups = init_weight.size(0)
            in_channels = init_weight.size(0)
            out_channels = init_weight.size(0)
        else:
            new_groups = 1
            in_channels = init_weight.size(1)
            out_channels = init_weight.size(0)

        new_layer = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding,
                              bias=bias_flag,
                              groups=new_groups)

        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, (MaskLinear)) and keeping:
        new_layer = MaskLinear(
            init_weight.size(1),
            init_weight.size(0),
            bias=bias_flag)

        new_layer.pruned_weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)
        new_layer.d.copy_(old_layer.d)
        if old_layer.beta is not None:
            new_layer.beta.copy_(old_layer.beta)
        new_layer.float_weight.data.copy_(old_layer.d)

    elif isinstance(old_layer, (nn.Linear, MaskLinear)):
        in_features = init_weight.size(1)
        out_features = init_weight.size(0)

        new_layer = nn.Linear(in_features,
                              out_features,
                              bias=bias_flag)

        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.PReLU):
        new_layer = nn.PReLU(init_weight.size(0))
        new_layer.weight.data.copy_(init_weight)

    else:
        assert False, "unsupport layer type:" + \
                      str(type(old_layer))
    return new_layer


class BlockPrune(object):
    """
    RCAB block pruning
    """

    def __init__(self, block, block_type):
        self.block = block
        self.block_type = block_type
        self.select_channels = None

    def pruning(self):
        """
        Perform pruning
        """

        # prune model
        if self.block_type == "DPCAB":
            if self.block.body1[2].d.sum() == 0:
                logger.info("remove whole block")
                return None
            
            # compute selected channels
            select_channels = get_select_channels(self.block.body1[2].d)
            self.select_channels = select_channels

            # prune and replace in_channels of body1_2 (conv2d)
            thin_weight, thin_bias = get_thin_params(self.block.body1[2], select_channels, 1)
            self.block.body1[2] = replace_layer(self.block.body1[2], thin_weight, thin_bias)
            
            # prune and replace out_features of body1_0 (conv2d)
            thin_weight, thin_bias = get_thin_params(self.block.body1[0], select_channels, 0)
            self.block.body1[0] = replace_layer(self.block.body1[0], thin_weight, thin_bias)
            
            self.block.cuda()

        elif self.block_type == "ResidualBlock_noBN":
            if self.block.conv2.d.sum() == 0:
                logger.info("remove whole block")
                return None
            # compute selected channels
            select_channels = get_select_channels(self.block.conv2.d)
            self.select_channels = select_channels

            # prune and replace conv2 (linear)
            thin_weight, thin_bias = get_thin_params(
                self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(
                self.block.conv2, thin_weight, thin_bias)

            # prune and replace conv1 (linear)
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(
                self.block.conv1, thin_weight, thin_bias)
            
            self.block.cuda()
        
        elif self.block_type == "Estimator":
            if self.block.dec[-1].d.sum() == 0:
                logger.info("remove whole block")
                return None
            
            # compute selected channels 
            select_channels = get_select_channels(self.block.dec[-1].d)
            self.select_channels = select_channels

            # prune and replace the in_features of dec layer (linear)
            for idx, module in enumerate(self.block.dec):
                thin_weight, thin_bias = get_thin_params(
                    module, select_channels, 1)
                self.block.dec[idx] = replace_layer(
                    module, thin_weight, thin_bias)
            
            # prune and replace the out_channels 
            # of the last conv2d layer in tail module 
            thin_weight, thin_bias = get_thin_params(
                self.block.tail[-2], select_channels, 0)
            self.block.tail[-2] = replace_layer(
                self.block.tail[-2], thin_weight, thin_bias)
            
            self.block.cuda()
        else:
            assert False, "invalid block type: " + self.block_type



class DCLSPrune(object):
    """
    Prune SwinIR networks
    """

    def __init__(self, args, model, net_type='SwinIR'):
        self.args = args
        self.scale = args.scale
        self.model = model
        if isinstance(model, nn.DataParallel):
            self.model = model.module
        self.net_type = net_type
        logger.info("|===>Init DCLSPrune")
        
        self.block_prune = []

        # init prune module
        for module in self.model.modules():
            if isinstance(module, DPCAB):
                self.block_prune.append(
                    BlockPrune(module, block_type='DPCAB'))
            elif isinstance(module, ResidualBlock_noBN):
                self.block_prune.append(
                    BlockPrune(module, block_type='ResidualBlock_noBN'))
            elif isinstance(module, Estimator):
                self.block_prune.append(
                    BlockPrune(module, block_type='Estimator'))

    def run(self):
        """
        Perform pruning
        """

        for block in self.block_prune: 
            block.pruning()
        
        self.model.cuda()
        logger.info(self.model)
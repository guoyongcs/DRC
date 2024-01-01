import logging
import torch
import torch.nn as nn
from .common_module import MaskLinear, MaskOutLinear
from models.network_swinir_pruned import Mlp, WindowAttention

__all__ = ['DRNModelPrune', 'get_select_channels']

logger = logging.getLogger('channel_selection')


def get_select_channels(d):
    """
    Get select channels
    """

    select_channels = (d > 0).nonzero().squeeze(1)
    return select_channels


def get_thin_params(layer, select_channels, dim=0):
    """
    Get params from layers after pruning
    """

    if isinstance(layer, (nn.Linear, MaskLinear, MaskOutLinear)):
        if hasattr(layer, 'beta') and layer.beta is not None:
            layer.pruned_weight.data.mul_(layer.beta.unsqueeze(0).\
                unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))

        if isinstance(layer, (MaskLinear, MaskOutLinear)):
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
    if isinstance(old_layer, (MaskLinear, MaskOutLinear)) and keeping:
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

    elif isinstance(old_layer, (nn.Linear, MaskLinear, MaskOutLinear)):
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
        if self.block_type.lower() == "windowattention":
            if self.block.qkv.d.sum() == 0:
                logger.info("remove whole block")
                return None
            
            # compute selected channels
            select_channels = get_select_channels(self.block.proj.d)
            self.select_channels = select_channels

            # prune and replace in_features of proj (linear)
            thin_weight, thin_bias = get_thin_params(self.block.proj, select_channels, 1)
            self.block.proj = replace_layer(self.block.proj, thin_weight, thin_bias)
            
            # prune and replace out_features of qkv (linear)
            select_qkv_channels = get_select_channels(self.block.qkv.d)
            
            thin_weight, thin_bias = get_thin_params(self.block.qkv, select_qkv_channels, 0)
            self.block.qkv = replace_layer(self.block.qkv, thin_weight, thin_bias)
            
            self.block.cuda()

        elif self.block_type.lower() == "mlp":
            if self.block.fc2.d.sum() == 0:
                logger.info("remove whole block")
                return None
            # compute selected channels
            select_channels = get_select_channels(self.block.fc2.d)
            self.select_channels = select_channels

            # prune and replace fc2 (linear)
            thin_weight, thin_bias = get_thin_params(
                self.block.fc2, select_channels, 1)
            self.block.fc2 = replace_layer(
                self.block.fc2, thin_weight, thin_bias)

            # prune and replace fc1 (linear)
            thin_weight, thin_bias = get_thin_params(
                self.block.fc1, select_channels, 0)
            self.block.fc1 = replace_layer(
                self.block.fc1, thin_weight, thin_bias)
            
            self.block.cuda()
        else:
            assert False, "invalid block type: " + self.block_type


class layerPrune(object):
    """
    Sequantial pruning
    """

    def __init__(self, layer, module_type='RSTB'):
        self.layer = layer
        self.module_type = module_type
        self.layer_length = layer.residual_group.depth
        self.block_num = 0
        self.block_prune = []

        for i in range(self.layer_length):
            for m in layer.residual_group.blocks[i].modules():
                if isinstance(m, WindowAttention):
                    self.block_prune.append(
                        BlockPrune(m, block_type='WindowAttention'))
                    self.block_num += 1
                if isinstance(m, Mlp):
                    self.block_prune.append(
                        BlockPrune(m, block_type='Mlp'))
                    self.block_num += 1

    def pruning(self):
        """
        Perform pruning
        """
        for i in range(self.block_num):
            self.block_prune[i].pruning()


class SwinIRPrune(object):
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
        logger.info("|===>Init SwinIRPrune")
        
        self.block_prune = []

        # init prune module
        for layer in self.model.layers:
            self.block_prune.append(
                layerPrune(layer, module_type='RSTB'))

    def run(self):
        """
        Perform pruning
        """

        for block in self.block_prune: 
            block.pruning()
        
        self.model.cuda()
        logger.info(self.model)
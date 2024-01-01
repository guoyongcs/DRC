import logging
import torch
import torch.nn as nn
from model.common import MaskConv2d, RCAB, ResBlock

__all__ = ['DRNModelPrune', 'get_select_channels']

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

    elif isinstance(old_layer, nn.PReLU):
        new_layer = nn.PReLU(init_weight.size(0))
        new_layer.weight.data.copy_(init_weight)

    else:
        assert False, "unsupport layer type:" + \
                      str(type(old_layer))
    return new_layer


class UpsamplerPrune(object):
    """
    Upsampler pruning
    """

    def __init__(self, block, block_type="upsampler"):
        self.block = block
        self.block_type = block_type
        self.select_channels = None

    def pruning(self):
        """
        Perform pruning
        """

        # prune model
        if self.block_type == "upsampler":
            # compute selected channels for block1 (conv1)
            select_channels = get_select_channels(self.block[1].d)
            self.select_channels = select_channels
            
            # compute selected channels for block0 (conv)
            d = torch.stack([self.block[1].d] * 4, dim=0)
            d = d.transpose(0,1).flatten()
            block0_select_channels = get_select_channels(d)
            
            # prune and replace block1 (conv)
            thin_weight, thin_bias = get_thin_params(
                                        self.block[1], select_channels, 1)
            self.block[1] = replace_layer(
                                self.block[1], thin_weight, thin_bias)

            # prune and replace block0 (conv)
            thin_weight, thin_bias = get_thin_params(
                                        self.block[0][0], block0_select_channels, 0)
            self.block[0][0] = replace_layer(
                                    self.block[0][0], thin_weight, thin_bias)
            
            self.block = list(self.block.cuda())
        else:
            assert False, "invalid block type: " + self.block_type


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
        if self.block_type == "rcab" or self.block_type == "resblock":
            if self.block.body[2].d.sum() == 0:
                logger.info("remove whole block")
                return None
            # compute selected channels
            select_channels = get_select_channels(self.block.body[2].d)
            self.select_channels = select_channels

            # prune and replace body2 (conv)
            thin_weight, thin_bias = get_thin_params(self.block.body[2], select_channels, 1)
            self.block.body[2] = replace_layer(self.block.body[2], thin_weight, thin_bias)

            # prune and replace body0 (conv)
            thin_weight, thin_bias = get_thin_params(self.block.body[0], select_channels, 0)
            self.block.body[0] = replace_layer(self.block.body[0], thin_weight, thin_bias)
            
            self.block.cuda()

        elif self.block_type == "downblock":
            if self.block.dual_module[1].d.sum() == 0:
                logger.info("remove whole block")
                return None
            # compute selected channels
            select_channels = get_select_channels(self.block.dual_module[1].d)
            self.select_channels = select_channels

            # prune and replace module1 (conv)
            thin_weight, thin_bias = get_thin_params(
                self.block.dual_module[1], select_channels, 1)
            self.block.dual_module[1] = replace_layer(
                self.block.dual_module[1], thin_weight, thin_bias)

            # prune and replace module00 (conv)
            thin_weight, thin_bias = get_thin_params(
                self.block.dual_module[0][0], select_channels, 0)
            self.block.dual_module[0][0] = replace_layer(
                self.block.dual_module[0][0], thin_weight, thin_bias)
            
            self.block.cuda()
        else:
            assert False, "invalid block type: " + self.block_type


class SeqPrune(object):
    """
    Sequantial pruning
    """

    def __init__(self, opt, sequential, seq_type):
        self.opt = opt
        self.sequential = sequential
        self.sequential_length = len(list(self.sequential))
        self.block_num = 0
        self.block_prune = []

        for i in range(self.sequential_length):
            if isinstance(self.sequential[i], (RCAB, ResBlock)):
                self.block_prune.append(
                    BlockPrune(self.sequential[i], block_type=seq_type))
                self.block_num += 1
        
        if self.opt.model.lower().find('drnt_res') < 0:
            self.block_prune.append(
                UpsamplerPrune(self.sequential[self.block_num:]))
        else:
            self.sequential_length -= 1

    def pruning(self):
        """
        Perform pruning
        """

        for i in range(self.sequential_length - 1):
            self.block_prune[i].pruning()

        temp_seq = []
        for i in range(self.block_num):
            if self.block_prune[i].block is not None:
                temp_seq.append(self.block_prune[i].block)
        
        if self.opt.model.lower().find('drnt_res') < 0:
            temp_seq.extend(self.block_prune[self.block_num].block)
        else:
            temp_seq.extend(self.sequential[self.block_num:])
        # for i in range(self.block_num, self.sequential_length):
        #     temp_seq.append(self.sequential[i])
            
        self.sequential = nn.Sequential(*temp_seq)


class DRNModelPrune(object):
    """
    Prune drn networks
    """

    def __init__(self, opt, model, net_type):
        self.opt = opt
        self.scale = opt.scale
        self.model = model
        self.net_type = net_type
        logger.info("|===>Init DRNModelPrune")

    def run(self):
        """
        Perform pruning
        """

        if self.net_type.lower() in ["attsrunet", "drn"] or \
                    self.net_type.lower().find('drnt_res') >= 0:
            down_block_prune = []
            up_seq_prune = []
            for i in range(len(self.scale)):
                if self.opt.model.lower().find('drnt_res') < 0:
                    down_block_prune.append(
                        BlockPrune(self.model.down[i], block_type='downblock'))
                up_seq_prune.append(
                    SeqPrune(self.opt, self.model.up_blocks[i], seq_type='rcab')) # or seq_type='resblock'

            for i in range(len(self.scale)):
                if self.opt.model.lower().find('drnt_res') < 0:
                    down_block_prune[i].pruning()
                    self.model.down[i] = down_block_prune[i].block
                up_seq_prune[i].pruning()
                self.model.up_blocks[i] = up_seq_prune[i].sequential
            
            self.model.cuda()
            logger.info(self.model)

        else:
            assert False, "unsupport model pruning: {}".format(self.opt.model)

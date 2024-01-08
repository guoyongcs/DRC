""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
import logging
from .search_model import SearchModel

PRIMITIVES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

class SearchController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, args, model, criterion, logger):
        super().__init__()
        self.n_layers = model.count_layers() # corresponding to n_nodes in drats
        self.criterion = criterion
        self.args = args
        self.logger = logger
        self.search_model = SearchModel(args, model, logger)
        # initialize architect parameters: alphas
        # alpha is the probability of the number of searched channels
        n_ops = len(PRIMITIVES)
        self.alphas_all = nn.ParameterList()
        self.channel_sets = []
        for i in range(self.n_layers):
            self.alphas_all.append(nn.Parameter(1e-3*torch.randn(n_ops)))
            self.channel_sets.append(
                self.init_channel_options(self.search_model.get_layers(i).in_channels))
        
        if self.args.search_multi_branch:
            self.search_model.init_search_modules(self.channel_sets, self.alphas_all)

        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

    def init_channel_options(self, channels_num):
        channels_options = []
        for op in PRIMITIVES:
            channels_option = int(channels_num * (1 - self.args.pruning_rate)  * op)
            channels_option = self.channel_constraints(channels_option, channels_num)
            channels_options.append(channels_option)
        return channels_options
        
    def forward(self, lr):
        return self.search_model(lr, self.alphas_all, self.channel_sets)

    def loss(self, lr, hr):
        if not isinstance(lr , list): lr = [lr]
        sr, sr2lr = self.forward(lr[0])
        return self.compute_loss(lr, hr, sr, sr2lr)

    def compute_loss(self, lr, hr, sr, sr2lr=None):
        total_loss = 0
        # compute primal regression loss
        if not isinstance(sr, list): sr = [sr]
        loss_primary = self.criterion(sr[-1], hr)
        for i in range(1, len(sr)):
            loss_primary += self.criterion(
                sr[i - 1 - len(sr)], lr[i - len(sr)])
        
        total_loss += loss_primary
        
        # compute dual regression loss
        if self.args.dual:
            loss_dual = self.criterion(sr2lr[0], lr[0])
            for i in range(1, len(self.args.scale)):
                loss_dual += self.criterion(
                    sr2lr[i - len(self.args.scale)], lr[i - len(self.args.scale)])
            
            total_loss += self.args.dual_weight * loss_dual

        return total_loss

    def print_alphas(self):
        # remove formats
        org_formatters = []
        for handler in self.logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        self.logger.info("####### ALPHA #######")
        for idx, alpha in enumerate(self.alphas_all):
            self.logger.info("layer: {}, alpha: {}".format(
                            idx, F.softmax(alpha.detach(), dim=-1)))

        # restore formats
        for handler, formatter in zip(self.logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def channel_constraints(self, channels_num, max_num, factor=4):
        # assure the numbers of channels is the multiple of 4 \
        # to speed up model with NC4HW4 format in mobilephone
        remainder = channels_num % factor
        if remainder * 2 > factor:
            channels_num += factor - remainder
        else:
            channels_num -= remainder
        # channels_num should be in the range of [factor, max_num]
        channels_num = min(max(channels_num, factor), max_num)
        return channels_num

    def weights(self):
        for n, p in self.search_model.named_parameters():
            if n.find('alpha') < 0:
                yield p

    def named_weights(self):
        for n, p in self.search_model.named_parameters():
            if n.find('alpha') < 0:
                yield n, p

    def get_alphas(self):
        return self.alphas_all
    
    def set_alphas(self, alphas):
        for (n, p), (ni, pi) in zip(self._alphas, alphas.items()):
            p.data.copy_(pi)
    
    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
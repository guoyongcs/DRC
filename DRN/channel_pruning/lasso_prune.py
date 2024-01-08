"""
channel pruning method based on LASSO
"""
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Lasso
from utils import get_logger




class LassoFilterPrune(object):
    """
    re-implementation of Channel pruning for accelerating very deep nerual networks
    """

    def __init__(self, opt, locations=10, prune_ratio=0.3):
        """
        filter level pruning
        :params locations: <int> number of locations sampling
        from output features of every input image.

        :params prune_ratio: <float> [0, 1), percentage of preserved channels
        """
        self.locations = locations
        self.prune_ratio = prune_ratio
        self.y_cache = None
        self.x_cache = None
        self.logger = get_logger(opt.save, 'lasso')

    def feature_extract(self, x, y, layer):
        """
        :params x: input feature
        :params y: output feature
        :params layer: pruned layer
        """
        assert isinstance(
            layer, nn.Conv2d), "unsupported layer type: " + str(type(layer))
        # padding
        padding_size = layer.padding
        padding_layer = nn.ZeroPad2d(
            (padding_size[1], padding_size[1], padding_size[0], padding_size[0]))
        x = padding_layer(x).data

        # generate random location of y
        y_h = torch.LongTensor(np.random.randint(
            y.size(2), size=self.locations * y.size(0))).cuda()
        y_w = torch.LongTensor(np.random.randint(
            y.size(3), size=self.locations * y.size(0))).cuda()

        # compute locations of x according to y
        x_h = y_h * layer.stride[0]
        x_w = y_w * layer.stride[1]

        # compute x of every channel
        temp_x_cache = tuple()
        temp_y_cache = tuple()

        # extract input features
        for i in range(y_h.size(0)):
            x_select = x[int(i / self.locations), :, x_h[i]:x_h[i] + layer.kernel_size[0],
                       x_w[i]:x_w[i] + layer.kernel_size[1]].unsqueeze(0)
            temp_x_cache = temp_x_cache + (x_select.cpu(),)

            y_select = y.data[int(i / self.locations),
                       :, y_h[i], y_w[i]].unsqueeze(0)
            temp_y_cache = temp_y_cache + (y_select.cpu(),)

        temp_x_cache = torch.cat(temp_x_cache, 0)
        temp_y_cache = torch.cat(temp_y_cache, 0)
        
        # add y to cache
        if self.y_cache is None:
            self.y_cache = temp_y_cache
        else:
            self.y_cache = torch.cat(
                (self.y_cache, temp_y_cache), 0)

        # add results to a larger cache
        if self.x_cache is None:
            self.x_cache = temp_x_cache
        else:
            self.x_cache = torch.cat(
                (self.x_cache, temp_x_cache), 0)

    def channel_select(self, layer, init_alpha=200.):
        """
        select channels according to value of x_hat
        :params layer: pruned layer
        :return beta, d: channel scale vector and channel selection vector
        """
        assert isinstance(
            layer, nn.Conv2d), "unsupported layer type: " + str(type(layer))

        in_channels = layer.in_channels
        out_channels = layer.out_channels
        n_samples = self.x_cache.size(0)

        # transform w: n c h w --> n c hw --> c n hw  --> c hw n
        weight = layer.weight.data.view(out_channels, in_channels, -1)
        weight = torch.transpose(torch.transpose(weight, 0, 1), 1, 2).cpu().numpy()
        
        # transform x: N c h w --> N c hw --> c N hw
        x = self.x_cache.view(n_samples, in_channels, -1)
        x = torch.transpose(x, 0, 1).numpy()

        # compute z: c N n --> c Nn --> Nn c
        z = np.matmul(x, weight).reshape((in_channels, -1)).T

        # transform y: N n --> Nn
        y = self.y_cache.view(-1).numpy()
        def _lasso_solver(left=1e-4, right=200.):
            alpha = left
            solver = Lasso(alpha=alpha, warm_start=False, selection='cyclic', precompute=True)
            return solver, left, right
        
        left = 0
        right = 2 * init_alpha
        solver, _, _ = _lasso_solver(left, right)
        alpha = left
        select_num = in_channels - \
                     int(math.floor(in_channels * self.prune_ratio))
        round_count = 0
        while True:
            if round_count >= 50:
                round_count = 0
                solver, left, right = _lasso_solver(left=alpha, right=alpha*2)
                self.logger.info("reset solver!!!")
                self.logger.info("reset alpha: {}".format((left + right) / 2.))
            alpha = (left + right) / 2.
            solver.alpha = alpha
            solver.fit(z, y)
            
            idxs = solver.coef_ != 0.
            beta = solver.coef_
            
            non_zero_num = sum(idxs)
            if non_zero_num > select_num:
                left = alpha
                self.logger.info("update left and alpha: {} {} {}".\
                                format(alpha, non_zero_num, select_num))
            elif non_zero_num < select_num:
                right = alpha
                self.logger.info("update right and alpha: {} {} {}".\
                                format(alpha, non_zero_num, select_num))
            else:
                break
            
            round_count += 1
            

        d = idxs + 1. - 1.
        remove_channels = np.where(d == 0.)[0]
        remove_channels = torch.LongTensor(remove_channels).cuda().long()
        select_channels = np.where(d == 1.)[0]
        select_channels = torch.LongTensor(select_channels).cuda().long()
        d = torch.Tensor(d).cuda()
        beta = torch.Tensor(beta).cuda()

        # transform x: N c h w --> N c hw
        x = self.x_cache.view(n_samples, in_channels, -1) * \
            beta.cpu().unsqueeze(0).unsqueeze(2)

        # select channels: N c hw --> N c' hw
        x = x.index_select(1, select_channels.cpu())
        # N c'hw
        x = x.view(n_samples, -1)
        # transform y: N n
        y = self.y_cache

        weight = torch.torch.mm(
            torch.mm(torch.mm(x.transpose(0, 1).cuda(), x.cuda()).inverse(), x.transpose(0, 1).cuda()),
            y.cuda()).squeeze()
        
        # w_norm: c'hw n --> c' hwn --> c'
        w_norm = weight.view(select_channels.size(0), -1).norm(2, 1)

        # n c' h w
        weight = weight.transpose(0, 1).contiguous().view(
            out_channels, select_channels.size(0), 
            layer.weight.size(2), layer.weight.size(3)
        ).div(w_norm.unsqueeze(0).unsqueeze(2).unsqueeze(3))

        for i in range(select_channels.size(0)):
            beta[select_channels[i]] = beta[select_channels[i]] * w_norm[i]
            layer.pruned_weight.data[:, select_channels[i], ...].copy_(
                weight[:, i, ...])

        layer.pruned_weight.data.index_fill_(1, remove_channels, 0)
        if not layer.pruned_weight.data.is_contiguous():
            layer.pruned_weight.data.contiguous()

        return beta, d

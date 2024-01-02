import math

import numpy as np
import torch
import torch.nn as nn


# re-implementation of ThiNet: a filter level pruning method for convolutional network compression


class ThinetFilterPrune(object):
    def __init__(self, locations=10, prune_ratio=0.3):
        """
        filter level pruning
        :params h: <int> height of feature maps. We always reshape the inputs of linear layer into 2D matrix which
        leads to changing of channels of inputs, so we need to define the height and weight of feature maps for linear layer
        :params w: <int> width of feature maps.
        :params locations: <int> number of locations sampling from output features of every input image.
        :params prune_ratio: <float> [0, 1), percentage of preserved channels
        """
        self.xhat_cache = None
        self.locations = locations
        self.prune_ratio = prune_ratio
        self.y_cache = None

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
        y_d = torch.LongTensor(np.random.randint(
            y.size(1), size=self.locations * y.size(0))).cuda()
        y_h = torch.LongTensor(np.random.randint(
            y.size(2), size=self.locations * y.size(0))).cuda()
        y_w = torch.LongTensor(np.random.randint(
            y.size(3), size=self.locations * y.size(0))).cuda()

        # select weight according to y
        w_select = layer.weight.data[y_d]

        # compute locations of x according to y
        x_h = y_h * layer.stride[0]
        x_w = y_w * layer.stride[1]

        # compute x of every channel
        temp_xhat_cache = tuple()
        temp_y_cache = []

        x_n = torch.LongTensor(
            np.arange(y_h.size(0)) / self.locations).cuda()

        for i in range(y_h.size(0)):
            x_select = x[int(i / self.locations), :, x_h[i]:x_h[i] + layer.kernel_size[0],
                       x_w[i]:x_w[i] + layer.kernel_size[1]].unsqueeze(0)

            temp_xhat_cache = temp_xhat_cache + (x_select,)

        temp_xhat_cache = torch.cat(temp_xhat_cache, 0)
        temp_xhat_cache = (temp_xhat_cache * w_select).sum(2).sum(2)

        temp_y_cache = y.data[x_n, y_d, y_h, y_w]

        # add y to cache
        if self.y_cache is None:
            self.y_cache = temp_y_cache
        else:
            self.y_cache = torch.cat(
                (self.y_cache, temp_y_cache), 0)

        # add results to a larger cache
        if self.xhat_cache is None:
            self.xhat_cache = temp_xhat_cache
        else:
            self.xhat_cache = torch.cat(
                (self.xhat_cache, temp_xhat_cache), 0)

    def least_square(self, layer, d):
        """
        select channels according to value of x_hat
        :params layer: pruned layer
        :return w_hat: scalar of weights
        """
        assert isinstance(
            layer, nn.Conv2d), "unsupported layer type: " + str(type(layer))

        channels = layer.in_channels
        remove_channels = np.where(d == 0.)[0]

        remove_channels = torch.LongTensor(remove_channels).cuda().long()
        select_channels = np.where(d == 1.)[0]
        select_channels = torch.LongTensor(select_channels).cuda().long()

        # compute scale
        x = self.xhat_cache.index_select(1, select_channels)
        x = x.view(x.size(0), -1)
        y = self.y_cache.unsqueeze(1)

        w_hat = torch.mm(
            torch.mm(torch.mm(x.transpose(0, 1), x).inverse(), x.transpose(0, 1)), y).squeeze()

        d = torch.zeros(channels).cuda().index_fill_(0, select_channels, 1)
        beta = torch.zeros(channels).cuda()
        for i in range(w_hat.size(0)):
            beta[select_channels[i]] = w_hat[i]

        return beta, d

    def channel_select(self, layer):
        """
        select channels according to value of x_hat
        :params layer: pruned layer
        :return w_hat: scalar of weights
        """
        assert isinstance(
            layer, nn.Conv2d), "unsupported layer type: " + str(type(layer))

        channels = layer.in_channels
        # init I and T: I is the set of all channels, T is the set of removed channels
        I = list(range(channels))
        T = []

        sum_cache = None
        for c in range(int(math.floor(channels * self.prune_ratio))):
            min_value = None
            # print len(I)
            select_value = None
            for i in I:
                tempT = T[:]
                tempT.append(i)
                tempT = torch.LongTensor(tempT).cuda()

                temp_value = self.xhat_cache.index_select(
                    1, torch.LongTensor([i]).cuda())
                if sum_cache is None:
                    value = temp_value.abs().sum()
                else:
                    value = (sum_cache + temp_value).abs().sum()

                if min_value is None or min_value > value:
                    select_value = temp_value
                    min_value = value
                    min_i = i
            if sum_cache is None:
                sum_cache = select_value
            else:
                sum_cache += select_value

            I.remove(min_i)
            T.append(min_i)

        S = list(range(channels))
        for c in T:
            if c in S:
                S.remove(c)
        select_channels = torch.LongTensor(S).cuda()

        # compute scale
        x = self.xhat_cache.index_select(1, select_channels)
        x = x.view(x.size(0), -1)
        y = self.y_cache.unsqueeze(1)

        try:
            w_hat = torch.mm(
                torch.mm(torch.mm(x.transpose(0, 1), x).inverse(), x.transpose(0, 1)), y).squeeze()
        except:
            w_hat = torch.ones(select_channels.size(0))

        d = torch.zeros(channels).cuda().index_fill_(0, select_channels, 1)
        beta = torch.zeros(channels).cuda()
        for i in range(w_hat.size(0)):
            beta[select_channels[i]] = w_hat[i]

        return beta, d

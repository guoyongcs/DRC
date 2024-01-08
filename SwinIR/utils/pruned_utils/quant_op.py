# -*- coding: utf-8 -*-

# BUG1989 is pleased to support the open source community by supporting ncnn available.
#
# Copyright (C) 2019 BUG1989. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os
import copy
import numpy as np
from scipy import stats
import torch 
from torch.autograd import Function

def add_pshadow(optimizer):
  #print('quantize....add_pshadow')
  for group in optimizer.param_groups:
    group['pshadow'] = []  
    for p in group['params']:
      group['pshadow'].append(p.clone())


def copy_pshadow2param(optimizer):
  # print('quantize....copy_pshadow2param')
  for group in optimizer.param_groups:
    shadow = group['pshadow']
    ps = group['params']
    for s, p in zip(shadow, ps):
      p.data[:] = s.data


def copy_param2pshadow(optimizer):
  # print('quantize....copy_param2pshadow')
  for group in optimizer.param_groups:
    shadow = group['pshadow']
    ps = group['params']
    for s, p in zip(shadow, ps):
      s.data[:] = p.data


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    #import ipdb;ipdb.set_trace()
    assert (hist <= 0).sum() == 0
    return hist


def quantize_data(param, intlen=0):
    data_ = param.data
    quant_scale = 127.0 / intlen
    dequant_scale = 1.0 / quant_scale
    data_.mul_(quant_scale).round_().clamp_(-127, 127)
    data_.mul_(dequant_scale)
    return data_


# group quantization
def quantize_weight(param):
    for i in range(param.shape[0]):
        data_ = param[i].data
        maxv = data_.abs().max()
        quant_scale = 127.0 / maxv
        dequant_scale = 1.0 / quant_scale
        param[i].data.mul_(quant_scale.cuda()).round_().clamp_(-127, 127)
        param[i].data.mul_(dequant_scale.cuda())


def quantize_params(optimizer):
    
    for group in optimizer.param_groups:
        ps = group['params']
        for p in ps:
            quantize_weight(p)
    

def get_best_activation_intlength(hist, hist_edges, target_bin=128):
    """
    Return the best threshold value. 
    Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    Args:
        hist: list, activations has been processed by histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL 
    """   
    hist = hist[1:]
    length = hist.size
    threshold_sum = sum(hist[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(hist[:threshold])

        # generate reference hist p
        p = sliced_nd_hist.copy()
        p[threshold-1] += threshold_sum
        threshold_sum = threshold_sum - hist[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        # 
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate quantized hist q
        num_merged_bins = sliced_nd_hist.size // target_bin
        
        # merge hist into num_quantized_bins bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
        
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        
        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = hist_edges[min_kl_divergence + target_bin]

    return threshold_value


def quantization_on_weights(x, quantize_num=127.0):
    c_out, c_in, k, w = x.shape
    x_reshape = x.reshape(c_out, -1)
    # threshold shape=c_out*1
    threshold, _ = torch.max(torch.abs(x_reshape), 1)
    weight_scale = quantize_num / threshold
    weight_scale = weight_scale.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)

    x = x*weight_scale
    x = RoundFunction.apply(x)
    x = torch.clamp(x, min=-127, max=127)
    x = x / weight_scale
    return x


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantization_on_input(x, quantize_num=127.0):
    threshold = torch.max(torch.abs(x))  # threshold shape=1
    activations_scale = quantize_num / threshold
    x = x*activations_scale
    x = RoundFunction.apply(x)
    x = torch.clamp(x, min=-127, max=127)
    x = x / activations_scale
    return x


def quantization_on_input_fix_scale(x, activation_value, quantize_num=127.0):
    threshold = activation_value
    activations_scale = quantize_num / threshold
    x = x*activations_scale
    x = RoundFunction.apply(x)
    x = torch.clamp(x, min=-127, max=127)
    x = x / activations_scale
    return x


def get_activation_value(table_path):
    activation_value_list = []
    for line in open(table_path):
        line = line.strip()
        key, value = line.split(' ')

        # remove 'net.'
        if "net" in key:
            key = key.replace('net.', '')
        activation_value_list.append(value)

    return activation_value_list


def replace_weight_name(sr_scale, logger, weight_path, calibration_path):
    count = 1
    calibration_file = open(calibration_path, 'a+')
    for line in open(weight_path, 'r'):
        line = line.strip()
        title = 'conv{}_param_0'.format(count)
        str_arr = line.split(' ', maxsplit=1)
        if str_arr[0].find('tail') >= 0 and \
                int(str_arr[0][-1]) < len(sr_scale):
            continue

        logger.info('{}-->{}'.format(str_arr[0], title))
        new_line = '{} {}\n'.format(title, str_arr[-1])
        calibration_file.write(new_line)
        count += 1
    calibration_file.close()


def replace_activation_name(sr_scale, logger, 
            activation_path, calibration_path, quantize_num=127.0):
    count = 1
    calibration_file = open(calibration_path, 'a+')
    for line in open(activation_path, 'r'):
        line = line.strip()
        title = 'conv{}'.format(count)
        str_arr = line.split(' ', maxsplit=1)
        if str_arr[0].find('tail') >= 0 and \
                int(str_arr[0][-1]) < len(sr_scale):
            continue

        logger.info('{}-->{}'.format(str_arr[0], title))

        threshold = float(str_arr[-1])
        scale = quantize_num / threshold
        new_line = '{} {}\n'.format(title, str(scale))
        calibration_file.write(new_line)
        count += 1
    calibration_file.close()
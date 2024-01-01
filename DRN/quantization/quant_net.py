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

import torch.nn as nn
import sys

import numpy as np
import _pickle as cPickle
from collections import *

import utils.quant_op as qp

# quant_table show the set when quantizing('quant' in phase)
# 'quant': whether it already has a quant_table
# 'test': it is used when eval the net
quant_table = OrderedDict({'quant': True, 'test': True, 'quant_bp_coef': 2})
quant_hist = OrderedDict()


class ConvQuant(nn.Module):
    def __init__(self, conv, phase='quant', name=''):
        super(ConvQuant, self).__init__()

        self.name = name
        self.add_module('conv', conv)
        if 'quant' in phase:
            self.fw_hook= self.conv.register_forward_pre_hook(self.pre_hook) 


    def remove_hooks(self):
        if self.fw_hook != 0:
            self.fw_hook.remove()


    def pre_hook(self, mdl, datain):
        if quant_table['quant']:
            # quantize input data inplace
            qp.quantize_data(datain[0], intlen=quant_table[self.name])
        else:
            if quant_hist['step'] == 1:
                if self.name not in quant_hist.keys():
                    quant_hist[self.name] = {'max_data': None}
                quant_hist[self.name]['max_data'] = datain[0].data.abs().max()

            if quant_hist['step'] == 2:
                th = quant_hist[self.name]['max_data'].cpu()
                th_cp = th.cpu().numpy()
                hist, hist_edges = np.histogram(
                    datain[0].data.cpu().numpy(), bins=8192, range=(0, th_cp))
                if 'hist' in quant_hist[self.name].keys():
                    quant_hist[self.name]['hist'] += hist
                else:
                    quant_hist[self.name]['hist'] = hist
                    quant_hist[self.name]['hist_edges'] = hist_edges

            if quant_hist['step'] == 3: # original test
                pass

            if quant_hist['step'] == 4:
                qp.quantize_data(datain[0], intlen=quant_table[self.name])


    def forward(self, x):
        out= self.conv(x)
        return out


def save_params(optimizer):
    qp.add_pshadow(optimizer)


def load_params(optimizer):
    qp.copy_pshadow2param(optimizer)

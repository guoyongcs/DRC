import os
import sys
import logging
import torch
import math
import time
import numpy as np
import datetime
from .model_analyse import *
from .model_transform import *
from .tensorboard_logger import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset all parameters
        """

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update parameters
        """

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def get_logger(save_path, logger_name, log_file="experiment.log"):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, log_file))
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger

def write_log(dir_name, file_name, log_str):
    """
    Write log to file
    :param dir_name:  the path of directory
    :param file_name: the name of the saved file
    :param log_str: the string that need to be saved
    """

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    with open(os.path.join(dir_name, file_name), "a+") as f:
        f.write(log_str)

def concat_gpu_data(data):
    """
    Concat gpu data from different gpu.
    """

    data_cat = data["0"]
    for i in range(1, len(data)):
        data_cat = torch.cat((data_cat, data[str(i)].cuda(0)))
    return data_cat

def concat_gpu_datalist(data_list):
    """
    Concat gpu data list from different gpu.
    """
    # data_cat is instance of list
    data_cat = data_list["0"]
    for j in range(len(data_cat)):
        for i in range(1, len(data_list)):
            data_cat[j] = torch.cat((data_cat[j], data_list[str(i)][j].cuda(0)))
    return data_cat

def print_result(epoch, nEpochs, count, iters, lr, data_time, iter_time, psnr, mode="Train", logger=None):
    log_str = "{}: [{:0>3d}|{:0>3d}], Iter: [{:0>3d}|{:0>3d}], Learning rate: {:.6f}, ".format(
        mode, epoch + 1, nEpochs, count, iters, lr, data_time, iter_time)
    if isinstance(psnr, list) or isinstance(psnr, np.ndarray):
        for i in range(len(psnr)):
            log_str += "PSNR_{:d}: {:.4f}\t".format(i, psnr[i])
    else:
        log_str += "PSNR: {:.4f}\t".format(psnr)


    time_str, total_time, left_time = compute_remain_time(epoch, nEpochs, count, iters, data_time, iter_time, mode)

    logger.info(log_str + time_str)

    return total_time, left_time

## gobal variant
single_train_time = 0
single_test_time = 0
single_train_iters = 0
single_test_iters = 0


def compute_remain_time(epoch, nEpochs, count, iters, data_time, iter_time, mode="Train"):
    global single_train_time, single_test_time
    global single_train_iters, single_test_iters

    # compute cost time
    if mode == "Train":
        single_train_time = single_train_time * \
                            0.95 + 0.05 * (data_time + iter_time)
        # single_train_time = data_time + iter_time
        single_train_iters = iters
        train_left_iter = single_train_iters - count + \
                          (nEpochs - epoch - 1) * single_train_iters
        # print "train_left_iters", train_left_iter
        test_left_iter = (nEpochs - epoch) * single_test_iters
    else:
        single_test_time = single_test_time * \
                           0.95 + 0.05 * (data_time + iter_time)
        # single_test_time = data_time+iter_time
        single_test_iters = iters
        train_left_iter = (nEpochs - epoch - 1) * single_train_iters
        test_left_iter = single_test_iters - count + \
                         (nEpochs - epoch - 1) * single_test_iters

    left_time = single_train_time * train_left_iter + \
                single_test_time * test_left_iter
    total_time = (single_train_time * single_train_iters +
                  single_test_time * single_test_iters) * nEpochs
    time_str = "DataTime: {:.4f}, IterTime: {:.4f}, TTime: {}, RTime: {}".format(
        data_time, iter_time,
        datetime.timedelta(seconds=total_time),
        datetime.timedelta(seconds=left_time))
    return time_str, total_time, left_time

def remain_time(epoch_time, current_epoch, total_epoch):
    remain_epochs = total_epoch - current_epoch
    log = 'Epoch Time: {} m {} s  '.format(
        epoch_time // 60, int(epoch_time) % 60)
    log += 'Remain Time: {} h {} m  '.format(
        epoch_time * remain_epochs // 3600, 
        epoch_time * remain_epochs // 60 % 60)
    log += 'Total Time: {} h {} m \n'.format(
        epoch_time * total_epoch // 3600, 
        epoch_time * total_epoch // 60 % 60)
    return log


def calc_psnr(sr, hr, scale, rgb_range=255, benchmark=True):
    if isinstance(sr, np.ndarray):
        sr = torch.from_numpy(sr).permute(2, 0, 1).float().unsqueeze(0)
    if isinstance(hr, np.ndarray):
        hr = torch.from_numpy(hr).permute(2, 0, 1).float().unsqueeze(0)

    if sr.size(-2) > hr.size(-2) or sr.size(-1) > hr.size(-1):
        print("the dimention of sr image is not equal to hr's! ")
        sr = sr[:, :, :hr.size(-2), :hr.size(-1)]
    diff = (sr - hr).data.div(rgb_range)

    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
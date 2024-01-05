import os
import sys
import logging
import torch
import numpy as np
import datetime
from .model_analyse import *
from .model_transform import *
from .tensorboard_logger import *
import utils.quant_op as quant_op


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


def get_logger(save_path, logger_name, log_file="experiment.log"):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, log_file))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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
    log_str = "{}: [{:0>3d}|{:0>3d}], Iter: [{:0>3d}|{:0>3d}], LR: {:.6f}, DataTime: {:.4f}, IterTime: {:.4f}, ".format(
        mode, epoch + 1, nEpochs, count, iters, lr, data_time, iter_time)
    if isinstance(psnr, list) or isinstance(psnr, np.ndarray):
        for i in range(len(psnr)):
            log_str += "PSNR_{:d}: {:.4f} ".format(i, psnr[i])
    else:
        log_str += "PSNR: {:.4f}".format(psnr)


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
    time_str = "TTime: {}, RTime: {}".format(datetime.timedelta(seconds=total_time),
                                             datetime.timedelta(seconds=left_time))
    return time_str, total_time, left_time


import argparse
import cv2
import glob
import random
import math
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import requests

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_image as util
from utils import utils_option as option
from utils import utils_logger
from channel_search import Search

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, help='Path to option JSON file.', 
                        default='options/swinir/train_swinir_sr_lightweight_search.json')
    parser.add_argument('--save_dir', type=str, default=None, help='patch to save results')
    args = parser.parse_args()

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    opt = option.parse(args.opt_path, is_train=True)
    # return None for missing key
    args.opt = option.dict_to_nonedict(opt)
    args.scale = opt['scale']
    if args.save_dir is None:
        args.save_dir = opt['path']['save_dir']
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set random seed
    seed = args.opt['train']['manual_seed']
    if seed is None: seed = 0 # random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # init logger
    logger_name = 'search'
    utils_logger.logger_info(logger_name, os.path.join(args.save_dir, logger_name + '_option.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(args.opt))

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in args.opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # set up model       
    model = define_Model(args.opt)
    model.init_train()
    
    # channel pruning
    experiment = Search(args, model, train_loader, test_loader)
    experiment.search()

if __name__ == '__main__':
    main()

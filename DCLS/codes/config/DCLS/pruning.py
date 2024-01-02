import argparse
import logging
import math
import os
import sys
import torch

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset

from channel_pruning import ChannelPruning



def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="options/setting2/train/pruning_setting2_x4_drn.yml",
                        help="Path to option YMAL file.")
    parser.add_argument('--save_dir', type=str, default="../../../../experiment/1_DCLS_experiments/pruning_debugs")
    parser.add_argument('--train_dataroot', type=str, default=None)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True, exp_root=args.save_dir)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    
    args.opt = opt
    args.scale = opt['scale']

    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(
        opt["pca_matrix_path"], map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    #### distributed training settings    
    util.set_random_seed(opt['train']['manual_seed'])

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### init logger
    logger_name = 'pruning'
    util.setup_logger(
        logger_name,
        args.save_dir,
        f"option_{opt['name']}",
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(args.opt))

    # setup dataset
    #### create train and val dataloader
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            logger.info(
                "Number of train images: {:,d}, iters: {:,d}".format(
                    len(train_set), train_size
                )
            )
            logger.info(
                "Total epochs needed: {:d} for iters {:,d}".format(
                    total_epochs, total_iters
                )
            )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info(
                "Number of val images in [{:s}]: {:d}".format(
                    dataset_opt["name"], len(val_set)
                )
            )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    # data process operation
    prepro = util.SRMDPreprocessing(
        scale=opt["scale"], pca_matrix=pca_matrix, cuda=True, **opt["degradation"]
    )

    #### create model
    model = create_model(opt)  # load pretrained model

    experiment = ChannelPruning(args, model, train_loader, val_loader, prepro)
    experiment.channel_selecton()


if __name__ == "__main__":
    main()




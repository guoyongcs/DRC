import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import datetime
import glob
import cv2


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

def calc_psnr(sr, hr, scale, rgb_range=255, benchmark=True):
    if len(sr.shape) < 4:
        sr = sr[None,:,:,:]
    if len(hr.shape) < 4:
        hr = hr[None,:,:,:]
    if sr.size(-2) > hr.size(-2) or sr.size(-1) > hr.size(-1):
        print("the dimention of sr image is not equal to hr's! ")
        sr = sr[:,:,:hr.size(-2),:hr.size(-1)]
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


def get_image_pair(folder_lq, scale, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{folder_lq}/{imgname}x{scale}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt


def test_model(opt, logger, test_loader, model, current_step, inf_E=False):
    avg_psnr = 0.0
    idx = 0

    for test_data in test_loader:
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        try:
            util.mkdir(img_dir)
        except:
            pass

        model.feed_data(test_data)
        # forward using G
        model.test(inf_E)
        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])
        # -----------------------
        # save estimated image E
        # -----------------------
        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
        util.imsave(E_img, save_img_path)
        # -----------------------
        # calculate PSNR
        # -----------------------
        # current_psnr = util.calculate_psnr(E_img, H_img, border=border)
        current_psnr = calc_psnr(visuals['E'], visuals['H'], 4, rgb_range=1)
        if opt['rank'] == 0:
            logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
        avg_psnr += current_psnr

    avg_psnr = avg_psnr / idx

    return avg_psnr



def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--opt_teacher', type=str, default=None, help='Path to option JSON file for teacher model.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--train_dataroot', type=str, default=None)


    opt = option.parse(parser.parse_args().opt, is_train=True, root=parser.parse_args().root)
    opt['dist'] = parser.parse_args().dist

    opt_teacher = None
    opt_teacher_tmp = parser.parse_args().opt_teacher
    if opt_teacher_tmp is not None:
        opt_teacher = option.parse(opt_teacher_tmp, is_train=True, root=parser.parse_args().root)
        opt_teacher['dist'] = parser.parse_args().dist
        opt_teacher['gpu_ids'] = opt['gpu_ids']

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
        # init_dist('pytorch', backend='gloo')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    if init_path_G is not None:
        opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)
    if opt_teacher is not None:
        opt_teacher = option.dict_to_nonedict(opt_teacher)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.txt'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    train_dataroot = parser.parse_args().train_dataroot
    if train_dataroot is not None:
        train_dataroot_H = opt['datasets']['train']['dataroot_H']
        new_train_dataroot_H = os.path.join(train_dataroot, train_dataroot_H.split('/sr/')[-1])
        opt['datasets']['train']['dataroot_H'] = new_train_dataroot_H

        train_dataroot_L = opt['datasets']['train']['dataroot_L']
        new_train_dataroot_L = os.path.join(train_dataroot, train_dataroot_L.split('/sr/')[-1])
        opt['datasets']['train']['dataroot_L'] = new_train_dataroot_L

        test_dataroot_H = opt['datasets']['test']['dataroot_H']
        new_test_dataroot_H = os.path.join(train_dataroot, test_dataroot_H.split('/sr/')[-1])
        opt['datasets']['test']['dataroot_H'] = new_test_dataroot_H

        test_dataroot_L = opt['datasets']['test']['dataroot_L']
        new_test_dataroot_L = os.path.join(train_dataroot, test_dataroot_L.split('/sr/')[-1])
        opt['datasets']['test']['dataroot_L'] = new_test_dataroot_L


    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
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

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info('Model created, param count: %d' % (sum([m.numel() for m in model.netG.parameters()])))

    # build teacher model
    teacher = None
    if opt_teacher is not None:
        teacher = define_Model(opt_teacher)
        teacher.init_train()
        if opt['rank'] == 0:
            logger.info('build the teacher model')

    '''
    # ----------------------------------------
    # Test before training
    # ----------------------------------------
    '''

    # test using netG
    avg_psnr = test_model(opt, logger if opt['rank'] == 0 else None, test_loader, model, current_step, inf_E=False)
    if opt['rank'] == 0:
        logger.info('<Average PSNR : {:<.2f}dB\n'.format(avg_psnr))
    # test using netE
    avg_psnr = test_model(opt, logger if opt['rank'] == 0 else None, test_loader, model, current_step, inf_E=True)
    if opt['rank'] == 0:
        logger.info('<EMA Average PSNR : {:<.2f}dB\n'.format(avg_psnr))
    # test using teacher
    if teacher is not None:
        avg_psnr = test_model(opt, logger if opt['rank'] == 0 else None, test_loader, teacher, current_step, inf_E=False)
        if opt['rank'] == 0:
            logger.info('<Teacher Average PSNR : {:<.2f}dB\n'.format(avg_psnr))

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    batch_time_m = utils_logger.AverageMeter()
    end = time.time()
    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)
            if 'warmup_steps' in opt and opt['warmup_steps'] > 0:
                model.adjust_lr(current_step, opt['warmup_steps'])

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step, teacher)
            batch_time_m.update(time.time() - end)
            end = time.time()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                eta_string = str(datetime.timedelta(seconds=int(batch_time_m.avg * (len(train_loader) - i))))

                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, eta:{}> '.format(epoch, current_step, model.current_learning_rate(), eta_string)
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                # test using netG
                avg_psnr = test_model(opt, logger if opt['rank'] == 0 else None, test_loader, model, current_step, inf_E=False)
                if opt['rank'] == 0:
                    logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))
                # test using netE
                avg_psnr = test_model(opt, logger if opt['rank'] == 0 else None, test_loader, model, current_step, inf_E=True)
                if opt['rank'] == 0:
                    logger.info('<epoch:{:3d}, iter:{:8,d}, EMA Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

            if 'warmup_steps' in opt and opt['warmup_steps'] > 0:
                model.recover_lr(epoch, opt['warmup_steps'])



if __name__ == '__main__':
    main()

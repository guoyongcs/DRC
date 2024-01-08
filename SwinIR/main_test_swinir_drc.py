import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from models.network_swinir import SwinIR as net
from utils import utils_image as util
import math
from models.select_model import define_Model
from utils import utils_option as option
from torch.utils.data import DataLoader
from data.select_dataset import define_Dataset


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


def test_model(img_dir, test_loader, model, inf_E=False):
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0

    for test_data in test_loader:
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

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
        save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
        util.imsave(E_img, save_img_path)
        # -----------------------
        # calculate PSNR
        # -----------------------
        current_psnr = calc_psnr(visuals['E'], visuals['H'], 4, rgb_range=1)

        # cal ssim
        output_y = util.bgr2ycbcr(E_img.astype(np.float32) / 255.) * 255.
        img_gt_y = util.bgr2ycbcr(H_img.astype(np.float32) / 255.) * 255.
        current_ssim = util.calculate_ssim(output_y, img_gt_y, border=4)

        print('{:->4d}--> {:>10s} | PSNR: {:<4.2f}dB, SSIM: {:<4.2f}'.format(idx, image_name_ext, current_psnr, current_ssim))
        avg_psnr += current_psnr
        avg_ssim += current_ssim

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--opt', type=str, default=None, help='Path to option JSON file.')
    parser.add_argument('--dataset', type=str, default='Set5', help='Name of dataset.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    dataset_opt = opt['datasets']['test']

    dataset_opt['dataroot_H'] = args.folder_gt
    dataset_opt['dataroot_L'] = args.folder_lq

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    model = define_Model(opt)
    model.init_train()

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    psnr_y, ssim_y = test_model(save_dir, test_loader, model)
    print('<Average PSNR : {:<.2f}dB, Average SSIM : {:<.4f}\n'.format(psnr_y, ssim_y))


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    save_dir = f'results/swinir_{args.task}_x{args.scale}/{args.dataset}'
    folder = args.folder_gt
    border = args.scale
    window_size = 8
    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
            np.float32) / 255.

    # 003 real-world image sr (load lq image only)
    elif args.task in ['real_sr']:
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # 004 grayscale image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ['gray_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    # 005 color image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ['color_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)

    # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ['jpeg_car']:
        img_gt = cv2.imread(path, 0)
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output, _ = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch, _ = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()

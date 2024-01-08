import math
import time
import random
import numpy as np
import cv2
import copy
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    
def parse_ft_args(args):
    args_ft = copy.deepcopy(args)
    args_ft.model = args.ft_model
    args_ft.pre_train = args.pruned_model
    args_ft.pre_train_dual = args.pruned_dual_model
    args_ft.n_resblocks = args.n_ft_resblocks
    args_ft.n_feats = args.n_ft_feats
    return args_ft

def parse_qt_args(args):
    args_qt = copy.deepcopy(args)
    args_qt.model = args.qt_model
    args_qt.pre_train = args.qt_model_path
    args_qt.pre_train_dual = args.qt_dual_model
    args_qt.n_resblocks = args.n_qt_resblocks
    args_qt.n_feats = args.n_qt_feats
    args_qt.lr = args.qt_lr
    return args_qt

def parse_search_args(args):
    args_search = copy.deepcopy(args)
    return args_search

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

def quantize(img, rgb_range=255):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range=255, benchmark=False):
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

def calc_ssim(sr, hr, scale, rgb_range, benchmark=False):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    def _Tensor2numpy(tensor, rgb_range):
        tensor = tensor.cpu() if tensor.is_cuda else tensor
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array
    
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    shave = scale if benchmark else scale + 6
    sr = sr[..., shave:-shave, shave:-shave].squeeze()
    hr = hr[..., shave:-shave, shave:-shave].squeeze()
    
    sr = _Tensor2numpy(sr, rgb_range)
    hr = _Tensor2numpy(hr, rgb_range)
    
    sr = np.dot(sr, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    hr = np.dot(hr, [65.481, 128.553, 24.966]) / 255.0 + 16.0

    assert sr.shape == hr.shape, 'Input images must have the same dimensions.'
    assert sr.ndim == 2, 'Wrong input image dimensions.'
        
    return ssim(sr, hr)

def make_optimizer(opt, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if opt.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': opt.momentum}
    elif opt.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (opt.beta1, opt.beta2),
            'eps': opt.epsilon
        }
    elif opt.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': opt.epsilon}

    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_dual_optimizer(opt, dual_models):
    dual_optimizers = []
    for dual_model in dual_models:
        print('base_dual optimizer init')
        temp_dual_optim = torch.optim.Adam(
            params=dual_model.parameters(),
            lr = opt.lr, 
            betas = (opt.beta1, opt.beta2),
            eps = opt.epsilon,
            weight_decay=opt.weight_decay)
        dual_optimizers.append(temp_dual_optim)
    
    print('dual_optimizers number:', len(dual_optimizers))
    return dual_optimizers

def make_alpha_optimizer(opt, alphas):
    if opt.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': opt.momentum}
    elif opt.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (opt.beta1, opt.beta2),
            'eps': opt.epsilon
        }
    elif opt.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': opt.epsilon}

    kwargs['lr'] = opt.alpha_lr
    kwargs['weight_decay'] = opt.alpha_weight_decay
    
    return optimizer_function(alphas, **kwargs)

def make_scheduler(opt, my_optimizer):
    if opt.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=opt.lr_decay,
            gamma=opt.gamma
        )
    elif opt.decay_type == "cosine":
        scheduler = lrs.CosineAnnealingLR(
            my_optimizer,
            float(opt.epochs),
            eta_min=opt.eta_min
        )
    elif opt.decay_type.find('step') >= 0:
        milestones = opt.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=opt.gamma
        )

    return scheduler

def make_dual_scheduler(opt, dual_optimizers):
    dual_scheduler = []
    for i in range(len(dual_optimizers)):
        if opt.decay_type == 'step':
            scheduler = lrs.StepLR(
                dual_optimizers[i],
                step_size=opt.lr_decay,
                gamma=opt.gamma
            )
        elif opt.decay_type == "cosine":
            scheduler = lrs.CosineAnnealingLR(
                dual_optimizers[i],
                float(opt.epochs),
                eta_min=opt.eta_min
            )
        elif opt.decay_type.find('step') >= 0:
            milestones = opt.decay_type.split('_')
            milestones.pop(0)
            milestones = list(map(lambda x: int(x), milestones))
            scheduler = lrs.MultiStepLR(
                dual_optimizers[i],
                milestones=milestones,
                gamma=opt.gamma
            )
        dual_scheduler.append(scheduler)

    return dual_scheduler

def make_drn_optimizer(opt, model):
    trainable = list(filter(lambda x: x.requires_grad, model.model.parameters()))
    if opt.dual:
        for i in range(len(model.dual_models)):
            trainable += model.dual_models[i].parameters()

    if opt.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': opt.momentum}
    elif opt.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (opt.beta1, opt.beta2),
            'eps': opt.epsilon
        }
    elif opt.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': opt.epsilon}

    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay
    
    return optimizer_function(trainable, **kwargs)
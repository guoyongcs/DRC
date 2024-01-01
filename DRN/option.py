import argparse
import template
import numpy as np
import time

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=12,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--data_dir', type=str, default='/home/datasets/sr',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension, img|sep')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=384,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--ycbcr', action='store_true', 
                    help="wether read image into ycbcr channel")
parser.add_argument('--single_scale', action='store_true',
                    help='if true, data scale is not the multiple scale')

# Model specifications
parser.add_argument('--model', default='drn',
                    help='model name EDSR|MDSR|DRN')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--pre_train_dual', type=str, default='.',
                    help='pre-trained dual model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=30,
                    help='number of residual blocks, 16|30|40|80')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

parser.add_argument('--dual', action="store_true", help='options: True | False, \
                    dual model can be used only when len(scale) is more than one')
parser.add_argument('--dual_type', type=str, default='base_dual', help="base_dual | all_dual,\
                    if it's base_dual, it would contain 3 dual blocks in 8x model; \
                    else it would contain 6 dual blocks.")

### Pruning specifications
parser.add_argument('--pruning', action='store_true',
                    help='pruning flag')
parser.add_argument('--prune_type', default='dcp',
                    help='prune type, dcp|cp|thinet')
parser.add_argument('--pruning_rate', type=float, default=0.5,
                    help='pruning rate of feature maps')
parser.add_argument('--layer_wise_lr', type=float, default=5e-5,
                    help='learning rate for feature reconstrcution')
parser.add_argument('--resume_pruning', default=None,
                    help='resume pruning from specific checkpoint')
parser.add_argument('--warm_start', default=True,
                    help='init pruned layer weight using origin layer')
parser.add_argument('--finetune_epochs', type=int, default=1,
                    help='number of epochs to finetune the pruned models')
parser.add_argument('--lasso_alpha', type=float, default=200.0,
                    help='alpha for lasso algorithm')
parser.add_argument('--configs_path', default=None,
                    help='path to a specific channel config file')

### Search specifications
parser.add_argument('--search', action='store_true',
                    help='the flag for channel search')
parser.add_argument('--resume_search', default=None,
                    help='resume search from specific checkpoint')
parser.add_argument('--w_momentum', type=float, default=0.9, 
                    help='momentum for weights')
parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                    help='weight decay for weights')
parser.add_argument('--w_grad_clip', type=float, default=5.0,
                    help='gradient clipping for weights')
parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
parser.add_argument('--alpha_weight_decay', type=float,
                    default=1e-3, help='weight decay for alpha')
parser.add_argument('--tau', type=float, default=10, help='tau for gumbel softmax')
parser.add_argument('--gamma_tau', type=float, default=0.9,
                    help='gamma for decreasing exponentially')
parser.add_argument('--min_tau', type=float, default=0.1,
                    help='minimum tau after decreasing.')
parser.add_argument('--unordered_channels', action='store_true',
                    help='do not sort channels during searching')
parser.add_argument('--search_multi_branch', action='store_true',
                    help='the type of the search module')


### Finetune specifications
parser.add_argument('--finetune', action='store_true',
                    help='finetune flag')
parser.add_argument('--ft_model', default='pdrn',
                    help='model name EDSR|MDSR|DRN')
parser.add_argument('--n_ft_resblocks', type=int, default=30,
                    help='number of residual blocks, 16|30|40|80')
parser.add_argument('--n_ft_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--pruned_model', type=str, default='.',
                    help='pruned model directory')
parser.add_argument('--pruned_dual_model', type=str, default='.',
                    help='pruned dual model directory')
parser.add_argument('--resume_finetune', default=None,
                    help='resume finetune from specific checkpoint')
parser.add_argument('--kd_weight', type=float, default=1,
                    help='the weight of konwledge distillation loss')
parser.add_argument('--kd_nums', type=int, default=0,
                    help='number of kd losses for each level, \
                    0|1|2, 0 for the input of the last conv only')

### Quantization specifications
parser.add_argument('--quantization', action='store_true',
                    help='quantization flag')
parser.add_argument('--qt_model', default='pdrn',
                    help='model name EDSR|MDSR|DRN')
parser.add_argument('--n_qt_resblocks', type=int, default=30,
                    help='number of residual blocks, 16|30|40|80')
parser.add_argument('--n_qt_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--qt_model_path', type=str, default='.',
                    help='pruned model directory')
parser.add_argument('--qt_dual_model', type=str, default='.',
                    help='pruned dual model directory')
parser.add_argument('--resume_quant', default=None,
                    help='resume quantization from specific checkpoint')
parser.add_argument('--activation_file', type=str, default='activate_scale.table',
                    help='filename to save activation scale')
parser.add_argument('--weight_file', type=str, default='weight_scale.table',
                    help='filename to save weight scale')
parser.add_argument('--calibration_file', type=str, default='calibration_file.table',
                    help='filename to save quantization scale')
parser.add_argument('--qt_lr', type=float, default=1e-5,
                    help='pruning rate of feature maps')

### Tiny model specifications
parser.add_argument('--pre_upsample', type=str, default='.',
                    help='pre-trained upsample model directory')
parser.add_argument('--fix_upsample', action='store_true',
                    help='flag to fix the upsample in cascaded model')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--save_train_sr', action='store_true', 
                    help='save sr or not in trian')
parser.add_argument('--save_every', type=int, default=20, 
                    help='do save sr images to compare the visual effect')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--negval', type=float, default=0.2, 
                    help='Negative value parameter for Leaky ReLU in discriminator network')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, 
                    help='learning rate') # 2.50e-5
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='cosine',
                    help='learning rate decay type: step | cosine')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration, L1|MSE|L1_Charbonnier')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')
parser.add_argument('--l1_reduction', type=str, default='mean', 
                    help="the reduction of L1 loss, sum|mean")
parser.add_argument('--alpha', type=float, default=1, 
                    help='weight for MSE in GM loss')
parser.add_argument('--ROBUST', type=float, default=0, 
                    help='option: 0 | 1e-3')
parser.add_argument('--mask_lambda', type=float, default=1, 
                    help='mask parameter to enlarge the distance between inputs')
parser.add_argument('--keep_ratio', type=float, default=0.8, 
                    help='the ratio of images')
parser.add_argument('--topk', action="store_true", 
                    help="whether use topk method to compute mask")
parser.add_argument('--divide', action="store_true", 
                    help='decide whether compute the gradient loss respectively')
parser.add_argument('--content_mask', action="store_true", 
                    help='whether use mask to compute content loss')
parser.add_argument('--cmask', action='store_true', 
                    help='use cmask to compute content loss or use mask, ture cmask, false mask')
parser.add_argument('--dual_weight', type=float, default=0.1,
                    help='the weight of dual loss')

# Log specifications
parser.add_argument('--save', type=str, default='../experiment/test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--print_model', action='store_true',
                    help='save output results')
parser.add_argument('--calc_ssim', action='store_true',
                    help='measure the ssim metrics')
parser.add_argument('--bicubic', action='store_true',
                    help='use bicubic to upsample')


args = parser.parse_args()

# add time stamp to save path
# if not args.test_only:
#     args.save = '{}_{}'.format(args.save, time.strftime("%Y%m%d_%H%M%S"))
if args.pruned_dual_model == '.':
    args.pruned_dual_model == args.pre_train_dual


template.set_template(args)

if not args.single_scale:
    args.scale = [pow(2, s+1) for s in range(int(np.log2(args.scale)))]
else:
    args.scale = [int(args.scale)]

print(args)
if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False


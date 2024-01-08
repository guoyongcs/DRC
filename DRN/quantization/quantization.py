import torch
import torch.nn as nn
import os.path as osp
import copy
import math
from decimal import Decimal
from collections import OrderedDict
import utility
from utils import get_logger, TensorboardLogger, concat_gpu_datalist
import utils.quant_op as qp
from utils.replace_layer import replace_layer_by_unique_name, replace_int8
from .quant_net import ConvQuant, quant_hist, quant_table
from .quant_checkpoint import QuantCheckpoint


class Quantization():
    def __init__(self, args, original_model, quant_model, loader):
        self.args = args
        self.scale = args.scale
        self.original_model = original_model
        self.quant_model = quant_model
        self.temp_model = copy.deepcopy(quant_model.get_model())
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.cache_original_feature = {}
        self.cache_quant_feature = {}

        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_L1 = nn.L1Loss().cuda()

        self.logger = get_logger(args.save, "quantization", log_file='quantization.log')
        self.tensorboard_logger = TensorboardLogger(osp.join(args.save, 'qt_tb'))
        self.logger.info("|===>Result will be saved at {}".format(args.save))
        
        self.start_epoch = 1
        
        self.quant_best_psnr, self.quant_best_epoch = 0, 0

        self.logger_counter = 0
        
        self.checkpoint = QuantCheckpoint(args)
        
    def quantization(self):
        #### get activate table
        self.gen_activate_table()

        #### prepare for finetuning quant model
        self.replace_conv_with_convint8()
        self.create_optimizer()
        self.fix_original_model()
        self.register_hooks()

        #### resume checkpoint
        if self.args.resume_quant is not None:
            self.checkpoint.resume(self, self.args.resume_quant)
        
        #### test only
        if self.args.test_only:
            self.test(0)
            return

        #### finetune quant model
        self.finetune()

        #### get weigth table
        self.gen_weight_table()

        #### merge the activate scale table and weigth table
        self.merge_table()
    
    def merge_table(self):
        qp.replace_weight_name(self.scale, self.logger, 
                self.checkpoint.weight_file, 
                self.checkpoint.calibration_file)
        qp.replace_activation_name(self.scale, self.logger,
                self.checkpoint.activation_file, 
                self.checkpoint.calibration_file)
    
    def gen_activate_table(self):
        if osp.exists(self.checkpoint.activation_file):
            return 

        self.logger.info('-----Repalace model-----')
        self.temp_model = self.replace_layer(self.temp_model)
        self.logger.info('-----Generate activate table-----')
        activation_file = open(self.checkpoint.activation_file, 'a+')
        
        # no quant_table at the beginning
        quant_table['quant'] = False
        # step 1: get the range of data_in
        quant_hist['step'] = 1
        self.logger.info('-----step 1-----')
        with torch.no_grad():
            for batch_idx, (lr, _, _ ) in enumerate(self.loader_train):
                lr = lr[0].cuda() if not self.args.cpu else lr[0]
                _ = self.temp_model(lr)

        # step 2: get the hist of data_in 
        quant_hist['step'] = 2
        self.logger.info('-----step 2-----')
        with torch.no_grad():
            # for batch_idx in range(forward_round):
            #     lr, _, _ = loader_train_iter.next()
            for batch_idx, (lr, _, _ ) in enumerate(self.loader_train):
                lr = lr[0].cuda() if not self.args.cpu else lr[0]
                _ = self.temp_model(lr)

        self.logger.info('-----step 2 done-----')

        # step 3: get quant_table 
        self.logger.info('-----step 3-----')

        for key in quant_hist.keys():
            if key == 'step':
                continue
            quant_table[key] = qp.get_best_activation_intlength(
                quant_hist[key]['hist'], quant_hist[key]['hist_edges'])
            
            lines = "{} {}".format(key, quant_table[key])
            self.logger.info(lines)
            activation_file.write("{}\n".format(lines))

        # output quant_table
        activation_file.close()
        self.logger.info(quant_table)
        self.logger.info("====> Save activation scale table success...")
        self.temp_model = None
    
    def replace_layer(self, model):
        conv_counter = 0 
        for name, module in model.named_modules(): 
            if isinstance(module, nn.Conv2d): 
                self.logger.info("Replace Conv2d({}) with QuantConv2d, {}.".format(name, module)) 
                temp_conv = ConvQuant(module, name=name) 
                replace_layer_by_unique_name(model, name, temp_conv) 
                conv_counter += 1 
        self.logger.info("Replace complete.({} Conv2d have been replaced.)".format(conv_counter)) 
        return model

    def gen_weight_table(self, quantize_num=127.0):
        self.logger.info("------save the scale table----")
        weight_file = open(self.checkpoint.weight_file, 'w')
        self.logger.info("------quant weight----")
        weight_table = OrderedDict()
        model = self.quant_model.get_model()
        for name, layer in model.named_modules():
            # find the convolution layers to get out the weight_scale
            if isinstance(layer, nn.Conv2d):
                per_ch = []
                for i in range(layer.weight.shape[0]):
                    data_ = layer.weight[i].data
                    maxv = data_.abs().max()
                    scale = quantize_num / maxv.cpu().numpy()
                    if name not in weight_table.keys():
                        weight_table[name] = per_ch
                        weight_table[name].append(scale)
                    else:
                        weight_table[name].append(scale)
        self.logger.info("------quant weight done----")

        # save the weight blob scales
        for key, value_list in weight_table.items():
            weight_file.write(key + " ")
            for value in value_list:
                weight_file.write(str(value) + " ")
            weight_file.write("\n")

        weight_file.close()
        self.logger.info("====> Save calibration table success...")

    def fix_original_model(self):
        self.original_model.eval()
        for p in self.original_model.parameters():
            p.required_grad = False

    def replace_conv_with_convint8(self):
        self.activation_value_list = self._get_activation_value()
        model = self.quant_model.get_model()
        self.last_scale = replace_int8(
            model, self.activation_value_list, self.logger)
    
    def _get_activation_value(self):
        activation_value_list = {}
        for line in open(self.checkpoint.activation_file):
            line = line.strip()
            key, value = line.split(' ')

            activation_value_list[key] = value

        for key in activation_value_list:
            self.logger.info('{} {}'.format(key, activation_value_list[key]))
        return activation_value_list

    def create_optimizer(self):
        self.quant_optimizer = utility.make_drn_optimizer(self.args, self.quant_model)
        self.quant_scheduler = utility.make_scheduler(self.args, self.quant_optimizer)

    def finetune(self):
        timer_epoch = utility.timer()
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            
            epoch_time = int(timer_epoch.toc())
            log = utility.remain_time(epoch_time, epoch, self.args.epochs)
            self.logger.info(log)
            timer_epoch.tic()
        self.remove_layer_hook()

    def train(self, epoch):
        lr = self.quant_scheduler.get_lr()[0]

        self.quant_model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lrs, hr, _) in enumerate(self.loader_train):
            if not self.args.cpu:
                lrs = [lr.cuda().detach() for lr in lrs]
                hr = hr.cuda().detach()
            timer_data.hold()
            timer_model.tic()

            self.cache_original_feature = {}
            self.cache_quant_feature = {}
            
            quant_sr = self.quant_model(lrs[0])
            quant_sr2lr = self.quant_model.dual_forward(quant_sr) if self.args.dual else None
            
            ## obtain the intermedia features of the original model
            self.original_model(lrs[0]) 
            
            kd_loss, rec_loss, total_loss = \
                self.compute_loss(lrs, hr, quant_sr, quant_sr2lr)
            
            self.quant_optimizer.zero_grad()
            total_loss.backward()
            self.quant_optimizer.step()
            
            timer_model.hold()
            
            if (batch + 1) % self.args.print_every == 0:
                self.tensorboard_logger.scalar_summary(
                    tag="KD_loss",
                    value=kd_loss,
                    step=self.logger_counter)
                self.tensorboard_logger.scalar_summary(
                    tag="Reconstruction_loss",
                    value=rec_loss,
                    step=self.logger_counter)
                self.tensorboard_logger.scalar_summary(
                    tag="Total_loss",
                    value=total_loss,
                    step=self.logger_counter)
                self.logger_counter += 1
                log = 'Epoch: {:0>4d} lr: {:.2e} '.format(epoch, Decimal(lr))
                log += "[{:0>5d}/{:d}]  ".format(
                                (batch + 1) * self.args.batch_size, 
                                len(self.loader_train.dataset))
                log += 'KD_loss:{:.4e}  Rec_loss:{:.4e}  Total_loss:{:.4e}  '\
                    .format(kd_loss, rec_loss, total_loss)
                log += '{:.1f}+{:.1f}s'.format(
                            timer_model.release(), timer_data.release())
                self.logger.info(log)
            
            timer_data.tic()        

        self.quant_scheduler.step()
    
    def test(self, epoch):
        
        self.original_model.eval()
        self.quant_model.eval()
        
        quant_psnr = 0
        scale = max(self.scale)
        with torch.no_grad():
            for _, (lr, hr, filename) in enumerate(self.loader_test):
                filename = filename[0]
                if isinstance(lr, list): lr = lr[0]
                
                no_eval = (hr.nelement() == 1)
                if not no_eval and not self.args.cpu:
                    lr, hr = lr.cuda(), hr.cuda()
                elif not self.args.cpu:
                    lr = lr.cuda()

                sr_quant = self.quant_model(lr)
                if isinstance(sr_quant, list):
                    sr_quant = sr_quant[-1]

                sr_quant = utility.quantize(sr_quant, self.args.rgb_range)

                if not no_eval:
                    quant_psnr += utility.calc_psnr(
                        sr_quant, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )

                # save test results
                if self.args.save_results:
                    self.checkpoint.save_results(filename, [sr_quant], scale)

            quant_psnr = quant_psnr / len(self.loader_test)
            
            if quant_psnr >= self.quant_best_psnr:
                self.quant_best_psnr = quant_psnr
                self.quant_best_epoch = epoch
            quant_log = '[{} x{}]  '.format(self.args.data_test, scale)
            quant_log += 'PSNR:  {:.3f} '.format(quant_psnr)
            quant_log += '(Best: {:.3f} @epoch {})'.format(
                            self.quant_best_psnr, self.quant_best_epoch)
            self.logger.info(quant_log)
        
        self.tensorboard_logger.scalar_summary(
                "quant_model_psnr", quant_psnr, epoch)
        
        self.checkpoint.save(self, epoch, self.args.save)

    def register_hooks(self):
        if self.args.n_GPUs > 1:
            original_model = self.original_model.model.module
            quant_model = self.quant_model.model.module
        else:
            original_model = self.original_model.model
            quant_model = self.quant_model.model
        
        self.original_hooks = [] 
        self.quant_hooks = []

        for i in range(len(self.scale)):
            # kd_nums: the number of kd losses for each level
            # if kd_nums is zero, only one kd loss for the last block
            if self.args.kd_nums == 2:
                mid_idx = len(original_model.up_blocks[i]) // 2 - 2
                self.original_hooks.append(
                    original_model.up_blocks[i][mid_idx].\
                    register_forward_hook(self._hook_origin_feature))
            if self.args.kd_nums >= 1 or i == len(self.scale) - 1:
                if self.args.model.lower() == 'fsrcnn':
                    self.original_hooks.append(
                        original_model.mid_part[-1][-2].\
                        register_forward_hook(self._hook_origin_feature))
                    
                if self.args.model.lower().find('drn')>=0:
                    self.original_hooks.append(
                        original_model.tail[i + 1].\
                        register_forward_hook(self._hook_origin_feature))
            
            if self.args.kd_nums == 2:
                mid_idx = len(quant_model.up_blocks[i]) // 2 - 2
                self.quant_hooks.append(
                    quant_model.up_blocks[i][mid_idx].\
                    register_forward_hook(self._hook_quant_feature))
            if self.args.kd_nums >= 1 or i == len(self.scale) - 1:
                if self.args.model.lower() == 'fsrcnn':
                    self.quant_hooks.append(
                        quant_model.mid_part[-1][-2].\
                        register_forward_hook(self._hook_quant_feature))
                
                if self.args.model.lower().find('drn')>=0:
                    self.quant_hooks.append(
                        quant_model.tail[i + 1].\
                        register_forward_hook(self._hook_quant_feature))
        self.logger.info(
            'The total number of KD loss: {}'.format(len(self.quant_hooks)))
    
    def _hook_origin_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        if gpu_id not in self.cache_original_feature.keys():
            self.cache_original_feature[gpu_id] = []
        if isinstance(module, nn.Conv2d): # the tail conv of drn
            self.cache_original_feature[gpu_id].append(
                qp.quantization_on_input_fix_scale(
                    input[0].mean(dim=1, keepdim=True),
                    activation_value=self.last_scale))
        else:
            self.cache_original_feature[gpu_id].append(
                            output.mean(dim=1, keepdim=True))

    def _hook_quant_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        if gpu_id not in self.cache_quant_feature.keys():
            self.cache_quant_feature[gpu_id] = []
        if isinstance(module, nn.Conv2d): # the tail conv of drn
            self.cache_quant_feature[gpu_id].append(
                            input[0].mean(dim=1, keepdim=True))
        else:
            self.cache_quant_feature[gpu_id].append(
                            output.mean(dim=1, keepdim=True))
    
    def remove_layer_hook(self):
        for i in range(len(self.original_hooks)):
            self.original_hooks[i].remove()
            self.quant_hooks[i].remove()
        self.cache_original_feature = {}
        self.cache_quant_feature = {}
        self.logger.info("|===>remove hook")

    def compute_loss(self, lr, hr, sr, sr2lr=None):
        # compute features reconstruction loss
        kd_loss = 0
        
        # note that cache features have been averaged across the channel dimension
        # with the shape of (B, 1, H, W)
        original_features = concat_gpu_datalist(self.cache_original_feature)
        quant_features = concat_gpu_datalist(self.cache_quant_feature)
        assert quant_features[0].size(1) == 1, \
            "did not average the feature across channels"
        for j in range(len(original_features)):
            kd_loss += self.criterion_mse(quant_features[j],
                                    original_features[j].detach())

        # compute primal regression loss
        if not isinstance(sr, list): sr = [sr]
        loss_primary = self.criterion_L1(sr[-1], hr)
        for i in range(1, len(sr)):
            loss_primary += self.criterion_L1(
                sr[i - 1 - len(sr)], lr[i - len(sr)])
        
        # compute dual regression loss
        if self.args.dual:
            loss_dual = self.criterion_L1(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.criterion_L1(
                    sr2lr[i - len(self.scale)], lr[i - len(self.scale)])
            
            # compute total loss
            rec_loss = loss_primary + self.args.dual_weight * loss_dual
        else:
            rec_loss = loss_primary
        total_loss = rec_loss + self.args.kd_weight * kd_loss
        return kd_loss, rec_loss, total_loss
import os
import math
import shutil
from importlib import import_module
import torch
import torch.nn as nn
from model.common import DownBlock

def dataparallel(model, gpu_list):
    ngpus = len(gpu_list)
    if ngpus == 0:
        assert False, "only support gpu mode"
    assert torch.cuda.device_count() >= ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if isinstance(model[i], list):
                    for m in model[i]:
                        m = torch.nn.DataParallel(m, gpu_list).cuda()

                elif not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()

            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model

class Model(nn.Module):
    def __init__(self, opt, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.opt = opt
        self.scale = opt.scale
        self.idx_scale = 0
        self.self_ensemble = opt.self_ensemble
        self.chop = opt.chop
        self.precision = opt.precision
        self.cpu = opt.cpu
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.n_GPUs = opt.n_GPUs
        self.save_models = opt.save_models

        module = import_module('model.' + opt.model.lower())
        self.model = module.make_model(opt).to(self.device)
        self.dual_models = None
        if self.opt.dual:
            self.dual_models = []
            for _ in self.opt.scale:
                dual_model = DownBlock(opt, 2).to(self.device)
                self.dual_models.append(dual_model)

        if opt.precision == 'half': 
            self.model.half()
            if opt.dual:
                for idx in range(len((self.dual_models))):
                    self.dual_models[idx].half()
        
        if not opt.cpu and opt.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(opt.n_GPUs))
            if self.opt.dual:
                self.dual_models = dataparallel(self.dual_models, range(opt.n_GPUs))

        self.load(
            opt.pre_train,
            opt.pre_train_dual,
            cpu=opt.cpu
        )

        if not opt.test_only or opt.print_model:
            print(self.model, file=ckp.log_file)
            if self.opt.dual:
                print(self.dual_models, file=ckp.log_file)
        
        # compute parameter
        parameter = self.count_parameters(self.model)
        if opt.dual:
            for dual_model in self.dual_models:
                parameter += self.count_parameters(dual_model)
        print("The number of Parameters is {}".format(parameter))

    def forward(self, x, idx_scale=0):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)

        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            return self.model(x)
    
    def dual_forward(self, sr):
        # dual forward
        sr2lr = []
        for i in range(len(self.dual_models)):
            sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
            sr2lr.append(sr2lr_i)
        return sr2lr

    def count_layers(self):
        """count the layers that are searched or pruned."""
        return len(self.scale) * (1 + 1 + self.opt.n_resblocks)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module
    
    def get_dual_model(self, idx):
        if self.n_GPUs == 1:
            return self.dual_models[idx]
        else:
            return self.dual_models[idx].module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
    
    def count_parameters(self, model):
        if self.opt.n_GPUs > 1:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self, apath, epoch, save_folder='model', is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, save_folder, 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, save_folder, 'model_best.pt')
            )
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, save_folder, 'model_{}.pt'.format(epoch))
            )

        if self.opt.dual:
            #### save dual models ####
            dual_models = []
            for i in range(len(self.dual_models)):
                dual_models.append(self.get_dual_model(i).state_dict())
            torch.save(
                dual_models,
                os.path.join(apath, save_folder, 'dual_model_latest.pt')
            )
            if is_best:
                torch.save(
                    dual_models,
                    os.path.join(apath, save_folder, 'dual_model_best.pt')
                )
            if self.save_models:
                torch.save(
                    dual_models,
                    os.path.join(apath, save_folder, 'dual_model_{}.pt'.format(epoch))
                )

    def load(self, pre_train='.', pre_train_dual='.', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        #### load primal model ####
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=True
            )
        #### load dual model ####
        if pre_train_dual != '.':
            print('Loading dual model from {}'.format(pre_train_dual))
            dual_models = torch.load(pre_train_dual, **kwargs)
            for i in range(len(self.dual_models)):
                self.get_dual_model(i).load_state_dict(
                    dual_models[i], strict=False
                )

    def combine_chop(self, sr_list, x, shave=10):
        """combine the patch that forward into model"""
        scale = max(self.scale)
        h, w = x.size(-2), x.size(-1)
        h_half, w_half = scale * (h // 2), scale * (w // 2)
        h_size, w_size = h_half + scale * shave, w_half + scale * shave
        h, w = scale * h, scale * w

        b, c = x.size(0), x.size(-3)
        output = sr_list[0].new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output
        
    def forward_chop(self, x, shave=10, min_size=160000):
        # scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        _, _, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)[-1]
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        output = self.combine_chop(sr_list, x, shave)
        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        # sr_list = [forward_function(aug) for aug in lr_list]
        sr_list = [forward_function(aug)[-1] for aug in lr_list]
        
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


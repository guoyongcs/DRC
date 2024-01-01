import os
import os.path as osp
import torch
# import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from models.network_swinir_pruned import SwinTransformerBlock


class SearchCheckpoint():
    def __init__(self, args):
        self.dir = args.save_dir
        self.search_folder = 'search_model'

    def save(self, searcher, epoch, save_path=None):
        if not osp.exists(osp.join(save_path, self.search_folder)):
            os.makedirs(osp.join(save_path, self.search_folder))
        if isinstance(searcher.model, (DataParallel, DistributedDataParallel)):
            state_dict = searcher.model.module.state_dict()
        else:
            state_dict = searcher.model.state_dict()
        torch.save(state_dict, osp.join(save_path, 
                    self.search_folder, f'search_models_{epoch}.pt'))
        state_dict = {}
        state_dict['epoch'] = epoch
        state_dict['best_epoch'] = searcher.best_epoch
        state_dict['best_psnr'] = searcher.best_psnr
        state_dict['w_optimizer'] = searcher.w_optimizer.state_dict()
        state_dict['w_scheduler'] = searcher.w_scheduler.state_dict()
        state_dict['alpha_optimizer'] = searcher.alpha_optimizer.state_dict()
        state_dict['alpha_scheduler'] = searcher.alpha_scheduler.state_dict()
        state_dict['alpha_parameter'] = OrderedDict(tuple(searcher.controller.named_alphas()))
        torch.save(state_dict, osp.join(save_path, 'search_state.pt'))
    
    def resume(self, searcher, resume_path):
        search_path = osp.join(resume_path, 
                            self.search_folder, 'model_latest.pt')
        state_dict = torch.load(search_path)
        if isinstance(searcher.model, (DataParallel, DistributedDataParallel)):
            searcher.model.module.load_state_dict(state_dict, strict=False)
        else:
            searcher.model.load_state_dict(state_dict, strict=False)
        
        resume_state = torch.load(osp.join(resume_path, 'search_state.pt'))
        searcher.start_epoch = resume_state['epoch'] + 1
        searcher.best_epoch = resume_state['best_epoch']
        searcher.best_psnr = resume_state['best_psnr']

        searcher.w_optimizer.load_state_dict(resume_state['w_optimizer'])
        searcher.w_scheduler.load_state_dict(resume_state['w_scheduler'])
        searcher.alpha_optimizer.load_state_dict(resume_state['alpha_optimizer'])
        searcher.alpha_scheduler.load_state_dict(resume_state['alpha_scheduler'])
        searcher.controller.set_alphas(resume_state['alpha_parameter'])

    def save_config(self, controller, is_best=False):
        alphas = controller.get_alphas()
        channel_sets = controller.channel_sets
        channel_configs = [torch.max(a.detach(), dim=-1) for a in alphas]
        configs = []
        for m_idx, ((_, c_idx), channels) in enumerate(
                        zip(channel_configs, channel_sets)):
            module = controller.search_model.search_modules[m_idx]
            if isinstance(module, SwinTransformerBlock):
                # the search num_channels repeat num_heads
                configs.append(int(channels[c_idx] * module.num_heads))
            else:
                configs.append(channels[c_idx])
        configs = np.array(configs)
        save_paths = [osp.join(self.dir, 'channel_configs.txt')]
        if is_best:
            save_paths.append(osp.join(self.dir, 'channel_configs_best.txt'))
        for path in save_paths:
            np.savetxt(path, configs, fmt='%d')

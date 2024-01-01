import os
import os.path as osp
import torch

class FinetuneCheckpoints():
    def __init__(self, args):
        self.dir = args.save
        self.ori_folder = 'original_ft_model'
        self.pruned_folder = 'pruned_ft_model'

    def save(self, finetuner, epoch, save_path=None):
        if not osp.exists(osp.join(save_path, self.ori_folder)):
            os.makedirs(osp.join(save_path, self.ori_folder))
        if not osp.exists(osp.join(save_path, self.pruned_folder)):
            os.makedirs(osp.join(save_path, self.pruned_folder))
        finetuner.original_model.save(save_path, epoch, self.ori_folder)
        finetuner.pruned_model.save(save_path, epoch, 
            self.pruned_folder, is_best=(finetuner.pruned_best_epoch == epoch))
        state_dict = {}
        state_dict['epoch'] = epoch
        state_dict['pruned_best_epoch'] = finetuner.pruned_best_epoch
        state_dict['pruned_best_psnr'] = finetuner.pruned_best_psnr
        state_dict['pruned_optimizer'] = finetuner.pruned_optimizer.state_dict()
        state_dict['pruned_scheduler'] = finetuner.pruned_scheduler.state_dict()
        torch.save(state_dict, osp.join(save_path, 'finetune_state.pt'))
    
    def resume(self, finetuner, resume_path):
        ori_path = osp.join(resume_path, 
                        self.ori_folder, 'model_latest.pt')
        ori_dual_path = osp.join(resume_path, 
                        self.ori_folder, 'dual_model_latest.pt')
        finetuner.original_model.load(ori_path, ori_dual_path)
        
        pruned_path = osp.join(resume_path, 
                            self.pruned_folder, 'model_latest.pt')
        pruned_dual_path = osp.join(resume_path, 
                            self.pruned_folder, 'dual_model_latest.pt')
        finetuner.pruned_model.load(pruned_path, pruned_dual_path)
        
        resume_state = torch.load(osp.join(resume_path, 'finetune_state.pt'))
        finetuner.start_epoch = resume_state['epoch'] + 1
        finetuner.pruned_best_epoch = resume_state['pruned_best_epoch']
        finetuner.pruned_best_psnr = resume_state['pruned_best_psnr']

        finetuner.pruned_optimizer.load_state_dict(resume_state['pruned_optimizer'])
        finetuner.pruned_scheduler.load_state_dict(resume_state['pruned_scheduler'])

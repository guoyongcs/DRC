import os
import os.path as osp
import torch
import imageio

class QuantCheckpoint():
    def __init__(self, args):
        self.args = args
        self.dir = args.save
        self.ori_folder = 'original_ft_model'
        self.quant_folder = 'quant_model'
        self.activation_file = osp.join(self.dir, args.activation_file)
        self.weight_file = osp.join(self.dir, args.weight_file)
        self.calibration_file = osp.join(self.dir, args.calibration_file)

    def save(self, quantization, epoch, save_path=None):
        if not osp.exists(osp.join(save_path, self.ori_folder)):
            os.makedirs(osp.join(save_path, self.ori_folder))
        if not osp.exists(osp.join(save_path, self.quant_folder)):
            os.makedirs(osp.join(save_path, self.quant_folder))
        quantization.original_model.save(save_path, epoch, self.ori_folder)
        quantization.quant_model.save(save_path, epoch, 
            self.quant_folder, is_best=(quantization.quant_best_epoch == epoch))
        state_dict = {}
        state_dict['epoch'] = epoch
        state_dict['quant_best_epoch'] = quantization.quant_best_epoch
        state_dict['quant_best_psnr'] = quantization.quant_best_psnr
        state_dict['quant_optimizer'] = quantization.quant_optimizer.state_dict()
        state_dict['quant_scheduler'] = quantization.quant_scheduler.state_dict()
        torch.save(state_dict, osp.join(save_path, 'quantization_state.pt'))
    
    def resume(self, quantization, resume_path):
        ori_path = osp.join(resume_path, 
                        self.ori_folder, 'model_latest.pt')
        ori_dual_path = osp.join(resume_path, 
                        self.ori_folder, 'dual_model_latest.pt')
        quantization.original_model.load(ori_path, ori_dual_path)
        
        quant_path = osp.join(resume_path, 
                            self.quant_folder, 'model_latest.pt')
        quant_dual_path = osp.join(resume_path, 
                            self.quant_folder, 'dual_model_latest.pt')
        quantization.quant_model.load(quant_path, quant_dual_path)
        
        resume_state = torch.load(osp.join(resume_path, 'quantization_state.pt'))
        quantization.start_epoch = resume_state['epoch'] + 1
        quantization.quant_best_epoch = resume_state['quant_best_epoch']
        quantization.quant_best_psnr = resume_state['quant_best_psnr']

        quantization.quant_optimizer.load_state_dict(resume_state['quant_optimizer'])
        quantization.quant_scheduler.load_state_dict(resume_state['quant_scheduler'])

    def save_results(self, filename, save_list, scale):
        apath = '{}/results/{}/X{}'.format(self.dir, self.args.data_test, scale)
        if not os.path.exists(apath):
            os.makedirs(apath)
        filename = os.path.join(apath, filename)

        v = save_list[0]
        normalized = v[0].data.mul(255 / self.args.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        imageio.imwrite('{}.png'.format(filename), ndarr)
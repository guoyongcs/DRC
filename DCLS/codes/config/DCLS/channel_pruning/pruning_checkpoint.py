import os
import torch
import torch.nn as nn
from utils.pruned_utils import list2sequential


class PruningCheckPoint():
    """
    Save model state to file
    check_point_params: original_model, pruned_model, aux_fc, current_pivot, index, block_count
    """

    def __init__(self, save_path, logger):
        self.ckp_save_path = os.path.join(save_path, "checkpoint")
        self.model_save_path = os.path.join(save_path, "model")
        self.check_point_params = {'model': None,
                                   'optimizer': None,
                                   'epoch': None}
        self.logger = logger

        # make directory
        if not os.path.isdir(self.ckp_save_path):
            os.makedirs(self.ckp_save_path)
        if not os.path.isdir(self.model_save_path):
            os.makedirs(self.model_save_path)
    
    def load_state(self, model, state_dict):
        """
        load state_dict to model
        :params model:
        :params state_dict:
        :return: model
        """
        model.eval()
        model_dict = model.state_dict()

        for key, value in list(state_dict.items()):
            if key in list(model_dict.keys()):
                model_dict[key] = value
            else:
                if self.logger:
                    self.logger.error("key error: {} {}".format(key, value.size))
                # assert False
        model.load_state_dict(model_dict)
        return model

    def load_checkpoint(self, checkpoint_path):
        """
        load checkpoint file
        :params checkpoint_path: path to the checkpoint file
        :return: model_state_dict, optimizer_state_dict, epoch
        """
        if os.path.isfile(checkpoint_path):
            if self.logger:
                self.logger.info("|===>Load resume check-point from: {}".format(checkpoint_path))
            self.check_point_params = torch.load(checkpoint_path)
            model_state_dict = self.check_point_params['model']
            optimizer_state_dict = self.check_point_params['optimizer']
            epoch = self.check_point_params['epoch']
            return model_state_dict, optimizer_state_dict, epoch
        else:
            assert False, "file not exits" + checkpoint_path
    
    def save_models(self, pruned_model, block_count=0, pruned=False):
        pruned_model = list2sequential(pruned_model)
        if isinstance(pruned_model, nn.DataParallel):
            pruned_model_state = pruned_model.module.state_dict()
        else:
            pruned_model_state = pruned_model.state_dict()

        model_save_name = "model_cs_{:0>3d}.pth".format(block_count)
        if pruned: model_save_name = 'pruned_' + model_save_name
        torch.save(pruned_model_state, 
            os.path.join(self.model_save_path, model_save_name))

    def save_checkpoints(self, pruned_model, block_count=0):
        # save state of the network
        check_point_params = {}

        pruned_model = list2sequential(pruned_model)
        if isinstance(pruned_model, nn.DataParallel):
            check_point_params["pruned_model"] = pruned_model.module.state_dict()
        else:
            check_point_params["pruned_model"] = pruned_model.state_dict()

        check_point_params["block_num"] = block_count
        checkpoint_save_name = "checkpoint_cs_{:0>3d}.pth".format(block_count)
        torch.save(check_point_params, os.path.join(self.ckp_save_path, checkpoint_save_name))
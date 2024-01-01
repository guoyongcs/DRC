import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.hooks as hooks
# import visdom 
from PIL import Image
import torchvision.transforms as transforms

class MaskedGradient(nn.Module):
    """docstring for MaskedGradient"""
    def __init__(self, opt):
        super(MaskedGradient, self).__init__()
        self.opt = opt
        # filter of image gradient
        self.dx = nn.Conv2d(in_channels=opt.n_colors, out_channels=opt.n_colors, \
            kernel_size=(1, 3), stride=1, padding=0, bias=False, groups=3)
        self.dy = nn.Conv2d(in_channels=opt.n_colors, out_channels=opt.n_colors, \
            kernel_size=(3, 1), stride=1, padding=0, bias=False, groups=3)

        # stable the parameter of dx and dy
        self.dx.weight.requires_grad = False
        self.dy.weight.requires_grad = False

        # initialize the weights of the kernels
        self._init_weights()

        self.criterion = nn.L1Loss(reduction=self.opt.l1_reduction)

        self.gamma = 1

    def _init_weights(self):
        weights_dx = torch.FloatTensor([1,0,-1])
        weights_dy = torch.FloatTensor([[1], [0], [-1]])
        if not self.opt.ycbcr:
            for i in range(self.dx.weight.size(0)):
                for j in range(self.dx.weight.size(1)):
                    self.dx.weight.data[i][j].copy_(weights_dx)

            for i in range(self.dy.weight.size(0)):
                for j in range(self.dy.weight.size(1)):
                    self.dy.weight.data[i][j].copy_(weights_dy)

    def _normalize(self, inputs):
        eps = 1e-5
        inputs_view = inputs.view(inputs.size(0), -1)
        min_element, _ = torch.min(inputs_view, 1)
        min_element = min_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        max_element, _ = torch.max(inputs_view, 1)
        max_element = max_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        outputs = (inputs - min_element) / (max_element - min_element + eps)
        return outputs

    def _abs_normalize(self, inputs):
        eps = 1e-5
        inputs_view = inputs.view(inputs.size(0), -1)
        f_norm = torch.norm(inputs_view, 2, 1)
        f_norm = f_norm.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        outputs = inputs / (f_norm + eps)
        return outputs

    def _combine_gradient_xy(self, gradient_x, gradient_y):
        eps = 1e-4
        return torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2) + eps)

    def _padding_gradient(self, gradient_x, gradient_y):
        # padding with 0 to ensure the same size of the masks and images
        output_x = F.pad(gradient_x, (1,1,0,0), "constant", 0)
        output_y = F.pad(gradient_y, (0,0,1,1), "constant", 0)
        return output_x, output_y

    def forward(self, inputs, targets):
        # input gradient
        inputs_grad_x = self.dx(inputs)
        inputs_grad_y = self.dy(inputs)
        
        # target gradient
        targets_grad_x = self.dx(targets.detach())
        targets_grad_y = self.dy(targets.detach())

        # padding with 0 to ensure the same size of the masks and images
        inputs_grad_x, inputs_grad_y = self._padding_gradient(inputs_grad_x, inputs_grad_y)
        targets_grad_x, targets_grad_y = self._padding_gradient(targets_grad_x, targets_grad_y)
        
        # inputs_grad = self._combine_gradient_xy(inputs_grad_x, inputs_grad_y)
        targets_grad = self._combine_gradient_xy(targets_grad_x, targets_grad_y)

        mask = self._normalize(targets_grad)
        # mask = self._abs_normalize(targets_grad)

        mask = mask.expand_as(inputs)
        P = mask**self.gamma
        CP = 1-P

        mse_inputs = CP * inputs
        mse_target = CP * targets

        # real P x I gradient
        inputs_x_new = self.dx(P * inputs)
        inputs_y_new = self.dy(P * inputs)
        targets_x_new = self.dx(P * targets)
        targets_y_new = self.dy(P * targets)
        
        # padding with 0 to ensure the same size of the grad_x and grad_y
        inputs_x_new, inputs_y_new = self._padding_gradient(inputs_x_new, inputs_y_new)
        targets_x_new, targets_y_new = self._padding_gradient(targets_x_new.detach(), targets_y_new.detach())
        
        # inputs_new_xy_grad = self._combine_gradient_xy(inputs_x_new, inputs_y_new)
        # target_new_xy_grad = self._combine_gradient_xy(targets_x_new, targets_y_new)

        # calculate the loss of gradient dividually
        grad_loss_x = self.criterion(inputs_x_new + self.opt.ROBUST, targets_x_new.detach() + self.opt.ROBUST)
        grad_loss_y = self.criterion(inputs_y_new + self.opt.ROBUST, targets_y_new.detach() + self.opt.ROBUST)
        grad_loss = grad_loss_x + grad_loss_y
        
        # combine the grad_x and grad_y to caculate the gradient loss
        # grad_loss = self.criterion(inputs_new_xy_grad + self.opt.ROBUST, target_new_xy_grad.detach() + self.opt.ROBUST)

        # calculate the loss of (1-P) x I (content loss)
        mse_loss = self.criterion(mse_inputs + self.opt.ROBUST, mse_target.detach() + self.opt.ROBUST)
        # calculate the total loss
        loss = grad_loss + self.opt.alpha * mse_loss
        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedGradientGrey(nn.Module):
    """docstring for MaskedGradientGrey"""
    def __init__(self, opt):
        super(MaskedGradientGrey, self).__init__()
        # filter of image gradient
        self.opt = opt
        self.dx = nn.Conv2d(in_channels=3, out_channels=1, \
            kernel_size=(1, 3), stride=1, padding=0, bias=False)
        self.dy = nn.Conv2d(in_channels=3, out_channels=1, \
            kernel_size=(3, 1), stride=1, padding=0, bias=False)

        # initialize the weights of the kernels
        self._init_weights()

        self.criterion = nn.MSELoss()

        self.gamma = 1

    def _init_weights(self):
        weights_dx = torch.FloatTensor([1,0,-1])
        weights_dy = torch.FloatTensor([[1], [0], [-1]])
        for i in range(1):
            for j in range(3):
                self.dx.weight.data[i][j].copy_(weights_dx)

        for i in range(1):
            for j in range(3):
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
        return torch.pow(torch.pow(gradient_x, 2)+torch.pow(gradient_y, 2), 0.5)

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

        inputs_grad = self._combine_gradient_xy(inputs_grad_x, inputs_grad_y)
        targets_grad = self._combine_gradient_xy(targets_grad_x, targets_grad_y)

        mask = self._normalize(targets_grad)
        # mask = self._abs_normalize(targets_grad)

        mask = mask.expand_as(inputs)
        P = mask**self.gamma
        CP = 1-P
        # masked input gradient
        masked_inputs_x = P * inputs_grad_x
        masked_inputs_y = P * inputs_grad_y
        masked_inputs_grad = P * inputs_grad
        # masked target gradient
        masked_targets_x = P * targets_grad_x
        masked_targets_y = P * targets_grad_y
        masked_targets_grad = P * targets_grad

        mse_inputs = CP * inputs
        mse_target = CP * targets

        # calculate the loss of the x axis and the y axis
        grad_loss_x = self.criterion(masked_inputs_x, masked_targets_x.detach())
        grad_loss_y = self.criterion(masked_inputs_y, masked_targets_y.detach())
        grad_loss = grad_loss_x + grad_loss_y

        # grad_loss = self.criterion(masked_inputs_grad, masked_targets_grad.detach())

        # calculate mse loss
        mse_loss = self.criterion(mse_inputs, mse_target.detach())

        loss = self.opt.alpha * grad_loss + mse_loss

        return loss

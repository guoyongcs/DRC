import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageGradient(nn.Module):
    def __init__(self, opt):
        super(ImageGradient, self).__init__()
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
        # self.vis = visdom.Visdom(port=8080)

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
    
    def _combine_gradient_xy(self, gradient_x, gradient_y):
        eps = 1e-4 # use to avoid the sqrt equal to 0, and avoid the emerge
        return torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2) + eps)

    def _padding_gradient(self, gradient_x, gradient_y):
        # padding with 0 to ensure the same size of the masks and images
        output_x = F.pad(gradient_x, (1,1,0,0), "constant", 0)
        output_y = F.pad(gradient_y, (0,0,1,1), "constant", 0)
        return output_x, output_y

    def forward(self, inputs, targets):
        # inputs gradient
        inputs_grad_x = self.dx(inputs)
        inputs_grad_y = self.dy(inputs)
        
        # targets gradient
        targets_grad_x = self.dx(targets.detach())
        targets_grad_y = self.dy(targets.detach())
        
        # targets gradient
        # padding with 0 to ensure the same size of the masks and images
        inputs_grad_x, inputs_grad_y = self._padding_gradient(inputs_grad_x, inputs_grad_y)
        targets_grad_x, targets_grad_y = self._padding_gradient(targets_grad_x, targets_grad_y)
        
        # combine the grad_x and grad_y
        inputs_grad = self._combine_gradient_xy(inputs_grad_x, inputs_grad_y)
        targets_grad = self._combine_gradient_xy(targets_grad_x, targets_grad_y)

        # caculate the weight of image gradient
        # if self.opt.cmask:
        #     # caculate mask
        #     if self.opt.topk:
        #         mask = self.grad_topK(targets, targets_grad)
        #     else:
        #         mask = self.caculate_mask(targets_grad)
        #     mask = mask.expand_as(inputs)
        #     complement_mask = 1 - mask
        #     weight = (1 + self.opt.alpha * complement_mask)
        # else:
        weight = 1
        
        # caculate gradient loss
        # grad_loss = self.criterion(inputs_grad + self.opt.ROBUST, targets_grad.detach() + self.opt.ROBUST)
        if self.opt.divide:
            # caculate the loss separately
            # print('gradient loss divide')
            grad_loss_x = self.criterion(weight * inputs_grad_x + self.opt.ROBUST, weight * targets_grad_x.detach() + self.opt.ROBUST)
            grad_loss_y = self.criterion(weight * inputs_grad_y + self.opt.ROBUST, weight * targets_grad_y.detach() + self.opt.ROBUST)
            grad_loss = grad_loss_x + grad_loss_y
        else:
            # caculate the loss after combine grad_x with grad_y
            grad_loss = self.criterion(weight * inputs_grad + self.opt.ROBUST, weight * targets_grad.detach() + self.opt.ROBUST)
        
        return grad_loss
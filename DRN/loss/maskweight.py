import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskWeightLoss(nn.Module):
    def __init__(self, opt):
        super(MaskWeightLoss, self).__init__()
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

    def _normalize(self, inputs):
        eps = 1e-5
        inputs_view = inputs.view(inputs.size(0), -1)
        min_element, _ = torch.min(inputs_view, 1)
        min_element = min_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        max_element, _ = torch.max(inputs_view, 1)
        max_element = max_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        outputs = (inputs - min_element) / (max_element - min_element + eps)
        return outputs
    
    def softer_softmax(self, inputs):
        mask_exp = torch.exp(self.opt.mask_lambda * inputs)
        cmask_exp = torch.exp(self.opt.mask_lambda * (1-inputs) )
        mask = mask_exp / (mask_exp+cmask_exp)
        return mask
    
    def caculate_mask(self, gradient):
        norm = self._normalize(gradient)
        return self.softer_softmax(norm)
    
    def grad_topK(self, targets, targets_grad):
        nelements = targets_grad.size(-2) * targets_grad.size(-1)
        k = int(self.opt.keep_ratio * nelements)
        if k > 0:
            temp_grad = targets_grad.view(targets.size(0), 1, nelements)
            topk = temp_grad.topk(k, 2)[0][:,:,-1]
            topk = topk.view(topk.size(0), topk.size(1), 1)
            mask = (temp_grad > topk).float()
            mask = mask.view(targets.size(0), 1, targets_grad.size(-2), targets_grad.size(-1))
        else:
            mask = torch.zeros(targets.size(0), 1, targets_grad.size(-2), targets_grad.size(-1)).cuda()
        return mask

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
        
        # caculate mask
        if self.opt.topk:
            mask = self.grad_topK(targets, targets_grad)
        else:
            mask = self.caculate_mask(targets_grad)
        mask = mask.expand_as(inputs)

        # caculate the content loss
        complement_mask = 1 - mask
        content_loss = self.criterion(complement_mask * inputs + self.opt.ROBUST, complement_mask * targets.detach() + self.opt.ROBUST)
        
        # caculate the gradient loss 
        if self.opt.divide:
            # caculate the loss separately
            grad_loss_x = self.criterion(mask * inputs_grad_x + self.opt.ROBUST, mask * targets_grad_x.detach() + self.opt.ROBUST)
            grad_loss_y = self.criterion(mask * inputs_grad_y + self.opt.ROBUST, mask * targets_grad_y.detach() + self.opt.ROBUST)
            gradient_loss = grad_loss_x + grad_loss_y
        else:
            # caculate the loss after combine grad_x with grad_y
            gradient_loss = self.criterion(mask * inputs_grad + self.opt.ROBUST, mask * targets_grad.detach() + self.opt.ROBUST)
        
        # caculate the total loss
        loss = content_loss + self.opt.alpha * gradient_loss
        return loss


class CMaskWeightLoss(nn.Module):
    def __init__(self, opt):
        super(CMaskWeightLoss, self).__init__()
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
    
    def grad_topK(self, targets, targets_grad):
        nelements = targets_grad.size(-2) * targets_grad.size(-1)
        k = int(self.opt.keep_ratio * nelements)
        if k > 0:
            temp_grad = targets_grad.view(targets.size(0), 1, nelements)
            topk = temp_grad.topk(k, 2)[0][:,:,-1]
            topk = topk.view(topk.size(0), topk.size(1), 1)
            mask = (temp_grad > topk).float()
            mask = mask.view(targets.size(0), 1, targets_grad.size(-2), targets_grad.size(-1))
        else:
            mask = torch.zeros(targets.size(0), 1, targets_grad.size(-2), targets_grad.size(-1)).cuda()
        return mask
    
    def _normalize(self, inputs):
        eps = 1e-5
        inputs_view = inputs.view(inputs.size(0), -1)
        min_element, _ = torch.min(inputs_view, 1)
        min_element = min_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        max_element, _ = torch.max(inputs_view, 1)
        max_element = max_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        outputs = (inputs - min_element) / (max_element - min_element + eps)
        return outputs
    
    def softer_softmax(self, inputs):
        mask_exp = torch.exp(self.opt.mask_lambda * inputs)
        cmask_exp = torch.exp(self.opt.mask_lambda * (1-inputs) )
        mask = mask_exp / (mask_exp+cmask_exp)
        return mask
    
    def caculate_mask(self, gradient):
        norm = self._normalize(gradient)
        return self.softer_softmax(norm)

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
        
        # caculate mask
        if self.opt.topk:
            mask = self.grad_topK(targets, targets_grad)
        else:
            mask = self.caculate_mask(targets_grad)
        mask = mask.expand_as(inputs)

        # caculate the content loss
        content_loss = self.criterion(inputs + self.opt.ROBUST, targets.detach() + self.opt.ROBUST)
        
        # caculate weight
        complement_mask = 1 - mask

        # caculate the gradient loss 
        if self.opt.divide:
            # caculate the loss separately
            grad_loss_x = self.criterion(complement_mask * inputs_grad_x + self.opt.ROBUST, complement_mask * targets_grad_x.detach() + self.opt.ROBUST)
            grad_loss_y = self.criterion(complement_mask * inputs_grad_y + self.opt.ROBUST, complement_mask * targets_grad_y.detach() + self.opt.ROBUST)
            gradient_loss = grad_loss_x + grad_loss_y
        else:
            # caculate the loss after combine grad_x with grad_y
            gradient_loss = self.criterion(complement_mask * inputs_grad + self.opt.ROBUST, complement_mask * targets_grad.detach() + self.opt.ROBUST)
        
        # caculate the total loss
        loss = content_loss + self.opt.alpha * gradient_loss
        return loss
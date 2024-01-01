import utility
from model import common
from types import SimpleNamespace
from loss import discriminator
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class AdversarialGrad(nn.Module):
    def __init__(self, args, gan_type):
        super(AdversarialGrad, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.dis = discriminator.Discriminator(args)
        if gan_type == 'WGAN_GP_G':
            # see https://arxiv.org/pdf/1704.00028.pdf pp.4
            optim_dict = {
                'optimizer': 'ADAM',
                'beta1': 0,
                'beta2': 0.9,
                'epsilon': 1e-8,
                'lr': 1e-5,
                'weight_decay': args.weight_decay,
                'decay': args.lr_decay,
                'gamma': args.gamma
            }
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_args = args

        self.optimizer = utility.make_optimizer(optim_args, self.dis)
        self.get_grad = Get_gradient()

    def forward(self, fake, real):
        ## get grad
        fake = self.get_grad(fake)
        real = self.get_grad(real)
        # updating discriminator...
        self.loss = 0
        fake_detach = fake.detach()     # do not backpropagate through G
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            # d: B x 1 tensor
            d_fake = self.dis(fake_detach)
            d_real = self.dis(real)
            retain_graph = False
            if self.gan_type == 'GAN_G':
                loss_d = self.bce(d_real, d_fake)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(d_fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.dis(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            # from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
            elif self.gan_type == 'RGAN_G':
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = self.bce(better_real, better_fake)
                retain_graph = True

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()

            if self.gan_type == 'WGAN_G':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        # updating generator...
        d_fake_bp = self.dis(fake)      # for backpropagation, use fake as it is
        if self.gan_type == 'GAN_G':
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_bp.mean()
        elif self.gan_type == 'RGAN_G':
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
            loss_g = self.bce(better_fake, better_real)

        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
from . import search_util

class ArchitectUpdate():
    """ Compute gradients of alphas """
    def __init__(self, controller, w_momentum, w_weight_decay):
        """
        Args:
            controller
            w_momentum: weights momentum
        """
        self.controller = controller
        self.v_controller = copy.deepcopy(controller)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t, w_lr, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            w_lr: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.controller.loss(trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.controller.weights())

        # avoid nan value
        gradients = search_util.correct_nan(gradients)
        # clip grad for stability
        gradients = search_util.clip_value_norm(gradients)
        for g in gradients: g.clamp_(min=-.1, max=.1)
        
        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.controller.weights(), self.v_controller.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - w_lr * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.controller.alphas(), self.v_controller.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t, 
                          val_lr, val_hr, val_lr_blured_t, val_lr_t, 
                          w_lr, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            w_lr: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t, w_lr, w_optim)

        # calc unrolled loss
        loss = self.v_controller.loss(val_lr, val_hr, val_lr_blured_t, val_lr_t) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_controller.alphas())
        v_weights = tuple(self.v_controller.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        
        # avoid nan value
        v_grads = search_util.correct_nan(v_grads)
        # clip grad for stability
        v_grads = search_util.clip_value_norm(v_grads)
        for vg in v_grads: vg.clamp_(min=-.1, max=.1)

        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t)

        # avoid nan value
        hessian = search_util.correct_nan(hessian)
        # # clip grad for stability
        hessian = search_util.clip_value_norm(hessian)
        for h in hessian: h.clamp_(min=-.1, max=.1)

        # update final gradient = dalpha - w_lr*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.controller.alphas(), dalpha, hessian):
                alpha.grad = da - w_lr*h
        
        return loss

    def compute_hessian(self, dw, trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.reshape(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.controller.weights(), dw):
                p += eps * d
        loss = self.controller.loss(trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t)
        dalpha_pos = torch.autograd.grad(loss, self.controller.alphas()) # dalpha { L_trn(w+) }

        # avoid nan value
        dalpha_pos = search_util.correct_nan(dalpha_pos)
        # clip grad for stability
        dalpha_pos = search_util.clip_value_norm(dalpha_pos)
        for da_p in dalpha_pos: da_p.clamp_(min=-.1, max=.1)
        

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.controller.weights(), dw):
                p -= 2. * eps * d
        loss = self.controller.loss(trn_lr, trn_hr, trn_lr_blured_t, trn_lr_t)
        dalpha_neg = torch.autograd.grad(loss, self.controller.alphas()) # dalpha { L_trn(w-) }

        # avoid nan value
        dalpha_neg = search_util.correct_nan(dalpha_neg)
        # clip grad for stability
        dalpha_neg = search_util.clip_value_norm(dalpha_neg)
        for da_n in dalpha_neg: da_n.clamp_(min=-.1, max=.1)
        

        # recover w
        with torch.no_grad():
            for p, d in zip(self.controller.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

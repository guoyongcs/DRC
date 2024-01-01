""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch


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

    def virtual_step(self, trn_lr, trn_hr, w_lr, w_optim):
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
        loss = self.controller.loss(trn_lr, trn_hr) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.controller.weights())

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

    def unrolled_backward(self, trn_lr, trn_hr, val_lr, val_hr, w_lr, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            w_lr: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_lr, trn_hr, w_lr, w_optim)

        # calc unrolled loss
        loss = self.v_controller.loss(val_lr, val_hr) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_controller.alphas())
        v_weights = tuple(self.v_controller.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_lr, trn_hr)

        # update final gradient = dalpha - w_lr*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.controller.alphas(), dalpha, hessian):
                alpha.grad = da - w_lr*h
        
        return loss

    def compute_hessian(self, dw, trn_lr, trn_hr):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.controller.weights(), dw):
                p += eps * d
        loss = self.controller.loss(trn_lr, trn_hr)
        dalpha_pos = torch.autograd.grad(loss, self.controller.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.controller.weights(), dw):
                p -= 2. * eps * d
        loss = self.controller.loss(trn_lr, trn_hr)
        dalpha_neg = torch.autograd.grad(loss, self.controller.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.controller.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

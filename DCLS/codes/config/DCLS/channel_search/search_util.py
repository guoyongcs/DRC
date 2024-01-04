import torch

def clip_value_norm(gradients, max_norm=.1, norm_type=2.0):
    """
    Inputs:
        gradients: tuple of tensors,
    
    return:
        gradients after clamp norm
    """
    device = gradients[0].device
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    total_norm = torch.norm(torch.stack(
        [torch.norm(grad.detach(), norm_type).to(device) for grad in gradients]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in gradients:
            grad.detach().mul_(clip_coef.to(grad.device))

    return gradients


def correct_nan(grad):
    """
    Fix nan
    :param grad: gradient input
    """
    if isinstance(grad, (tuple, list)):
        for g in grad:
            g.masked_fill_(g.ne(g), 0)
    else:
        grad.masked_fill_(grad.ne(grad), 0)
    return grad
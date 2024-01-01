import torch.nn as nn
from model.common import Conv2dInt8

__all__ = ['replace_layer_by_unique_name', 
            'get_layer_by_unique_name', 'replace_int8']

def replace_layer_by_unique_name(module, unique_name, layer): 
    unique_names = unique_name.split(".") 
    if len(unique_names) == 1: 
        module._modules[unique_names[0]] = layer 
    else: 
        replace_layer_by_unique_name( 
            module._modules[unique_names[0]], 
            ".".join(unique_names[1:]),
            layer
        )

def get_layer_by_unique_name(module, unique_name): 
    unique_names = unique_name.split(".") 
    if len(unique_names) == 1: 
        return module._modules[unique_names[0]] 
    else:
        return get_layer_by_unique_name(
            module._modules[unique_names[0]], 
            ".".join(unique_names[1:]), 
        )

def replace_int8(model, activation_value_list, logger):
    count = 0
    last_tail_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            logger.info("count={}, name={}, activation_value={}".\
                    format(count, name, activation_value_list[name]))
            temp_conv = Conv2dInt8(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                groups=module.groups,
                bias=(module.bias is not None),
                activation_value=float(activation_value_list[name]))
            temp_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                temp_conv.bias.data.copy_(module.bias.data)
            # put the temp conv into target device
            device = module.weight.device
            temp_conv = temp_conv.to(device)
            replace_layer_by_unique_name(model, name, temp_conv)
            count += 1
            if name.find('tail') >= 0:
                last_tail_name = name
            if name.find('mid_part') >= 0:
                last_tail_name = name
    last_scale = float(activation_value_list[last_tail_name])
    return last_scale

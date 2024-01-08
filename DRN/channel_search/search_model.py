import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import RCAB, DownBlock, ResBlock, SearchConv2d, SearchMultiConv2d

class SearchModel(nn.Module):
    def __init__(self, args, model, logger):
        super(SearchModel, self).__init__()
        self.args = args
        self.model = model
        self.logger = logger
        
        self.replace_all_layers()
    
    def forward(self, lr, alphas, channel_sets):
        if not self.args.search_multi_branch:
            self.set_model_width(alphas, channel_sets)
        if isinstance(lr , list): lr = lr[0]
        sr = self.model(lr)
        sr2lr = self.model.dual_forward(sr) if self.args.dual else None
        return sr, sr2lr

    def set_model_width(self, alphas, channel_sets):
        alphas_softmax = [F.softmax(a, dim=-1) for a in alphas]
        for idx, module in enumerate(self.search_modules):
            if isinstance(module, (RCAB, ResBlock)):
                layer = module.body[2]
            elif isinstance(module, DownBlock):
                layer = module.dual_module[1]
            elif isinstance(module, SearchConv2d):
                layer = module
            else:
                assert False, "Wrong type: {}".format(type(module))
            assert isinstance(layer, SearchConv2d), 'Wrong type of the search layer'

            # sum up the channel weights rather than suming up the output, 
            # which avoids forward for servel times 
            channel_weights = self.compute_channel_weights(
                alphas_softmax[idx], channel_sets[idx], layer)
            # set weights for all channels
            layer.set_channel_weights(channel_weights)

    def compute_channel_weights(self, alphas_softmax, channel_options, layer):
        if not self.args.unordered_channels:
            # sort the channels based on L1-norm
            with torch.no_grad():
                values, indices = torch.sort(
                    layer.weight.data.abs().sum(dim=(0,2,3)), descending=True)
                indices = indices.cpu().numpy()
        else:
            # do not sort the channels
            indices = torch.arange(layer.in_channels).numpy()
        
        channel_weights = torch.zeros(layer.in_channels).cuda()
        for i, channels_num in enumerate(channel_options):
            # sum up the weights for each channels
            for j in range(channels_num):
                channel_weights[indices[j]] = channel_weights[indices[j]] + alphas_softmax[i]
        return channel_weights  
    
    def get_search_modules(self):
        search_modules = None
        for module in self.model.modules():
            if isinstance(module, (DownBlock, RCAB, ResBlock, nn.Conv2d)):
                if isinstance(module, nn.Conv2d):
                    # search the 1x1 conv after Upsamplers
                    k_size = module.kernel_size
                    in_chans = module.in_channels
                    out_chans = module.out_channels
                    if k_size != (1, 1): continue
                    if in_chans / out_chans not in [2., 4.]: continue
                    
                self.logger.info("enter block: {}".format(type(module)))
                if search_modules is not None:
                    search_modules.add_module(str(len(search_modules)), module)
                else:
                    search_modules = nn.Sequential(module)

        self.search_modules = search_modules
        assert len(search_modules) == self.model.count_layers(), \
            "The number of the search modules are wrong."
        self.logger.info('The number of search modules: {}'.format(len(search_modules)))
        self.logger.info(search_modules)

    def replace_all_layers(self):
        """
        Replace the convolutional layer with mask convolutional layer
        """
        self.get_search_modules()
        for idx, module in enumerate(self.search_modules):
            self.replace_one_layer(module, idx)

        self.logger.info('Replace layers done ... ')
        self.logger.info(self.search_modules)

    def replace_one_layer(self, module, module_idx):
        """replace the layer in original model to selected channels"""
        if isinstance(module, (RCAB, ResBlock)):
            layer = module.body[2]
        elif isinstance(module, DownBlock):
            layer = module.dual_module[1]
        elif isinstance(module, nn.Conv2d):
            layer = module

        if not isinstance(layer, (SearchConv2d, SearchMultiConv2d)):
            if not self.args.search_multi_branch:
                # use the SearchConv2d for searching by default
                temp_conv = SearchConv2d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=(layer.bias is not None))
                device = layer.weight.device
                temp_conv = temp_conv.to(device)

                temp_conv.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    temp_conv.bias.data.copy_(layer.bias.data)
            else:
                # use the module with multi conv2d branch for searching
                temp_conv = SearchMultiConv2d(self.args, layer)
            
            if isinstance(module, (RCAB, ResBlock)):
                module.body[2] = temp_conv
            elif isinstance(module, DownBlock):
                module.dual_module[1] = temp_conv
            elif isinstance(module, nn.Conv2d):
                for i in range(len(self.args.scale)):
                    conv1 = self.model.get_model().up_blocks[i][-1]
                    if conv1.out_channels == temp_conv.out_channels:
                        self.model.get_model().up_blocks[i][-1] = temp_conv
                        self.search_modules[module_idx] = temp_conv

    def get_layers(self, idx):
        module = self.search_modules[idx]
        if isinstance(module, (RCAB, ResBlock)):
            layer = module.body[2]
        elif isinstance(module, DownBlock):
            layer = module.dual_module[1]
        elif isinstance(module, (nn.Conv2d, SearchMultiConv2d)):
            layer = module
        else:
            assert False, "Unsupport type of modules to be searched"
        return layer

    def init_search_modules(self, channel_sets, alpha_all):
        for idx in range(len(self.search_modules)):
            layer = self.get_layers(idx)
            layer.init_multi_branchs(channel_sets[idx], alpha_all[idx])
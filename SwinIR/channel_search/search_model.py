import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network_swinir_pruned import WindowAttention, Mlp, SwinTransformerBlock
from .common_module import SearchLinear, SearchOutLinear, SearchWindowAttention

class SearchModel(nn.Module):
    def __init__(self, args, model, logger):
        super(SearchModel, self).__init__()
        self.args = args
        self.model = model
        self.logger = logger
        
        self.replace_all_layers()
    
    def forward(self, lr, alphas, channel_sets):
        self.set_model_width(alphas, channel_sets)
        sr, sr2lr = self.model(lr)
        return sr, sr2lr

    def set_model_width(self, alphas, channel_sets):
        alphas_softmax = [F.softmax(a, dim=-1) for a in alphas]
        for idx, module in enumerate(self.search_modules):
            if isinstance(module, Mlp):
                layer = module.fc2
                assert isinstance(layer, SearchLinear), \
                    'Wrong type of the search layer'

                # sum up the channel weights rather than suming up the output, 
                # which avoids forward for servel times 
                channel_weights = self.compute_channel_weights(
                    alphas_softmax[idx], channel_sets[idx], layer)
                # set weights for all channels
                layer.set_channel_weights(channel_weights)
            

    def compute_channel_weights(self, alphas_softmax, channel_options, layer):
        if not self.args.opt['search']['unordered_channels']:
            # sort the channels based on L1-norm
            with torch.no_grad():
                values, indices = torch.sort(
                    layer.weight.data.abs().sum(dim=(0)), descending=True)
                indices = indices.cpu().numpy()
        else:
            # do not sort the channels
            indices = torch.arange(layer.in_features).numpy()
        
        channel_weights = torch.zeros(layer.in_features).cuda()
        for i, channels_num in enumerate(channel_options):
            # sum up the weights for each channels
            for j in range(channels_num):
                channel_weights[indices[j]] = channel_weights[indices[j]] + alphas_softmax[i]
        return channel_weights
    
    def get_search_modules(self):
        search_modules = None
        for module in self.model.modules():
            if isinstance(module, Mlp):
                self.logger.info("enter block: {}".format(type(module)))
                if search_modules is not None:
                    search_modules.add_module(str(len(search_modules)), module)
                else:
                    search_modules = nn.Sequential(module)
            
            # record the module for the search process of WindowAttention 
            if isinstance(module, (SwinTransformerBlock)): 
                self.logger.info("enter block: {}".format(type(module)))
                if search_modules is not None:
                    search_modules.add_module(str(len(search_modules)), module)
                else:
                    search_modules = nn.Sequential(module)

        self.search_modules = search_modules
        self.logger.info('The number of search modules: {}'.format(len(search_modules)))
        self.logger.info(search_modules)
        return None

    def replace_all_layers(self):
        """
        Replace the linear layer with search linear layer
        Replace the winatt module with search winatt module
        """
         # replace WindowAttention with SearchWindowAttention
        for module in self.model.modules():
            # replace windowattion with search windowattion
            if isinstance(module, SwinTransformerBlock):
                module.attn = SearchWindowAttention(
                    self.args, module.attn, alpha=None)

        self.get_search_modules()
        for idx, module in enumerate(self.search_modules):
            self.replace_one_layer(module, idx)

        self.logger.info('Replace layers done ... ')
        self.logger.info(self.search_modules)

    def replace_one_layer(self, module, module_idx): 
        """replace the layer in original model to selected channels"""
        if isinstance(module, WindowAttention):
            layer = module.proj
            layer_qkv = module.qkv
        elif isinstance(module, SwinTransformerBlock):
            layer = module.attn.win_att.proj
            layer_qkv = module.attn.win_att.qkv
        elif isinstance(module, Mlp):
            layer = module.fc2
        else:
            assert False, "unsupport layer: {}".format(type(layer))

        if not isinstance(layer, (SearchLinear)):
            # use the SearchConv2d for searching by default
            temp_layer = SearchLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=(layer.bias is not None))
            device = layer.weight.device
            temp_layer = temp_layer.to(device)

            temp_layer.weight.data.copy_(layer.weight.data)
            if layer.bias is not None:
                temp_layer.bias.data.copy_(layer.bias.data)
            
            if isinstance(module, (WindowAttention, SwinTransformerBlock)):
                temp_qkv = SearchOutLinear(
                    in_features=layer_qkv.in_features,
                    out_features=layer_qkv.out_features,
                    bias=(layer_qkv.bias is not None))

                temp_qkv = temp_qkv.to(device)

                temp_qkv.weight.data.copy_(layer_qkv.weight.data)
                if layer.bias is not None:
                    temp_qkv.bias.data.copy_(layer_qkv.bias.data)
            
        if isinstance(module, WindowAttention):
            module.proj = temp_layer
            module.qkv = temp_qkv
        elif isinstance(module, SwinTransformerBlock):
            module.attn.win_att.proj = temp_layer
            module.attn.win_att.qkv = temp_qkv
        elif isinstance(module, Mlp):
            module.fc2  = temp_layer

    def get_layers(self, idx):
        module = self.search_modules[idx]
        if isinstance(module, WindowAttention):
            layer = module.proj
        elif isinstance(module, SwinTransformerBlock):
            layer = module.attn.win_att.proj
        elif isinstance(module, SearchWindowAttention):
            layer = module.win_att.proj
        elif isinstance(module, Mlp):
            layer = module.fc2
        else:
            assert False, "Unsupport type of modules to be searched"
        return layer
    
    def get_search_channel_num(self, idx):
        module = self.search_modules[idx]
        num = 0
        if isinstance(module, WindowAttention):
            num = module.proj.in_features // module.num_heads
        if isinstance(module, SwinTransformerBlock):
            module = module.attn.win_att
            num = module.proj.in_features // module.num_heads
        elif isinstance(module, SearchWindowAttention):
            module = module.win_att
            num = module.proj.in_features // module.num_heads
        elif isinstance(module, Mlp):
            num = module.fc2.in_features
        else:
            assert False, "Unsupport type of modules to be searched"
        return num
    
    def init_search_modules(self, channel_sets, alpha_all):
        """initialize alpha for search window attention"""
        for idx, module in enumerate(self.search_modules):
            if isinstance(module, SwinTransformerBlock):
                module.attn.set_alpha(alpha_all[idx])
                module.attn.set_channel_options(channel_sets[idx])
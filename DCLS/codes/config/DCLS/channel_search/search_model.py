import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.dcls_arch import DPCAB, Estimator
from models.modules.module_util import ResidualBlock_noBN
from .common_module import SearchConv2d, SearchLinear, SearchModuleList

class SearchModel(nn.Module):
    def __init__(self, args, model, logger):
        super(SearchModel, self).__init__()
        self.args = args
        self.model = model
        self.logger = logger
        
        self.replace_all_layers()
    
    def forward(self, lr, alphas, channel_sets):
        self.set_model_width(alphas, channel_sets)
        sr, est_kernel, sr2lr = self.model(lr)
        return sr, est_kernel, sr2lr

    def set_model_width(self, alphas, channel_sets):
        alphas_softmax = [F.softmax(a, dim=-1) for a in alphas]
        for idx, module in enumerate(self.search_modules):
            if isinstance(module, DPCAB):
                layer = module.body1[2]
            elif isinstance(module, ResidualBlock_noBN):
                layer = module.conv2
            elif isinstance(module, Estimator):
                # all modules in dec share the same weight
                layer = module.dec[-1] 
            else:
                assert False, "Wrong type: {}".format(type(module))
            assert isinstance(layer, (SearchConv2d, SearchLinear)), \
                    'Wrong type of the search layer'

            # sum up the channel weights rather than suming up the output, 
            # which avoids forward for servel times 
            channel_weights = self.compute_channel_weights(
                alphas_softmax[idx], channel_sets[idx], layer)
            
            # set weights for all channels
            if isinstance(module, Estimator):
                module.dec.set_channel_weights(channel_weights)
            else:
                layer.set_channel_weights(channel_weights)

    def compute_channel_weights(self, alphas_softmax, channel_options, layer):
        if isinstance(layer, SearchLinear):
            num_channels = layer.in_features
            sum_dims = (0)
        else:
            num_channels = layer.in_channels
            sum_dims = (0,2,3)
        
        if not self.args.opt['search']['unordered_channels']:
            # sort the channels based on L1-norm
            with torch.no_grad():
                values, indices = torch.sort(
                    layer.weight.data.abs().sum(dim=sum_dims), descending=True)
                indices = indices.cpu().numpy()
        else:
            # do not sort the channels
            indices = torch.arange(num_channels).numpy()
        
        channel_weights = torch.zeros(num_channels).cuda()
        for i, channels_num in enumerate(channel_options):
            # sum up the weights for each channels
            for j in range(channels_num):
                channel_weights[indices[j]] = channel_weights[indices[j]] + alphas_softmax[i]
        return channel_weights
    
    def get_search_modules(self):
        search_modules = None
        for module in self.model.modules():
            if isinstance(module, (DPCAB, ResidualBlock_noBN, nn.ModuleList)): 
                self.logger.info("enter block: {}".format(type(module)))

                if isinstance(module, nn.ModuleList):
                    # replace the ModuleList with the prarent module
                    for m in self.model.modules():
                        if isinstance(m, Estimator):
                            module = m
                            break
                    assert isinstance(module, Estimator)
                
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
        self.get_search_modules()
        for idx, module in enumerate(self.search_modules):
            self.replace_one_layer(module, idx)

        self.logger.info('Replace layers done ... ')
        self.logger.info(self.search_modules)

    def replace_one_layer(self, module, module_idx):
        """replace the layer in original model to selected channels"""
        if isinstance(module, DPCAB):
            layer = module.body1[2]
        elif isinstance(module, ResidualBlock_noBN):
            layer = module.conv2
        elif isinstance(module, Estimator):
            layer = module.dec
        else:
            assert False, "unsupport layer: {}".format(type(layer))

        if not isinstance(layer, (SearchConv2d)):
            
            if isinstance(module, Estimator):
                temp_layer = SearchModuleList(layer[:])

                # put into the same device
                device = layer[-1].weight.device
                temp_layer = temp_layer.to(device)
            
            else:
                # use the SearchConv2d for searching by default
                temp_layer = SearchConv2d(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        bias=(layer.bias is not None))
                device = layer.weight.device
                temp_layer = temp_layer.to(device)

                temp_layer.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    temp_layer.bias.data.copy_(layer.bias.data)
            
        if isinstance(module, DPCAB):
            module.body1[2] = temp_layer
        elif isinstance(module, ResidualBlock_noBN):
            module.conv2  = temp_layer
        elif isinstance(module, Estimator):
            module.dec = temp_layer

    def get_layers(self, idx):
        module = self.search_modules[idx]
        if isinstance(module, DPCAB):
            layer = module.body1[2]
        elif isinstance(module, ResidualBlock_noBN):
            layer = module.conv2
        elif isinstance(module, Estimator):
            layer = module.dec
        else:
            assert False, "Unsupport type of modules to be searched"
        return layer
    
    def get_search_channel_num(self, idx):
        module = self.search_modules[idx]
        if isinstance(module, DPCAB):
            num = module.body1[2].in_channels
        elif isinstance(module, ResidualBlock_noBN):
            num = module.conv2.in_channels
        elif isinstance(module, Estimator):
            num = module.dec[-1].in_features
        else:
            assert False, "Unsupport type of modules to be searched"
        return num

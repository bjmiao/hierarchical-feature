import torch
import torch.nn as nn
import torchvision.models as models

import sys
class AlexNetDeconv(nn.Module):
    def __init__(self):
        super(AlexNetDeconv, self).__init__()
        self.features = nn.Sequential(
            #1st
            nn.MaxUnpool2d(stride = 2, kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 384, 3, padding = 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(384, 192, 3, padding = 1),
            
            #2nd
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 64, 5, padding = 2),
            
            #3rd
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 11, stride= 4, padding = 2),
        )
        
        self.conv_deconv_idx = {0:12 , 3:9 , 6:6 , 8:4, 10:2} # Mapping location of conv2d layer in both Convnet and Deconvnet.
        self.unpool_pool_idx = {10:2, 7:5, 0:12} # Mapping location of Un-Pooling layer:Pooling Layer
        self._init_weights() # assigning same weights of alexnet pretrained to our deconvnet model 
    
    def _init_weights(self):
        alexnet_pre = models.alexnet(pretrained = True)
        for ids, layer in enumerate(alexnet_pre.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv_deconv_idx[ids]].weight.data = layer.weight.data
    
    def forward(self, x, layer, pool_locs):
        if layer in self.conv_deconv_idx:
            start_idx = self.conv_deconv_idx[layer]
        else:
            print("It is not a conv layer")
        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d): 
        #If the layer is unpooling we need to pass 2 inputs. First is x and second is switches stored in pool_locs 
        # when we call return indices = true in MaxPool2d
                x = self.features[idx](x, pool_locs[self.unpool_pool_idx[idx]])
            else:
                x = self.features[idx](x)
        return x        

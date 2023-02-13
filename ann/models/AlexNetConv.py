import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

from collections import OrderedDict
class AlexNetConv(nn.Module):
    def __init__(self):
        super(AlexNetConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride = 4, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride = 2, return_indices=True), 
    # return indices has to be true as it act as switches which holds the location of max pixel values from receptive-field
            
            #2nd
            nn.Conv2d(64, 192, kernel_size = 5, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride = 2, return_indices = True),
            
            #3rd
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride = 2, return_indices = True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.Softmax(dim = 1),
        )
        self.feature_maps = OrderedDict() # For storing all the feature maps of *this* convnet_model {0: [[[[]]]], .....}
        self.pool_locs = OrderedDict() # For storing the switch indices for pool layers {2: [[[[]]]], .....}
        self.conv_layer_indices = [0, 3, 6, 8, 10] #storing the conv2d layer indices
        self.init_weights() # calling function to assign the pretrained alexnet weights to our model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def init_weights(self):
        alexnet_pre = self.check()
        for idx, layer in enumerate(alexnet_pre.features):
        # 0 , Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)). This how idx, layer will look like
            if isinstance(layer, nn.Conv2d): #checking if layer is nn.Conv2d or not if  yes then proceed
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data
                
        for idx, layer in enumerate(alexnet_pre.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data

    def check(self):
        model = models.alexnet(pretrained=True)
        return model

if __name__ == '__main__':
    model = models.alexnet(pretrained=True)
    print(model)

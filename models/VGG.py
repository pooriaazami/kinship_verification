import torch
import torch.nn as nn

class VGGModel(nn.Module):
    def __init__(self, include_top=False):
        super(VGGModel, self).__init__()
        self.network = nn.Sequential(
            nn.Sequential(
                self.vgg_conv_layer(3, 64),
                self.vgg_conv_layer(64, 128),
                self.vgg_conv_layer(128, 256, True),
                self.vgg_conv_layer(256, 512, True),
                self.vgg_conv_layer(512, 512, True),
            ),
            nn.Flatten(),
            self.fc_layers(include_top)
        )
        
    def vgg_conv_layer(self, input_shape, output_shape, three_layered=False):
        if three_layered:
            block = nn.Sequential(
                nn.Conv2d(input_shape, output_shape, 3, padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(output_shape, output_shape, 3, padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(output_shape, output_shape, 3, padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        else:
            block = nn.Sequential(
                nn.Conv2d(input_shape, output_shape, 3, padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(output_shape, output_shape, 3, padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            
        return block
            
    def fc_layers(self, include_top):
        if include_top:
            block = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 2622)
            )
        else:
            block = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
            )
        
        return block
    
    def forward(self, x):
        return self.network(x)

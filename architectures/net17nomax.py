from architectures.template import LayerBlock
from torch import nn
import torch

class Net17NoMax(nn.Module):
    def __init__(self, numberFeatures):
        super(Net17NoMax, self).__init__()
        layers = [
            LayerBlock(1, numberFeatures, 3, False), 
            LayerBlock(numberFeatures, numberFeatures, 3, False), 
            LayerBlock(numberFeatures, numberFeatures, 3, False), 
            LayerBlock(numberFeatures, numberFeatures, 3, False),
            LayerBlock(numberFeatures, numberFeatures, 3, False),
            LayerBlock(numberFeatures, numberFeatures, 3, False),
            LayerBlock(numberFeatures, numberFeatures, 3, False),
            nn.Conv2d(numberFeatures, 1, kernel_size=3, padding="valid", stride=(1, 1)),
            nn.Sigmoid() 
        ]
        self.net = nn.Sequential(*layers)
        self.numberLayers = 8

        self.seed = 0
        self.apply(self.reset_parameters)

    def reset_parameters(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.manual_seed(self.seed)
            m.reset_parameters()



    def forward(self, x):
        return self.net(x)

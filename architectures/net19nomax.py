from architectures.template import LayerBlock
from torch import nn

class Net19NoMax(nn.Module):
    def __init__(self, numberFeatures):
        super(Net19NoMax, self).__init__()
        layers = [
            LayerBlock(1, numberFeatures, 3, False), 
            LayerBlock(numberFeatures, numberFeatures, 3, False), 
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
        self.numberLayers = 9


    def forward(self, x):
        return self.net(x)


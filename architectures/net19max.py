from architectures.template import LayerBlock
from torch import nn

class Net19Max(nn.Module):
    def __init__(self, numberFeatures):
        super(Net19Max, self).__init__()
        layers = [
            LayerBlock(1, numberFeatures, 3, True),                     #19 - 15 
            LayerBlock(numberFeatures, numberFeatures, 3, True),        #15 - 11
            LayerBlock(numberFeatures, numberFeatures, 3, True),        #11 - 7
            LayerBlock(numberFeatures, numberFeatures, 3, True),        #7-3
            nn.Conv2d(numberFeatures, 1, kernel_size=3, padding="valid", stride=(1, 1)), # 3-1
            nn.MaxPool2d(kernel_size=1, stride=(1, 1)),                 # 1
            nn.Sigmoid() 
        ]
        self.net = nn.Sequential(*layers)
        self.numberLayers = 4


    def forward(self, x):
        return self.net(x)

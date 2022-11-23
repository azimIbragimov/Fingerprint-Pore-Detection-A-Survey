from architectures.template import LayerBlock
from torch import nn

class Net17Max(nn.Module):
    def __init__(self, numberFeatures):
        super(Net17Max, self).__init__()
        layers = [
            LayerBlock(1, numberFeatures, 3, True),             #17 - 14
            LayerBlock(numberFeatures, numberFeatures, 3, True), #14 - 9
            LayerBlock(numberFeatures, numberFeatures, 3, True), #9 - 5
            nn.Conv2d(numberFeatures, 1, kernel_size=3, padding="valid", stride=(1, 1)),#5-3
            nn.MaxPool2d(kernel_size=3, stride=(1, 1)), # 3-1
            nn.Sigmoid() 
        ]
        self.net = nn.Sequential(*layers)
        self.numberLayers = 3


    def forward(self, x):
        return self.net(x)

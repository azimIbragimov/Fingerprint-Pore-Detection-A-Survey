from architectures.template import LayerBlock
from torch import nn

class Gabriel(nn.Module):
    def __init__(self, numberFeatures):
        super(Gabriel, self).__init__()
        layers = [
            LayerBlock(1, numberFeatures, 3, True),                 #17 -> 13
            LayerBlock(numberFeatures, numberFeatures*2, 3, True),  #13 -> 9
            LayerBlock(numberFeatures*2, numberFeatures*4, 3, True), #9->5 
            nn.Dropout(p=0.2),
            nn.Conv2d(                                              #5 -> 1
                numberFeatures*4, 1, 
                kernel_size=5, 
                padding="valid", 
                stride=(1, 1)),
            nn.BatchNorm2d(1), 
            nn.Sigmoid() 
        ]
        self.net = nn.Sequential(*layers)
        self.numberLayers = 3


    def forward(self, x):
        return self.net(x)

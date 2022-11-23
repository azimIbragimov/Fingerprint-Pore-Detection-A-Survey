from architectures.template import LayerBlock
from torch import nn

class ResNet17Max(nn.Module):
    def __init__(self, numberFeatures):
        super(ResNet17Max, self).__init__()
        self.l1 = LayerBlock(1, numberFeatures, 3, True)   #17->13
        self.l2 = LayerBlock(numberFeatures, numberFeatures, 3, True)   #13->9
        self.l3 = LayerBlock(numberFeatures, 1, 3, True)    #9->5
        self.l4 = nn.Conv2d(numberFeatures, 1, kernel_size=3, padding="valid", stride=(1, 1))
        self.l5 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))       
        self.sigmoid = nn.Sigmoid() 
        self.numberLayers = 3


    def forward(self, x):
        x = self.l1(x)
        x = x[:,:,self.l2.resValue:-self.l2.resValue,self.l2.resValue:-self.l2.resValue] + self.l2(x)
        x = x[:,:,self.l3.resValue:-self.l3.resValue,self.l3.resValue:-self.l3.resValue] + self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.sigmoid(x)
        return x


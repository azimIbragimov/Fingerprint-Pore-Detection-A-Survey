from architectures.template import LayerBlock
from torch import nn

class ResNet13NoMax(nn.Module):
    def __init__(self, numberFeatures):
        super(ResNet13NoMax, self).__init__()
        self.l1 = LayerBlock(1, numberFeatures, 3, False) #15->13
        self.l2 = LayerBlock(numberFeatures, numberFeatures, 3, False) #13->11
        self.l3 = LayerBlock(numberFeatures, numberFeatures, 3, False)  #11->9
        self.l4 = LayerBlock(numberFeatures, numberFeatures, 3, False)  #9->7
        self.l5 = LayerBlock(numberFeatures, numberFeatures, 3, False)  #7->5
        self.l7 = nn.Conv2d(numberFeatures, 1, kernel_size=3, padding="valid", stride=(1, 1))
        self.sigmoid = nn.Sigmoid() 
        self.numberLayers = 7


    def forward(self, x):
        x = self.l1(x)
        x = x[:,:,self.l2.resValue:-self.l2.resValue,self.l2.resValue:-self.l2.resValue] + self.l2(x)
        x = x[:,:,self.l3.resValue:-self.l3.resValue,self.l3.resValue:-self.l3.resValue] + self.l3(x)
        x = x[:,:,self.l4.resValue:-self.l4.resValue,self.l4.resValue:-self.l4.resValue] + self.l4(x)
        x = x[:,:,self.l5.resValue:-self.l5.resValue,self.l5.resValue:-self.l5.resValue] + self.l5(x)
        x = self.l7(x)

        x = self.sigmoid(x)
        return x


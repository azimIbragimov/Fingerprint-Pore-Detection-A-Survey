## Su _et al._'s (2017) CNN reimplementation model

from architectures.template import LayerBlock
from torch import nn
import torch

'''
During testing, the individual pore recognition task can be extended to the pore extraction
 of a full fingerprint image. However, since fully-connected layers are computed with matrix 
 multiplication, the number of weights is heavily dependent on the output size of the previous 
 layer, which is indirectly controlled by the input image size. In order to employ the entire 
 fingerprint image in the same network described above, the fully-connected layers have to be 
 converted into convolutional layers [18]. It can be done by replacing a K-output fully-connected 
 layer with a convolutional layer with K filters of size equivalent to the size of the 
 feature maps in the previous layer. The output of the testing network is a binary map 
 with 1's indicating the locations of pores.
'''



class Su(nn.Module):
    def __init__(self, NUMBERFEATURES, batchSize):
        super(Su, self).__init__()
        layers = [
            LayerBlock(1, filter(1), 3, False),         #17 - 15
            LayerBlock(filter(1), filter(2), 3, False), #15 -> 13
            LayerBlock(filter(2), filter(3), 3, False), #13 -> 11
            LayerBlock(filter(3), filter(4), 3, False), #11 -> 9
            LayerBlock(filter(4), filter(5), 3, False), #9 -> 7
            LayerBlock(filter(5), filter(6), 3, False), #7 -> 5
            LayerBlock(filter(6), filter(7), 3, False), #5 -> 3
            LayerBlock(filter(7), 4096, 3, False),      #3 -> 1
            LayerBlock(4096, 1, 1, False),      #1 -> 1
            torch.nn.Sigmoid()
        ]
        self.net = nn.Sequential(*layers)


        self.numberLayers = 8

        

    def forward(self, x):
        x = self.net(x)
        return x


def filter(i):
    return 2**((i + 1) // 2 + 5)


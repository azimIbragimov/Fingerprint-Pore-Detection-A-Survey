import torch 

class LayerBlock(torch.nn.Module):
    def __init__(self, inChanels, outChanels, kernelSize, maxPoolingAllowed):

        self.resValue = 1 if not maxPoolingAllowed else 2

        super(LayerBlock, self).__init__()
        self.block = [
            torch.nn.Conv2d(
                inChanels, outChanels, 
                kernel_size=kernelSize, 
                padding="valid", 
                stride=(1, 1)),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(outChanels)
        ]

        if maxPoolingAllowed: 
            self.block += [torch.nn.MaxPool2d(kernel_size=kernelSize, stride=(1, 1))]

        self.block = torch.nn.Sequential(*self.block)
    
    def forward(self, x):
        x = self.block(x)
        return x


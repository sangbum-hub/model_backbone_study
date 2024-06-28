import torch
from torch import nn

class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLu(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLu(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


##########



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
        )

        self.l_conv = nn.Sequential(
            nn.Conv2d(in_channels= , out_channels= , kernel_size= , stride= , padding= ),
            nn.BatchNorm2d(),
            nn.Softmax(inplace=True)
        )


    def forward(self, x):
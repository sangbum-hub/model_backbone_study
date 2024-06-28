import torch
import torch.nn as nn


class InvertedBlock(nn.Module) :
    # self, input_channel : inch, output_channel : c, expansion : t, repeat : n, stride : s
    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion):
        super(InvertedBlock, self).__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch*expansion, kernel_size=1, stride=1, padding=0),   # 1x1 conv2d
            nn.BatchNorm2d(in_ch*expansion),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=in_ch*expansion, out_channels=in_ch*expansion, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_ch*expansion), # 3x3 dwise s=s
            nn.BatchNorm2d(in_ch*expansion),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=in_ch*expansion, out_channels=out_ch, kernel_size=1, stride=1, padding=0),  # 1x1 conv2d
            nn.BatchNorm2d(out_ch)
        )
        self.convert = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.bottleneck_1(x)
        if self.stride > 1 or self.in_ch != self.out_ch:
            x = self.convert(x)

        return self.act(x1 + x)




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.bottleneck1 = InvertedBlock(in_ch=32, out_ch=16, kernel_size=3, stride=1, expansion=1)
        self.bottleneck2 = InvertedBlock(in_ch=16, out_ch=24, kernel_size=3, stride=2, expansion=6)
        self.bottleneck3 = InvertedBlock(in_ch=24, out_ch=32, kernel_size=3, stride=2, expansion=6)
        self.bottleneck4 = InvertedBlock(in_ch=32, out_ch=64, kernel_size=3, stride=2, expansion=6)
        self.bottleneck5 = InvertedBlock(in_ch=64, out_ch=96, kernel_size=3, stride=1, expansion=6)
        self.bottleneck6 = InvertedBlock(in_ch=96, out_ch=160, kernel_size=3, stride=1, expansion=6)
        self.bottleneck7 = InvertedBlock(in_ch=160, out_ch=320, kernel_size=3, stride=1, expansion=6)


    def forward(self, x):
        x = self.conv(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)

        return x



import time
# torch.rand :batch, channel, height, width
img = torch.rand(2, 3, 224, 224)
model = Model()
start = time.time()
result = model(img)
print(time.time() - start)
print(result.shape)

from torchinfo import summary
summary(model, img.shape)

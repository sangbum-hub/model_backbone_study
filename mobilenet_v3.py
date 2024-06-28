import torch
import torch.nn as nn


# class SEBlock(nn.Module):
#     def __init__(self, in_ch, ratio=16):
#         super().__init__()
#         self.squeeze = nn.AdaptiveAvgPool2d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(in_ch, in_ch // ratio),
#             nn.ReLU(),
#             nn.Linear(in_ch // ratio, in_ch),
#             nn.Hardsigmoid()
#         )
#
#     def forward(self, x):
#         x = self.squeeze(x)
#         x = x.view(x.size(0), -1)
#         x = self.excitation(x)
#         x = x.view(x.size(0), x.size(1), 1, 1)
#         return x

class SEBlock(nn.Module):
    def __init__(self, middle_channels):
        super(SEBlock, self).__init__()
        r = 4
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels // r, kernel_size=1),
            # nn.Linear(in_features=middle_channels, out_features=middle_channels // r),
            nn.ReLU(inplace=True),
            # nn.Linear(in_features=middle_channels // r, out_features=middle_channels),
            nn.Conv2d(in_channels=middle_channels // r, out_channels=middle_channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        return self.se(x) * x

class InvertedBlock(nn.Module) :

    def __init__(self, in_ch, out_ch, mid_ch, kernel_size, stride, se=False):
        super(InvertedBlock, self).__init__()
        self.in_ch, self.out_ch, self.stride, self.mid_ch, self.se = in_ch, out_ch, stride, mid_ch, se

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=1, stride=1, padding=0),   # 1x1 conv2d
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=mid_ch), # 3x3 dwise s=s
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            SEBlock(mid_ch) if se else nn.Identity(),
            nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),  # 1x1 conv2d
            nn.BatchNorm2d(out_ch)
        )
        self.convert = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        x1 = self.bottleneck(x)
        if self.stride > 1 or self.in_ch != self.out_ch:
            x = self.convert(x)

        return x1 + x




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardsigmoid(inplace=True)
        )
        self.bottleneck1 = InvertedBlock(in_ch=16, out_ch=16, mid_ch=16, kernel_size=3, stride=2, se=True)
        self.bottleneck2 = InvertedBlock(in_ch=16, out_ch=24, mid_ch=72, kernel_size=3, stride=2)
        self.bottleneck3 = InvertedBlock(in_ch=24, out_ch=24, mid_ch=88, kernel_size=3, stride=1)
        self.bottleneck4 = InvertedBlock(in_ch=24, out_ch=40, mid_ch=96, kernel_size=3, stride=2, se=True)
        self.bottleneck5 = InvertedBlock(in_ch=40, out_ch=40, mid_ch=240, kernel_size=5, stride=1, se=True)
        self.bottleneck6 = InvertedBlock(in_ch=40, out_ch=40, mid_ch=240, kernel_size=5, stride=1, se=True)
        self.bottleneck7 = InvertedBlock(in_ch=40, out_ch=48, mid_ch=120, kernel_size=5, stride=1, se=True)
        self.bottleneck8 = InvertedBlock(in_ch=48, out_ch=48, mid_ch=144, kernel_size=5, stride=1, se=True)
        self.bottleneck9 = InvertedBlock(in_ch=48, out_ch=96, mid_ch=288, kernel_size=5, stride=2, se=True)
        self.bottleneck10 = InvertedBlock(in_ch=96, out_ch=96, mid_ch=576, kernel_size=5, stride=1, se=True)
        self.bottleneck11 = InvertedBlock(in_ch=96, out_ch=96, mid_ch=576, kernel_size=5, stride=1, se=True)

        self.hs = nn.Hardsigmoid(inplace=True)
        self.re = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bottleneck1(x)
        x = self.re(x)
        x = self.bottleneck2(x)
        x = self.re(x)
        x = self.bottleneck3(x)
        x = self.re(x)
        x = self.bottleneck4(x)
        x = self.hs(x)
        x = self.bottleneck5(x)
        x = self.hs(x)
        x = self.bottleneck6(x)
        x = self.hs(x)
        x = self.bottleneck7(x)
        x = self.hs(x)
        x = self.bottleneck8(x)
        x = self.hs(x)
        x = self.bottleneck9(x)
        x = self.hs(x)
        x = self.bottleneck10(x)
        x = self.hs(x)
        x = self.bottleneck11(x)
        x = self.hs(x)

        return x



import time
# torch.rand :batch, channel, height, width
img = torch.rand(1, 3, 224, 224)
model = Model()
start = time.time()
result = model(img)

from torchinfo import summary
summary(model, img.shape, device='cpu')
print(time.time() - start)
print(result.shape)
import torch
import torch.nn as nn


class DepthwiseSep(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_ch)
        )
        self.convert = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        self.act = nn.ReLU()

    def forward(self, x):
        y = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        # if x.shape[1] != y.shape[1]:
        #     y = self.convert(y)
        if self.stride > 1 or self.in_ch != self.out_ch:
            y = self.convert(y)
        x = x+y
        return self.act(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv_s1 = DepthwiseSep(in_ch=32, out_ch=64, stride=1)
        self.conv_s2 = DepthwiseSep(in_ch=64, out_ch=128, stride=2)
        self.conv_s3 = DepthwiseSep(in_ch=128, out_ch=128, stride=1)
        self.conv_s4 = DepthwiseSep(in_ch=128, out_ch=256, stride=2)
        self.conv_s5 = DepthwiseSep(in_ch=256, out_ch=256, stride=1)
        self.conv_s6 = DepthwiseSep(in_ch=256, out_ch=512, stride=2)
        self.conv_s7_1 = DepthwiseSep(in_ch=512, out_ch=512, stride=1)
        self.conv_s7_2 = DepthwiseSep(in_ch=512, out_ch=512, stride=1)
        self.conv_s7_3 = DepthwiseSep(in_ch=512, out_ch=512, stride=1)
        self.conv_s7_4 = DepthwiseSep(in_ch=512, out_ch=512, stride=1)
        self.conv_s7_5 = DepthwiseSep(in_ch=512, out_ch=512, stride=1)
        self.conv_s8 = DepthwiseSep(in_ch=512, out_ch=1024, stride=2)
        self.conv_s9 = DepthwiseSep(in_ch=1024, out_ch=1024, stride=1)


    def forward(self, x):
        x = self.conv(x)
        x = self.conv_s1(x)
        x = self.conv_s2(x)
        x = self.conv_s3(x)
        x = self.conv_s4(x)
        x = self.conv_s5(x)
        x = self.conv_s6(x)
        x = self.conv_s7_1(x)
        x = self.conv_s7_2(x)
        x = self.conv_s7_3(x)
        x = self.conv_s7_4(x)
        x = self.conv_s7_5(x)
        x = self.conv_s8(x)
        x = self.conv_s9(x)

        return x

import time

img = torch.rand(2, 3, 224, 224)
model = Model()
start = time.time()
result = model(img)
print(time.time() - start)
print(result.shape)

from torchinfo import summary
summary(model, img.shape)

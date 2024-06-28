import torch
import torch.nn as nn

class seq(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding=1):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_ch)
        )

        self.relu = nn.ReLU(inplace=True)

        self.convert = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2)


    def forward(self, x):
        x1 = self.conv(x)
        if self.stride > 1 or self.in_ch != self.out_ch:
            x = self.convert(x)

        return self.relu(x1 + x)





class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.con2_x1 = seq(in_ch=64, out_ch=64, kernel_size=3, stride=1)
        self.con2_x2 = seq(in_ch=64, out_ch=64, kernel_size=3, stride=1)
        self.con3_x1 = seq(in_ch=64, out_ch=128, kernel_size=3, stride=2)
        self.con3_x2 = seq(in_ch=128, out_ch=128, kernel_size=3, stride=1)
        self.con4_x1 = seq(in_ch=128, out_ch=256, kernel_size=3, stride=2)
        self.con4_x2 = seq(in_ch=256, out_ch=256, kernel_size=3, stride=1)
        self.con5_x1 = seq(in_ch=256, out_ch=512, kernel_size=3, stride=2)
        self.con5_x2 = seq(in_ch=512, out_ch=512, kernel_size=3, stride=1)

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool1(x)
        x = self.con2_x1(x)
        x = self.con2_x2(x)
        x = self.con3_x1(x)
        x = self.con3_x2(x)
        x = self.con4_x1(x)
        x = self.con4_x2(x)
        x = self.con5_x1(x)
        x = self.con5_x2(x)

        return x


import time

img = torch.rand(2, 3, 224, 224)
model = Model()
start = time.time()
result = model(img)
print(time.time() - start)
print(result.shape)

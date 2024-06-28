import torch
import torch.nn as nn

## 1. model

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
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
        self.convert = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


    def forward(self, x):
        x1 = self.conv(x)
        if self.stride > 1 or self.in_ch != self.out_ch:
            x = self.convert(x)
        return self.relu(x1 + x)

class UpSample(nn.Module):
    # def __init__(self, in_ch, out_ch, kernel_size, stride):
    def __init__(self, in_ch, out_ch, stride):
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        super().__init__()
        self.stride = stride
        self.conv = nn.Sequential(
            # 1) k:2, s:2, p:0      2) k:4, s:2, p:1
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=stride, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.convert = nn.Upsample(scale_factor=stride, mode='nearest')
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        if self.stride > 1 or self.in_ch != self.out_ch:
            x = self.convert(x)
        return self.act(x1 + x)



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.con2_x1 = DownSample(in_ch=64, out_ch=64, kernel_size=3, stride=1)
        self.con2_x2 = DownSample(in_ch=64, out_ch=64, kernel_size=3, stride=1)
        self.con3_x1 = DownSample(in_ch=64, out_ch=128, kernel_size=3, stride=2)
        self.con3_x2 = DownSample(in_ch=128, out_ch=128, kernel_size=3, stride=1)
        self.con4_x1 = DownSample(in_ch=128, out_ch=256, kernel_size=3, stride=2)
        self.con4_x2 = DownSample(in_ch=256, out_ch=256, kernel_size=3, stride=1)
        self.con5_x1 = DownSample(in_ch=256, out_ch=512, kernel_size=3, stride=2)
        self.con5_x2 = DownSample(in_ch=512, out_ch=512, kernel_size=3, stride=1)

        self.upconv1 = UpSample(in_ch=512, out_ch=512, stride=2)
        self.upconv2 = UpSample(in_ch=512, out_ch=512, stride=2)
        self.upconv3 = UpSample(in_ch=512, out_ch=512, stride=2)



    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        x = self.maxpool1(x)
        x = self.con2_x1(x)
        x = self.con2_x2(x)
        x = self.con3_x1(x)
        x = self.con3_x2(x)
        x = self.con4_x1(x)
        x = self.con4_x2(x)
        x = self.con5_x1(x)
        x = self.con5_x2(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)

        return x


import time

height_s = 224
width_s = 224
img = torch.rand(2, 3, height_s, width_s)
model = Model()
start = time.time()
result = model(img)
print(time.time() - start)
print('result shape : ', result.shape)
# print(result[0], type(result[0]))
print(len(result[0]))
print(len(result[0][0]))
print(len(result[0][0][0]))


## 2. loss

# heatmap loss
nn.GaussianNLLLoss()


# dimension loss
h_loss = nn.L1Loss(reduction='sum')


# offset loss

# 중앙점 찾아서 loss 만들기
# for i in range(height_s/2):
#     for j in range(width_s/2):

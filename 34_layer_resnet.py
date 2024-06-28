import torch
from torch import nn

class Seq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels)
        )
        if (in_channels != out_channels) or (stride > 1):
            self.t = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=(kernel_size - 1) // 2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        iden = x
        if hasattr(self, "t"):
            iden = self.t(x)
        x1 = self.seq(x)

        return self.relu(iden + x1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(kernel_size=3, strid=1, padding=1)
        self.layer = nn.Sequential(
            Seq(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            Seq(64, 64, 3, 1),
            Seq(64, 64, 3, 1),
            Seq(64, 128, 3, 2),
            Seq(128, 128, 3, 1),
            Seq(128, 128, 3, 1),
            Seq(128, 128, 3, 1),
            Seq(128, 256, 3, 2),
            Seq(256, 256, 3, 1),
            Seq(256, 256, 3, 1),
            Seq(256, 256, 3, 1),
            Seq(256, 256, 3, 1),
            Seq(256, 256, 3, 1),
            Seq(256, 512, 3, 2),
            Seq(512, 512, 3, 1),
            Seq(512, 512, 3, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer(x)
        return x


if __name__ == '__main__':
    import time
    from torchinfo import summary

    model = Model()

    sizex = 640
    sizey = 416
    in_ch = 3
    summary(model, input_size=(1, in_ch, sizey, sizex), device='cpu')

    x = torch.randn(1, in_ch, sizey, sizex)
    start = time.time()
    y = model.forward(x)
    print(time.time() - start)
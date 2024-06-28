import torch
from torch import nn

class SE(nn.Module):
    def __init__(self, middle_channels):
        super(SE, self).__init__()
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



class Seq(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels, kernel_size, stride, NL, se=False):
        super(Seq, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(middle_channels),
            NL,
            nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=3, stride=1, padding=1, groups=middle_channels),
            nn.BatchNorm2d(middle_channels),
            NL,
            SE(middle_channels) if se else nn.Identity(),
            nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels or stride != 1:
            self.x_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
                nn.BatchNorm2d(out_channels)
            )
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        tmp = x
        if hasattr(self, "x_block"):
            tmp = self.x_block(x)
        x1 = self.block(x)

        return self.silu(tmp + x1)

        # # b, c, _, _ = x.shape
        # x1 = self.block1(x)
        # l1 = self.sq(x1)
        # # l1 = l1.view(b, c)
        # # l1 = l1.squeeze(-1)
        # # l1 = l1.squeeze(-1)
        # l2 = self.excite(l1)
        #
        # # l2 = l2.unsqueeze(-1)
        # # l2 = l2.unsqueeze(-1)
        #
        # a = x1 * l2
        # # a = l2.view(b, c, 1, 1) * x1
        # x2 = self.block2(a)

        # return self.silu(x + x2)




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )
        self.bneck = nn.Sequential(
            Seq(in_channels=16, out_channels=16, middle_channels=16, kernel_size=3, stride=2, NL=nn.ReLU(inplace=True), se=True),
            Seq(in_channels=16, out_channels=24, middle_channels=72, kernel_size=3, stride=2, NL=nn.ReLU(inplace=True)),
            Seq(in_channels=24, out_channels=24, middle_channels=88, kernel_size=3, stride=1, NL=nn.ReLU(inplace=True)),
            Seq(in_channels=24, out_channels=40, middle_channels=96, kernel_size=5, stride=2, NL=nn.Hardswish(inplace=True), se=True),
            Seq(in_channels=40, out_channels=40, middle_channels=240, kernel_size=5, stride=1, NL=nn.Hardswish(inplace=True), se=True),
            Seq(in_channels=40, out_channels=40, middle_channels=240, kernel_size=5, stride=1, NL=nn.Hardswish(inplace=True), se=True),
            Seq(in_channels=40, out_channels=48, middle_channels=120, kernel_size=5, stride=1, NL=nn.Hardswish(inplace=True), se=True),
            Seq(in_channels=48, out_channels=48, middle_channels=144, kernel_size=5, stride=1, NL=nn.Hardswish(inplace=True), se=True),
            Seq(in_channels=48, out_channels=96, middle_channels=277, kernel_size=5, stride=2, NL=nn.Hardswish(inplace=True), se=True),
            Seq(in_channels=96, out_channels=96, middle_channels=576, kernel_size=5, stride=1, NL=nn.Hardswish(inplace=True), se=True),
            Seq(in_channels=96, out_channels=96, middle_channels=576, kernel_size=5, stride=1, NL=nn.Hardswish(inplace=True), se=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.bneck(x)


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
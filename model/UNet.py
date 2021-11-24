import torch
import numpy as np
from torch import nn
from torch.nn import functional


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, conn_channel=None):
        super().__init__()
        if conn_channel is None:
            conn_channel = out_channel
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, conn_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(conn_channel),
            nn.LeakyReLU(),
            nn.Conv2d(conn_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownLayer(nn.Module):
    def __init__(self, in_channel, out_channel, pool_size=2):
        super().__init__()
        self.down = nn.Sequential(
            DoubleConv(in_channel, out_channel),
            nn.MaxPool2d(pool_size)
        )

    def forward(self, x):
        return self.down(x)


class UpLayer(nn.Module):
    def __init__(self, in_channel, out_channel, pool_size=2, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel//2, pool_size, stride)
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x, ori_x):
        x = self.up(x)
        diffY = ori_x.size()[2] - x.size()[2]
        diffX = ori_x.size()[3] - x.size()[3]
        x = functional.pad(x, (diffX // 2, diffX - diffX//2, diffY//2, diffY-diffY//2))
        x = torch.cat([ori_x, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=19):
        super().__init__()
        self.dc_max1 = DownLayer(in_channel, 64)
        self.dc_max2 = DownLayer(64, 128)
        self.dc_max3 = DownLayer(128, 256)
        self.dc_max4 = DownLayer(256, 512)
        self.dc5 = DoubleConv(512, 1024)
        self.up_conv1 = UpLayer(1024, 512)
        self.up_conv2 = UpLayer(512, 256)
        self.up_conv3 = UpLayer(256, 128)
        self.up_conv4 = UpLayer(128, 64)
        self.out_conv = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.dc_max1(x)
        x = self.dc_max2(x)
        x = self.dc_max3(x)
        x = self.dc_max4(x)
        x = self.dc5(x)
        x = self.up_conv1(1)
        x = self.up_conv2(1)
        x = self.up_conv3(1)
        x = self.up_conv4(1)
        x = self.out_conv(x)
        return x












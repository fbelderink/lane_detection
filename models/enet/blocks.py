import torch
import torch.nn as nn


class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super.__init__()

        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False,
        )

        self.pooling = nn.MaxPool2d(2, 2)

        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.activation = nn.PReLU()

    def forward(self, x):
        conv = self.conv2d(x)
        pool = self.pooling(x)

        out = torch.cat((conv, pool), 1)

        out = self.batch_norm(out)

        return self.activation(out)


class DownsamplingBottleneck(nn.Module):
    def __init__(self, channels,):
        super.__init__()

        pass

    def forward(self, x):
        pass

class UpsamplingBottleneck(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
import torch
import torch.nn as nn


class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
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


class Norm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        return self.activation(self.norm(x))


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, dropout=0.01, return_indices=False):
        super().__init__()

        self.return_indices = return_indices

        self.pool = nn.MaxPool2d(2, 2, return_indices=return_indices)

        internal_channels = (in_channels + out_channels) // 2

        self.ext_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=(2, 2),
                stride=(2, 2),
                bias=False
            ),
            Norm(internal_channels),
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=(3, 3),
                padding=padding,
                bias=False
            ),
            Norm(internal_channels),
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.Dropout2d(dropout)
        )

        self.activation = nn.PReLU()

        pass

    def forward(self, x):

        ext = self.ext_branch(x)

        indices = []

        # max pooling
        if self.return_indices:
            main, indices = self.pool(x)
        else:
            main = self.pool(x)

        # padding
        b, c, h, w = main.shape
        zeros = torch.zeros(b, abs(ext.size()[1] - c), h, w).to(x.device)

        main = torch.cat((main, zeros), 1)

        return self.activation(main + ext), indices


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, dropout=0.01):
        super().__init__()

        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.unpool = nn.MaxUnpool2d(2, 2)

        internal_channels = (in_channels + out_channels) // 2

        self.ext_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            Norm(internal_channels),
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=(3, 3),
                stride=2,
                padding=padding,
                output_padding=1
            ),
            Norm(internal_channels),
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.Dropout2d(dropout)
        )

        self.activation = nn.PReLU()

    def forward(self, x, max_indices):
        ext = self.ext_branch(x)

        main = self.main_conv(x)
        main = self.unpool(main, max_indices)

        return self.activation(main + ext)


class Bottleneck(nn.Module):

    def __init__(self, channels, kernel_size=3, padding=0, dropout=0.01, dilation=1, asymmetric=False):
        super().__init__()

        self.activation = nn.PReLU()

        if asymmetric:
            conv = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(kernel_size, 1),
                    dilation=dilation,
                    padding=(padding, 0)
                ),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, kernel_size),
                    dilation=dilation,
                    padding=(0, padding)
                )
            )
        else:
            conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            )

        self.ext_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            ),
            Norm(channels),
            conv,
            Norm(channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            ),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.activation(self.ext_branch(x))

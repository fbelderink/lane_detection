import torch
import torch.nn as nn
from blocks import UpsamplingBottleneck, Bottleneck
from encoder import ENetEncoder

class ENetDecoder(nn.Module):

    def __init__(self, channels_sizes, C):
        super().__init__()

        assert len(channels_sizes) == 3

        self.stage4 = _Stage4(channels_sizes[0], channels_sizes[1])
        self.stage5 = _Stage5(channels_sizes[1], channels_sizes[2])
        self.full = nn.ConvTranspose2d(channels_sizes[2],
                                       C,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1)

    def forward(self, x, indices):
        x = self.stage4(x, indices[0])
        x = self.stage5(x, indices[1])
        x = self.full(x)
        return x


class _Stage4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = UpsamplingBottleneck(in_channels, out_channels, padding=1, dropout=0.1)

        self.bottlenecks = nn.ModuleList([
            Bottleneck(out_channels, padding=1, dropout=0.1) for _ in range(2)
        ])

    def forward(self, x, indices):
        x = self.upsample(x, indices)

        for bottleneck in self.bottlenecks:
            x = bottleneck(x)

        return x


class _Stage5(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = UpsamplingBottleneck(in_channels, out_channels, padding=1, dropout=0.1)
        self.bottleneck = Bottleneck(out_channels, padding=1, dropout=0.1)

    def forward(self, x, indices):
        x = self.upsample(x, indices)

        x = self.bottleneck(x)

        return x

if __name__ == "__main__":
    encoder = ENetEncoder(channel_sizes=[3, 16, 64, 128])
    v_x = torch.randn(1, 3, 512, 512)

    y_e, indices = encoder(v_x)
    print(y_e.shape, [i.shape for i in indices])
    decoder = ENetDecoder([128, 64, 16], 2)

    y_d = decoder(y_e, indices)
    print(y_d.shape)



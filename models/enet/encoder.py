import torch
import torch.nn as nn
from blocks import InitialBlock, DownsamplingBottleneck, Bottleneck


class ENetEncoder(nn.Module):
    def __init__(self, channel_sizes, cut_stage=-1):
        super().__init__()
        assert cut_stage <= 3
        assert len(channel_sizes) == (4 if cut_stage == -1 else (cut_stage + 1))

        self.cut_stage = cut_stage
        self.initial = InitialBlock(channel_sizes[0], channel_sizes[1])

        self.stage1 = Stage1(channel_sizes[1], channel_sizes[2])
        self.stage2 = Stage2(channel_sizes[2], channel_sizes[3])
        self.stage3 = Stage3(channel_sizes[3])

    def forward(self, x):

        stages = [self.initial, self.stage1, self.stage2, self.stage3]

        indices = []
        for i, stage in enumerate(stages):
            if i < self.cut_stage:
                ret = stage(x)

                if isinstance(ret, tuple):
                    x = ret[0]
                    indices.insert(0, ret[1])
                else:
                    x = ret

        return x, indices


class Stage1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = DownsamplingBottleneck(in_channels,
                                                 out_channels,
                                                 padding=1,
                                                 dropout=0.01,
                                                 return_indices=True)

        self.bottlenecks = nn.ModuleList([
            Bottleneck(out_channels,
                       padding=1,
                       dropout=0.01)
            for _ in range(4)
        ])

    def forward(self, x):
        out, indices = self.downsample(x)

        for bottleneck in self.bottlenecks:
            out = bottleneck(out)

        return out, indices


class Stage2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = DownsamplingBottleneck(in_channels,
                                                 out_channels,
                                                 padding=1,
                                                 dropout=0.1,
                                                 return_indices=True)

        self.shared_stage3 = Stage3(out_channels)

    def forward(self, x):
        out, indices = self.downsample(x)

        out = self.shared_stage3(out)

        return out, indices


class Stage3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ls_dilation = [1, 2, 1, 4, 1, 8, 1, 16]
        padding = [1, 2, 2, 4, 1, 8, 2, 16]
        asym_blocks = [3, 7]

        self.bottlenecks = nn.ModuleList(
            [Bottleneck(channels,
                        dropout=0.1,
                        kernel_size=(5 if (i + 1) in asym_blocks else 3),
                        padding=padding[i],
                        dilation=ls_dilation[i],
                        asymmetric=(True if (i + 1) in asym_blocks else False))
             for i in range(8)]
        )

    def forward(self, x):
        for bottleneck in self.bottlenecks:
            x = bottleneck(x)

        return x


if __name__ == "__main__":
    encoder = ENetEncoder(channel_sizes=[3, 16, 64, 128])
    v_x = torch.randn(1, 3, 512, 512)

    y, idxs = encoder(v_x)

    print(y.shape, idxs[0].shape, idxs[1].shape)

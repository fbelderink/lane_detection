import torch
import torch.nn as nn
from models.enet.encoder import ENetEncoder, Stage3
from models.enet.decoder import ENetDecoder


class LaneNet(nn.Module):
    def __init__(self, channel_sizes, embedding):
        super().__init__()

        assert len(channel_sizes) == 6

        self.shared_stages = ENetEncoder(channel_sizes[:4], cut_stage=3)

        self.binary_stage3 = Stage3(channel_sizes[3])
        self.instance_stage3 = Stage3(channel_sizes[3])

        self.binary_decoder = ENetDecoder(channel_sizes[3:6], 1)
        self.instance_decoder = ENetDecoder(channel_sizes[3:6], embedding)

    def forward(self, x):
        x, indices = self.shared_stages(x)

        binary_out = self.binary_stage3(x)
        binary_out = self.binary_decoder(binary_out, indices)

        instance_out = self.instance_stage3(x)
        instance_out = self.instance_decoder(instance_out, indices)

        return binary_out, instance_out

if __name__ == '__main__':
    lanenet = LaneNet([3, 16, 64, 128, 64, 16], 4)

    x = torch.rand(1, 3, 512, 256)
    y_bin, y_inst = lanenet(x)
    print(y_bin.shape, y_inst.shape)


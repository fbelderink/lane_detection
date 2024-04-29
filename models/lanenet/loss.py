import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import get_project_root
from data.dataset import LaneDataset
import os


class CrossEntropyLoss(nn.Module):
    def __init__(self, c=1.02):
        super().__init__()
        self.c = c

    def forward(self, pred, target):
        lane_markings_mask = (target == 1).float()
        street_mask = (target == 0).float()

        target_classed = torch.cat((lane_markings_mask, street_mask), dim=1)

        prob_lane = torch.count_nonzero(lane_markings_mask) / torch.numel(target)

        weights = torch.tensor(
            [np.divide(1, np.log(self.c + prob)) for prob in [prob_lane, 1 - prob_lane]],
            dtype=torch.float32
        )

        return F.cross_entropy(pred, target_classed, weight=weights)

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1, beta=1, gamma=0.001):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(self, pred, target):

        batch_size = pred.shape[0]

        for b in range(batch_size):
            pass

        return torch.tensor(0, dtype=torch.float32)

if __name__ == '__main__':
    ce_criterion = CrossEntropyLoss()
    discriminative_criterion = DiscriminativeLoss(0.5, 3)

    root_dir = get_project_root()
    dataset = LaneDataset(os.path.join(root_dir, "data/train_set_example"))

    image, bin_label, inst_label = dataset[0]
    bin_label = torch.reshape(bin_label, (1, *bin_label.shape))
    inst_label = torch.reshape(inst_label, (1, *inst_label.shape))

    y_bin = torch.randint(0, 5, (1, 2, 256, 512)).to(torch.float)
    y_bin.requires_grad_()

    y_inst = torch.randint(0, 5, (1, 4, 256, 512)).to(torch.float)

    bin_loss = ce_criterion(y_bin, bin_label)
    inst_loss = discriminative_criterion(y_inst, inst_label)

    bin_loss.backward()

    print(y_bin.grad.shape)

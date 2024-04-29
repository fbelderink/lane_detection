import os.path

import torch
from torch.utils.data import Dataset
from common.utils import *
import json
import cv2


class LaneDataset(Dataset):
    def __init__(self, path, reshape=(512, 256)):
        data_desc = get_file_paths(path, ['txt'])[0]

        paths = []
        with open(data_desc, 'r') as f:
            for line in f:
                paths.append(json.loads(line))

        self.data_paths = list(zip(*paths))
        self.reshape = reshape

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        root = get_project_root()

        image = cv2.imread(os.path.join(root, self.data_paths[idx][0]), cv2.IMREAD_COLOR)
        bin_mask = cv2.imread(os.path.join(root, self.data_paths[idx][1]), cv2.IMREAD_UNCHANGED)
        instance_mask = cv2.imread(os.path.join(root, self.data_paths[idx][2]), cv2.IMREAD_UNCHANGED)

        if self.reshape:
            image = cv2.resize(image, self.reshape, interpolation=cv2.INTER_AREA)
            bin_mask = cv2.resize(bin_mask, self.reshape, interpolation=cv2.INTER_NEAREST)
            instance_mask = cv2.resize(instance_mask, self.reshape, interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(torch.float)
        bin_mask = torch.from_numpy(bin_mask).to(torch.float)
        instance_mask = torch.from_numpy(instance_mask).to(torch.float)

        bin_mask = torch.reshape(bin_mask, (1, *bin_mask.shape))
        instance_mask = torch.reshape(instance_mask, (1, *instance_mask.shape))

        return image, bin_mask, instance_mask

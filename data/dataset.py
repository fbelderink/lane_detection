import os.path

from torch.utils.data import Dataset
from common.utils import *
import json
import cv2


class LaneDataset(Dataset):
    def __init__(self, path):
        data_desc = get_file_paths(path, ['txt'])[0]

        paths = []
        with open(data_desc, 'r') as f:
            for line in f:
                paths.append(json.loads(line))

        self.data_paths = list(zip(*paths))

    def __len__(self):
        if not self.data_paths:
            return 0
        return len(self.data_paths[0])

    def __getitem__(self, idx):
        root = get_project_root()

        image = cv2.imread(os.path.join(root, self.data_paths[idx][0]), cv2.IMREAD_COLOR)
        bin_mask = cv2.imread(os.path.join(root, self.data_paths[idx][1]), cv2.IMREAD_UNCHANGED)
        instance_mask = cv2.imread(os.path.join(root, self.data_paths[idx][2]), cv2.IMREAD_UNCHANGED)

        return image, bin_mask, instance_mask

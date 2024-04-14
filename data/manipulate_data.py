import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from common.utils import get_project_root, get_file_paths


def download_dataset(target_dir='data'):
    api = KaggleApi()
    api.authenticate()

    datasets = api.dataset_list(search='tusimple')
    path = os.path.join(get_project_root(), target_dir)
    api.dataset_download_files(str(datasets[0]), path=path, unzip=True, quiet=False)


def transform_dataset(source_dir, target_dir):
    json_files = get_file_paths(source_dir, ['json'])

    img_dir = os.path.join(target_dir, 'images')
    mask_dir = os.path.join(target_dir, 'masks')

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    num_img = 0
    for jf in json_files:
        for img, bin_mask in _transform_json(jf):
            img_name = f"{num_img:04d}.jpg"
            cv2.imwrite(os.path.join(img_dir, img_name), img)
            cv2.imwrite(os.path.join(mask_dir, img_name), bin_mask)
            num_img += 1


def _transform_json(json_file_path):
    images = []
    with open(json_file_path, 'r') as fp:
        for line in fp:
            image_mask = json.loads(line)
            image_path = os.path.join(os.path.dirname(json_file_path), str(image_mask['raw_file']))
            lanes = image_mask['lanes']
            h_samples = image_mask['h_samples']

            img = cv2.imread(image_path)

            bin_mask = np.zeros(img.shape, np.uint8)

            for lane_idx, lane in enumerate(lanes):
                for x, y in zip(lane, h_samples):
                    if x != -2:
                        bin_mask[y][x] = 255

            images.append((img, bin_mask))

    return images


def display_data(path):
    image_paths = get_file_paths(path, ['jpg'])
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.imshow(images[i])
    plt.show()


if __name__ == '__main__':
    display_data("tu_simple_example/images/")
    transform_dataset("tu_simple_example/", "train_set_example/")

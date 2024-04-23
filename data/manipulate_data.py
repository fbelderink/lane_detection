import os
import cv2
import json
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from common.utils import get_project_root, get_file_paths


def download_dataset(target_dir='data', unzip=True):
    api = KaggleApi()
    api.authenticate()

    datasets = api.dataset_list(search='tusimple')
    path = os.path.join(get_project_root(), target_dir)
    api.dataset_download_files(str(datasets[0]), path=path, unzip=unzip, quiet=False)


def transform_dataset(source_dir, target_dir, descr_filename):
    json_files = get_file_paths(source_dir, ['json'])

    img_dir = os.path.join(target_dir, 'images')
    bin_mask_dir = os.path.join(target_dir, 'bin_masks')
    instance_mask_dir = os.path.join(target_dir, 'instance_masks')

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if not os.path.exists(bin_mask_dir):
        os.makedirs(bin_mask_dir)

    if not os.path.exists(instance_mask_dir):
        os.makedirs(instance_mask_dir)

    data_desc = [[], [], []]

    num_img = 0
    for jf in json_files:
        for img, bin_mask, instance_mask in _transform_json(jf):
            img_name = f"{num_img:04d}.jpg"

            paths = [os.path.join(img_dir, img_name), os.path.join(bin_mask_dir, img_name),
                     os.path.join(instance_mask_dir, img_name)]
            data_desc = [data_desc[i] + [os.path.relpath(paths[i], get_project_root())] for i in range(len(paths))]

            cv2.imwrite(paths[0], img)
            cv2.imwrite(paths[1], bin_mask)
            cv2.imwrite(paths[2], instance_mask)
            num_img += 1

    with open(os.path.join(target_dir, descr_filename), 'w') as f:
        for paths in data_desc:
            f.write(json.dumps(paths) + '\n')


def _transform_json(json_file_path):
    with open(json_file_path, 'r') as fp:
        for line in fp:
            image_mask = json.loads(line)
            image_path = os.path.join(os.path.dirname(json_file_path), str(image_mask['raw_file']))
            lanes = image_mask['lanes']
            h_samples = image_mask['h_samples']

            img = cv2.imread(image_path)

            # compress image to 2D
            bin_mask = np.zeros(img.shape[:2], np.uint8)
            instance_mask = np.zeros(img.shape[:2], np.uint8)

            cv2_line_type = cv2.LINE_8

            for lane_idx, lane in enumerate(lanes):
                lane_pts = []
                for x, y in zip(lane, h_samples):
                    if x != -2:
                        lane_pts.append([x, y])

                lane_pts = np.array(lane_pts)
                lane_pts = lane_pts.reshape((1, -1, 2))

                cv2.polylines(bin_mask, lane_pts, isClosed=False, color=(255, 255, 255), thickness=3,
                              lineType=cv2_line_type)

                lane_color = (lane_idx + 1) / len(lanes) * 255
                cv2.polylines(instance_mask, lane_pts, isClosed=False, color=(lane_color, lane_color, lane_color),
                              thickness=3, lineType=cv2_line_type)

            yield img, bin_mask, instance_mask


if __name__ == '__main__':
    transform_dataset("tu_simple_example/", "train_set_example/", "train.txt")

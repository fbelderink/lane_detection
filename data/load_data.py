import os
from kaggle.api.kaggle_api_extended import KaggleApi
from common.utils import get_project_root


def download_dataset():
    api = KaggleApi()
    api.authenticate()

    datasets = api.dataset_list(search='tusimple')
    path = os.path.join(get_project_root(), 'data')
    api.dataset_download_files(str(datasets[0]), path=path, unzip=True, quiet=False)


if __name__ == "__main__":
    download_dataset()

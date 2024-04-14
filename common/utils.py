from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parents[1]


def get_file_paths(path, file_types) -> list:
    paths = []
    for ft in file_types:
        paths.extend([str(p) for p in Path(path).glob(f"*.{ft}")])
    return paths

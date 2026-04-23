
from pathlib import Path
import os

"""
The functions operate on the assumption that the dataset has the following structure:
Train data:
    /nyu_data/data/nyu2_train/[location_name]_out/(1.jpg, 1.png, 2.jpg, 2.png, ...)
    RGB images are named as [index].jpg, and depth maps are named as [index].png. 

Validation data:
    /nyu_data/data/nyu2_test/(00000_colors.png, 00000_depth.png, 00001_colors.png, 00001_depth.png, ...)
    RGB images are named as [index]_colors.png, and depth maps are named as [index]_depth.png. 

The RGB and depth files with the same index correspond to each other.
"""


def get_train_pathes(dataset_path: str) -> dict[str, list[str]]:
    """
    Get the file paths for the training data.
    
    Args:
        dataset_path: The base path to the dataset (folder 'nyu_data')
    Returns:
        A dictionary containing lists of (local) file paths for 'gt_depth', 'sparse_depth', and 'rgb'.
    """
    folder = Path(os.path.join(dataset_path, 'nyu_data', 'data', 'nyu2_train'))

    assert folder.exists(), f"Training data folder not found at {folder}. Please check the dataset path and structure."

    gt_depth_paths = [p for p in folder.rglob('*.png')]
    rgb_paths = [p for p in folder.rglob('*.jpg')]
    gt_depth_paths.sort()
    rgb_paths.sort()
    return {
        'gt_depth': [str(p) for p in gt_depth_paths],
        'rgb': [str(p) for p in rgb_paths]
    }

def get_val_pathes(dataset_path: str) -> dict[str, list[str]]:
    """
    Get the file paths for the validation data.
    
    Args:
        dataset_path: The base path to the dataset (folder 'nyu_data')
    Returns:
        A dictionary containing lists of (local) file paths for 'gt_depth', 'sparse_depth', and 'rgb'.
    """
    folder = Path(os.path.join(dataset_path, 'nyu_data', 'data', 'nyu2_test'))

    assert folder.exists(), f"Validation data folder not found at {folder}. Please check the dataset path and structure."

    gt_depth_paths = [p for p in folder.rglob('*_depth.png')]
    rgb_paths = [p for p in folder.rglob('*_colors.png')]
    gt_depth_paths.sort()
    rgb_paths.sort()
    return {
        'gt_depth': [str(p) for p in gt_depth_paths],
        'rgb': [str(p) for p in rgb_paths]
    }
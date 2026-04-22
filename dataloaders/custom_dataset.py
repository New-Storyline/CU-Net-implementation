from dataclasses import dataclass
import enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from utils import *
from model_arcs.geo_features import CameraIntrinsics, GeoFeatures

class CustomDepthDataset(data.Dataset):

    def __init__(
            self, 
            image_size: tuple,
            get_image_pathes_fn,
            transform_fn,
            create_sparse_depth_fn,
            load_calib_fn,
            use_image=False,
            depth_read_fn=read_depth,
            rgb_read_fn=read_rgb,
        ):
        """
        Create a custom dataset for depth completion.
        This dataset dynamically loads RGB images and depth maps from disk, applies transformations, 
        and generates sparse depth maps on the fly.

        Args:
            get_image_pathes_fn: Callable that returns a dictionary with image
                paths for the dataset, for example `{'rgb': [...], 'gt_depth': [...]}`.
            create_sparse_depth_fn: Callable that converts a dense depth map to
                a sparse depth map. Expected signature:
                `(depth_map: np.ndarray) -> np.ndarray`.
            transform_fn: Callable that applies dataset transforms to the loaded
                arrays. Expected signature:
                `(sparse_depth, gt_depth, rgb, position) -> (sparse_depth, gt_depth, rgb, position)`.
            load_calib_fn: Callable that loads the camera calibration matrix K.
            use_image: Whether to load and return RGB images. If False, the dataset will only return depth maps and positions.
            depth_read_fn: Callable that reads a depth map from a file path.
            rgb_read_fn: Callable that reads an RGB image from a file path.
        """
        self.depth_read = depth_read_fn
        self.rgb_read = rgb_read_fn
        self.image_size = image_size
        self.use_image = use_image
        self.paths: dict[str, list[str]] = get_image_pathes_fn()
        self.create_sparse_depth = create_sparse_depth_fn
        self.transform = transform_fn
        self.camera_intrinsics: Optional[CameraIntrinsics] = load_calib_fn() if load_calib_fn is not None else None
        self.position = GeoFeatures.calc_position_map(self.image_size)

    def __getitem__(self, index):
        """
        Get the transformed data for a given index.
        
        Returns:
            A DataElement containing:
            - rgb: The RGB image (if use_image is True, otherwise None).
            - sparse_depth: The generated sparse depth map.
            - gt_depth: The ground truth depth map.
            - position: The position map for the image.
        """

        gt = self.depth_read(self.paths['gt_depth'][index])
        rgb = self.rgb_read(self.paths['rgb'][index]) if self.use_image else None
        sparse_dirty = self.create_sparse_depth(gt)

        sparse_dirty, gt, rgb, position = self.transform(sparse_dirty, gt, rgb, self.position)

        return DataElement(rgb=rgb, sparse_depth=sparse_dirty, gt_depth=gt, position=position)

    def __len__(self):
        return len(self.paths['gt_depth'])

@dataclass
class DataElement:
    rgb: Optional[torch.Tensor]
    sparse_depth: torch.Tensor
    gt_depth: torch.Tensor
    position: torch.Tensor


def TEST_as_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def TEST_visualize_first_sample(dataset: CustomDepthDataset):
    sample = dataset[0]

    rgb = TEST_as_numpy(sample.rgb) if sample.rgb is not None else None
    sparse_depth = TEST_as_numpy(sample.sparse_depth)
    gt_depth = TEST_as_numpy(sample.gt_depth)

    if rgb is not None and rgb.ndim == 3 and rgb.shape[0] in (1, 3):
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb is not None:
        rgb = np.squeeze(rgb)

    sparse_depth = np.squeeze(sparse_depth)
    gt_depth = np.squeeze(gt_depth)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    if rgb is not None:
        axes[0].imshow(rgb)
        axes[0].set_title("RGB image")
    else:
        axes[0].text(0.5, 0.5, "RGB disabled", ha="center", va="center")
        axes[0].set_title("RGB image")

    axes[1].imshow(sparse_depth, cmap="viridis")
    axes[1].set_title("Sparse depth")

    axes[2].imshow(gt_depth, cmap="viridis")
    axes[2].set_title("GT depth")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    plt.show()

def TEST_create_sparse_depth(depth):
    sparse = np.zeros_like(depth)
    sparse[::2, ::2] = depth[::2, ::2]
    return sparse

def test_dataset():


    
    dataset = CustomDepthDataset(
        image_size=(480, 640),
        get_image_pathes_fn=lambda: {
            'rgb': ['datasets\\nyu_data\\data\\nyu2_test\\00000_colors.png', 'datasets\\nyu_data\\data\\nyu2_test\\00001_colors.png'],
            'gt_depth': ['datasets\\nyu_data\\data\\nyu2_test\\00000_depth.png', 'datasets\\nyu_data\\data\\nyu2_test\\00001_depth.png'],
        },
        transform_fn=lambda sparse, gt, rgb, pos: (sparse, gt, rgb, pos),
        create_sparse_depth_fn=TEST_create_sparse_depth,
        load_calib_fn=None,
        use_image=True,
    )
    TEST_visualize_first_sample(dataset)


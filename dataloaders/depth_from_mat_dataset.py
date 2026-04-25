from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.utils.data as data

from model_arcs.geo_features import CameraIntrinsics, GeoFeatures

import matplotlib.pyplot as plt

from utils.data_utils import create_sparse_depth


def identity_transform(sparse_depth, gt_depth, rgb, position):
    return sparse_depth, gt_depth, rgb, position


class DepthFromMatDataset(data.Dataset):
    """
    Dataset for depth completion tasks backed by in-memory numpy arrays.

    The dataset mirrors the sample format used by DepthFromFilesDataset and
    returns dictionaries with ``sparse_depth``, ``gt_depth``, and ``position``.
    The ``rgb`` tensor is included only when RGB arrays are provided and
    ``use_image`` is True.
    """

    def __init__(
        self,
        gt_depths,
        transform_fn=identity_transform,
        image_size: Optional[tuple[int, int]] = None,
        load_calib_fn=None,
        sparse_depths=None,
        create_sparse_depth_fn=None,
        rgb_images=None,
        use_image: bool = False,
    ):
        self.gt_depths = self._normalize_depth_batch(gt_depths, name="gt_depths")
        self.sparse_depths = None
        if sparse_depths is not None:
            self.sparse_depths = self._normalize_depth_batch(sparse_depths, name="sparse_depths")

        self.rgb_images = None
        if rgb_images is not None:
            self.rgb_images = self._normalize_rgb_batch(rgb_images, name="rgb_images")

        inferred_image_size = tuple(self.gt_depths.shape[1:3])
        self.image_size = tuple(image_size) if image_size is not None else inferred_image_size
        self.use_image = use_image
        self.create_sparse_depth = create_sparse_depth_fn
        self.transform = transform_fn or identity_transform
        self.camera_intrinsics: Optional[CameraIntrinsics] = load_calib_fn() if load_calib_fn is not None else None
        self.position = GeoFeatures.calc_position_map(self.image_size)

        self._validate_arrays()

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        assert index < len(self), f"Index {index} out of range for dataset of size {len(self)}"

        gt = np.array(self.gt_depths[index], copy=True)
        rgb = np.array(self.rgb_images[index], copy=True) if self.use_image else None
        if self.sparse_depths is not None:
            sparse_dirty = np.array(self.sparse_depths[index], copy=True)
        else:
            sparse_dirty = self.create_sparse_depth(np.array(gt, copy=True))

        sample_position = self.position.squeeze(0).clone()
        sparse_dirty, gt, rgb, position = self.transform(sparse_dirty, gt, rgb, sample_position)

        sample = {
            "sparse_depth": self._to_tensor(sparse_dirty),
            "gt_depth": self._to_tensor(gt),
            "position": self._to_tensor(position),
        }
        if rgb is not None:
            sample["rgb"] = self._to_tensor(rgb)

        return sample

    def __len__(self):
        return self.gt_depths.shape[0]

    def print_mat_file_keys(mat_path: str | Path):
        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

        with h5py.File(mat_path, "r") as mat_file:
            print(f"Keys in MAT file '{mat_path}':")
            for key in mat_file.keys():
                print(f"  - {key}")

    def visualize_first_sample(self):
        sample = self[0]

        rgb = self._as_numpy(sample['rgb']) if 'rgb' in sample else None
        sparse_depth = self._as_numpy(sample['sparse_depth'])
        gt_depth = self._as_numpy(sample['gt_depth'])

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

    def _validate_arrays(self):
        if len(self) == 0:
            raise ValueError("gt_depths must contain at least one sample")

        if self.sparse_depths is None and self.create_sparse_depth is None:
            raise ValueError("Either sparse_depths or create_sparse_depth_fn must be provided")

        if tuple(self.gt_depths.shape[1:3]) != self.image_size:
            raise ValueError(
                f"image_size={self.image_size} does not match GT depth shape {self.gt_depths.shape[1:3]}"
            )

        if self.sparse_depths is not None:
            if self.sparse_depths.shape[0] != len(self):
                raise ValueError(
                    f"Number of sparse depth samples ({self.sparse_depths.shape[0]}) does not match "
                    f"GT depth samples ({len(self)})"
                )
            if self.sparse_depths.shape[1:3] != self.gt_depths.shape[1:3]:
                raise ValueError(
                    "Sparse depth maps must have the same spatial shape as GT depth maps: "
                    f"got {self.sparse_depths.shape[1:3]} vs {self.gt_depths.shape[1:3]}"
                )

        if self.rgb_images is not None:
            if self.rgb_images.shape[0] != len(self):
                raise ValueError(
                    f"Number of RGB samples ({self.rgb_images.shape[0]}) does not match GT depth samples ({len(self)})"
                )
            if self.rgb_images.shape[1:3] != self.gt_depths.shape[1:3]:
                raise ValueError(
                    f"RGB images must have the same spatial shape as GT depth maps: got {self.rgb_images.shape[1:3]} "
                    f"vs {self.gt_depths.shape[1:3]}"
                )

        if self.use_image and self.rgb_images is None:
            raise ValueError("use_image=True requires rgb_images to be provided")

    @staticmethod
    def create_train_val_datasets_from_mat(
        mat_path: str | Path,
        gt_depth_key: str,
        transform_fn=identity_transform,
        train_split: float = 0.8,
        rgb_key: Optional[str] = None,
        sparse_depth_key: Optional[str] = None,
        create_sparse_depth_fn=None,
        use_image: bool = False,
        load_calib_fn=None,
        image_size: Optional[tuple[int, int]] = None,
        shuffle: bool = True,
        random_seed: int = 42,
        transpose_from_mat: bool = True,
    ) -> tuple["DepthFromMatDataset", "DepthFromMatDataset"]:
        """
        Load arrays from a MATLAB v7.3 .mat file and build train/validation datasets.

        Args:
            mat_path: Path to the .mat file.
            gt_depth_key: Dataset key inside the .mat file for dense depth maps.
            transform_fn: Per-sample transform with the same signature as in
                DepthFromFilesDataset.
            train_split: Fraction of samples assigned to the training split.
            rgb_key: Optional dataset key for RGB images.
            sparse_depth_key: Optional dataset key for sparse depth maps.
            create_sparse_depth_fn: Factory used when sparse depth maps are not
                stored in the file.
            use_image: Whether returned samples should include RGB tensors.
            load_calib_fn: Optional callable returning camera intrinsics.
            image_size: Optional explicit (H, W). If omitted, inferred from GT.
            shuffle: Whether to shuffle samples before splitting.
            random_seed: Seed used when shuffle=True.
            transpose_from_mat: Whether to swap the last two axes of loaded
                arrays. This should stay True for most MATLAB v7.3 files loaded
                through h5py, including NYU Depth V2.
        """

        if not 0 < train_split < 1:
            raise ValueError(f"train_split must be in the open interval (0, 1), got {train_split}")

        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

        with h5py.File(mat_path, "r") as mat_file:

            gt_depths = DepthFromMatDataset._load_array_from_mat(
                mat_file,
                gt_depth_key,
                transpose_from_mat=transpose_from_mat,
            )
            rgb_images = None
            if rgb_key is not None:
                rgb_images = DepthFromMatDataset._load_array_from_mat(
                    mat_file,
                    rgb_key,
                    transpose_from_mat=transpose_from_mat,
                )

            sparse_depths = None
            if sparse_depth_key is not None:
                sparse_depths = DepthFromMatDataset._load_array_from_mat(
                    mat_file,
                    sparse_depth_key,
                    transpose_from_mat=transpose_from_mat,
                )

        num_samples = int(np.asarray(gt_depths).shape[0])
        if num_samples == 0:
            raise ValueError(f"Dataset '{gt_depth_key}' in {mat_path} is empty")

        indices = np.arange(num_samples)
        if shuffle:
            np.random.default_rng(random_seed).shuffle(indices)

        split_index = int(round(num_samples * train_split))
        if num_samples > 1:
            split_index = min(max(split_index, 1), num_samples - 1)

        train_indices = indices[:split_index]
        val_indices = indices[split_index:]

        common_kwargs = {
            "transform_fn": transform_fn,
            "image_size": image_size,
            "load_calib_fn": load_calib_fn,
            "create_sparse_depth_fn": create_sparse_depth_fn,
            "use_image": use_image,
        }

        dataset_train = DepthFromMatDataset(
            gt_depths=DepthFromMatDataset._slice_batch(gt_depths, train_indices),
            sparse_depths=DepthFromMatDataset._slice_batch(sparse_depths, train_indices),
            rgb_images=DepthFromMatDataset._slice_batch(rgb_images, train_indices),
            **common_kwargs,
        )
        dataset_val = DepthFromMatDataset(
            gt_depths=DepthFromMatDataset._slice_batch(gt_depths, val_indices),
            sparse_depths=DepthFromMatDataset._slice_batch(sparse_depths, val_indices),
            rgb_images=DepthFromMatDataset._slice_batch(rgb_images, val_indices),
            **common_kwargs,
        )
        return dataset_train, dataset_val

    @staticmethod
    def _load_array_from_mat(mat_file: h5py.File, key: str, transpose_from_mat: bool) -> np.ndarray:
        try:
            obj = mat_file[key]
        except KeyError as exc:
            raise KeyError(f"Key '{key}' not found in MAT file") from exc

        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"Key '{key}' points to an HDF5 group, not a dataset")

        array = np.asarray(obj[()])
        if transpose_from_mat and array.ndim >= 2:
            array = np.swapaxes(array, -1, -2)
        return array

    @staticmethod
    def _slice_batch(array_like, indices: np.ndarray):
        if array_like is None:
            return None
        return np.asarray(array_like)[indices]

    @staticmethod
    def _normalize_depth_batch(array_like, name: str) -> np.ndarray:
        array = np.asarray(array_like)

        if array.ndim == 2:
            array = array[np.newaxis, :, :, np.newaxis]
        elif array.ndim == 3:
            array = array[..., np.newaxis]
        elif array.ndim == 4:
            if array.shape[-1] == 1:
                pass
            elif array.shape[1] == 1:
                array = np.moveaxis(array, 1, -1)
            else:
                raise ValueError(
                    f"{name} must have shape (N, H, W), (N, H, W, 1), or (N, 1, H, W); got {array.shape}"
                )
        else:
            raise ValueError(
                f"{name} must have shape (N, H, W), (N, H, W, 1), or (N, 1, H, W); got {array.shape}"
            )

        return array.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_rgb_batch(array_like, name: str) -> np.ndarray:
        array = np.asarray(array_like)

        if array.ndim == 3:
            if array.shape[-1] in (1, 3):
                array = array[np.newaxis, ...]
            elif array.shape[0] in (1, 3):
                array = np.moveaxis(array, 0, -1)[np.newaxis, ...]
            else:
                raise ValueError(
                    f"{name} must have shape (N, H, W, C) or (N, C, H, W) with C in (1, 3); got {array.shape}"
                )
        elif array.ndim == 4:
            if array.shape[-1] in (1, 3):
                pass
            elif array.shape[1] in (1, 3):
                array = np.moveaxis(array, 1, -1)
            else:
                raise ValueError(
                    f"{name} must have shape (N, H, W, C) or (N, C, H, W) with C in (1, 3); got {array.shape}"
                )
        else:
            raise ValueError(
                f"{name} must have shape (N, H, W, C) or (N, C, H, W) with C in (1, 3); got {array.shape}"
            )

        return array

    @staticmethod
    def _to_tensor(array_like) -> torch.Tensor:
        if isinstance(array_like, torch.Tensor):
            tensor = array_like.detach().clone()
        else:
            tensor = torch.from_numpy(np.asarray(array_like))

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)

        return tensor.contiguous()
    
    @staticmethod
    # Example usage:
    def test():
        mat_file = "datasets/nyu_labled (depthes in float16)/nyu_depth_v2_labeled.mat"
        gt_depth_key = "depths"
        rgb_key = "images"
        sparse_depth_key = None
        
        dataset_train, dataset_val = DepthFromMatDataset.create_train_val_datasets_from_mat(
            mat_path=mat_file,
            gt_depth_key=gt_depth_key,
            rgb_key=rgb_key,
            sparse_depth_key=sparse_depth_key,
            use_image=True,
            image_size=(480, 640),
        )
        dataset_train.visualize_first_sample()
        dataset_val.visualize_first_sample()

    @staticmethod
    def test_create_sparse_depth(gt_depths):
        
        return create_sparse_depth(gt_depths[0], num_points=500)
        

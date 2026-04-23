
import os
import numpy as np
from PIL import Image

def read_depth(path, bit_depth=None, scale_factor=None):
    """
    Load a depth map from a given path.

    Args:
        path: Path to the depth image.
        bit_depth: Expected source bit depth. Supported values are 8, 16,
            or None for auto-detection from the loaded data.
        scale_factor: Divisor applied after loading. If omitted, defaults to
            256.0 for 16-bit depth maps and 1.0 for 8-bit depth maps.

    Returns:
        np.ndarray: Depth map of shape (H, W, 1) and dtype float32.
    """
    assert os.path.exists(path), "file not found: {}".format(path)
    if bit_depth not in (None, 8, 16):
        raise ValueError(f"Unsupported bit_depth={bit_depth}. Expected one of: None, 8, 16")

    with Image.open(path) as img_file:
        pil_mode = img_file.mode
        depth_array = np.asarray(img_file)

    if depth_array.ndim == 3:
        if depth_array.shape[2] != 1:
            raise ValueError(f"Expected a single-channel depth image, got shape={depth_array.shape}, path={path}")
        depth_array = depth_array[..., 0]

    depth_max = int(np.max(depth_array))
    if np.issubdtype(depth_array.dtype, np.uint16) or pil_mode.startswith("I;16"):
        detected_bit_depth = 16
    elif np.issubdtype(depth_array.dtype, np.uint8):
        detected_bit_depth = 8
    else:
        detected_bit_depth = 16 if depth_max > 255 else 8
    effective_bit_depth = detected_bit_depth if bit_depth is None else bit_depth

    if bit_depth is not None and detected_bit_depth != bit_depth:
        raise ValueError(
            f"Depth bit depth mismatch for {path}: expected {bit_depth}-bit, "
            f"but loaded values fit into {detected_bit_depth}-bit range (max={depth_max})"
        )

    if scale_factor is None:
        scale_factor = 256.0 if effective_bit_depth == 16 else 1.0

    depth = depth_array.astype(np.float32) / float(scale_factor)
    depth = np.expand_dims(depth, -1)
    return depth

def read_rgb(path):
    """
    Load an RGB image from a given path.

    Returns:
        np.ndarray: The loaded RGB image.
    """
    assert os.path.exists(path), "file not found: {}".format(path)
    img_file = Image.open(path)
    rgb = np.array(img_file, dtype=np.uint8)
    img_file.close()
    return rgb

def create_sparse_depth(gt_depth, num_points=500):
    """
    Create a sparse depth map by randomly sampling points from the ground truth depth map.

    Args:
        gt_depth: The ground truth depth map as a numpy array of shape (H, W, 1).
        num_points: The number of random points to sample for the sparse depth map.
    Returns:
        np.ndarray: A sparse depth map of the same shape as gt_depth, with values at the sampled points and zeros elsewhere.
    """
    H, W, _ = gt_depth.shape
    sparse_depth = np.zeros_like(gt_depth)

    valid_mask = gt_depth > 0
    valid_indices = np.argwhere(valid_mask)

    if len(valid_indices) == 0:
        raise ValueError("No valid depth points found in the ground truth depth map.")

    num_points = min(num_points, len(valid_indices))
    sampled_indices = valid_indices[np.random.choice(len(valid_indices), num_points, replace=False)]

    for idx in sampled_indices:
        y, x, _ = idx
        sparse_depth[y, x] = gt_depth[y, x]

    return sparse_depth
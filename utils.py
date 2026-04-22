
import os
import numpy as np
from PIL import Image

def read_depth(path):
    """
    Load a depth map from a given path.

    Returns:
        np.ndarray: The loaded depth map.
    """
    assert os.path.exists(path), "file not found: {}".format(path)
    img_file = Image.open(path)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, "np.max(depth_png)={}, path={}".format(np.max(depth_png), path)

    depth = depth_png.astype(np.float32) / 256.
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
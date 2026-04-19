import torch.nn as nn

class CU_Net(nn.Module):
    """
    CU-Net model for LiDAR depth completion.

    Additionally, this model performs position encoding in different modes
    (which was not mentioned in the paper but is used in the authors'
    implementation). There are four encoding modes in total:
    - xyz: From the depth map and normalized pixel coordinates, we compute
      3D point positions using the camera intrinsic matrix (back-projection).
    - uv: We use two matrices of normalized pixel coordinates (unorm, vnorm),
      each ranging from -1 to 1.
    - z: We simply use the depth map as the positional feature.
    - std: No position encoding is applied.

    For each spatial scale of the encoder, a position feature tensor is
    created and concatenated to the input of each convolutional block
    at that scale.
    """
    def __init__(self):
        super(CU_Net, self).__init__()

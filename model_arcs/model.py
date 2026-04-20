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

    Local U-Net architecture:
    - Get geometry features based on encoding mode for each scale.
    - Encoder: 
      - 5 convolutional blocks with downsampling (max pooling).
        Each block consists of 4 blocks of (x + geo encoding for corresponding scale)Conv2D -> BatchNorm -> ReLU -> repeat one more time.
    - Decoder:
      - 5 convolutional blocks with upsampling (ConvTranspose2d -> BatchNorm -> ReLU).
    - One base conv block
    - conv1x1 to get the depth map output.
    - conv1x1 + sigmoid to get the confidence map output.

    Global U-Net architecture:
    - Encoder:
      - Delete outliers from the sparse depth map using algorithm from the paper:
        Create confidence, confusion and valid pixels masks by binary thresholding: 
        - confidence mask: 1 where depth >= 0.7, 0 elsewhere.
        - confusion mask: 1 where depth < 0.3 and depth <= 0
        - valid pixels mask: 1 where depth > 0.1, 0 elsewhere.
        Using the predefined convolution kernels (7x7, 13x13), which are designed to capture 
        neighboring pixels, we perform outliers removal:
        - calculate mean depth value of neighboring pixels for each pixel using both kernels:
          - lidar_sum = conv(depth)
          - lidar_count = conv(valid_pixels_mask)
          - lidar_mean = lidar_sum / (lidar_count + 1e-6)
        - create potential outliers mask: 1 where abs(depth - lidar_mean) > 1 (for kernel 13x13 -> 0.7), 0 elsewhere.
        - calculate same for 13x13 kernel...
        - final potential outliers mask = potential_outliers_mask_7x7 + potential_outliers_mask_13x13 
        - final cleared depth map = 
            depth * confidence_mask + 
            depth * (1 - final_potential_outliers_mask) * confusion_mask * valid_pixels_mask
      - the same architecture as the local U-Net encoder
    - Decoder:
      - the same architecture as the local U-Net decoder 
    - Fusion:
      - concatenate both confidence maps from the local and global U-Nets, and apply softmax, then divide back.
      - calculate the final depth map as local_depth * local_confidence + global_depth * global_confidence
    
    Loss computation:
    - compute the L2 loss with ingnoring invalid pixels (where depth <= 0) for local depth map, global depth map and final depth map, and do a weighted sum.
      L(D) = ||(D - Dgt) * 1{Dgt > 0}||_2^2
      L = lambda_local * L(local_depth) + lambda_global * L(global_depth) + lambda_fusion * L(final_depth)

    
    """
    def __init__(self):
        super(CU_Net, self).__init__()


from dataclasses import dataclass
import enum

from typing import List
import torch.nn as nn
import torch

from model_arcs.base_layers import SparseDownSampleClosest

class GeoEncodingType(enum.Enum):
  STD = 0
  Z = 1
  UV = 2
  XYZ = 3

@dataclass
class CameraIntrinsics:
  c_h: float # center height
  c_w: float # center width
  f_h: float # focal height
  f_w: float # focal width


class GeoFeatures(nn.Module):
    """
    Module creates geometric features at multiple scales from the input sparse depth map.
    """
    def __init__(self, geo_encoding_type: GeoEncodingType, img_size : tuple, scales_num: int = 6, camera_intrinsics: CameraIntrinsics = None):
        """
        Args:
            geo_encoding_type: Type of geometric encoding to use (STD, Z, UV, XYZ)
            scales_num: Number of scales to generate geometric features for (default is 6 for CU-Net architecture)
            camera_intrinsics: CameraIntrinsics dataclass containing camera parameters for geometric feature calculation
                Since the original repository used the same intrinsic matrix for all images, we define it once at the beginning.
        """
        super().__init__()

        assert geo_encoding_type != GeoEncodingType.XYZ or camera_intrinsics is not None, "Camera intrinsics must be provided for XYZ geo encoding type"

        self.geo_encoding_type = geo_encoding_type
        self.img_size = img_size
        self.scales_num = scales_num
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sparse_pool = SparseDownSampleClosest(stride=2)

        if camera_intrinsics is not None:
            self.c_h_tensor = torch.tensor(camera_intrinsics.c_h, dtype=torch.float32).view(1, 1, 1, 1)
            self.c_w_tensor = torch.tensor(camera_intrinsics.c_w, dtype=torch.float32).view(1, 1, 1, 1)
            self.f_h_tensor = torch.tensor(camera_intrinsics.f_h, dtype=torch.float32).view(1, 1, 1, 1)
            self.f_w_tensor = torch.tensor(camera_intrinsics.f_w, dtype=torch.float32).view(1, 1, 1, 1)
        else:
            self.c_h_tensor = None
            self.c_w_tensor = None
            self.f_h_tensor = None
            self.f_w_tensor = None

    def forward(self, sparse_depth : torch.Tensor, positions_map : torch.Tensor):
        """
        Args:
            sparse_depth: Input sparse depth map of shape (B, 1, H, W)
            positions_map: Tensor of shape (B, 2, H, W) containing normalized [-1, 1] vertical and horizontal coordinates for each pixel
        Returns:
            List of geometric feature tensors at different scales (length equals self.scales_num), each of shape (B, C_geo, H/2^i, W/2^i) for i in [0, self.scales_num-1]
        """

        geo_features_by_scale = []

        unorm = positions_map[:, 0:1, :, :]
        vnorm = positions_map[:, 1:2, :, :]
        valid_mask = (sparse_depth > 0).to(sparse_depth.dtype)

        geo_features_by_scale.append(
           self._calc_geofeatures(sparse_depth, vnorm, unorm)
        )

        for _ in range(1, self.scales_num):
            unorm = self.pool(unorm)
            vnorm = self.pool(vnorm)
            sparse_depth, valid_mask = self.sparse_pool(sparse_depth, valid_mask)
            geo_features_by_scale.append(
                self._calc_geofeatures(sparse_depth, vnorm, unorm)
            )

        return geo_features_by_scale

    def _calc_geofeatures(self, d, vnorm, unorm):
        """
        Calculate geometric features for position encoding in CU-Net architecture.
        
        Arguments:
            d: Depth map
            vnorm: Normalized vertical coordinates
            unorm: Normalized horizontal coordinates
        """
        
        if self.geo_encoding_type == GeoEncodingType.STD:
            return None # No geometric features, return None and handle this case in UNetBase
        elif self.geo_encoding_type == GeoEncodingType.Z:
            return d
        elif self.geo_encoding_type == GeoEncodingType.UV:
            return self._calc_geo_uv(vnorm, unorm)
        elif self.geo_encoding_type == GeoEncodingType.XYZ:
            return self._calc_geo_xyz(d, vnorm, unorm)
        else:
            raise ValueError(f"Unsupported geo_encoding_type: {self.geo_encoding_type}")
    
    def _calc_geo_xyz(self, d, vnorm, unorm):
        """
        Calculate XYZ coordinates as geometric features for position encoding in CU-Net architecture.
        
        Arguments:
            d: Depth map
            vnorm: Normalized vertical coordinates
            unorm: Normalized horizontal coordinates
        """
        x = d * (0.5 * self.img_size[0] * (vnorm+1) - self.c_h_tensor) / self.f_h_tensor
        y = d * (0.5 * self.img_size[1] * (unorm+1) - self.c_w_tensor) / self.f_w_tensor
        z = d
        return torch.cat((x, y, z),1)
        
    def _calc_geo_uv(self, vnorm, unorm):
        """
        Calculate UV coordinates as geometric features for position encoding in CU-Net architecture.
        
        Arguments:
            vnorm: Normalized vertical coordinates
            unorm: Normalized horizontal coordinates
        """
        return torch.cat((vnorm, unorm),1)
    
    @staticmethod
    def get_geo_planes_num(geo_encoding_type: GeoEncodingType):
        if geo_encoding_type == GeoEncodingType.STD:
            return 0
        elif geo_encoding_type == GeoEncodingType.Z:
            return 1
        elif geo_encoding_type == GeoEncodingType.UV:
            return 2
        elif geo_encoding_type == GeoEncodingType.XYZ:
            return 3
        else:
            raise ValueError(f"Unsupported geo_encoding_type: {geo_encoding_type}")
        
    @staticmethod
    def calc_position_map(img_size: tuple):
        """
        Calculate normalized position map for the input image size.
        
        Arguments:
            img_size: Tuple (H, W) representing the height and width of the input image
            device: torch.device to create the position map on
        Returns:
            positions_map: Tensor of shape (1, 2, H, W) containing normalized [-1, 1] vertical and horizontal coordinates for each pixel
        """
        H, W = img_size
        v_coords = torch.linspace(-1, 1, steps=H).view(1, 1, H, 1).expand(-1, -1, -1, W)
        u_coords = torch.linspace(-1, 1, steps=W).view(1, 1, 1, W).expand(-1, -1, H, -1)
        positions_map = torch.cat((u_coords, v_coords), dim=1)
        return positions_map

if __name__ == "__main__":
    # GetFeatures test with all encoding types
    batch_size = 8
    img_size = (256, 256)
    sparse_depth = torch.rand(batch_size, 1, img_size[0], img_size[1])
    positions_map = GeoFeatures.calc_position_map(img_size) # normalized to [-1, 1]
    camera_intrinsics = CameraIntrinsics(c_h=128, c_w=128, f_h=256, f_w=256)

    for geo_encoding_type in GeoEncodingType:
        print(f"Testing GeoFeatures with encoding type: {geo_encoding_type}")
        geo_features_module = GeoFeatures(geo_encoding_type=geo_encoding_type, img_size=img_size, camera_intrinsics=camera_intrinsics)
        geo_features_by_scale = geo_features_module(sparse_depth, positions_map)
        for i, features in enumerate(geo_features_by_scale):
            if features is not None:
                print(f"Scale {i}: Geometric features shape: {features.shape}")
            else:
                print(f"Scale {i}: No geometric features (STD encoding)")




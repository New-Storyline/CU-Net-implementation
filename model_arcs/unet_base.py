
import torch
import torch.nn as nn
from base_layers import *

class UNetBase(nn.Module):
    def __init__(self, in_channels, out_channels, geoplanes):
        super(UNetBase, self).__init__()

        assert geoplanes >= 0 and geoplanes <= 3, "geoplanes should be between 0 and 3"

        self.cobv_block_1 = self.UNetConvBlock(in_channels, 64, stride=2, geoplanes=geoplanes)
        self.cobv_block_2 = self.UNetConvBlock(64, 128,  stride=2, geoplanes=geoplanes)
        self.cobv_block_3 = self.UNetConvBlock(128, 256, stride=2, geoplanes=geoplanes)
        self.cobv_block_4 = self.UNetConvBlock(256, 256, stride=2, geoplanes=geoplanes)
        self.cobv_block_5 = self.UNetConvBlock(256, 256, stride=2, geoplanes=geoplanes)

        self.deconv_block_1 = DeConvBlock(256, 256, stride=2)
        self.deconv_block_2 = DeConvBlock(256, 256, stride=2)
        self.deconv_block_3 = DeConvBlock(256, 128, stride=2)
        self.deconv_block_4 = DeConvBlock(128, 64 , stride=2)
        self.deconv_block_5 = DeConvBlock(64, in_channels, stride=2) # in_channels for skip connection

        self.pre_final_block = PreFinalConvBlock(in_channels, out_channels, act=False)

    def forward(self, x, geo_features_by_scale):
        """
        Args:
            x: Input image tensor of shape (B, C_in, H, W)
            geo_features_by_scale: List of geometric feature tensors at different scales 6 scales ((B, C_geo, H/2^i, W/2^i) for i in [0, 5])
        Returns:
            Output tensor of shape (B, C_out, H, W)
        """

        # Encoder
        x1 = self.cobv_block_1(x, geo_feature_1=geo_features_by_scale[0], geo_feature_2=geo_features_by_scale[1])
        x2 = self.cobv_block_2(x1, geo_feature_1=geo_features_by_scale[1], geo_feature_2=geo_features_by_scale[2])
        x3 = self.cobv_block_3(x2, geo_feature_1=geo_features_by_scale[2], geo_feature_2=geo_features_by_scale[3])
        x4 = self.cobv_block_4(x3, geo_feature_1=geo_features_by_scale[3], geo_feature_2=geo_features_by_scale[4])
        x5 = self.cobv_block_5(x4, geo_feature_1=geo_features_by_scale[4], geo_feature_2=geo_features_by_scale[5])

        # Decoder
        d4 = self.deconv_block_1(x5, resigual=x4)
        d3 = self.deconv_block_2(d4, resigual=x3)
        d2 = self.deconv_block_3(d3, resigual=x2)
        d1 = self.deconv_block_4(d2, resigual=x1)
        d0 = self.deconv_block_5(d1, resigual=x)

        out = self.pre_final_block(d0)

        return out

    class UNetConvBlock(nn.Module):
        
        def __init__(self, inplanes : int, outplanes: int, stride=1, geoplanes=None):
            super().__init__()
            self.block_1 = ResGeoConvBlock(inplanes, outplanes, stride=stride, geoplanes=geoplanes)
            self.block_2 = ResGeoConvBlock(outplanes, outplanes, stride=1, geoplanes=geoplanes)
            self.block_3 = ResGeoConvBlock(outplanes, outplanes, stride=1, geoplanes=geoplanes)
            self.block_4 = ResGeoConvBlock(outplanes, outplanes, stride=1, geoplanes=geoplanes)


        def forward(self, x, geo_feature_1=None, geo_feature_2=None):
            
            x = self.block_1(x, geo_feature_1, geo_feature_2)
            x = self.block_2(x, geo_feature_2, geo_feature_2)
            x = self.block_3(x, geo_feature_2, geo_feature_2)
            x = self.block_4(x, geo_feature_2, geo_feature_2)
            return x
        
if __name__ == "__main__":
    # Example usage
    model = UNetBase(in_channels=16, out_channels=1, geoplanes=3)
    input_tensor = torch.randn(2, 16, 256, 256)
    geo_features_by_scale = [torch.randn(2, 3, 256 // (2 ** i), 256 // (2 ** i)) for i in range(6)]
    output = model(input_tensor, geo_features_by_scale)
    print(output.shape)
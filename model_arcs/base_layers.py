
import torch.nn as nn
import torch.nn.functional as F
import torch

def conv1x1(inplanes, planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, groups=groups, bias=bias)

def conv3x3(inplanes, planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(
        inplanes, 
        planes,
        kernel_size=3, 
        stride=stride,
        padding=padding, 
        groups=groups,
        bias=bias, 
        dilation=dilation
    )

def calc_geofeatures(d, vnorm, unorm, h, w, ch, cw, fh, fw):
    """
    Calculate geometric features for position encoding in CU-Net architecture.
    
    Arguments:
        d: Depth map
        vnorm: Normalized vertical coordinates
        unorm: Normalized horizontal coordinates
        h: Height of the feature map
        w: Width of the feature map
        ch: Center height
        cw: Center width
        fh: Focal height
        fw: Focal width
    """
    x = d * (0.5 * h * (vnorm+1) - ch) / fh
    y = d * (0.5 * w * (unorm+1) - cw) / fw
    return torch.cat((x, y, d),1)

class ResGeoConvBlock(nn.Module):

    """
    Residual block with geometric features for CU-Net architecture. (BasicBlockGeo from original code)

    This block takes in the input feature map `x` and two geometric feature maps `g1` and `g2`.
    The geometric features are concatenated with the input and intermediate features before convolution.
    """
    def __init__(self, inplanes, planes, stride=1, geoplanes=3):
        super(ResGeoConvBlock, self).__init__()

        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x, g1=None, g2=None):
        identity = x

        if g1 is not None:
            x = torch.cat((x, g1), 1)
        shortcut = self.downsample(x) if hasattr(self, 'downsample') else identity
        out = F.relu(self.bn1(self.conv1(x)))
        if g2 is not None:
            out = torch.cat((g2,out), 1)
        out = self.bn2(self.conv2(out))

        out += shortcut
        out = F.relu(out)

        return out
    
class InitConvBlock(nn.Module):

    """
    Lightweight and initial convolutional block for CU-Net architecture. (Convbnrelu from original code)

    This block consists of a convolutional layer followed by batch normalization and ReLU activation.
    """
    def __init__(self, inplanes, planes, norm_layer=False, stride=1, kernel_size=3, padding=1):
        super(InitConvBlock, self).__init__()

        self.norm_layer = norm_layer

        self.conv = nn.Conv2d(
            in_channels=inplanes, 
            out_channels=planes, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            bias=not norm_layer
        )

        if norm_layer:
            self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv(x)
        if self.norm_layer:
            out = self.bn(out)
        out = F.relu(out)
        return out
    
class PreFinalConvBlock(nn.Module):

    """
    This block is used before depth and confidence prediction in the CU-Net architecture. (BasicBlock from original code)
    """
    def __init__(self, inplanes, planes, stride=1, act=True):
        super(PreFinalConvBlock, self).__init__()
        self.act = act
        self.is_downsample = stride != 1 or inplanes != planes

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if self.is_downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):

        if self.is_downsample:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += shortcut
        if self.act:
            out = F.relu(out)

        return out
    
class DeConvBlock(nn.Module):
    """
    UNet decoder block with optional batch normalization for CU-Net architecture. (Deconvbnrelu_pre from original code)
    """
    def __init__(self, inplanes, planes, norm_layer=True, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        self.norm_layer = norm_layer
        self.deconv = nn.ConvTranspose2d(
            in_channels=inplanes, 
            out_channels=planes, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            output_padding=output_padding, 
            bias=not norm_layer
        )

        if norm_layer:
            self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, resigual=None):
        out = self.deconv(x)
        if self.norm_layer:
            out = self.bn(out)
        out = F.relu(out)
        if resigual is not None:
            out = out + resigual
        return out
    
class SparseDownSampleClosest(nn.Module):
    """Downsample a sparse depth map by selecting the closest (minimum) valid
    depth value within each pooling window.

    Standard pooling cannot be applied to sparse depth maps because zero
    (invalid) pixels would corrupt the result.  This module works around
    that by encoding the depth so that invalid pixels are guaranteed to
    lose the ``max`` competition:

    1. Negate the depth and subtract a large constant for invalid pixels::

           encode = -(1 - mask) * L - d

       Valid pixels (mask=1) become ``-d``; invalid ones become ``-L``,
       which is always smaller than any valid ``-d`` (assuming d < L).

    2. Apply ``MaxPool2d`` on the encoded values.  The maximum of ``-d``
       values equals ``-d_min``, i.e. the closest valid depth wins.

    3. Negate back: ``d_out = -MaxPool(encode)``.  If the entire window
       was invalid (mask_out=0), subtract ``L`` to reset the value to 0.
    """

    def __init__(self, stride):
        super(SparseDownSampleClosest, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
    def forward(self, d, mask):
        encode_d = - (1-mask)*self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1-mask_result)*self.large_number

        return d_result, mask_result



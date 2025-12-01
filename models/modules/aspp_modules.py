"""
ASPP (Atrous Spatial Pyramid Pooling) for 3D
표준 ASPP 블록 구현 (DeepLab v3 스타일)

Reference:
    Rethinking Atrous Convolution for Semantic Image Segmentation (Chen et al., arXiv 2017)
    DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, 
    and Fully Connected CRFs (Chen et al., TPAMI 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_3d_unet import _make_norm3d


class ASPPConv3D(nn.Module):
    """3D Atrous Convolution for ASPP
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        dilation: Dilation rate
        norm: 정규화 타입 ('bn', 'in', 'gn')
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int, norm: str = 'bn'):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=padding, 
            dilation=dilation, 
            bias=False
        )
        self.bn = _make_norm3d(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)
        
        Returns:
            출력 텐서 (B, out_channels, D, H, W)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPooling3D(nn.Module):
    """Global Average Pooling + 1x1x1 Conv + Upsampling for ASPP
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        norm: 정규화 타입 ('bn', 'in', 'gn')
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = _make_norm3d(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)
        
        Returns:
            출력 텐서 (B, out_channels, D, H, W)
        """
        # Global Average Pooling
        x = self.global_pool(x)  # (B, C, 1, 1, 1)
        # 1x1x1 Convolution
        x = self.conv(x)  # (B, out_channels, 1, 1, 1)
        x = self.bn(x)
        x = self.relu(x)
        # Upsample to original spatial size
        x = F.interpolate(x, size=x.shape[2:], mode='trilinear', align_corners=True)
        # Actually, we need to upsample to match the input size
        # Store input size for proper upsampling
        _, _, d, h, w = x.shape
        # We'll handle upsampling in the ASPP module itself
        return x


class ASPP3D(nn.Module):
    """3D Atrous Spatial Pyramid Pooling (ASPP) Module
    
    표준 ASPP 구조:
    - 1x1x1 convolution (dilation=1)
    - 3x3x3 convolution with dilation rate 6
    - 3x3x3 convolution with dilation rate 12
    - 3x3x3 convolution with dilation rate 18
    - Global Average Pooling + 1x1x1 conv + upsampling
    - 모든 브랜치를 concat하고 1x1x1 conv로 채널 수 조정
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        dilation_rates: Dilation rate 리스트 (기본값: [6, 12, 18])
        norm: 정규화 타입 ('bn', 'in', 'gn')
        use_image_pooling: Global Average Pooling 브랜치 사용 여부 (기본값: True)
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dilation_rates: list[int] = None,
        norm: str = 'bn',
        use_image_pooling: bool = True
    ):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [6, 12, 18]
        
        self.use_image_pooling = use_image_pooling
        
        # 1x1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3x3 atrous convolution branches
        self.aspp_convs = nn.ModuleList([
            ASPPConv3D(in_channels, out_channels, dilation=rate, norm=norm)
            for rate in dilation_rates
        ])
        
        # Global Average Pooling branch
        if self.use_image_pooling:
            self.image_pooling = ASPPPooling3D(in_channels, out_channels, norm=norm)
            num_branches = len(dilation_rates) + 2  # +1 for 1x1x1, +1 for pooling
        else:
            num_branches = len(dilation_rates) + 1  # +1 for 1x1x1
        
        # Final 1x1x1 convolution to reduce channels
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * num_branches, out_channels, kernel_size=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5)  # Dropout for regularization
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, in_channels, D, H, W)
        
        Returns:
            출력 텐서 (B, out_channels, D, H, W)
        """
        # Store input spatial size for upsampling
        _, _, d, h, w = x.shape
        
        # 1x1x1 convolution branch
        x1 = self.conv1x1(x)
        
        # 3x3x3 atrous convolution branches
        aspp_outs = [x1]
        for aspp_conv in self.aspp_convs:
            aspp_outs.append(aspp_conv(x))
        
        # Global Average Pooling branch
        if self.use_image_pooling:
            x_pool = self.image_pooling(x)  # (B, out_channels, 1, 1, 1)
            # Upsample to match input spatial size
            x_pool = F.interpolate(x_pool, size=(d, h, w), mode='trilinear', align_corners=True)
            aspp_outs.append(x_pool)
        
        # Concatenate all branches
        x = torch.cat(aspp_outs, dim=1)  # (B, out_channels * num_branches, D, H, W)
        
        # Project to output channels
        x = self.project(x)  # (B, out_channels, D, H, W)
        
        return x


class ASPP3D_Simplified(nn.Module):
    """간소화된 3D ASPP 모듈 (메모리 효율적)
    
    표준 ASPP보다 적은 브랜치를 사용하여 메모리 사용량을 줄인 버전입니다.
    Dilation rate [1, 2, 5] 조합을 사용하여 다양한 수용 영역을 효율적으로 포착합니다.
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        dilation_rates: Dilation rate 리스트 (기본값: [1, 2, 5])
        norm: 정규화 타입 ('bn', 'in', 'gn')
        use_image_pooling: Global Average Pooling 브랜치 사용 여부 (기본값: False)
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dilation_rates: list[int] = None,
        norm: str = 'bn',
        use_image_pooling: bool = False
    ):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 5]
        
        self.use_image_pooling = use_image_pooling
        
        # 1x1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3x3 atrous convolution branches
        self.aspp_convs = nn.ModuleList([
            ASPPConv3D(in_channels, out_channels, dilation=rate, norm=norm)
            for rate in dilation_rates
        ])
        
        # Global Average Pooling branch (optional)
        if self.use_image_pooling:
            self.image_pooling = ASPPPooling3D(in_channels, out_channels, norm=norm)
            num_branches = len(dilation_rates) + 2  # +1 for 1x1x1, +1 for pooling
        else:
            num_branches = len(dilation_rates) + 1  # +1 for 1x1x1
        
        # Final 1x1x1 convolution to reduce channels
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * num_branches, out_channels, kernel_size=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, in_channels, D, H, W)
        
        Returns:
            출력 텐서 (B, out_channels, D, H, W)
        """
        # Store input spatial size for upsampling
        _, _, d, h, w = x.shape
        
        # 1x1x1 convolution branch
        x1 = self.conv1x1(x)
        
        # 3x3x3 atrous convolution branches
        aspp_outs = [x1]
        for aspp_conv in self.aspp_convs:
            aspp_outs.append(aspp_conv(x))
        
        # Global Average Pooling branch (optional)
        if self.use_image_pooling:
            x_pool = self.image_pooling(x)  # (B, out_channels, 1, 1, 1)
            # Upsample to match input spatial size
            x_pool = F.interpolate(x_pool, size=(d, h, w), mode='trilinear', align_corners=True)
            aspp_outs.append(x_pool)
        
        # Concatenate all branches
        x = torch.cat(aspp_outs, dim=1)  # (B, out_channels * num_branches, D, H, W)
        
        # Project to output channels
        x = self.project(x)  # (B, out_channels, D, H, W)
        
        return x


__all__ = [
    'ASPPConv3D',
    'ASPPPooling3D',
    'ASPP3D',
    'ASPP3D_Simplified',
]


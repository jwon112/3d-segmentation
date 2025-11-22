"""
CBAM (Convolutional Block Attention Module) for 3D
Channel Attention and Spatial Attention modules for 3D feature maps

Reference:
    CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_3d_unet import _make_norm3d


class ChannelAttention3D(nn.Module):
    """3D Channel Attention Module (SE-Net style)
    
    채널 어텐션 메커니즘을 통해 중요한 채널에 집중합니다.
    Global Average Pooling과 Global Max Pooling을 모두 사용하여 더 풍부한 정보를 활용합니다.
    
    Args:
        channels: 입력 채널 수
        reduction: 채널 축소 비율 (기본값: 16)
    
    Reference:
        Squeeze-and-Excitation Networks (Hu et al., CVPR 2018)
        CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)
        
        Returns:
            Channel attention이 적용된 출력 텐서 (B, C, D, H, W)
        """
        b, c, _, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.mlp(avg_out)
        
        # Max pooling branch
        max_out = self.max_pool(x).view(b, c)
        max_out = self.mlp(max_out)
        
        # Combine both branches
        out = avg_out + max_out
        out = self.sigmoid(out)
        
        # Store latest channel attention weights for logging/analysis
        self.last_channel_weights = out.detach().cpu()  # (B, C)
        
        out = out.view(b, c, 1, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module (CBAM style)
    
    공간 어텐션 메커니즘을 통해 중요한 공간 위치에 집중합니다.
    채널 차원에서 Average Pooling과 Max Pooling을 수행한 후, 
    convolution을 통해 공간 어텐션 맵을 생성합니다.
    
    Args:
        kernel_size: Convolution kernel size (기본값: 7)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)
        
        Returns:
            Spatial attention이 적용된 출력 텐서 (B, C, D, H, W)
        """
        # Channel-wise average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        # Channel-wise max pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)  # (B, 2, D, H, W)
        out = self.conv(out)  # (B, 1, D, H, W)
        out = self.sigmoid(out)  # (B, 1, D, H, W)
        
        # Store latest spatial attention weights for logging/analysis
        self.last_spatial_weights = out.detach().cpu()  # (B, 1, D, H, W)
        
        return x * out


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module
    
    Channel Attention과 Spatial Attention을 순차적으로 적용합니다.
    
    Args:
        channels: 입력 채널 수
        reduction: Channel attention의 채널 축소 비율 (기본값: 16)
        spatial_kernel: Spatial attention의 kernel size (기본값: 7)
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)
        
        Returns:
            CBAM attention이 적용된 출력 텐서 (B, C, D, H, W)
        """
        # Apply channel attention first
        out = self.channel_attention(x)
        # Then apply spatial attention
        out = self.spatial_attention(out)
        
        # Store latest weights from both attention modules for logging/analysis
        # Channel attention weights are already stored in self.channel_attention.last_channel_weights
        # Spatial attention weights are already stored in self.spatial_attention.last_spatial_weights
        self.last_channel_weights = getattr(self.channel_attention, 'last_channel_weights', None)
        self.last_spatial_weights = getattr(self.spatial_attention, 'last_spatial_weights', None)
        
        return out


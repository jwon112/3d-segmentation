"""
GhostNet Modules
GhostNet 스타일의 3D 모듈들
"""

import torch
import torch.nn as nn

from ..model_3d_unet import _make_norm3d


class GhostModule3D(nn.Module):
    """3D Ghost Module: cheap operations to generate more feature maps."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, 
                 ratio: int = 2, norm: str = 'bn'):
        super().__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            _make_norm3d(norm, init_channels),
            nn.ReLU(inplace=True) if ratio == 2 else nn.ReLU(inplace=True),
        )
        
        # new_channels가 정확히 필요한 만큼만 생성되도록 조정
        actual_new_channels = min(new_channels, out_channels - init_channels)
        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, actual_new_channels, kernel_size=3, stride=1, padding=1, 
                     groups=init_channels, bias=False),
            _make_norm3d(norm, actual_new_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # 필요한 채널만 반환 (슬라이싱은 여전히 필요하지만, 모든 파라미터가 사용됨)
        return out[:, :self.out_channels, :, :, :]


class GhostBottleneck3D(nn.Module):
    """3D Ghost Bottleneck block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = 'bn'):
        super().__init__()
        self.stride = stride
        
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
            )
        else:
            self.downsample = None
        
        mid_channels = out_channels // 2
        self.ghost1 = GhostModule3D(in_channels, mid_channels, stride=stride, norm=norm)
        self.ghost2 = GhostModule3D(mid_channels, out_channels, stride=1, norm=norm)
        self.use_res_connect = (stride == 1) and (in_channels == out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        
        out = self.ghost1(x)
        out = self.ghost2(out)
        
        if self.use_res_connect:
            return out + residual
        return out


class Down3DGhostNet(nn.Module):
    """Downsampling using GhostNet bottleneck (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = GhostBottleneck3D(in_channels, out_channels, stride=2, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


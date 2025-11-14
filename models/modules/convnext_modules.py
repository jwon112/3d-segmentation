"""
ConvNeXt Modules
ConvNeXt 스타일의 3D 모듈들
"""

import torch
import torch.nn as nn

from ..model_3d_unet import _make_norm3d


class ConvNeXtBlock3D(nn.Module):
    """3D ConvNeXt block: depthwise conv + layer scale + stochastic depth."""
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6, norm: str = 'bn'):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = _make_norm3d(norm, dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # Stochastic depth can be added if needed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1, 1) * x
        
        x = input + self.drop_path(x)
        return x


class Down3DConvNeXt(nn.Module):
    """Downsampling using ConvNeXt block (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', num_blocks: int = 2):
        super().__init__()
        # Downsampling layer
        self.downsample = nn.Sequential(
            _make_norm3d(norm, in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
        )
        # ConvNeXt blocks
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock3D(out_channels, norm=norm) for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


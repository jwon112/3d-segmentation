"""
3D Segmentation Models

- baseline/: UNet3D 빌딩 블록, modal comparison. (UNETR/SwinUNETR는 MONAI 외부 로드)
- custom/: project-specific variants (dual-branch 등)
- modules/: shared building blocks
- channel_configs.py: shared channel configs
"""

from .baseline import (
    UNet3D_Medium,
    UNet3D_Small,
    DoubleConv3D,
    Down3D,
    Up3D,
    OutConv3D,
    _make_norm3d,
    _make_activation,
)

__all__ = [
    "UNet3D_Medium",
    "UNet3D_Small",
    "DoubleConv3D",
    "Down3D",
    "Up3D",
    "OutConv3D",
    "_make_norm3d",
    "_make_activation",
]

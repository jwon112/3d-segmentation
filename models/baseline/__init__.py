"""
Baseline / reference models (paper implementations).
"""

from .model_3d_unet import (
    UNet3D_Medium,
    UNet3D_Small,
    DoubleConv3D,
    Down3D,
    Up3D,
    OutConv3D,
    _make_norm3d,
    _make_activation,
)
from .model_unetr import (
    UNETR,
    UNETR_Simplified,
    PatchEmbedding3D as UNETRPatchEmbedding,
    PositionalEncoding3D,
    TransformerBlock,
)
from .model_swin_unetr import (
    SwinUNETR,
    SwinUNETR_Simplified,
    PatchEmbedding3D as SwinPatchEmbedding,
    WindowAttention3D,
    SwinTransformerBlock3D,
    PatchMerging3D,
)
from .mobileunetr import MobileUNETR

__all__ = [
    "UNet3D_Medium",
    "UNet3D_Small",
    "DoubleConv3D",
    "Down3D",
    "Up3D",
    "OutConv3D",
    "_make_norm3d",
    "_make_activation",
    "UNETR",
    "UNETR_Simplified",
    "UNETRPatchEmbedding",
    "PositionalEncoding3D",
    "TransformerBlock",
    "SwinUNETR",
    "SwinUNETR_Simplified",
    "SwinPatchEmbedding",
    "WindowAttention3D",
    "SwinTransformerBlock3D",
    "PatchMerging3D",
    "MobileUNETR",
]

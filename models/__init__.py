"""
3D Segmentation Models
"""

from .model_3d_unet import (
    UNet3D_Medium, 
    UNet3D_Small, 
    DoubleConv3D, 
    Down3D, 
    Up3D, 
    OutConv3D,
)

from .model_unetr import (
    UNETR,
    UNETR_Simplified,
    PatchEmbedding3D as UNETRPatchEmbedding,
    PositionalEncoding3D,
    TransformerBlock
)

from .model_swin_unetr import (
    SwinUNETR,
    SwinUNETR_Simplified,
    PatchEmbedding3D as SwinPatchEmbedding,
    WindowAttention3D,
    SwinTransformerBlock3D,
    PatchMerging3D
)

from .mobileunetr import MobileUNETR

from .quadbranch_unet import (
    QuadBranchUNet3D,
    QuadBranchUNet3D_XS,
    QuadBranchUNet3D_Small,
    QuadBranchUNet3D_Medium,
    QuadBranchUNet3D_Large,
)

from .quadbranch_unet_attention import (
    QuadBranchAttentionBase,
    QuadBranchUNet3D_Channel_Centralized_Concat,
    QuadBranchUNet3D_Channel_Distributed_Concat,
    QuadBranchUNet3D_Channel_Distributed_Conv,
    QuadBranchUNet3D_Spatial_Centralized_Concat,
    QuadBranchUNet3D_Spatial_Distributed_Concat,
    QuadBranchUNet3D_Spatial_Distributed_Conv,
)

__all__ = [
    # U-Net models
    'UNet3D_Medium',
    'UNet3D_Small',
    'DoubleConv3D',
    'Down3D', 
    'Up3D',
    'OutConv3D',
    
    # UNETR models
    'UNETR',
    'UNETR_Simplified',
    'UNETRPatchEmbedding',
    'PositionalEncoding3D',
    'TransformerBlock',
    
    # Swin UNETR models
    'SwinUNETR',
    'SwinUNETR_Simplified',
    'SwinPatchEmbedding',
    'WindowAttention3D',
    'SwinTransformerBlock3D',
    'PatchMerging3D',
    
    # MobileUNETR models
    'MobileUNETR',
    
    # Quad-Branch UNet models
    'QuadBranchUNet3D',
    'QuadBranchUNet3D_XS',
    'QuadBranchUNet3D_Small',
    'QuadBranchUNet3D_Medium',
    'QuadBranchUNet3D_Large',
    'QuadBranchAttentionBase',
    'QuadBranchUNet3D_Channel_Centralized_Concat',
    'QuadBranchUNet3D_Channel_Distributed_Concat',
    'QuadBranchUNet3D_Channel_Distributed_Conv',
    'QuadBranchUNet3D_Spatial_Centralized_Concat',
    'QuadBranchUNet3D_Spatial_Distributed_Concat',
    'QuadBranchUNet3D_Spatial_Distributed_Conv',
]

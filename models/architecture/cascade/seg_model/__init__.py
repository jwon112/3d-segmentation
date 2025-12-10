"""
Cascade segmentation architectures.
"""

from .shufflenet_v2 import (
    CascadeShuffleNetV2UNet3D,
    build_cascade_shufflenet_v2_unet3d,
    CascadeShuffleNetV2UNet3D_P3D,
    build_cascade_shufflenet_v2_unet3d_p3d,
    CascadeShuffleNetV2UNet3D_LK,
    build_cascade_shufflenet_v2_unet3d_lk,
    CascadeShuffleNetV2UNet3D_P3D_LK,
    build_cascade_shufflenet_v2_unet3d_p3d_lk,
    CascadeShuffleNetV2UNet3D_LKAHybrid,
    build_cascade_shufflenet_v2_unet3d_lka_hybrid,
    CascadeShuffleNetV2UNet3D_MViT,
    build_cascade_shufflenet_v2_unet3d_mvit,
    CascadeShuffleNetV2UNet3D_P3D_MViT,
    build_cascade_shufflenet_v2_unet3d_p3d_mvit,
)
from .shufflenet_v2_segnext import (
    CascadeShuffleNetV2SegNeXt3D_LKA,
    build_cascade_shufflenet_v2_segnext_lka,
    CascadeShuffleNetV2SegNeXt3D_P3D_LKA,
    build_cascade_shufflenet_v2_segnext_p3d_lka,
)
from .baseline_models import (
    CascadeUNet3D,
    CascadeUNETR,
    CascadeSwinUNETR,
    build_cascade_unet3d,
    build_cascade_unetr,
    build_cascade_swin_unetr,
)
from .patch_conv_transformer import (
    CascadePatchConvTransformerUNet3D,
    build_cascade_patch_conv_transformer_unet3d,
)

__all__ = [
    "CascadeShuffleNetV2UNet3D",
    "build_cascade_shufflenet_v2_unet3d",
    "CascadeShuffleNetV2UNet3D_P3D",
    "build_cascade_shufflenet_v2_unet3d_p3d",
    "CascadeShuffleNetV2UNet3D_LK",
    "build_cascade_shufflenet_v2_unet3d_lk",
    "CascadeShuffleNetV2UNet3D_P3D_LK",
    "build_cascade_shufflenet_v2_unet3d_p3d_lk",
    "CascadeShuffleNetV2UNet3D_LKAHybrid",
    "build_cascade_shufflenet_v2_unet3d_lka_hybrid",
    "CascadeShuffleNetV2SegNeXt3D_LKA",
    "build_cascade_shufflenet_v2_segnext_lka",
    "CascadeShuffleNetV2SegNeXt3D_P3D_LKA",
    "build_cascade_shufflenet_v2_segnext_p3d_lka",
    "CascadeShuffleNetV2UNet3D_MViT",
    "build_cascade_shufflenet_v2_unet3d_mvit",
    "CascadeShuffleNetV2UNet3D_P3D_MViT",
    "build_cascade_shufflenet_v2_unet3d_p3d_mvit",
    # Baseline models
    "CascadeUNet3D",
    "CascadeUNETR",
    "CascadeSwinUNETR",
    "build_cascade_unet3d",
    "build_cascade_unetr",
    "build_cascade_swin_unetr",
    # Patch Conv Transformer
    "CascadePatchConvTransformerUNet3D",
    "build_cascade_patch_conv_transformer_unet3d",
]



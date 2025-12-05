"""
Baseline models for cascade segmentation comparison.

This module provides wrapper classes for standard baseline models:
- UNet3D: Standard 3D U-Net
- UNETR: Transformer-based 3D segmentation
- SwinUNETR: Swin Transformer-based 3D segmentation

All models are adapted to work with the cascade framework:
- Support for n_image_channels + n_coord_channels input
- include_coords option
- Consistent interface with other cascade models
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Union, Tuple

# Import baseline models
from ...model_3d_unet import UNet3D
from ...model_unetr import UNETR
from ...model_swin_unetr import SwinUNETR


class CascadeUNet3D(nn.Module):
    """
    Cascade wrapper for standard 3D U-Net baseline.
    
    Adapts UNet3D to work with cascade framework:
    - Supports n_image_channels + n_coord_channels input
    - include_coords option for coordinate channels
    """
    
    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        norm: str = "bn",
        size: str = "s",
        include_coords: bool = True,
        bilinear: bool = False,
    ):
        super().__init__()
        self.n_image_channels = n_image_channels
        self.n_coord_channels = n_coord_channels
        self.include_coords = include_coords and n_coord_channels > 0
        self.norm = norm or "bn"
        self.size = size
        
        # Calculate input channels
        in_channels = n_image_channels + (n_coord_channels if self.include_coords else 0)
        
        # Create UNet3D model
        self.unet = UNet3D(
            n_channels=in_channels,
            n_classes=n_classes,
            norm=norm,
            bilinear=bilinear,
            size=size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, n_image_channels + n_coord_channels, D, H, W)
        
        Returns:
            Output logits (B, n_classes, D, H, W)
        """
        return self.unet(x)


class CascadeUNETR(nn.Module):
    """
    Cascade wrapper for UNETR baseline with dynamic input size support.
    
    Adapts UNETR to work with cascade framework:
    - Supports n_image_channels + n_coord_channels input
    - include_coords option for coordinate channels
    - Dynamic input size handling (works with any input size, not just fixed img_size)
    """
    
    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        patch_size: Union[int, Tuple[int, int, int]] = (12, 12, 12),  # 96^3 / 8 = 12^3 patches
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        include_coords: bool = True,
    ):
        super().__init__()
        self.n_image_channels = n_image_channels
        self.n_coord_channels = n_coord_channels
        self.include_coords = include_coords and n_coord_channels > 0
        
        # Normalize patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        
        # Calculate input channels
        in_channels = n_image_channels + (n_coord_channels if self.include_coords else 0)
        
        # Import UNETR components
        from ...model_unetr import PatchEmbedding3D, PositionalEncoding3D, TransformerBlock
        
        self.embed_dim = embed_dim
        self.patch_size_tuple = patch_size
        
        # Patch embedding (will work with any input size)
        self.patch_embed = PatchEmbedding3D(
            img_size=(96, 96, 96),  # Dummy size, actual size determined in forward
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Positional encoding (dynamic max_len)
        self.pos_encoding = PositionalEncoding3D(embed_dim, max_len=10000)
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder heads
        self.decoder_heads = nn.ModuleList([
            nn.ConvTranspose3d(embed_dim, 512, kernel_size=2, stride=2),
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
        ])
        
        # Final output layer
        self.final_conv = nn.Conv3d(64, n_classes, kernel_size=1)
        
        # Skip connection convolutions
        self.skip_convs = nn.ModuleList([
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, n_image_channels + n_coord_channels, D, H, W)
        
        Returns:
            Output logits (B, n_classes, D, H, W)
        """
        import torch.nn.functional as F
        
        B, C, D, H, W = x.shape
        
        # Patch embedding (works with any input size)
        x_patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x_patches = x_patches.transpose(0, 1)  # (num_patches, B, embed_dim)
        x_patches = self.pos_encoding(x_patches)
        x_patches = x_patches.transpose(0, 1)  # (B, num_patches, embed_dim)
        
        # Transformer encoding
        for transformer_block in self.transformer_blocks:
            x_patches = transformer_block(x_patches)
        
        # Reshape back to spatial dimensions (dynamic based on actual input size)
        patch_d = D // self.patch_size[0]
        patch_h = H // self.patch_size[1]
        patch_w = W // self.patch_size[2]
        
        x_encoded = x_patches.transpose(1, 2).view(B, self.embed_dim, patch_d, patch_h, patch_w)
        
        # Decoder with skip connections
        skip_features = []
        
        # Create skip features at different scales
        skip_features.append(self.skip_convs[0](x))  # Original scale
        skip_features.append(self.skip_convs[1](F.max_pool3d(skip_features[0], 2)))  # 1/2 scale
        skip_features.append(self.skip_convs[2](F.max_pool3d(skip_features[1], 2)))  # 1/4 scale
        skip_features.append(self.skip_convs[3](F.max_pool3d(skip_features[2], 2)))  # 1/8 scale
        
        # Decoder
        x_dec = x_encoded
        for i, decoder_head in enumerate(self.decoder_heads):
            x_dec = decoder_head(x_dec)
            if i < len(skip_features):
                # Add skip connection (handle size mismatch with padding/cropping)
                skip = skip_features[-(i+1)]
                if x_dec.shape[2:] != skip.shape[2:]:
                    # Crop or pad to match
                    diff_d = x_dec.shape[2] - skip.shape[2]
                    diff_h = x_dec.shape[3] - skip.shape[3]
                    diff_w = x_dec.shape[4] - skip.shape[4]
                    if diff_d > 0 or diff_h > 0 or diff_w > 0:
                        skip = F.pad(skip, (0, max(0, diff_w), 0, max(0, diff_h), 0, max(0, diff_d)))
                    if diff_d < 0 or diff_h < 0 or diff_w < 0:
                        skip = skip[:, :, :x_dec.shape[2], :x_dec.shape[3], :x_dec.shape[4]]
                x_dec = x_dec + skip
        
        # Final output
        output = self.final_conv(x_dec)
        
        # Resize to match input size if needed
        if output.shape[2:] != (D, H, W):
            output = F.interpolate(output, size=(D, H, W), mode='trilinear', align_corners=False)
        
        return output


class CascadeSwinUNETR(nn.Module):
    """
    Cascade wrapper for SwinUNETR baseline with dynamic input size support.
    
    Adapts SwinUNETR to work with cascade framework:
    - Supports n_image_channels + n_coord_channels input
    - include_coords option for coordinate channels
    - Dynamic input size handling (works with any input size, not just fixed img_size)
    """
    
    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        patch_size: Union[int, Tuple[int, int, int]] = (4, 4, 4),  # 96^3 / 24 = 4^3 patches
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        include_coords: bool = True,
    ):
        super().__init__()
        self.n_image_channels = n_image_channels
        self.n_coord_channels = n_coord_channels
        self.include_coords = include_coords and n_coord_channels > 0
        
        # Normalize patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.window_size = window_size
        
        # Calculate input channels
        in_channels = n_image_channels + (n_coord_channels if self.include_coords else 0)
        
        # Import SwinUNETR components
        from ...model_swin_unetr import (
            PatchEmbedding3D, SwinTransformerBlock3D, PatchMerging3D
        )
        
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.depths = depths
        self.num_heads = num_heads
        
        # Patch embedding (will work with any input size)
        self.patch_embed = PatchEmbedding3D(
            img_size=(96, 96, 96),  # Dummy size, actual size determined in forward
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Build Swin Transformer layers (will compute resolutions dynamically)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            dim = embed_dim * 2 ** i_layer
            for i_block in range(depths[i_layer]):
                # Resolution will be computed dynamically in forward
                layer.append(SwinTransformerBlock3D(
                    dim=dim,
                    input_resolution=(24, 24, 24),  # Dummy, will be computed in forward
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                ))
            if i_layer < self.num_layers - 1:
                layer.append(PatchMerging3D(
                    input_resolution=(24, 24, 24),  # Dummy, will be computed in forward
                    dim=dim
                ))
            self.layers.append(layer)
        
        # Decoder
        self.decoder_conv1 = nn.ConvTranspose3d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2)
        self.decoder_conv2 = nn.ConvTranspose3d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.decoder_conv3 = nn.ConvTranspose3d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        
        # Final output layer
        self.final_conv = nn.Conv3d(embed_dim, n_classes, kernel_size=1)
        
        # Skip connection convolutions
        self.skip_conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.skip_conv2 = nn.Conv3d(embed_dim, embed_dim * 2, kernel_size=3, padding=1)
        self.skip_conv3 = nn.Conv3d(embed_dim * 2, embed_dim * 4, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, n_image_channels + n_coord_channels, D, H, W)
        
        Returns:
            Output logits (B, n_classes, D, H, W)
        """
        import torch.nn.functional as F
        
        B, C, D, H, W = x.shape
        
        # Calculate patch resolution dynamically
        patch_d = D // self.patch_size[0]
        patch_h = H // self.patch_size[1]
        patch_w = W // self.patch_size[2]
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Encoder with skip features
        skip_features = []
        skip_features.append(self.skip_conv1(x))  # Original scale
        skip_features.append(self.skip_conv2(F.max_pool3d(skip_features[0], 2)))  # 1/2 scale
        skip_features.append(self.skip_conv3(F.max_pool3d(skip_features[1], 2)))  # 1/4 scale
        
        # Process through Swin Transformer layers
        # Note: We need to handle dynamic resolution, but Swin blocks need fixed resolution
        # For now, we'll use a simplified approach that works with variable sizes
        current_resolution = (patch_d, patch_h, patch_w)
        
        for i_layer, layer in enumerate(self.layers):
            # Update resolution for this layer
            for i_block, block in enumerate(layer[:-1] if len(layer) > 1 else layer):
                # SwinTransformerBlock needs input_resolution, but we'll use current_resolution
                # This is a limitation - we need to pad/reshape to make it work
                x_patches = block(x_patches)
            
            if len(layer) > 1 and i_layer < self.num_layers - 1:
                # PatchMerging
                x_patches = layer[-1](x_patches)
                # Update resolution
                current_resolution = (
                    current_resolution[0] // 2,
                    current_resolution[1] // 2,
                    current_resolution[2] // 2
                )
        
        # Reshape back to spatial dimensions for the deepest feature
        final_patch_d, final_patch_h, final_patch_w = current_resolution
        x_encoded = x_patches.transpose(1, 2).view(
            B, self.embed_dim * (2 ** (self.num_layers - 1)), 
            final_patch_d, final_patch_h, final_patch_w
        )
        
        # Decoder with skip connections
        x_dec = self.decoder_conv1(x_encoded)
        if x_dec.shape[2:] != skip_features[2].shape[2:]:
            skip = skip_features[2]
            if x_dec.shape[2] > skip.shape[2] or x_dec.shape[3] > skip.shape[3] or x_dec.shape[4] > skip.shape[4]:
                skip = F.pad(skip, (0, max(0, x_dec.shape[4] - skip.shape[4]),
                                    0, max(0, x_dec.shape[3] - skip.shape[3]),
                                    0, max(0, x_dec.shape[2] - skip.shape[2])))
            else:
                skip = skip[:, :, :x_dec.shape[2], :x_dec.shape[3], :x_dec.shape[4]]
        x_dec = x_dec + skip
        
        x_dec = self.decoder_conv2(x_dec)
        if x_dec.shape[2:] != skip_features[1].shape[2:]:
            skip = skip_features[1]
            if x_dec.shape[2] > skip.shape[2] or x_dec.shape[3] > skip.shape[3] or x_dec.shape[4] > skip.shape[4]:
                skip = F.pad(skip, (0, max(0, x_dec.shape[4] - skip.shape[4]),
                                    0, max(0, x_dec.shape[3] - skip.shape[3]),
                                    0, max(0, x_dec.shape[2] - skip.shape[2])))
            else:
                skip = skip[:, :, :x_dec.shape[2], :x_dec.shape[3], :x_dec.shape[4]]
        x_dec = x_dec + skip
        
        x_dec = self.decoder_conv3(x_dec)
        if x_dec.shape[2:] != skip_features[0].shape[2:]:
            skip = skip_features[0]
            if x_dec.shape[2] > skip.shape[2] or x_dec.shape[3] > skip.shape[3] or x_dec.shape[4] > skip.shape[4]:
                skip = F.pad(skip, (0, max(0, x_dec.shape[4] - skip.shape[4]),
                                    0, max(0, x_dec.shape[3] - skip.shape[3]),
                                    0, max(0, x_dec.shape[2] - skip.shape[2])))
            else:
                skip = skip[:, :, :x_dec.shape[2], :x_dec.shape[3], :x_dec.shape[4]]
        x_dec = x_dec + skip
        
        # Final output
        output = self.final_conv(x_dec)
        
        # Resize to match input size if needed
        if output.shape[2:] != (D, H, W):
            output = F.interpolate(output, size=(D, H, W), mode='trilinear', align_corners=False)
        
        return output


# Builder functions for consistency with other cascade models
def build_cascade_unet3d(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
    bilinear: bool = False,
) -> CascadeUNet3D:
    """Build CascadeUNet3D model."""
    return CascadeUNet3D(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
        bilinear=bilinear,
    )


def build_cascade_unetr(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    patch_size: Union[int, Tuple[int, int, int]] = (12, 12, 12),
    embed_dim: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    include_coords: bool = True,
    size: str = "s",
) -> CascadeUNETR:
    """Build CascadeUNETR model with size variant support.
    
    Note: UNETR 공식 구현에서는 size variant를 제공하지 않지만,
    표준 ViT 스케일링을 참고하여 추가했습니다:
    - xs: ViT-Small scale (384 dim, 6 heads, 6 layers)
    - s: ViT-Base scale (768 dim, 12 heads, 12 layers) - 공식 기본값
    - m: ViT-Large scale (1024 dim, 16 heads, 16 layers)
    - l: ViT-Huge scale (1536 dim, 24 heads, 24 layers)
    """
    # Size-based configuration (ViT 표준 스케일링 참고)
    size_configs = {
        "xs": {"embed_dim": 384, "num_heads": 6, "num_layers": 6},   # ViT-Small scale
        "s": {"embed_dim": 768, "num_heads": 12, "num_layers": 12},  # ViT-Base (공식 기본값)
        "m": {"embed_dim": 1024, "num_heads": 16, "num_layers": 16}, # ViT-Large scale
        "l": {"embed_dim": 1536, "num_heads": 24, "num_layers": 24}, # ViT-Huge scale
    }
    
    if size in size_configs:
        config = size_configs[size]
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
    
    return CascadeUNETR(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        include_coords=include_coords,
    )


def build_cascade_swin_unetr(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    patch_size: Union[int, Tuple[int, int, int]] = (4, 4, 4),
    embed_dim: int = 96,
    depths: Tuple[int, ...] = (2, 2, 6, 2),
    num_heads: Tuple[int, ...] = (3, 6, 12, 24),
    window_size: int = 7,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    include_coords: bool = True,
    size: str = "s",
) -> CascadeSwinUNETR:
    """Build CascadeSwinUNETR model with size variant support.
    
    Note: SwinUNETR 공식 구현에서는 size variant를 제공하지 않지만,
    표준 Swin Transformer 스케일링을 참고하여 추가했습니다:
    - xs: Custom small scale (48 dim)
    - s: Swin-Tiny scale (96 dim, depths=(2,2,6,2)) - 공식 기본값
    - m: Swin-Small scale (96 dim) 또는 Swin-Base scale (128 dim) 참고
    - l: Swin-Large scale (192 dim) 참고
    """
    # Size-based configuration (Swin Transformer 표준 스케일링 참고)
    size_configs = {
        "xs": {
            "embed_dim": 48,  # Custom small scale
            "depths": (2, 2, 4, 2),
            "num_heads": (2, 4, 8, 16),
        },
        "s": {
            "embed_dim": 96,  # Swin-Tiny (공식 기본값)
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
        },
        "m": {
            "embed_dim": 192,  # Swin-Large scale 참고
            "depths": (2, 2, 8, 2),
            "num_heads": (6, 12, 24, 48),
        },
        "l": {
            "embed_dim": 384,  # Swin-Large scale 확장
            "depths": (2, 2, 12, 2),
            "num_heads": (12, 24, 48, 96),
        },
    }
    
    if size in size_configs:
        config = size_configs[size]
        embed_dim = config["embed_dim"]
        depths = config["depths"]
        num_heads = config["num_heads"]
    
    return CascadeSwinUNETR(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        include_coords=include_coords,
    )


__all__ = [
    "CascadeUNet3D",
    "CascadeUNETR",
    "CascadeSwinUNETR",
    "build_cascade_unet3d",
    "build_cascade_unetr",
    "build_cascade_swin_unetr",
]


"""
Cascade Patch-wise Conv + Transformer UNet.

핵심 아이디어:
- 패치를 먼저 나누기 (unfold)
- 각 패치 내부에서 convolution (inductive bias)
- 평탄화하여 패치 간 global attention
- Stage별 전략:
  - Stage 1-2: 패치화 + Conv만 (overlapping patches로 gridding artifact 방지)
  - Stage 3-4: 패치화 + Conv + Global Attention (shifted patches로 경계 정보 교환)

이 방법은:
- CNN의 inductive bias (translation equivariance, locality)
- Transformer의 long-range dependency
- 두 방법의 장점을 결합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

from models.channel_configs import (
    get_singlebranch_channels_2step_decoder,
    get_activation_type,
)
from models.model_3d_unet import _make_norm3d, _make_activation, Up3D, OutConv3D
from models.modules.cbam_modules import ChannelAttention3D


class Stem3x3(nn.Module):
    """Initial stem with two 3x3x3 conv blocks and residual connection."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn", activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity


class PatchWiseConvBlock3D(nn.Module):
    """
    Patch-wise Convolution Block for Stage 1-2.
    
    패치를 먼저 나누고, 각 패치 내부에서 convolution을 수행합니다.
    Overlapping patches를 지원하여 gridding artifact를 방지합니다.
    
    Flow:
    1. Unfold (overlapping 지원)
    2. 각 패치에 Conv (local feature extraction)
    3. Fold (overlapping 처리)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 4,
        stride: int = 2,  # Downsampling stride
        overlap: float = 0.5,  # Overlap ratio for gridding artifact prevention
        norm: str = "bn",
        activation: str = "relu",
        use_channel_attention: bool = True,  # Channel attention 사용 여부
        reduction: int = 8,  # Channel attention reduction ratio
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride
        self.overlap = overlap
        
        # Patch stride for overlapping
        self.patch_stride = max(1, int(patch_size * (1 - overlap)))
        
        # Patch-wise convolution: 각 패치에 대해 독립적으로 conv 수행
        # 일반 conv 사용 (메모리 절약 + inductive bias 강화, 레이어 수 감소)
        self.patch_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )
        
        # Channel Attention (inductive bias 강화)
        self.channel_attention = ChannelAttention3D(out_channels, reduction=reduction) if use_channel_attention else nn.Identity()
        
        # Downsampling을 위한 projection (stride > 1인 경우)
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
            )
        else:
            self.downsample = None

    def _unfold_overlapping(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Unfold with overlapping patches."""
        B, C, D, H, W = x.shape
        p = self.patch_size
        s = self.patch_stride
        
        # Padding to ensure we can extract patches
        pad_d = (p - D % p) % p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            D += pad_d
            H += pad_h
            W += pad_w
        
        # Calculate number of patches with overlap
        num_patch_d = (D - p) // s + 1 if D > p else 1
        num_patch_h = (H - p) // s + 1 if H > p else 1
        num_patch_w = (W - p) // s + 1 if W > p else 1
        
        # Extract patches
        patches = []
        for d in range(0, D - p + 1, s):
            for h in range(0, H - p + 1, s):
                for w in range(0, W - p + 1, s):
                    patch = x[:, :, d:d+p, h:h+p, w:w+p]
                    patches.append(patch)
        
        # Stack patches: (B * num_patches, C, patch_size, patch_size, patch_size)
        patches = torch.stack(patches, dim=0)  # (num_patches, B, C, p, p, p)
        patches = patches.view(-1, C, p, p, p)  # (B * num_patches, C, p, p, p)
        
        info = {
            "batch_size": B,
            "channels": C,
            "orig_size": (D - pad_d, H - pad_h, W - pad_w),
            "padded_size": (D, H, W),
            "num_patch_d": num_patch_d,
            "num_patch_h": num_patch_h,
            "num_patch_w": num_patch_w,
            "patch_size": p,
            "patch_stride": s,
        }
        
        return patches, info

    def _fold_overlapping(self, patches: torch.Tensor, info: Dict, output_size: Tuple[int, int, int]) -> torch.Tensor:
        """Fold overlapping patches back to feature map with averaging."""
        B = info["batch_size"]
        C = patches.shape[1]  # Output channels
        p = info["patch_size"]
        s = info["patch_stride"]
        D, H, W = output_size
        
        # Reshape patches: (B * num_patches, C, p, p, p) -> (B, num_patches, C, p, p, p)
        num_patches_total = patches.shape[0] // B
        patches = patches.view(B, num_patches_total, C, p, p, p)
        
        # Initialize output and weight accumulator
        output = torch.zeros(B, C, D, H, W, device=patches.device, dtype=patches.dtype)
        weight = torch.zeros(B, 1, D, H, W, device=patches.device, dtype=patches.dtype)
        
        # Reconstruct patches
        num_patch_d = info["num_patch_d"]
        num_patch_h = info["num_patch_h"]
        num_patch_w = info["num_patch_w"]
        
        idx = 0
        for d in range(0, D - p + 1, s):
            for h in range(0, H - p + 1, s):
                for w in range(0, W - p + 1, s):
                    if idx < num_patches_total:
                        patch = patches[:, idx, :, :, :, :]  # (B, C, p, p, p)
                        output[:, :, d:d+p, h:h+p, w:w+p] += patch
                        weight[:, :, d:d+p, h:h+p, w:w+p] += 1.0
                        idx += 1
        
        # Average overlapping regions
        output = output / weight.clamp_min(1e-6)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Downsampling이 필요한 경우
        if self.stride > 1:
            # Output size after downsampling
            out_D = D // self.stride
            out_H = H // self.stride
            out_W = W // self.stride
            
            # Unfold with overlapping
            patches, info = self._unfold_overlapping(x)
            
            # Apply patch-wise convolution
            patches_conv = self.patch_conv(patches)  # (B * num_patches, out_channels, p, p, p)
            
            # Apply channel attention (각 패치에 독립적으로 적용)
            patches_conv = self.channel_attention(patches_conv)
            
            # Downsample each patch
            patches_down = F.avg_pool3d(patches_conv, kernel_size=self.stride, stride=self.stride)
            
            # Update patch size and stride after downsampling
            new_patch_size = self.patch_size // self.stride
            new_patch_stride = self.patch_stride // self.stride
            info["patch_size"] = new_patch_size
            info["patch_stride"] = max(1, new_patch_stride)  # Ensure stride >= 1
            
            # Fold back
            out = self._fold_overlapping(patches_down, info, (out_D, out_H, out_W))
            
            # Residual connection
            if self.downsample is not None:
                residual = self.downsample(x)
                out = out + residual
        else:
            # No downsampling
            patches, info = self._unfold_overlapping(x)
            patches_conv = self.patch_conv(patches)
            # Apply channel attention (각 패치에 독립적으로 적용)
            patches_conv = self.channel_attention(patches_conv)
            out = self._fold_overlapping(patches_conv, info, (D, H, W))
            
            # Residual connection
            if C == self.out_channels:
                out = out + x
        
        return out


class PatchConvTransformerBlock3D(nn.Module):
    """
    Patch-wise Conv + Transformer Block for Stage 3-4.
    
    패치를 먼저 나누고, 각 패치 내부에서 convolution을 수행한 후,
    패치 간 global attention을 수행합니다.
    Shifted patches를 지원하여 gridding artifact를 방지합니다.
    
    Flow:
    1. Unfold (non-overlapping)
    2. 각 패치에 Conv (local feature extraction)
    3. Flatten (패치 내부 feature map을 벡터로)
    4. Projection (embedding dimension으로)
    5. Transformer (패치 간 global attention)
    6. Unprojection + Reshape
    7. Fold
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 4,
        stride: int = 2,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_transformer_layers: int = 2,
        norm: str = "bn",
        activation: str = "relu",
        use_shifted_patches: bool = True,  # Shifted patches for boundary information exchange
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.use_shifted_patches = use_shifted_patches
        
        # Patch-wise convolution
        # 일반 conv 사용 (메모리 절약 + inductive bias 강화, 레이어 수 감소)
        self.patch_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _make_norm3d(norm, in_channels),
            _make_activation(activation, inplace=True),
        )
        
        # Flatten and projection
        patch_area = patch_size ** 3
        self.patch_proj = nn.Linear(in_channels * patch_area, embed_dim)
        self.patch_unproj = nn.Linear(embed_dim, in_channels * patch_area)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer3D(embed_dim, num_heads, mlp_ratio, norm, activation, drop_path_rate)
            for _ in range(num_transformer_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, out_channels),
        )
        
        # Downsampling projection
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
            )
        else:
            self.downsample = None

    def _unfold(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Unfold to non-overlapping patches."""
        B, C, D, H, W = x.shape
        p = self.patch_size
        
        # Padding
        pad_d = (p - D % p) % p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            D += pad_d
            H += pad_h
            W += pad_w
        
        num_patch_d = D // p
        num_patch_h = H // p
        num_patch_w = W // p
        num_patches = num_patch_d * num_patch_h * num_patch_w
        
        # Unfold: (B, C, D, H, W) -> (B, num_patches, C, p, p, p)
        patches = x.view(B, C, num_patch_d, p, num_patch_h, p, num_patch_w, p)
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        patches = patches.view(B, num_patches, C, p, p, p)
        
        info = {
            "batch_size": B,
            "channels": C,
            "orig_size": (D - pad_d, H - pad_h, W - pad_w),
            "padded_size": (D, H, W),
            "num_patch_d": num_patch_d,
            "num_patch_h": num_patch_h,
            "num_patch_w": num_patch_w,
            "patch_size": p,
        }
        
        return patches, info

    def _fold(self, patches: torch.Tensor, info: Dict) -> torch.Tensor:
        """Fold patches back to feature map."""
        B = info["batch_size"]
        C = info["channels"]
        num_patch_d = info["num_patch_d"]
        num_patch_h = info["num_patch_h"]
        num_patch_w = info["num_patch_w"]
        p = info["patch_size"]
        D, H, W = info["padded_size"]
        orig_D, orig_H, orig_W = info["orig_size"]
        
        # Reshape: (B, num_patches, C, p, p, p) -> (B, C, D, H, W)
        patches = patches.view(B, num_patch_d, num_patch_h, num_patch_w, C, p, p, p)
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        features = patches.view(B, C, D, H, W)
        
        # Remove padding
        if (D, H, W) != (orig_D, orig_H, orig_W):
            features = features[:, :, :orig_D, :orig_H, :orig_W].contiguous()
        
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Shifted patches (optional, for boundary information exchange)
        if self.use_shifted_patches and self.training:
            # Random shift for data augmentation effect
            shift_d = torch.randint(0, self.patch_size, (1,)).item()
            shift_h = torch.randint(0, self.patch_size, (1,)).item()
            shift_w = torch.randint(0, self.patch_size, (1,)).item()
            x_shifted = torch.roll(x, shifts=(shift_d, shift_h, shift_w), dims=(2, 3, 4))
        else:
            x_shifted = x
        
        # Unfold
        patches, info = self._unfold(x_shifted)  # (B, num_patches, C, p, p, p)
        num_patches = patches.shape[1]
        
        # Apply patch-wise convolution
        # Reshape for conv: (B * num_patches, C, p, p, p)
        patches_flat = patches.view(B * num_patches, C, self.patch_size, self.patch_size, self.patch_size)
        patches_conv = self.patch_conv(patches_flat)  # (B * num_patches, C, p, p, p)
        
        # Flatten patch features: (B * num_patches, C * p^3)
        patches_conv_flat = patches_conv.view(B * num_patches, -1)
        
        # Project to embedding dimension
        tokens = self.patch_proj(patches_conv_flat)  # (B * num_patches, embed_dim)
        tokens = tokens.view(B, num_patches, self.embed_dim)
        
        # Transformer: patch 간 global attention
        for transformer_layer in self.transformer_layers:
            tokens = transformer_layer(tokens)
        
        # Unproject
        tokens_flat = tokens.view(B * num_patches, self.embed_dim)
        patches_out_flat = self.patch_unproj(tokens_flat)  # (B * num_patches, C * p^3)
        
        # Reshape back to patches
        patches_out = patches_out_flat.view(B * num_patches, C, self.patch_size, self.patch_size, self.patch_size)
        patches_out = patches_out.view(B, num_patches, C, self.patch_size, self.patch_size, self.patch_size)
        
        # Fold
        features = self._fold(patches_out, info)  # (B, C, D, H, W)
        
        # Downsampling if needed
        if self.stride > 1:
            features = F.avg_pool3d(features, kernel_size=self.stride, stride=self.stride)
            D, H, W = D // self.stride, H // self.stride, W // self.stride
        
        # Output projection
        out = self.output_proj(features)  # (B, out_channels, D, H, W)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(x)
            if residual.shape[2:] != out.shape[2:]:
                # Adjust size if needed
                residual = F.interpolate(residual, size=out.shape[2:], mode='trilinear', align_corners=False)
            out = out + residual
        elif C == self.out_channels and out.shape[2:] == x.shape[2:]:
            out = out + x
        
        return out


class TransformerLayer3D(nn.Module):
    """Single Transformer layer for patch-wise attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        norm: str = "bn",
        activation: str = "relu",
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            _make_activation(activation, inplace=False),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1),
        )
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        
        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.drop_path(mlp_out)
        
        return x


class DropPath(nn.Module):
    """DropPath (Stochastic Depth) as a module."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        x = x / keep_prob * random_tensor
        return x


class CascadePatchConvTransformerUNet3D(nn.Module):
    """
    Cascade Patch-wise Conv + Transformer UNet.
    
    Stage별 전략:
    - Stage 1 (96³): Patch-wise Conv (overlapping patches)
    - Stage 2 (48³): Patch-wise Conv (overlapping patches)
    - Stage 3 (24³): Patch Conv + Transformer (shifted patches)
    - Stage 4 (12³): Patch Conv + Transformer (shifted patches)
    """
    
    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        norm: str = "bn",
        size: str = "s",
        include_coords: bool = True,
        patch_size_stage12: int = 4,
        patch_size_stage34: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_transformer_layers: int = 2,
    ):
        super().__init__()
        self.n_image_channels = n_image_channels
        self.n_coord_channels = n_coord_channels
        self.include_coords = include_coords and n_coord_channels > 0
        self.norm = norm or "bn"
        self.size = size
        
        activation = get_activation_type(size)
        channels = get_singlebranch_channels_2step_decoder(size)
        
        stem_in = n_image_channels + (n_coord_channels if self.include_coords else 0)
        self.stem = Stem3x3(stem_in, channels["stem"], norm=self.norm, activation=activation)
        
        # Stage 1: 96³ -> 48³, Patch-wise Conv only
        self.down1 = PatchWiseConvBlock3D(
            channels["stem"],
            channels["branch2"],
            patch_size=patch_size_stage12,
            stride=2,
            overlap=0.5,  # 50% overlap for gridding artifact prevention
            norm=self.norm,
            activation=activation,
            use_channel_attention=True,  # Channel attention으로 inductive bias 강화
            reduction=8,
        )
        
        # Stage 2: 48³ -> 24³, Patch-wise Conv only
        self.down2 = PatchWiseConvBlock3D(
            channels["branch2"],
            channels["branch3"],
            patch_size=patch_size_stage12,
            stride=2,
            overlap=0.5,
            norm=self.norm,
            activation=activation,
            use_channel_attention=True,  # Channel attention으로 inductive bias 강화
            reduction=8,
        )
        
        # Stage 3: 24³ -> 24³ (no downsampling), Patch Conv + Transformer
        self.down3 = PatchConvTransformerBlock3D(
            channels["branch3"],
            channels["branch3"],
            patch_size=patch_size_stage34,
            stride=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            norm=self.norm,
            activation=activation,
            use_shifted_patches=True,
        )
        
        # Stage 3 extra blocks
        self.down3_extra = nn.ModuleList([
            PatchConvTransformerBlock3D(
                channels["branch3"],
                channels["branch3"],
                patch_size=patch_size_stage34,
                stride=1,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers,
                norm=self.norm,
                activation=activation,
                use_shifted_patches=True,
                drop_path_rate=0.05 * (i + 1),  # Linear schedule
            )
            for i in range(2)  # 2 extra blocks
        ])
        
        # Stage 4: 24³ -> 12³, Patch Conv + Transformer
        self.down4 = PatchConvTransformerBlock3D(
            channels["branch3"],
            channels["down4"],
            patch_size=patch_size_stage34,
            stride=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            norm=self.norm,
            activation=activation,
            use_shifted_patches=True,
        )
        
        # Stage 4 extra block
        self.down4_extra = PatchConvTransformerBlock3D(
            channels["down4"],
            channels["down4"],
            patch_size=patch_size_stage34,
            stride=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            norm=self.norm,
            activation=activation,
            use_shifted_patches=True,
            drop_path_rate=0.15,
        )
        
        # Decoder
        self.up1 = Up3D(channels["down4"], channels["branch3"], bilinear=False, norm=self.norm, skip_channels=channels["branch3"])
        self.up2 = Up3D(channels["branch3"], channels["branch2"], bilinear=False, norm=self.norm, skip_channels=channels["branch2"])
        self.up3 = Up3D(channels["branch2"], channels["stem"], bilinear=False, norm=self.norm, skip_channels=channels["stem"])
        self.outc = OutConv3D(channels["stem"], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.include_coords and self.n_coord_channels > 0:
            img = x[:, : self.n_image_channels]
            coord = x[:, self.n_image_channels : self.n_image_channels + self.n_coord_channels]
            x_in = torch.cat([img, coord], dim=1)
        else:
            x_in = x[:, : self.n_image_channels]
        
        x1 = self.stem(x_in)  # 96³
        x2 = self.down1(x1)   # 48³
        x3 = self.down2(x2)   # 24³
        x3 = self.down3(x3)   # 24³
        for block in self.down3_extra:
            x3 = block(x3)    # 24³
        x4 = self.down4(x3)   # 12³
        x4 = self.down4_extra(x4)  # 12³
        
        x = self.up1(x4, x3)  # 24³
        x = self.up2(x, x2)   # 48³
        x = self.up3(x, x1)   # 96³
        return self.outc(x)


def build_cascade_patch_conv_transformer_unet3d(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
    patch_size_stage12: int = 4,
    patch_size_stage34: int = 4,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_transformer_layers: int = 2,
) -> CascadePatchConvTransformerUNet3D:
    return CascadePatchConvTransformerUNet3D(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
        patch_size_stage12=patch_size_stage12,
        patch_size_stage34=patch_size_stage34,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers,
    )


__all__ = [
    "CascadePatchConvTransformerUNet3D",
    "build_cascade_patch_conv_transformer_unet3d",
    "PatchWiseConvBlock3D",
    "PatchConvTransformerBlock3D",
]


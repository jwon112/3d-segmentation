"""
MobileViT Modules
MobileViT 스타일의 3D Vision Transformer 모듈들 (HuggingFace 구현을 3D에 맞게 변형)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_3d_unet import _make_norm3d
from .replk_modules import Transition3D


class MobileViTTransformerLayer3D(nn.Module):
    """Single Transformer encoder layer used inside MobileViT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(attn_dropout)

        hidden_dim = dim * mlp_ratio
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        # Self-Attention block
        attn_input = self.norm1(tokens)
        attn_out, attn_weights = self.attn(attn_input, attn_input, attn_input, need_weights=return_attn)
        tokens = tokens + self.attn_dropout(attn_out)

        # Feed-forward block
        ffn_input = self.norm2(tokens)
        tokens = tokens + self.ffn_dropout(self.ffn(ffn_input))
        if return_attn:
            return tokens, attn_weights
        return tokens


class MobileViTTransformer3D(nn.Module):
    """Multi-layer Transformer stack for MobileViT."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MobileViTTransformerLayer3D(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        all_attn_weights = []
        for layer in self.layers:
            if return_attn:
                tokens, attn_weights = layer(tokens, return_attn=True)
                all_attn_weights.append(attn_weights)
            else:
                tokens = layer(tokens, return_attn=False)
        if return_attn:
            return tokens, all_attn_weights
        return tokens


class MobileViT3DBlock(nn.Module):
    """3D MobileViT block based on HF implementation (conv -> unfold -> multi-layer transformer -> fold)."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = None,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        norm: str = 'bn',
        patch_size: int = 2,
        num_transformer_layers: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim or channels
        self.patch_size = patch_size

        self.conv_kxk = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(channels, self.hidden_dim, kernel_size=1, bias=False),
            _make_norm3d(norm, self.hidden_dim),
        )

        self.transformer = MobileViTTransformer3D(
            dim=self.hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )

        self.conv_projection = nn.Sequential(
            nn.Conv3d(self.hidden_dim, channels, kernel_size=1, bias=False),
            _make_norm3d(norm, channels),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        residual = x
        features = self.conv_kxk(x)
        features = self.conv_1x1(features)

        patches, info = self._unfold(features)
        if return_attn:
            patches, attn_weights = self.transformer(patches, return_attn=True)
        else:
            patches = self.transformer(patches, return_attn=False)
        features = self._fold(patches, info)

        features = self.conv_projection(features)
        features = self.fusion(torch.cat((residual, features), dim=1))
        if return_attn:
            return features, attn_weights
        return features

    def _unfold(self, features: torch.Tensor):
        """Convert (B, C, D, H, W) to tokens (B*patch_area, num_patches, C)."""
        B, C, D, H, W = features.shape
        p = self.patch_size

        pad_d = (p - D % p) % p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_d or pad_h or pad_w:
            features = F.pad(features, (0, pad_w, 0, pad_h, 0, pad_d))
            D += pad_d
            H += pad_h
            W += pad_w

        num_patch_d = D // p
        num_patch_h = H // p
        num_patch_w = W // p
        num_patches = num_patch_d * num_patch_h * num_patch_w
        patch_area = p ** 3

        patches = features.view(B, C, num_patch_d, p, num_patch_h, p, num_patch_w, p)
        patches = patches.permute(0, 3, 5, 7, 2, 4, 6, 1).contiguous()
        patches = patches.view(B * patch_area, num_patches, C)

        info = {
            "batch_size": B,
            "channels": C,
            "orig_size": (D - pad_d, H - pad_h, W - pad_w),
            "padded_size": (D, H, W),
            "num_patch_d": num_patch_d,
            "num_patch_h": num_patch_h,
            "num_patch_w": num_patch_w,
            "patch_size": p,
            "patch_area": patch_area,
        }
        return patches, info

    def _fold(self, patches: torch.Tensor, info: dict) -> torch.Tensor:
        """Inverse of _unfold."""
        B = info["batch_size"]
        C = info["channels"]
        num_patch_d = info["num_patch_d"]
        num_patch_h = info["num_patch_h"]
        num_patch_w = info["num_patch_w"]
        p = info["patch_size"]
        patch_area = info["patch_area"]
        D, H, W = info["padded_size"]
        orig_D, orig_H, orig_W = info["orig_size"]

        num_patches = num_patch_d * num_patch_h * num_patch_w
        patches = patches.view(B, patch_area, num_patches, C)
        patches = patches.permute(0, 3, 2, 1).contiguous()
        patches = patches.view(B, C, num_patch_d, num_patch_h, num_patch_w, p, p, p)
        patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
        features = patches.view(B, C, D, H, W)

        if (D, H, W) != (orig_D, orig_H, orig_W):
            features = features[:, :, :orig_D, :orig_H, :orig_W]
        return features


class MobileViT3DBlockV3(nn.Module):
    """3D MobileViT v3 block with improved fusion mechanism.
    
    Key differences from v1:
    - Fusion uses 1x1 conv instead of 3x3 conv
    - Fuses local (conv_kxk output) and global (transformer output) features
    - Adds input feature as residual to fusion output
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = None,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        norm: str = 'bn',
        patch_size: int = 2,
        num_transformer_layers: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim or channels
        self.patch_size = patch_size

        # Local feature extraction
        self.conv_kxk = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
        )
        
        # Global feature extraction (Transformer path)
        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(channels, self.hidden_dim, kernel_size=1, bias=False),
            _make_norm3d(norm, self.hidden_dim),
        )

        self.transformer = MobileViTTransformer3D(
            dim=self.hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )

        self.conv_projection = nn.Sequential(
            nn.Conv3d(self.hidden_dim, channels, kernel_size=1, bias=False),
            _make_norm3d(norm, channels),
        )
        
        # V3: 1x1 conv fusion (instead of 3x3 in v1)
        self.fusion = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # Input feature (for residual)
        input_feat = x
        
        # Local feature
        local_feat = self.conv_kxk(x)
        
        # Global feature path
        global_input = self.conv_1x1(local_feat)
        patches, info = self._unfold(global_input)
        if return_attn:
            patches, attn_weights = self.transformer(patches, return_attn=True)
        else:
            patches = self.transformer(patches, return_attn=False)
        global_feat = self._fold(patches, info)
        global_feat = self.conv_projection(global_feat)
        
        # V3: Fuse local and global features (instead of input and global in v1)
        fused = self.fusion(torch.cat((local_feat, global_feat), dim=1))
        
        # V3: Add input feature as residual
        if return_attn:
            return fused + input_feat, attn_weights
        return fused + input_feat

    def _unfold(self, features: torch.Tensor):
        """Convert (B, C, D, H, W) to tokens (B*patch_area, num_patches, C)."""
        B, C, D, H, W = features.shape
        p = self.patch_size

        pad_d = (p - D % p) % p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_d or pad_h or pad_w:
            features = F.pad(features, (0, pad_w, 0, pad_h, 0, pad_d))
            D += pad_d
            H += pad_h
            W += pad_w

        num_patch_d = D // p
        num_patch_h = H // p
        num_patch_w = W // p
        num_patches = num_patch_d * num_patch_h * num_patch_w
        patch_area = p ** 3

        patches = features.view(B, C, num_patch_d, p, num_patch_h, p, num_patch_w, p)
        patches = patches.permute(0, 3, 5, 7, 2, 4, 6, 1).contiguous()
        patches = patches.view(B * patch_area, num_patches, C)

        info = {
            "batch_size": B,
            "channels": C,
            "orig_size": (D - pad_d, H - pad_h, W - pad_w),
            "padded_size": (D, H, W),
            "num_patch_d": num_patch_d,
            "num_patch_h": num_patch_h,
            "num_patch_w": num_patch_w,
            "patch_size": p,
            "patch_area": patch_area,
        }
        return patches, info

    def _fold(self, patches: torch.Tensor, info: dict) -> torch.Tensor:
        """Inverse of _unfold."""
        B = info["batch_size"]
        C = info["channels"]
        num_patch_d = info["num_patch_d"]
        num_patch_h = info["num_patch_h"]
        num_patch_w = info["num_patch_w"]
        p = info["patch_size"]
        patch_area = info["patch_area"]
        D, H, W = info["padded_size"]
        orig_D, orig_H, orig_W = info["orig_size"]

        num_patches = num_patch_d * num_patch_h * num_patch_w
        patches = patches.view(B, patch_area, num_patches, C)
        patches = patches.permute(0, 3, 2, 1).contiguous()
        patches = patches.view(B, C, num_patch_d, num_patch_h, num_patch_w, p, p, p)
        patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
        features = patches.view(B, C, D, H, W)

        if (D, H, W) != (orig_D, orig_H, orig_W):
            features = features[:, :, :orig_D, :orig_H, :orig_W]
        return features


class Down3DStrideMViT(nn.Module):
    """Downsample via Transition3D then apply MobileViT3DBlock."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn',
                 num_heads: int = 4, mlp_ratio: int = 2, patch_size: int = 2,
                 num_transformer_layers: int = 2, attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.0):
        super().__init__()
        self.trans = Transition3D(in_channels, out_channels, norm=norm)
        self.mvit = MobileViT3DBlock(out_channels, hidden_dim=out_channels,
                                     num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     norm=norm, patch_size=patch_size,
                                     num_transformer_layers=num_transformer_layers,
                                     attn_dropout=attn_dropout,
                                     ffn_dropout=ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trans(x)
        x = self.mvit(x)
        return x


class Down3DStrideMViTV3(nn.Module):
    """Downsample via Transition3D then apply MobileViT3DBlockV3."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn',
                 num_heads: int = 4, mlp_ratio: int = 2, patch_size: int = 2,
                 num_transformer_layers: int = 2, attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.0):
        super().__init__()
        self.trans = Transition3D(in_channels, out_channels, norm=norm)
        self.mvit = MobileViT3DBlockV3(out_channels, hidden_dim=out_channels,
                                       num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       norm=norm, patch_size=patch_size,
                                       num_transformer_layers=num_transformer_layers,
                                       attn_dropout=attn_dropout,
                                       ffn_dropout=ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trans(x)
        x = self.mvit(x)
        return x


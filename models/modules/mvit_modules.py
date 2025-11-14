"""
MobileViT Modules
MobileViT 스타일의 3D Vision Transformer 모듈들
"""

import torch
import torch.nn as nn

from ..model_3d_unet import _make_norm3d
from .replk_modules import Transition3D


class MobileViT3DBlock(nn.Module):
    """A lightweight 3D MobileViT-like block.

    conv3d (local) -> tokens (global) with MHSA -> MLP -> fuse back -> residual.
    """
    def __init__(self, channels: int, embed_dim: int = None, num_heads: int = 4, mlp_ratio: int = 2, norm: str = 'bn'):
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim or channels
        self.local = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm or 'bn', channels),
            nn.ReLU(inplace=True),
        )
        # projection to transformer dim
        self.proj_in = nn.Conv3d(channels, self.embed_dim, kernel_size=1, bias=True)
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * mlp_ratio), nn.GELU(), nn.Linear(self.embed_dim * mlp_ratio, self.embed_dim)
        )
        self.proj_out = nn.Conv3d(self.embed_dim, channels, kernel_size=1, bias=True)
        self.out_bn = _make_norm3d(norm or 'bn', channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # local conv
        y = self.local(x)
        b, c, d, h, w = y.shape
        # to tokens: (B, N, C)
        t = self.proj_in(y)  # (B, E, D, H, W)
        e = t.size(1)
        t = t.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, e)
        # transformer
        t = self.attn_norm(t)
        attn_out, _ = self.attn(t, t, t)
        t = t + attn_out
        t = t + self.ffn(self.attn_norm(t))
        # back to 3D
        t = t.view(b, d, h, w, e).permute(0, 4, 1, 2, 3).contiguous()
        t = self.proj_out(t)
        out = self.out_bn(t + x)
        return out


class Down3DStrideMViT(nn.Module):
    """Downsample then MobileViT3D block."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        # Use RepLKNet-style Transition for downsample for stability
        self.trans = Transition3D(in_channels, out_channels, norm=norm)
        self.mvit = MobileViT3DBlock(out_channels, embed_dim=out_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trans(x)
        x = self.mvit(x)
        return x


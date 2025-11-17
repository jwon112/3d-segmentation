"""
ShuffleNetV2 Hybrid Modules
ShuffleNetV2 + Transformer 하이브리드 블록

핵심 아이디어 (도식 기준):
- Stem: 3x3 depthwise stride s (s∈{1,2}) → 1x1 pointwise conv
- Branch Conv: 3x3 depthwise → 1x1 conv (local representation)
- Branch Transformer: 3x3 depthwise → 1x1 conv (expansion ratio 적용) → Transformer → 1x1 conv (reduction)
- 두 branch 출력 concat → Channel shuffle → 1x1 conv (fuse)
- Residual 연결 (stride≠1 또는 채널 mismatch일 때는 1x1 conv로 정렬)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_3d_unet import _make_norm3d
from .shufflenet_modules import channel_shuffle_3d


class ShuffleNetV2HybridUnit3D(nn.Module):
    """3D ShuffleNetV2 Hybrid Unit (Conv + Transformer).
    
    Stem → (Conv branch, Transformer branch) → concat → shuffle → fuse → residual
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: str = 'bn', expand_ratio: float = 4.0,
                 num_heads: int = 4, mlp_ratio: int = 2, patch_size: int = 4):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        mid_channels = out_channels // 2
        expanded_channels = max(int(mid_channels * expand_ratio), mid_channels)
        self.expanded_channels = expanded_channels
        self.patch_size = patch_size

        # Stem: 3x3 depthwise (stride) + 1x1 conv
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            _make_norm3d(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

        # Conv branch (local)
        self.conv_branch = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                      groups=out_channels, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, mid_channels),
            nn.ReLU(inplace=True),
        )

        # Transformer branch (MobileViT 스타일)
        self.transformer_local = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                      groups=out_channels, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, expanded_channels),
            nn.ReLU(inplace=True),
        )
        self.transformer_norm = nn.LayerNorm(expanded_channels)
        self.transformer_attn = nn.MultiheadAttention(expanded_channels, num_heads=num_heads, batch_first=True)
        self.transformer_ffn = nn.Sequential(
            nn.Linear(expanded_channels, expanded_channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(expanded_channels * mlp_ratio, expanded_channels),
        )
        self.transformer_reduce = nn.Sequential(
            nn.Conv3d(expanded_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, mid_channels),
            nn.ReLU(inplace=True),
        )
        patch_dim = (patch_size ** 3) * expanded_channels
        self.patch_embed = nn.Linear(patch_dim, expanded_channels)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, expanded_channels),
            nn.GELU(),
            nn.Linear(expanded_channels, expanded_channels),
        )

        # Fusion after concat + shuffle
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, out_channels),
        )

        # Residual path
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_out = self.stem(x)

        # Conv branch
        conv_feat = self.conv_branch(stem_out)

        # Transformer branch
        trans_local = self.transformer_local(stem_out)
        b, c, d, h, w = trans_local.shape
        orig_d, orig_h, orig_w = d, h, w
        p = self.patch_size
        d_pad = (p - d % p) % p
        h_pad = (p - h % p) % p
        w_pad = (p - w % p) % p
        if d_pad or h_pad or w_pad:
            trans_local = F.pad(trans_local, (0, w_pad, 0, h_pad, 0, d_pad))
            d += d_pad
            h += h_pad
            w += w_pad
        Dz, Hy, Wx = d // p, h // p, w // p
        trans_local = trans_local.view(b, c, Dz, p, Hy, p, Wx, p)
        trans_local = trans_local.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        tokens = trans_local.view(b, Dz * Hy * Wx, -1)
        tokens = self.patch_embed(tokens)
        pos = self._positional_encoding(Dz, Hy, Wx, tokens.device, tokens.dtype)
        tokens = tokens + pos.unsqueeze(0)
        tokens_norm = self.transformer_norm(tokens)
        attn_out, _ = self.transformer_attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out
        tokens = tokens + self.transformer_ffn(self.transformer_norm(tokens))
        tokens = tokens.view(b, Dz, Hy, Wx, p, p, p, self.expanded_channels)
        tokens = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(
            b, self.expanded_channels, d, h, w
        )
        if d_pad or h_pad or w_pad:
            tokens = tokens[:, :, :orig_d, :orig_h, :orig_w]
        trans_feat = self.transformer_reduce(tokens)

        # Concat -> shuffle -> fuse
        out = torch.cat([conv_feat, trans_feat], dim=1)
        out = channel_shuffle_3d(out, groups=2)
        out = self.fuse(out)

        # Residual add
        out = out + self.shortcut(x)
        return torch.relu(out)

    def _positional_encoding(self, Dz: int, Hy: int, Wx: int, device, dtype):
        z = torch.linspace(-1.0, 1.0, Dz, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, Hy, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, Wx, device=device, dtype=dtype)
        grid = torch.stack(torch.meshgrid(z, y, x, indexing='ij'), dim=-1)  # (Dz,Hy,Wx,3)
        pos = grid.view(-1, 3)
        return self.pos_mlp(pos)


class Down3DShuffleNetV2Hybrid(nn.Module):
    """Downsampling using ShuffleNetV2 Hybrid unit (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn',
                 expand_ratio: float = 4.0, num_heads: int = 4, mlp_ratio: int = 2,
                 patch_size: int = 4):
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = ShuffleNetV2HybridUnit3D(in_channels, out_channels, stride=2,
                                              norm=norm, expand_ratio=expand_ratio,
                                              num_heads=num_heads, mlp_ratio=mlp_ratio,
                                              patch_size=patch_size)
        # Second unit: stride=1 for feature refinement
        self.unit2 = ShuffleNetV2HybridUnit3D(out_channels, out_channels, stride=1,
                                              norm=norm, expand_ratio=expand_ratio,
                                              num_heads=num_heads, mlp_ratio=mlp_ratio,
                                              patch_size=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


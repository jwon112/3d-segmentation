"""
ShuffleNetV2 Hybrid Modules
ShuffleNetV2 + Transformer 하이브리드 블록

핵심 아이디어 (업데이트된 구조):
- 입력을 먼저 channel-wise split (ShuffleNetV2 스타일)
- Conv Branch: 3x3 depthwise(stride s) → 1x1 conv → GELU
- Transformer Branch: (기존 Stem) 3x3 depthwise(stride s) → 1x1 conv → Transformer 경로
- 두 branch 출력 concat → Channel shuffle → 1x1 conv (fuse)
- Residual 연결 (stride≠1 또는 채널 mismatch일 때는 1x1 conv로 정렬)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from ..model_3d_unet import _make_norm3d
from .shufflenet_modules import channel_shuffle_3d
from .mvit_modules import MobileViT3DBlock


def _make_layernorm3d(channels: int) -> nn.Module:
    """Channel-first LayerNorm implemented via GroupNorm(Group=1)."""
    return nn.GroupNorm(1, channels)


class ShuffleNetV2HybridUnit3D(nn.Module):
    """3D ShuffleNetV2 Hybrid Unit (Conv + Transformer).
    
    Input split → (Conv branch, Transformer branch) → concat → shuffle → fuse → residual
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: str = 'bn', expand_ratio: float = 4.0,
                 num_heads: int = 4, mlp_ratio: int = 2, patch_size: int = 4,
                 num_transformer_layers: int = 2,
                 attn_dropout: float = 0.0, ffn_dropout: float = 0.0,
                 force_layernorm: bool = False, log_stats: bool = False,
                 stats_dict: Optional[Dict[str, list]] = None, stats_prefix: str = ''):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        self.force_layernorm = force_layernorm
        self.norm_type = norm or 'bn'
        self.log_stats = log_stats
        self.stats_dict = stats_dict
        self.stats_prefix = stats_prefix
        norm_fn = (lambda c: _make_layernorm3d(c)) if self.force_layernorm else (lambda c: _make_norm3d(self.norm_type, c))
        mid_channels = out_channels // 2
        if mid_channels == 0:
            raise ValueError("out_channels must be >= 2 for ShuffleNetV2HybridUnit3D")
        expanded_channels = max(int(mid_channels * expand_ratio), mid_channels)
        self.expanded_channels = expanded_channels
        self.conv_in_channels = in_channels // 2
        self.trans_in_channels = in_channels - self.conv_in_channels
        if self.conv_in_channels == 0 or self.trans_in_channels == 0:
            raise ValueError("in_channels must be >= 2 to split channels for hybrid unit.")

        # Conv branch (local)
        self.conv_branch = nn.Sequential(
            nn.Conv3d(self.conv_in_channels, self.conv_in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=self.conv_in_channels, bias=False),
            _make_layernorm3d(self.conv_in_channels),
            nn.GELU(),
            nn.Conv3d(self.conv_in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_layernorm3d(mid_channels),
            nn.GELU(),
        )

        # Transformer branch (MobileViT 스타일)
        if stride == 1 and self.trans_in_channels == mid_channels:
            self.transformer_align = nn.Identity()
        else:
            kernel_size = 3 if stride > 1 else 1
            padding = 1 if stride > 1 else 0
            self.transformer_align = nn.Sequential(
                nn.Conv3d(self.trans_in_channels, mid_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False),
                norm_fn(mid_channels),
                nn.ReLU(inplace=True),
            )
        self.transformer_mvit = MobileViT3DBlock(
            channels=mid_channels,
            hidden_dim=expanded_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm=self.norm_type,
            patch_size=patch_size,
            num_transformer_layers=num_transformer_layers,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )

        # Fusion after concat + shuffle
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            norm_fn(out_channels),
        )

        # Residual path
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                norm_fn(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv, x_trans = torch.split(x, [self.conv_in_channels, self.trans_in_channels], dim=1)

        # Conv branch
        conv_feat = self.conv_branch(x_conv)

        # Transformer branch
        trans_feat_input = self.transformer_align(x_trans)
        trans_feat = self.transformer_mvit(trans_feat_input)
        self._record_stat('trans_feat', trans_feat)

        # Concat -> shuffle -> fuse
        out = torch.cat([conv_feat, trans_feat], dim=1)
        out = channel_shuffle_3d(out, groups=2)
        out = self.fuse(out)

        # Residual add
        out = out + self.shortcut(x)
        return torch.relu(out)

    def _record_stat(self, name: str, tensor: torch.Tensor) -> None:
        if not self.log_stats or self.stats_dict is None:
            return
        key_mean = f"{self.stats_prefix}{name}/mean"
        key_std = f"{self.stats_prefix}{name}/std"
        det = tensor.detach()
        mean_val = det.float().mean().item()
        std_val = det.float().std(unbiased=False).item()
        self.stats_dict.setdefault(key_mean, []).append(mean_val)
        self.stats_dict.setdefault(key_std, []).append(std_val)

class Down3DShuffleNetV2Hybrid(nn.Module):
    """Downsampling using ShuffleNetV2 Hybrid unit (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn',
                 expand_ratio: float = 4.0, num_heads: int = 4, mlp_ratio: int = 2,
                 patch_size: int = 4, num_transformer_layers: int = 2,
                 attn_dropout: float = 0.0, ffn_dropout: float = 0.0,
                 force_layernorm: bool = False,
                 log_stats: bool = False, stats_dict: Optional[Dict[str, list]] = None,
                 stats_prefix: str = ''):
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = ShuffleNetV2HybridUnit3D(in_channels, out_channels, stride=2,
                                              norm=norm, expand_ratio=expand_ratio,
                                              num_heads=num_heads, mlp_ratio=mlp_ratio,
                                              patch_size=patch_size,
                                              num_transformer_layers=num_transformer_layers,
                                              attn_dropout=attn_dropout,
                                              ffn_dropout=ffn_dropout,
                                              force_layernorm=force_layernorm,
                                              log_stats=log_stats, stats_dict=stats_dict,
                                              stats_prefix=f"{stats_prefix}unit1/")
        # Second unit: stride=1 for feature refinement
        self.unit2 = ShuffleNetV2HybridUnit3D(out_channels, out_channels, stride=1,
                                              norm=norm, expand_ratio=expand_ratio,
                                              num_heads=num_heads, mlp_ratio=mlp_ratio,
                                              patch_size=patch_size,
                                              num_transformer_layers=num_transformer_layers,
                                              attn_dropout=attn_dropout,
                                              ffn_dropout=ffn_dropout,
                                              force_layernorm=force_layernorm,
                                              log_stats=log_stats, stats_dict=stats_dict,
                                              stats_prefix=f"{stats_prefix}unit2/")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


class ShuffleNetV2HybridUnit3D_AllLN(ShuffleNetV2HybridUnit3D):
    """Convenience class forcing LayerNorm/GroupNorm(1) everywhere."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('force_layernorm', True)
        super().__init__(*args, **kwargs)


class Down3DShuffleNetV2Hybrid_AllLN(Down3DShuffleNetV2Hybrid):
    """Down block variant that forces LayerNorm in every normalization site."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('force_layernorm', True)
        super().__init__(*args, **kwargs)


"""
Quad-Branch UNet variants with Channel/Spatial Attention for ablation study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import DoubleConv3D, Down3D, Up3D, OutConv3D, _make_norm3d
from .channel_configs import get_quadbranch_channels
from .modules.cbam_modules import ChannelAttention3D, SpatialAttention3D


class QuadBranchAttentionBase(nn.Module):
    """Backbone that keeps stage 1-4 modality-specific branches, stage 5 fused."""

    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 4, "QuadBranchAttentionBase expects 4 input channels (T1, T1CE, T2, FLAIR)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear
        self.size = size

        channels = get_quadbranch_channels(size)

        # Helper dimensions for subclasses
        self.stage1_channels = channels['stem'] * 4
        self.stage2_channels = channels['branch2'] * 4
        self.stage3_channels = channels['branch3'] * 4
        self.stage4_channels = channels['branch4'] * 4
        self.fused_channels = self.stage4_channels
        self.factor = 2 if self.bilinear else 1
        self.bottleneck_channels = channels['down5'] // self.factor
        self.output_channels = channels['stem'] * 4 // self.factor

        # Stage 1: stems
        self.stem_t1 = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t2 = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)

        # Stage 2-4: modality-specific branches
        self.branch_t1_2 = Down3D(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce_2 = Down3D(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t2_2 = Down3D(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_flair_2 = Down3D(channels['stem'], channels['branch2'], norm=self.norm)

        self.branch_t1_3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce_3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t2_3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_flair_3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)

        self.branch_t1_4 = Down3D(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce_4 = Down3D(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t2_4 = Down3D(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_flair_4 = Down3D(channels['branch3'], channels['branch4'], norm=self.norm)

        # Stage 5: fused branch
        self.down5 = Down3D(self.fused_channels, self.bottleneck_channels, norm=self.norm)

        # Decoder (4 stages)
        if self.bilinear:
            self.up1 = Up3D(self.bottleneck_channels, self.fused_channels // self.factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(self.fused_channels, self.stage3_channels // self.factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(self.stage3_channels, self.stage2_channels // self.factor, self.bilinear, norm=self.norm)
            self.up4 = Up3D(self.stage2_channels, self.stage1_channels // self.factor, self.bilinear, norm=self.norm)
        else:
            self.up1 = Up3D(self.bottleneck_channels, self.fused_channels // self.factor, self.bilinear, norm=self.norm, skip_channels=self.fused_channels)
            self.up2 = Up3D(self.fused_channels, self.stage3_channels // self.factor, self.bilinear, norm=self.norm, skip_channels=self.stage3_channels)
            self.up3 = Up3D(self.stage3_channels, self.stage2_channels // self.factor, self.bilinear, norm=self.norm, skip_channels=self.stage2_channels)
            self.up4 = Up3D(self.stage2_channels, self.stage1_channels // self.factor, self.bilinear, norm=self.norm, skip_channels=self.stage1_channels)

        self.outc = OutConv3D(self.output_channels, self.n_classes)

    # Hooks for subclasses to override -------------------------------------------------
    def _apply_encoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        return x

    def _apply_decoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        return x

    # ----------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x1_t1 = self.stem_t1(x[:, 0:1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1_t2 = self.stem_t2(x[:, 2:3])
        x1_flair = self.stem_flair(x[:, 3:4])
        x1 = torch.cat([x1_t1, x1_t1ce, x1_t2, x1_flair], dim=1)

        # Stage 2
        b_t1_2 = self.branch_t1_2(x1_t1)
        b_t1ce_2 = self.branch_t1ce_2(x1_t1ce)
        b_t2_2 = self.branch_t2_2(x1_t2)
        b_flair_2 = self.branch_flair_2(x1_flair)
        x2 = torch.cat([b_t1_2, b_t1ce_2, b_t2_2, b_flair_2], dim=1)
        x2 = self._apply_encoder_attention(2, x2)

        # Stage 3
        b_t1_3 = self.branch_t1_3(b_t1_2)
        b_t1ce_3 = self.branch_t1ce_3(b_t1ce_2)
        b_t2_3 = self.branch_t2_3(b_t2_2)
        b_flair_3 = self.branch_flair_3(b_flair_2)
        x3 = torch.cat([b_t1_3, b_t1ce_3, b_t2_3, b_flair_3], dim=1)
        x3 = self._apply_encoder_attention(3, x3)

        # Stage 4
        b_t1_4 = self.branch_t1_4(b_t1_3)
        b_t1ce_4 = self.branch_t1ce_4(b_t1ce_3)
        b_t2_4 = self.branch_t2_4(b_t2_3)
        b_flair_4 = self.branch_flair_4(b_flair_3)
        x4 = torch.cat([b_t1_4, b_t1ce_4, b_t2_4, b_flair_4], dim=1)
        x4 = self._apply_encoder_attention(4, x4)

        # Stage 5
        x5 = self.down5(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self._apply_decoder_attention(1, x)
        x = self.up2(x, x3)
        x = self._apply_decoder_attention(2, x)
        x = self.up3(x, x2)
        x = self._apply_decoder_attention(3, x)
        x = self.up4(x, x1)
        x = self._apply_decoder_attention(4, x)

        return self.outc(x)


class QuadBranchUNet3D_Channel_Centralized_Concat(QuadBranchAttentionBase):
    """Channel attention applied only after decoder double conv blocks."""

    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear: bool = False, size: str = 's', reduction: int = 8):
        super().__init__(n_channels, n_classes, norm, bilinear, size)
        self.dec_attn = nn.ModuleList([
            ChannelAttention3D(self.fused_channels // self.factor, reduction),
            ChannelAttention3D(self.stage3_channels // self.factor, reduction),
            ChannelAttention3D(self.stage2_channels // self.factor, reduction),
            ChannelAttention3D(self.stage1_channels // self.factor, reduction),
        ])

    def _apply_decoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.dec_attn[stage_idx - 1](x)


class QuadBranchUNet3D_Channel_Distributed_Concat(QuadBranchAttentionBase):
    """Channel attention at encoder concatenations + decoder."""

    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear: bool = False, size: str = 's', reduction: int = 16):
        super().__init__(n_channels, n_classes, norm, bilinear, size)
        self.enc_attn = nn.ModuleDict({
            '2': ChannelAttention3D(self.stage2_channels, reduction),
            '3': ChannelAttention3D(self.stage3_channels, reduction),
            '4': ChannelAttention3D(self.stage4_channels, reduction),
        })
        self.dec_attn = nn.ModuleList([
            ChannelAttention3D(self.fused_channels // self.factor, reduction),
            ChannelAttention3D(self.stage3_channels // self.factor, reduction),
            ChannelAttention3D(self.stage2_channels // self.factor, reduction),
            ChannelAttention3D(self.stage1_channels // self.factor, reduction),
        ])

    def _apply_encoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        attn = self.enc_attn.get(str(stage_idx))
        return attn(x) if attn is not None else x

    def _apply_decoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.dec_attn[stage_idx - 1](x)


class QuadBranchUNet3D_Channel_Distributed_Conv(QuadBranchAttentionBase):
    """Channel attention with pointwise conv before attention (encoder & decoder)."""

    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear: bool = False, size: str = 's', reduction: int = 16):
        super().__init__(n_channels, n_classes, norm, bilinear, size)

        def pw_conv(c):
            return nn.Sequential(
                nn.Conv3d(c, c, kernel_size=1, bias=False),
                _make_norm3d(self.norm, c),
                nn.ReLU(inplace=True),
            )

        self.enc_convs = nn.ModuleDict({
            '2': pw_conv(self.stage2_channels),
            '3': pw_conv(self.stage3_channels),
            '4': pw_conv(self.stage4_channels),
        })
        self.enc_attn = nn.ModuleDict({
            '2': ChannelAttention3D(self.stage2_channels, reduction),
            '3': ChannelAttention3D(self.stage3_channels, reduction),
            '4': ChannelAttention3D(self.stage4_channels, reduction),
        })

        self.dec_convs = nn.ModuleDict({
            '1': pw_conv(self.fused_channels // self.factor),
            '2': pw_conv(self.stage3_channels // self.factor),
            '3': pw_conv(self.stage2_channels // self.factor),
            '4': pw_conv(self.stage1_channels // self.factor),
        })
        self.dec_attn = nn.ModuleList([
            ChannelAttention3D(self.fused_channels // self.factor, reduction),
            ChannelAttention3D(self.stage3_channels // self.factor, reduction),
            ChannelAttention3D(self.stage2_channels // self.factor, reduction),
            ChannelAttention3D(self.stage1_channels // self.factor, reduction),
        ])

    def _apply_encoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        key = str(stage_idx)
        if key in self.enc_convs:
            x = self.enc_convs[key](x)
            x = self.enc_attn[key](x)
        return x

    def _apply_decoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        key = str(stage_idx)
        if key in self.dec_convs:
            x = self.dec_convs[key](x)
            x = self.dec_attn[stage_idx - 1](x)
        return x


class QuadBranchUNet3D_Spatial_Centralized_Concat(QuadBranchAttentionBase):
    """Spatial attention applied only at decoder stages."""

    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__(n_channels, n_classes, norm, bilinear, size)
        self.dec_attn = nn.ModuleList([SpatialAttention3D() for _ in range(4)])

    def _apply_decoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.dec_attn[stage_idx - 1](x)


class QuadBranchUNet3D_Spatial_Distributed_Concat(QuadBranchAttentionBase):
    """Spatial attention at encoder concatenations + decoder."""

    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__(n_channels, n_classes, norm, bilinear, size)
        self.enc_attn = nn.ModuleDict({
            '2': SpatialAttention3D(),
            '3': SpatialAttention3D(),
            '4': SpatialAttention3D(),
        })
        self.dec_attn = nn.ModuleList([SpatialAttention3D() for _ in range(4)])

    def _apply_encoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        attn = self.enc_attn.get(str(stage_idx))
        return attn(x) if attn is not None else x

    def _apply_decoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.dec_attn[stage_idx - 1](x)


class QuadBranchUNet3D_Spatial_Distributed_Conv(QuadBranchAttentionBase):
    """Spatial attention with depthwise conv before attention (encoder & decoder)."""

    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__(n_channels, n_classes, norm, bilinear, size)

        def dw_conv(c):
            return nn.Sequential(
                nn.Conv3d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
                _make_norm3d(self.norm, c),
                nn.ReLU(inplace=True),
            )

        self.enc_convs = nn.ModuleDict({
            '2': dw_conv(self.stage2_channels),
            '3': dw_conv(self.stage3_channels),
            '4': dw_conv(self.stage4_channels),
        })
        self.enc_attn = nn.ModuleDict({
            '2': SpatialAttention3D(),
            '3': SpatialAttention3D(),
            '4': SpatialAttention3D(),
        })

        self.dec_convs = nn.ModuleDict({
            '1': dw_conv(self.fused_channels // self.factor),
            '2': dw_conv(self.stage3_channels // self.factor),
            '3': dw_conv(self.stage2_channels // self.factor),
            '4': dw_conv(self.stage1_channels // self.factor),
        })
        self.dec_attn = nn.ModuleList([SpatialAttention3D() for _ in range(4)])

    def _apply_encoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        key = str(stage_idx)
        if key in self.enc_convs:
            x = self.enc_convs[key](x)
            x = self.enc_attn[key](x)
        return x

    def _apply_decoder_attention(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        key = str(stage_idx)
        if key in self.dec_convs:
            x = self.dec_convs[key](x)
            x = self.dec_attn[stage_idx - 1](x)
        return x


"""
Dual-Branch MobileNetV2 UNet (Shuffle-inspired, stage3 fused only)

- Stage 1/2: 각 2개 블록
- Stage 3: 6개 블록 (1 down + 5 extra)
- Stage 4: 단일 MobileNetV2 블록 (depthwise rates [1, 2, 5])

Fixed/Half decoder 구성을 `channel_configs.py` (lines 163-220 등)과 일치시키고,
MobileNetV2 블록 내부에 채널 어텐션(CBAM channel-only)을 삽입합니다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import OutConv3D, DoubleConv3D, _make_norm3d, _make_activation
from .channel_configs import get_dualbranch_channels_stage3_fused, get_activation_type
from .modules.cbam_modules import ChannelAttention3D


# ============================================================================
# MobileNetV2 Blocks with Channel Attention
# ============================================================================

class MobileNetV2Block3D(nn.Module):
    """3D MobileNetV2 inverted residual block with optional channel attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        expand_ratio: float = 2.0,
        norm: str = "bn",
        activation: str = "relu",
        use_channel_attention: bool = False,
        reduction: int = 16,
        dilation: int = 1,
        dilation_rates: list[int] | None = None,
    ):
        super().__init__()
        assert stride in (1, 2), "Stride must be 1 or 2."
        self.use_residual = stride == 1 and in_channels == out_channels
        self.norm = norm or "bn"
        hidden_dim = max(out_channels, int(round(in_channels * expand_ratio)))

        self.expand = None
        if hidden_dim != in_channels:
            self.expand = nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(self.norm, hidden_dim),
                _make_activation(activation, inplace=True),
            )

        dilation_list = dilation_rates or [dilation]
        current_stride = stride
        dw_layers = []
        for rate in dilation_list:
            padding = rate
            dw_layers.append(
                nn.Conv3d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=current_stride,
                    padding=padding,
                    dilation=rate,
                    groups=hidden_dim,
                    bias=False,
                )
            )
            dw_layers.append(_make_norm3d(self.norm, hidden_dim))
            dw_layers.append(_make_activation(activation, inplace=True))
            current_stride = 1  # 이후 블록은 stride=1
        self.depthwise = nn.Sequential(*dw_layers)

        self.channel_attention = (
            ChannelAttention3D(hidden_dim, reduction=reduction) if use_channel_attention else None
        )

        self.project = nn.Conv3d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = _make_norm3d(self.norm, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x) if self.expand is not None else x
        out = self.depthwise(out)
        if self.channel_attention is not None:
            out = self.channel_attention(out)
        out = self.project(out)
        out = self.pw_bn(out)
        if self.use_residual:
            out = out + x
        return out


class Down3DMobileNetV2(nn.Module):
    """Stride-2 wrapper around MobileNetV2Block3D (kept for legacy models)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: str = "bn",
        expand_ratio: float = 2.0,
        activation: str = "relu",
        use_channel_attention: bool = False,
        reduction: int = 16,
        dilation_rates: list[int] | None = None,
    ):
        super().__init__()
        self.block = MobileNetV2Block3D(
            in_channels,
            out_channels,
            stride=2,
            expand_ratio=expand_ratio,
            norm=norm,
            activation=activation,
            use_channel_attention=use_channel_attention,
            reduction=reduction,
            dilation_rates=dilation_rates,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _build_mobile_stage(
    in_channels: int,
    out_channels: int,
    *,
    activation: str,
    norm: str,
    expand_ratio: float,
    stride: int,
    num_post_blocks: int,
    use_channel_attention: bool,
    reduction: int,
    first_dilation_rates: list[int] | None = None,
    post_dilation_rates: list[int] | None = None,
) -> nn.Sequential:
    blocks = [
        MobileNetV2Block3D(
            in_channels,
            out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            norm=norm,
            activation=activation,
            use_channel_attention=use_channel_attention,
            reduction=reduction,
            dilation_rates=first_dilation_rates,
        )
    ]
    for _ in range(num_post_blocks):
        blocks.append(
            MobileNetV2Block3D(
                out_channels,
                out_channels,
                stride=1,
                expand_ratio=expand_ratio,
                norm=norm,
                activation=activation,
                use_channel_attention=use_channel_attention,
                reduction=reduction,
                dilation_rates=post_dilation_rates,
            )
        )
    return nn.Sequential(*blocks)


def _build_mobile_stem(
    in_channels: int,
    out_channels: int,
    *,
    activation: str,
    norm: str,
    expand_ratio: float,
    use_channel_attention: bool,
    reduction: int,
) -> nn.Sequential:
    return nn.Sequential(
        MobileNetV2Block3D(
            in_channels,
            out_channels,
            stride=1,
            expand_ratio=expand_ratio,
            norm=norm,
            activation=activation,
            use_channel_attention=use_channel_attention,
            reduction=reduction,
        ),
        MobileNetV2Block3D(
            out_channels,
            out_channels,
            stride=1,
            expand_ratio=expand_ratio,
            norm=norm,
            activation=activation,
            use_channel_attention=use_channel_attention,
            reduction=reduction,
        ),
    )


# ============================================================================
# Upsampling Block with Channel Attention
# ============================================================================

class Up3DMobileCBAM(nn.Module):
    """Upsampling block with channel attention (CBAM channel-only)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bilinear: bool = True,
        norm: str = "bn",
        skip_channels: int | None = None,
        target_skip_channels: int | None = None,
        use_channel_attention: bool = True,
        reduction: int = 16,
        activation: str = "relu",
    ):
        super().__init__()
        self.use_channel_attention = use_channel_attention

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            if skip_channels is None:
                skip_channels = in_channels // 2
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if skip_channels is None:
                skip_channels = in_channels // 2
            up_channels = in_channels // 2

        processed_skip = skip_channels
        if target_skip_channels is not None and skip_channels is not None and skip_channels != target_skip_channels:
            self.skip_compress = nn.Sequential(
                nn.Conv3d(skip_channels, target_skip_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, target_skip_channels),
                _make_activation(activation, inplace=True),
            )
            processed_skip = target_skip_channels
        else:
            self.skip_compress = None

        total_channels = up_channels + processed_skip
        self.channel_attention = (
            ChannelAttention3D(total_channels, reduction=reduction) if use_channel_attention else None
        )
        self.double_conv = DoubleConv3D(total_channels, out_channels, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        if self.skip_compress is not None:
            x2 = self.skip_compress(x2)
        x = torch.cat([x2, x1], dim=1)
        if self.channel_attention is not None:
            x = self.channel_attention(x)
        return self.double_conv(x)


# ============================================================================
# Dual-Branch MobileNetV2 (Stage 3 fused)
# ============================================================================

class DualBranchUNet3D_MobileNetV2(nn.Module):
    """Dual-branch MobileNetV2 UNet aligned with stage3-fused channel configs."""

    def __init__(
        self,
        n_channels: int = 2,
        n_classes: int = 4,
        *,
        norm: str = "bn",
        bilinear: bool = False,
        expand_ratio: float = 2.0,
        size: str = "s",
        half_decoder: bool = False,
        fixed_decoder: bool = False,
    ):
        super().__init__()
        assert n_channels == 2, "This model expects 2 input modalities (T1CE, FLAIR)."
        self.norm = norm or "bn"
        self.bilinear = bilinear
        self.expand_ratio = expand_ratio
        activation = get_activation_type(size)
        channels = get_dualbranch_channels_stage3_fused(size, half_decoder=half_decoder, fixed_decoder=fixed_decoder)

        # Stage 1 stems (각 2블록)
        self.stem_flair = _build_mobile_stem(
            1,
            channels["stem"],
            activation=activation,
            norm=self.norm,
            expand_ratio=self.expand_ratio,
            use_channel_attention=True,
            reduction=16,
        )
        self.stem_t1ce = _build_mobile_stem(
            1,
            channels["stem"],
            activation=activation,
            norm=self.norm,
            expand_ratio=self.expand_ratio,
            use_channel_attention=True,
            reduction=16,
        )

        # Stage 2 branches (stride=2, post 1블록)
        self.branch_flair = _build_mobile_stage(
            channels["stem"],
            channels["branch2"],
            activation=activation,
            norm=self.norm,
            expand_ratio=self.expand_ratio,
            stride=2,
            num_post_blocks=1,
            use_channel_attention=True,
            reduction=8,
        )
        self.branch_t1ce = _build_mobile_stage(
            channels["stem"],
            channels["branch2"],
            activation=activation,
            norm=self.norm,
            expand_ratio=self.expand_ratio,
            stride=2,
            num_post_blocks=1,
            use_channel_attention=True,
            reduction=8,
        )

        # Stage 3 branches (총 6블록: 1 down + 5 extra)
        self.branch_flair3 = _build_mobile_stage(
            channels["branch2"],
            channels["branch3"],
            activation=activation,
            norm=self.norm,
            expand_ratio=self.expand_ratio,
            stride=2,
            num_post_blocks=5,
            use_channel_attention=True,
            reduction=8,
        )
        self.branch_t1ce3 = _build_mobile_stage(
            channels["branch2"],
            channels["branch3"],
            activation=activation,
            norm=self.norm,
            expand_ratio=self.expand_ratio,
            stride=2,
            num_post_blocks=5,
            use_channel_attention=True,
            reduction=8,
        )

        # Stage 4 (fused, 단일 dilated 블록 with rates [1,2,5])
        fused_channels = channels["branch3"] * 2
        self.down4 = MobileNetV2Block3D(
            fused_channels,
            channels["down4"],
            stride=2,
            expand_ratio=self.expand_ratio,
            norm=self.norm,
            activation=activation,
            use_channel_attention=True,
            reduction=16,
            dilation_rates=[1, 2, 5],
        )

        # Decoder channel targets
        target_skip1 = channels.get("up1")
        target_skip2 = channels.get("up2")
        target_skip3 = channels.get("up3")

        up1_out = target_skip1 or channels["branch3"]
        up2_out = target_skip2 or channels["branch2"]
        up3_out = target_skip3 or channels["out"]
        final_out_channels = channels.get("out", up3_out)

        skip1 = fused_channels                  # From x3 (Stage 3 fused)
        skip2 = channels["branch2"] * 2         # From x2 (Stage 2 fused)
        skip3 = channels["stem"] * 2            # From x1 (Stage 1 fused)

        self.up1 = Up3DMobileCBAM(
            channels["down4"],
            up1_out,
            bilinear=self.bilinear,
            norm=self.norm,
            skip_channels=skip1,
            target_skip_channels=target_skip1,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        self.up2 = Up3DMobileCBAM(
            up1_out,
            up2_out,
            bilinear=self.bilinear,
            norm=self.norm,
            skip_channels=skip2,
            target_skip_channels=target_skip2,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        self.up3 = Up3DMobileCBAM(
            up2_out,
            up3_out,
            bilinear=self.bilinear,
            norm=self.norm,
            skip_channels=skip3,
            target_skip_channels=target_skip3,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        self.outc = OutConv3D(final_out_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)

        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)

        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)

        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


# Convenience classes for sizes / options
class DualBranchUNet3D_MobileNetV2_XS(DualBranchUNet3D_MobileNetV2):
    def __init__(self, **kwargs):
        super().__init__(size="xs", **kwargs)


class DualBranchUNet3D_MobileNetV2_Small(DualBranchUNet3D_MobileNetV2):
    def __init__(self, **kwargs):
        super().__init__(size="s", **kwargs)


class DualBranchUNet3D_MobileNetV2_Medium(DualBranchUNet3D_MobileNetV2):
    def __init__(self, **kwargs):
        super().__init__(size="m", **kwargs)


class DualBranchUNet3D_MobileNetV2_Large(DualBranchUNet3D_MobileNetV2):
    def __init__(self, **kwargs):
        super().__init__(size="l", **kwargs)


class DualBranchUNet3D_MobileNetV2_FixedDecoder(DualBranchUNet3D_MobileNetV2):
    def __init__(self, size: str = "s", **kwargs):
        super().__init__(size=size, fixed_decoder=True, **kwargs)


__all__ = [
    "DualBranchUNet3D_MobileNetV2",
    "Down3DMobileNetV2",
    "DualBranchUNet3D_MobileNetV2_XS",
    "DualBranchUNet3D_MobileNetV2_Small",
    "DualBranchUNet3D_MobileNetV2_Medium",
    "DualBranchUNet3D_MobileNetV2_Large",
    "DualBranchUNet3D_MobileNetV2_FixedDecoder",
]


"""
Shared building blocks for remaining dual-branch models.
Extracted from dualbranch_basic and dualbranch_mobile (removed) for use by
dualbranch_backbone_unet, dualbranch_mvit, dualbranch_replk.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..baseline.model_3d_unet import _make_norm3d, _make_activation
from .cbam_modules import ChannelAttention3D


# ============================================================================
# From dualbranch_basic
# ============================================================================

class Down3DStride(nn.Module):
    """Downsampling with stride-2 Conv instead of MaxPool.
    Pattern: Conv(stride=2) -> Norm -> ReLU -> Conv -> Norm -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3DStrideDilated(nn.Module):
    """Downsampling with stride-2 Conv and dilated convolutions for wider ERF.
    Pattern: Conv(stride=2) -> Norm -> ReLU -> DilatedConv(rate=2) -> ... -> DilatedConv(rate=5) -> ...
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================================
# From dualbranch_mobile
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
            current_stride = 1
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
    """Stride-2 wrapper around MobileNetV2Block3D."""

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


__all__ = [
    "Down3DStride",
    "Down3DStrideDilated",
    "MobileNetV2Block3D",
    "Down3DMobileNetV2",
]

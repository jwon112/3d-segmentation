"""
Light-weight ROI-specific 3D U-Net architectures.

기존 세그멘테이션 모델과 분리된 독립 구조로,
Cascade 1단계(ROI 탐지)에 최적화된 네트워크를 정의한다.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm3d(norm: str, num_features: int) -> nn.Module:
    norm = (norm or "bn").lower()
    if norm in ("in", "instancenorm", "instance"):
        return nn.InstanceNorm3d(num_features, affine=True, track_running_stats=False)
    if norm in ("gn", "groupnorm", "group"):
        num_groups = 8 if num_features % 8 == 0 else (4 if num_features % 4 == 0 else 1)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    return nn.BatchNorm3d(num_features)


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn"):
        super().__init__()
        mid_channels = max(out_channels, in_channels)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn"):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(2),
            _DoubleConv(in_channels, out_channels, norm=norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm: str = "bn"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = _DoubleConv(in_channels + skip_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(
            x,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class _OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ROICascadeUNet3D(nn.Module):
    """
    ROI Detector 전용 3D U-Net.

    - 기본 채널 수(base_channels)를 줄여 파라미터 수를 감소
    - depth를 조절하여 입력 해상도/메모리 상황에 맞춤
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        *,
        base_channels: int = 16,
        depth: int = 4,
        norm: str = "bn",
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2 for UNet encoder/decoder symmetry.")

        enc_channels = [base_channels * (2 ** i) for i in range(depth)]

        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        self.enc_blocks.append(_DoubleConv(in_channels, enc_channels[0], norm=norm))
        for idx in range(1, depth):
            self.down_blocks.append(_Down(enc_channels[idx - 1], enc_channels[idx], norm=norm))

        bottleneck_channels = enc_channels[-1] * 2
        self.bottleneck = _DoubleConv(enc_channels[-1], bottleneck_channels, norm=norm)

        self.up_blocks = nn.ModuleList()
        decoder_in = bottleneck_channels
        for ch in reversed(enc_channels):
            self.up_blocks.append(_Up(decoder_in, ch, ch, norm=norm))
            decoder_in = ch

        self.out_conv = _OutConv(decoder_in, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = self.enc_blocks[0](x)
        skips.append(h)
        for down in self.down_blocks:
            h = down(skips[-1])
            skips.append(h)

        h = self.bottleneck(skips[-1])

        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip)

        return self.out_conv(h)


def build_roi_unet3d_small(
    *,
    in_channels: int = 7,
    out_channels: int = 2,
    norm: str = "bn",
    base_channels: int = 16,
    depth: int = 4,
) -> ROICascadeUNet3D:
    """
    Factory helper that instantiates the ROI-specific UNet3D.
    """

    return ROICascadeUNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        norm=norm,
    )


__all__ = ["ROICascadeUNet3D", "build_roi_unet3d_small"]



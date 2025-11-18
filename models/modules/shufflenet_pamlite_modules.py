"""
ShuffleNetV2 PAM-Lite Modules
 - Split/shuffle 구조 유지
 - Conv branch를 dilated depthwise conv (rate 1/2/5) + GhostModule3D로 경량화
 - GhostNet 연산을 활용하여 PAM 증가 없이 전역 receptive field 확보
"""

import torch
import torch.nn as nn

from ..model_3d_unet import _make_norm3d
from .shufflenet_modules import channel_shuffle_3d
from .ghostnet_modules import GhostModule3D


class DilatedGhostStage3D(nn.Module):
    """Dilated depthwise conv (rate 1/2/5) followed by GhostModule."""

    def __init__(self, channels: int, out_channels: int, norm: str = "bn"):
        super().__init__()
        self.dw1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1,
                      dilation=1, groups=channels, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
        )
        self.dw2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=2,
                      dilation=2, groups=channels, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
        )
        self.dw3 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=5,
                      dilation=5, groups=channels, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
        )
        self.ghost = GhostModule3D(channels, out_channels, kernel_size=1, stride=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        return self.ghost(x)


class ShuffleNetV2PamLiteUnit3D(nn.Module):
    """ShuffleNetV2 unit with PAM-lite (DW dilated + Ghost) branch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = "bn"):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride == 1:
            assert in_channels == out_channels, "Stride=1 requires in/out channels equal."
            mid_channels = out_channels // 2
            self.branch1 = nn.Identity()
            self.branch2 = DilatedGhostStage3D(mid_channels, mid_channels, norm=norm)
        else:
            half_out = out_channels // 2
            self.branch1 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,
                          groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.ReLU(inplace=True),
                GhostModule3D(in_channels, half_out, kernel_size=1, stride=1, norm=norm),
            )
            self.branch2_prep = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,
                          groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.ReLU(inplace=True),
            )
            self.branch2 = DilatedGhostStage3D(in_channels, half_out, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(self.branch2_prep(x))

        out = torch.cat([out1, out2], dim=1)
        out = channel_shuffle_3d(out, groups=2)
        return out


class Down3DShuffleNetV2PamLite(nn.Module):
    """Downsampling block using PAM-Lite ShuffleNet units."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn"):
        super().__init__()
        self.unit1 = ShuffleNetV2PamLiteUnit3D(in_channels, out_channels, stride=2, norm=norm)
        self.unit2 = ShuffleNetV2PamLiteUnit3D(out_channels, out_channels, stride=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


__all__ = [
    "ShuffleNetV2PamLiteUnit3D",
    "Down3DShuffleNetV2PamLite",
]


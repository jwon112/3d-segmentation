"""
Light-weight ROI-specific 3D U-Net architectures.

기존 세그멘테이션 모델과 채널 구성이 다르기 때문에
Cascade 1단계(ROI 탐지)에 최적화된 전용 모델을 정의한다.
"""

import torch
import torch.nn as nn

from models.model_3d_unet import DoubleConv3D, Down3D, Up3D, OutConv3D


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
        bilinear: bool = True,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2 for UNet encoder/decoder symmetry.")
        enc_channels = [base_channels * (2 ** i) for i in range(depth)]

        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        self.enc_blocks.append(DoubleConv3D(in_channels, enc_channels[0], norm=norm))
        for idx in range(1, depth):
            self.down_blocks.append(Down3D(enc_channels[idx - 1], enc_channels[idx], norm=norm))

        factor = 2 if bilinear else 1
        bottleneck_channels = enc_channels[-1] * 2
        self.bottleneck = DoubleConv3D(enc_channels[-1], bottleneck_channels // factor, norm=norm)

        self.up_blocks = nn.ModuleList()
        decoder_in = bottleneck_channels
        for ch in reversed(enc_channels):
            self.up_blocks.append(
                Up3D(
                    decoder_in,
                    ch // factor,
                    bilinear=bilinear,
                    norm=norm,
                    skip_channels=ch,
                )
            )
            decoder_in = ch

        self.out_conv = OutConv3D(enc_channels[0] // factor, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = self.enc_blocks[0](x)
        skips.append(h)
        for down in self.down_blocks:
            h = down(h)
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
    bilinear: bool = True,
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
        bilinear=bilinear,
    )


__all__ = ["ROICascadeUNet3D", "build_roi_unet3d_small"]



"""
Cascade ShuffleNet V2 encoder + SegNeXt-style decoder (3D, LKA-hybrid encoder).

- Encoder:
    Stem3x3 (96^3) ->
    Down3DShuffleNetV2 (48^3) ->
    Down3DShuffleNetV2_LKAHybrid + extra LKA blocks (24^3) ->
    Stage4 Hybrid LKA (12^3)  [reused from CascadeShuffleNetV2UNet3D_LKAHybrid]

- Decoder (SegNeXt-style):
    - Trilinear upsample (x2)
    - 1x1x1 projections for high-level and skip feature
    - Add fusion (no concat)
    - Depthwise 3x3x3 + Pointwise 1x1x1 (lightweight refinement)
    - Hierarchical multi-scale fusion:
        x4(12^3) -> 24^3 (fuse x3) -> 48^3 (fuse x2) -> 96^3 (fuse x1) -> 1x1x1 seg head
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.channel_configs import get_singlebranch_channels_2step_decoder, get_activation_type
from models.model_3d_unet import _make_norm3d, _make_activation
from models.modules.shufflenet_modules import Down3DShuffleNetV2
from models.modules.lka_hybrid_modules import LKAHybridCBAM3D
from .shufflenet_v2 import (
    Stem3x3,
    Down3DShuffleNetV2_LKAHybrid,
    _build_shufflenet_v2_lka_extra_blocks,
)


class SegNeXtDecoderBlock3D(nn.Module):
    """
    3D SegNeXt-style decoder block:

    - Upsample high-level feature by 2x (trilinear)
    - Project high-level and skip feature to the same channels with 1x1x1 conv
    - Fuse via addition (no concat)
    - Lightweight Depthwise 3x3x3 + Pointwise 1x1x1 refinement
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.norm = norm or "bn"

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        # Project high-level feature to out_channels
        self.proj_high = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, out_channels),
        )

        # Project skip feature to out_channels
        self.proj_skip = nn.Sequential(
            nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, out_channels),
        )

        # Depthwise separable refinement: DW 3x3x3 + PW 1x1x1
        self.refine = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,  # depthwise
                bias=False,
            ),
            _make_norm3d(self.norm, out_channels),
            _make_activation(activation, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_norm3d(self.norm, out_channels),
            _make_activation(activation, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample high-level feature
        x_up = self.up(x)

        # Spatial alignment with skip (padding if needed)
        diffZ = skip.size(2) - x_up.size(2)
        diffY = skip.size(3) - x_up.size(3)
        diffX = skip.size(4) - x_up.size(4)
        if diffZ != 0 or diffY != 0 or diffX != 0:
            x_up = F.pad(
                x_up,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )

        # Project to same channel dimension
        x_high = self.proj_high(x_up)
        x_skip = self.proj_skip(skip)

        # Fuse via addition
        x_fused = x_high + x_skip

        # Lightweight refinement
        out = self.refine(x_fused)
        return out


class CascadeShuffleNetV2SegNeXt3D_LKA(nn.Module):
    """
    Cascade ShuffleNet V2 encoder with LKA-hybrid (same as CascadeShuffleNetV2UNet3D_LKAHybrid)
    + SegNeXt-style lightweight decoder.

    - Encoder: 96^3 -> 48^3 -> 24^3 -> 12^3
      * Stage2: Down3DShuffleNetV2
      * Stage3: Down3DShuffleNetV2_LKAHybrid + extra LKA units
      * Stage4: LKAHybridCBAM3D (stride=2) + LKAHybridCBAM3D (stride=1)

    - Decoder: hierarchical upsample + add-fusion + DW/PW conv refinement
      * 12^3 -> 24^3 (fuse Stage3)
      * 24^3 -> 48^3 (fuse Stage2)
      * 48^3 -> 96^3 (fuse Stem)
    """

    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        norm: str = "bn",
        size: str = "s",
        include_coords: bool = True,
    ) -> None:
        super().__init__()
        self.n_image_channels = n_image_channels
        self.n_coord_channels = n_coord_channels
        self.include_coords = include_coords and n_coord_channels > 0
        self.norm = norm or "bn"
        self.size = size

        activation = get_activation_type(size)
        channels = get_singlebranch_channels_2step_decoder(size)

        stem_in = n_image_channels + (n_coord_channels if self.include_coords else 0)
        self.stem = Stem3x3(stem_in, channels["stem"], norm=self.norm, activation=activation)

        # Stage 2: ShuffleNetV2 Down block (48^3)
        self.down2 = Down3DShuffleNetV2(
            channels["stem"],
            channels["branch2"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )

        # Stage 3: Hybrid LKA Down block (24^3) + extra LKA units
        self.down3 = Down3DShuffleNetV2_LKAHybrid(
            channels["branch2"],
            channels["branch3"],
            norm=self.norm,
            reduction=4,
            activation=activation,
        )
        # down3에 이미 2개 블록이 있으므로 extra의 첫 블록부터 drop path 적용
        self.down3_extra = _build_shufflenet_v2_lka_extra_blocks(
            channels["branch3"],
            3,
            self.norm,
            reduction=4,
            activation=activation,
            # down3의 2개 블록 이후이므로 첫 블록부터 drop path 적용
            drop_path_rates=[0.05, 0.1, 0.15],
            # Spatial dropout (width 앙상블) - Stage 3에 적용
            drop_channel_rates=[0.05, 0.05, 0.05],
        )

        fused_channels = channels["branch3"]
        expanded_channels = channels["down4"]

        # Stage 4: 24^3 -> 12^3 via LKA stride=2, then one more LKA (stride=1)
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # lka1은 항상 실행되어 Stage 4가 완전히 사라지지 않도록 보장
        self.down4_lka1 = LKAHybridCBAM3D(
            channels=expanded_channels,
            reduction=4,
            norm=self.norm,
            activation=activation,
            use_residual=True,
            stride=2,
            drop_path_rate=0.0,  # 첫 번째 블록은 항상 실행
            drop_channel_rate=0.0,  # 첫 번째 블록은 spatial dropout 미적용
        )
        # lka2만 drop path 및 spatial dropout 적용하여 regularization 효과
        self.down4_lka2 = LKAHybridCBAM3D(
            channels=expanded_channels,
            reduction=4,
            norm=self.norm,
            activation=activation,
            use_residual=True,
            stride=1,
            drop_path_rate=0.15,  # 두 번째 블록에만 drop path 적용
            drop_channel_rate=0.05,  # Bottleneck이므로 낮은 spatial dropout
        )

        # SegNeXt-style decoder
        # Level 3: 12^3 -> 24^3, fuse Stage3 (branch3)
        self.dec3 = SegNeXtDecoderBlock3D(
            in_channels=channels["down4"],
            skip_channels=channels["branch3"],
            out_channels=channels["up1"],
            norm=self.norm,
            activation=activation,
        )
        # Level 2: 24^3 -> 48^3, fuse Stage2 (branch2)
        self.dec2 = SegNeXtDecoderBlock3D(
            in_channels=channels["up1"],
            skip_channels=channels["branch2"],
            out_channels=channels["up2"],
            norm=self.norm,
            activation=activation,
        )
        # Level 1: 48^3 -> 96^3, fuse Stem
        self.dec1 = SegNeXtDecoderBlock3D(
            in_channels=channels["up2"],
            skip_channels=channels["stem"],
            out_channels=channels["out"],
            norm=self.norm,
            activation=activation,
        )

        # Final segmentation head
        self.outc = nn.Conv3d(channels["out"], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input split: image + coords
        if self.include_coords and self.n_coord_channels > 0:
            img = x[:, : self.n_image_channels]
            coord = x[:, self.n_image_channels : self.n_image_channels + self.n_coord_channels]
            x_in = torch.cat([img, coord], dim=1)
        else:
            x_in = x[:, : self.n_image_channels]

        # Encoder
        x1 = self.stem(x_in)        # 96^3
        x2 = self.down2(x1)         # 48^3
        x3 = self.down3(x2)         # 24^3
        x3 = self.down3_extra(x3)   # 24^3 (refined)

        x4 = self.down4_expand(x3)  # 24^3
        x4 = self.down4_lka1(x4)    # 12^3
        x4 = self.down4_lka2(x4)    # 12^3

        # SegNeXt-style decoder
        d3 = self.dec3(x4, x3)      # 12^3 -> 24^3
        d2 = self.dec2(d3, x2)      # 24^3 -> 48^3
        d1 = self.dec1(d2, x1)      # 48^3 -> 96^3

        out = self.outc(d1)
        return out


def build_cascade_shufflenet_v2_segnext_lka(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2SegNeXt3D_LKA:
    return CascadeShuffleNetV2SegNeXt3D_LKA(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
    )


__all__ = [
    "CascadeShuffleNetV2SegNeXt3D_LKA",
    "build_cascade_shufflenet_v2_segnext_lka",
]



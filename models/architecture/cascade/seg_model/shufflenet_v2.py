"""
Cascade ShuffleNet V2 UNet (single branch).

- 입력: MRI n_image_channels + CoordConv n_coord_channels (기본 4+3)
- Coord 채널은 stem 이전에 concat하여 공간 정보를 직접 주입
- Stage/채널 구성은 Stage3 Fused DualBranch 버전과 동일한 helper를 재사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.channel_configs import (
    get_singlebranch_channels_2step_decoder,
    get_activation_type,
)
from models.modules.shufflenet_modules import (
    ShuffleNetV2Unit3D,
    Down3DShuffleNetV2,
    MultiScaleDilatedDepthwise3D,
)
from models.model_3d_unet import _make_norm3d, _make_activation


def _build_shufflenet_v2_extra_blocks(
    channels: int,
    num_blocks: int,
    norm: str,
    use_channel_attention: bool,
    reduction: int,
    activation: str,
) -> nn.Module:
    if num_blocks <= 0:
        return nn.Identity()
    layers = [
        ShuffleNetV2Unit3D(
            channels,
            channels,
            stride=1,
            norm=norm,
            use_channel_attention=use_channel_attention,
            reduction=reduction,
            activation=activation,
        )
        for _ in range(num_blocks)
    ]
    return nn.Sequential(*layers)


class Stem3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn", activation: str = "relu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up3DShuffleNetV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "bn",
        reduction: int = 8,
        activation: str = "relu",
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        total_channels = in_channels + skip_channels

        self.unit1 = ShuffleNetV2Unit3D(
            total_channels,
            total_channels,
            stride=1,
            norm=norm,
            use_channel_attention=True,
            reduction=reduction,
            activation=activation,
        )
        if total_channels != out_channels:
            self.channel_adjust = nn.Sequential(
                nn.Conv3d(total_channels, out_channels, kernel_size=1, bias=False),
                _make_norm3d(norm, out_channels),
                _make_activation(activation, inplace=True),
            )
        else:
            self.channel_adjust = None
        self.unit2 = ShuffleNetV2Unit3D(
            out_channels,
            out_channels,
            stride=1,
            norm=norm,
            use_channel_attention=True,
            reduction=reduction,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        diffZ = skip.size(2) - x_up.size(2)
        diffY = skip.size(3) - x_up.size(3)
        diffX = skip.size(4) - x_up.size(4)
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
        x = torch.cat([skip, x_up], dim=1)
        x = self.unit1(x)
        if self.channel_adjust is not None:
            x = self.channel_adjust(x)
        x = self.unit2(x)
        return x


class CascadeShuffleNetV2UNet3D(nn.Module):
    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        norm: str = "bn",
        size: str = "s",
        include_coords: bool = True,
    ):
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

        self.down2 = Down3DShuffleNetV2(
            channels["stem"],
            channels["branch2"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        self.down3 = Down3DShuffleNetV2(
            channels["branch2"],
            channels["branch3"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=2,
            activation=activation,
        )
        self.down3_extra = _build_shufflenet_v2_extra_blocks(
            channels["branch3"], 4, self.norm, True, 2, activation
        )

        fused_channels = channels["branch3"]
        expanded_channels = channels["down4"]

        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        self.down4_depth = MultiScaleDilatedDepthwise3D(
            expanded_channels, dilation_rates=[1, 2, 3], norm=self.norm, activation=activation
        )
        self.down4_compress = nn.Identity()

        # up1: down4 -> up1 (skip from branch3)
        # upsampled: down4(256) + skip: branch3(128) -> out: up1(64)
        self.up1 = Up3DShuffleNetV2(
            channels["down4"],
            skip_channels=fused_channels,
            out_channels=channels["up1"],
            norm=self.norm,
            reduction=2,
            activation=activation,
        )
        # up2: up1 -> up2 (skip from branch2)
        # upsampled: up1(64) + skip: branch2(64) -> out: up2(64) [1:1 ratio]
        self.up2 = Up3DShuffleNetV2(
            channels["up1"],
            skip_channels=channels["branch2"],
            out_channels=channels["up2"],
            norm=self.norm,
            reduction=8,
            activation=activation,
        )
        # up3: up2 -> out (skip from stem)
        # upsampled: up2(64) + skip: stem(32) -> out: out(32)
        self.up3 = Up3DShuffleNetV2(
            channels["up2"],
            skip_channels=channels["stem"],
            out_channels=channels["out"],
            norm=self.norm,
            reduction=2,
            activation=activation,
        )
        self.outc = nn.Conv3d(channels["out"], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.include_coords and self.n_coord_channels > 0:
            img = x[:, : self.n_image_channels]
            coord = x[:, self.n_image_channels : self.n_image_channels + self.n_coord_channels]
            x_in = torch.cat([img, coord], dim=1)
        else:
            x_in = x[:, : self.n_image_channels]

        x1 = self.stem(x_in)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = self.down3_extra(x3)

        x4 = self.down4_expand(x3)
        x4 = self.down4_depth(x4)
        x4 = self.down4_compress(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def build_cascade_shufflenet_v2_unet3d(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2UNet3D:
    return CascadeShuffleNetV2UNet3D(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
    )


__all__ = [
    "CascadeShuffleNetV2UNet3D",
    "build_cascade_shufflenet_v2_unet3d",
]




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
from models.modules.shufflenet_modules import Down3DShuffleNetV2, ShuffleNetV2Unit3D_LKAHybrid, channel_shuffle_3d
from models.modules.lka_hybrid_modules import LKAHybridCBAM3D
from .shufflenet_v2 import (
    Stem3x3,
    Down3DShuffleNetV2_LKAHybrid,
    _build_shufflenet_v2_lka_extra_blocks,
    P3DStem3x3,
    P3DDown3DShuffleNetV2,
    P3DConv3d,
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


# ============================================================================
# P3D (Pseudo-3D) Variants
# ============================================================================


class P3DSegNeXtDecoderBlock3D(nn.Module):
    """
    3D SegNeXt-style decoder block with P3D:

    - Upsample high-level feature by 2x (trilinear)
    - Project high-level and skip feature to the same channels with 1x1x1 conv
    - Fuse via addition (no concat)
    - P3D Depthwise 3x3x3 + Pointwise 1x1x1 refinement
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

        # P3D Depthwise separable refinement: P3D DW 3x3x3 + PW 1x1x1
        self.refine = nn.Sequential(
            P3DConv3d(
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


class P3DShuffleNetV2Unit3D_LKAHybrid(nn.Module):
    """3D ShuffleNetV2 Unit with Hybrid LKA branch (P3D version).

    - Stride=1:
        Split -> Branch1 (identity) +
        Branch2 (1x1 -> LKAHybridCBAM3D -> 출력) -> Concat -> Shuffle
    - Stride=2:
        No split ->
        Branch1 (P3D DWConv stride=2 -> 1x1) +
        Branch2 (1x1 -> LKAHybridCBAM3D -> 출력 채널) -> Concat -> Shuffle

    LKAHybridCBAM3D 내부는 일반 3D Conv 유지 (P3D 적용 안 함).
    Branch1의 depthwise conv만 P3D로 변경.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "bn",
        reduction: int = 16,
        activation: str = "relu",
        drop_path: float = 0.0,
        drop_channel: float = 0.0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.drop_path = float(drop_path)
        self.drop_channel = float(drop_channel)

        activation_fn = _make_activation(activation, inplace=True)

        if stride == 1:
            # Stride=1: Split operation
            assert in_channels == out_channels, "For stride=1, in_channels must equal out_channels"
            mid_channels = out_channels // 2

            # Branch 1: Identity (no operation)
            self.branch1 = nn.Identity()

            # Branch 2: 1x1 -> Hybrid LKA
            self.branch2_conv1 = nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.branch2_bn1 = _make_norm3d(norm, mid_channels)
            self.branch2_activation = activation_fn

            # Hybrid LKA block (includes depthwise dense + sparse + CBAM channel attention + residual)
            # LKA block은 일반 3D Conv 유지
            self.branch2_lka = LKAHybridCBAM3D(
                channels=mid_channels,
                reduction=reduction,
                norm=norm,
                activation=activation,
                use_residual=True,
                stride=1,
                drop_path_rate=self.drop_path,
                drop_channel_rate=self.drop_channel,
            )
        else:
            # Stride=2: No split, both branches process full input
            mid_out = out_channels // 2

            # Branch 1: P3D DWConv stride=2 -> 1x1
            self.branch1_conv1 = P3DConv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            )
            self.branch1_bn1 = _make_norm3d(norm, in_channels)
            self.branch1_conv2 = nn.Conv3d(
                in_channels,
                mid_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.branch1_bn2 = _make_norm3d(norm, mid_out)
            self.branch1_activation = activation_fn

            # Branch 2: 1x1 -> Hybrid LKA (stride=2) for downsampling
            self.branch2_conv1 = nn.Conv3d(
                in_channels,
                mid_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.branch2_bn1 = _make_norm3d(norm, mid_out)
            self.branch2_activation = activation_fn

            # Hybrid LKA block (stride=2) - 일반 3D Conv 유지
            self.branch2_lka = LKAHybridCBAM3D(
                channels=mid_out,
                reduction=reduction,
                norm=norm,
                activation=activation,
                use_residual=True,
                stride=2,
                drop_path_rate=self.drop_path,
                drop_channel_rate=self.drop_channel,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            # Split channels
            x1, x2 = x.chunk(2, dim=1)
            # Branch 1: Identity
            out1 = self.branch1(x1)
            # Branch 2: Processing
            out2 = self.branch2_conv1(x2)
            out2 = self.branch2_bn1(out2)
            out2 = self.branch2_activation(out2)
            out2 = self.branch2_lka(out2)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        else:
            # Both branches process full input
            # Branch 1
            out1 = self.branch1_conv1(x)
            out1 = self.branch1_bn1(out1)
            out1 = self.branch1_conv2(out1)
            out1 = self.branch1_bn2(out1)
            out1 = self.branch1_activation(out1)
            # Branch 2
            out2 = self.branch2_conv1(x)
            out2 = self.branch2_bn1(out2)
            out2 = self.branch2_activation(out2)
            out2 = self.branch2_lka(out2)
            # Concat
            out = torch.cat([out1, out2], dim=1)

        # Channel Shuffle
        out = channel_shuffle_3d(out, groups=2)
        return out


class P3DDown3DShuffleNetV2_LKAHybrid(nn.Module):
    """Downsampling using P3DShuffleNetV2Unit3D_LKAHybrid (stride=2 then stride=1).

    - unit1: stride=2 (해상도 절반으로 감소)
    - unit2: stride=1 (해상도 유지, 특징 정제)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "bn",
        reduction: int = 16,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = P3DShuffleNetV2Unit3D_LKAHybrid(
            in_channels,
            out_channels,
            stride=2,
            norm=norm,
            reduction=reduction,
            activation=activation,
        )
        # Second unit: stride=1 for feature refinement
        self.unit2 = P3DShuffleNetV2Unit3D_LKAHybrid(
            out_channels,
            out_channels,
            stride=1,
            norm=norm,
            reduction=reduction,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


def _build_p3d_shufflenet_v2_lka_extra_blocks(
    channels: int,
    num_blocks: int,
    norm: str,
    reduction: int,
    activation: str,
    drop_path_rates: list[float] | None = None,
    drop_channel_rates: list[float] | None = None,
) -> nn.Module:
    """P3D ShuffleNetV2 + Hybrid LKA extra blocks (stride=1).

    Stage3/Stage4에서 해상도는 유지하고, 채널만 유지한 채로
    P3DShuffleNetV2Unit3D_LKAHybrid 블록을 반복적으로 적용합니다.
    """
    if num_blocks <= 0:
        return nn.Identity()

    layers: list[nn.Module] = []
    for idx in range(num_blocks):
        drop_rate = 0.0
        if drop_path_rates is not None and idx < len(drop_path_rates):
            drop_rate = float(drop_path_rates[idx])

        drop_channel_rate = 0.0
        if drop_channel_rates is not None and idx < len(drop_channel_rates):
            drop_channel_rate = float(drop_channel_rates[idx])

        layers.append(
            P3DShuffleNetV2Unit3D_LKAHybrid(
                channels,
                channels,
                stride=1,
                norm=norm,
                reduction=reduction,
                activation=activation,
                drop_path=drop_rate,
                drop_channel=drop_channel_rate,
            )
        )
    return nn.Sequential(*layers)


class CascadeShuffleNetV2SegNeXt3D_P3D_LKA(nn.Module):
    """
    Cascade ShuffleNet V2 encoder with P3D + LKA-hybrid
    + SegNeXt-style lightweight decoder with P3D.

    - Encoder: 96^3 -> 48^3 -> 24^3 -> 12^3
      * Stem: P3DStem3x3
      * Stage2: P3DDown3DShuffleNetV2
      * Stage3: P3DDown3DShuffleNetV2_LKAHybrid + extra P3D LKA units
      * Stage4: LKAHybridCBAM3D (stride=2) + LKAHybridCBAM3D (stride=1) [일반 3D Conv 유지]

    - Decoder: hierarchical upsample + add-fusion + P3D DW/PW conv refinement
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
        self.stem = P3DStem3x3(stem_in, channels["stem"], norm=self.norm, activation=activation)

        # Stage 2: P3D ShuffleNetV2 Down block (48^3)
        self.down2 = P3DDown3DShuffleNetV2(
            channels["stem"],
            channels["branch2"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )

        # Stage 3: P3D Hybrid LKA Down block (24^3) + extra P3D LKA units
        self.down3 = P3DDown3DShuffleNetV2_LKAHybrid(
            channels["branch2"],
            channels["branch3"],
            norm=self.norm,
            reduction=4,
            activation=activation,
        )
        # down3에 이미 2개 블록이 있으므로 extra의 첫 블록부터 drop path 적용
        self.down3_extra = _build_p3d_shufflenet_v2_lka_extra_blocks(
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
        # LKA block은 일반 3D Conv 유지 (P3D 적용 안 함)
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

        # P3D SegNeXt-style decoder
        # Level 3: 12^3 -> 24^3, fuse Stage3 (branch3)
        self.dec3 = P3DSegNeXtDecoderBlock3D(
            in_channels=channels["down4"],
            skip_channels=channels["branch3"],
            out_channels=channels["up1"],
            norm=self.norm,
            activation=activation,
        )
        # Level 2: 24^3 -> 48^3, fuse Stage2 (branch2)
        self.dec2 = P3DSegNeXtDecoderBlock3D(
            in_channels=channels["up1"],
            skip_channels=channels["branch2"],
            out_channels=channels["up2"],
            norm=self.norm,
            activation=activation,
        )
        # Level 1: 48^3 -> 96^3, fuse Stem
        self.dec1 = P3DSegNeXtDecoderBlock3D(
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


def build_cascade_shufflenet_v2_segnext_p3d_lka(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2SegNeXt3D_P3D_LKA:
    return CascadeShuffleNetV2SegNeXt3D_P3D_LKA(
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
    "CascadeShuffleNetV2SegNeXt3D_P3D_LKA",
    "build_cascade_shufflenet_v2_segnext_p3d_lka",
]



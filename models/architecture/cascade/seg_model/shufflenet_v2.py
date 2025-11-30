"""
Cascade ShuffleNet V2 UNet (single branch).

- 입력: MRI n_image_channels + CoordConv n_coord_channels (기본 4+3)
- Coord 채널은 stem 이전에 concat하여 공간 정보를 직접 주입
- Stage/채널 구성은 Stage3 Fused DualBranch 버전과 동일한 helper를 재사용

P3D 변형:
- 3D convolution을 2D spatial conv + 1D depth conv로 분리
- 메모리 효율적이고 파라미터 수 감소
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Any

from models.channel_configs import (
    get_singlebranch_channels_2step_decoder,
    get_activation_type,
)
from models.modules.shufflenet_modules import (
    ShuffleNetV2Unit3D,
    Down3DShuffleNetV2,
    MultiScaleDilatedDepthwise3D,
    ChannelAttention3D,
    channel_shuffle_3d,
)
from models.modules.mvit_modules import MobileViT3DBlockV3
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


# ============================================================================
# P3D (Pseudo-3D) Variants
# ============================================================================

class P3DConv3d(nn.Module):
    """
    P3D Convolution: 3D conv를 2D spatial conv + 1D depth conv로 분리
    
    P3D-A 방식: 2D spatial conv → 1D depth conv
    Conv3d(k, k, k) → Conv2d(k, k) + Conv1d(k)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # 2D spatial convolution (H, W)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride), padding=(padding, padding),
            groups=groups, bias=False
        )
        # 1D depth convolution (D)
        self.conv1d = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Reshape for 2D conv: (B*D, C, H, W)
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        x_2d = self.conv2d(x_2d)  # (B*D, C_out, H', W')
        _, C_out, H_out, W_out = x_2d.shape
        
        # Reshape back: (B, D, C_out, H_out, W_out) -> (B, C_out, D, H_out, W_out)
        x_3d = x_2d.view(B, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()
        
        # 1D depth conv: (B, C_out, D, H_out, W_out) -> (B*H_out*W_out, C_out, D)
        x_1d = x_3d.permute(0, 3, 4, 1, 2).contiguous().view(B * H_out * W_out, C_out, D)
        x_1d = self.conv1d(x_1d)  # (B*H_out*W_out, C_out, D')
        _, _, D_out = x_1d.shape
        
        # Reshape back: (B, H_out, W_out, C_out, D_out) -> (B, C_out, D_out, H_out, W_out)
        x_out = x_1d.view(B, H_out, W_out, C_out, D_out).permute(0, 3, 4, 1, 2).contiguous()
        
        return x_out


class P3DStem3x3(nn.Module):
    """P3D Stem: 3x3x3 conv를 3x3 2D + 3 1D로 분리"""
    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn", activation: str = "relu"):
        super().__init__()
        self.block = nn.Sequential(
            P3DConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
            P3DConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class P3DShuffleNetV2Unit3D(nn.Module):
    """
    P3D ShuffleNetV2 Unit: Depthwise conv를 P3D로 변환
    
    기존: Conv3d(k, k, k, groups=channels)
    P3D: Conv2d(k, k, groups=channels) + Conv1d(k, groups=channels)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = 'bn',
                 use_channel_attention: bool = False, reduction: int = 16, activation: str = 'relu'):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_channel_attention = use_channel_attention
        
        activation_fn = _make_activation(activation, inplace=True)
        
        if stride == 1:
            assert in_channels == out_channels, "For stride=1, in_channels must equal out_channels"
            mid_channels = out_channels // 2
            
            # Branch 1: Identity
            self.branch1 = nn.Identity()
            
            # Branch 2: P3D DWConv -> 1x1 -> P3D DWConv -> Channel Attention -> 1x1
            # P3D Depthwise Conv (3x3x3 -> 3x3 2D + 3 1D)
            self.branch2_conv1 = P3DConv3d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
            self.branch2_bn1 = _make_norm3d(norm, mid_channels)
            # Pointwise Conv
            self.branch2_conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.branch2_bn2 = _make_norm3d(norm, mid_channels)
            # P3D Depthwise Conv
            self.branch2_conv3 = P3DConv3d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
            self.branch2_bn3 = _make_norm3d(norm, mid_channels)
            # Channel Attention
            if use_channel_attention:
                self.branch2_channel_attention = ChannelAttention3D(mid_channels, reduction=reduction)
            # Pointwise Conv
            self.branch2_conv4 = nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.branch2_bn4 = _make_norm3d(norm, mid_channels)
            self.branch2_activation = activation_fn
        else:
            # Stride=2: No split, both branches process full input
            # Branch 1: P3D DWConv stride=2 -> 1x1
            self.branch1_conv1 = P3DConv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
            self.branch1_bn1 = _make_norm3d(norm, in_channels)
            self.branch1_conv2 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
            self.branch1_bn2 = _make_norm3d(norm, out_channels // 2)
            self.branch1_activation = activation_fn
            
            # Branch 2: 1x1 -> P3D DWConv stride=2 -> Channel Attention -> 1x1
            self.branch2_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.branch2_bn1 = _make_norm3d(norm, in_channels)
            self.branch2_conv2 = P3DConv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
            self.branch2_bn2 = _make_norm3d(norm, in_channels)
            # Channel Attention
            if use_channel_attention:
                self.branch2_channel_attention = ChannelAttention3D(in_channels, reduction=reduction)
            # Pointwise Conv
            self.branch2_conv3 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
            self.branch2_bn3 = _make_norm3d(norm, out_channels // 2)
            self.branch2_activation = activation_fn
    
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
            out2 = self.branch2_conv2(out2)
            out2 = self.branch2_bn2(out2)
            out2 = self.branch2_activation(out2)
            out2 = self.branch2_conv3(out2)
            out2 = self.branch2_bn3(out2)
            # Channel Attention
            if self.use_channel_attention:
                out2 = self.branch2_channel_attention(out2)
            out2 = self.branch2_conv4(out2)
            out2 = self.branch2_bn4(out2)
            out2 = self.branch2_activation(out2)
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
            out2 = self.branch2_conv2(out2)
            out2 = self.branch2_bn2(out2)
            # Channel Attention
            if self.use_channel_attention:
                out2 = self.branch2_channel_attention(out2)
            out2 = self.branch2_conv3(out2)
            out2 = self.branch2_bn3(out2)
            out2 = self.branch2_activation(out2)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        
        # Channel Shuffle
        out = channel_shuffle_3d(out, groups=2)
        return out


class P3DDown3DShuffleNetV2(nn.Module):
    """P3D Downsampling using ShuffleNetV2 unit (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn',
                 use_channel_attention: bool = False, reduction: int = 16, activation: str = 'relu'):
        super().__init__()
        self.unit1 = P3DShuffleNetV2Unit3D(in_channels, out_channels, stride=2, norm=norm,
                                          use_channel_attention=False, activation=activation)
        self.unit2 = P3DShuffleNetV2Unit3D(out_channels, out_channels, stride=1, norm=norm,
                                          use_channel_attention=use_channel_attention, reduction=reduction, activation=activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


def _build_p3d_shufflenet_v2_extra_blocks(
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
        P3DShuffleNetV2Unit3D(
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


class P3DUp3DShuffleNetV2(nn.Module):
    """P3D Upsampling block"""
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

        self.unit1 = P3DShuffleNetV2Unit3D(
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
        self.unit2 = P3DShuffleNetV2Unit3D(
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


class CascadeShuffleNetV2UNet3D_P3D(nn.Module):
    """
    Cascade ShuffleNet V2 UNet with P3D (Pseudo-3D) convolutions.
    
    P3D 구조: 3D conv를 2D spatial conv + 1D depth conv로 분리
    - 메모리 효율적
    - 파라미터 수 감소
    - 96^3 -> 48^3 -> 24^3 -> 12^3 구조 유지
    """
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
        self.stem = P3DStem3x3(stem_in, channels["stem"], norm=self.norm, activation=activation)

        self.down2 = P3DDown3DShuffleNetV2(
            channels["stem"],
            channels["branch2"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        self.down3 = P3DDown3DShuffleNetV2(
            channels["branch2"],
            channels["branch3"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=2,
            activation=activation,
        )
        self.down3_extra = _build_p3d_shufflenet_v2_extra_blocks(
            channels["branch3"], 4, self.norm, True, 2, activation
        )

        fused_channels = channels["branch3"]
        expanded_channels = channels["down4"]

        # down4는 기존 MultiScaleDilatedDepthwise3D 사용 (P3D 변환 복잡)
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        self.down4_depth = MultiScaleDilatedDepthwise3D(
            expanded_channels, dilation_rates=[1, 2, 3], norm=self.norm, activation=activation
        )
        self.down4_compress = nn.Identity()

        # Upsampling blocks
        self.up1 = P3DUp3DShuffleNetV2(
            channels["down4"],
            skip_channels=fused_channels,
            out_channels=channels["up1"],
            norm=self.norm,
            reduction=2,
            activation=activation,
        )
        self.up2 = P3DUp3DShuffleNetV2(
            channels["up1"],
            skip_channels=channels["branch2"],
            out_channels=channels["up2"],
            norm=self.norm,
            reduction=8,
            activation=activation,
        )
        self.up3 = P3DUp3DShuffleNetV2(
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


def build_cascade_shufflenet_v2_unet3d_p3d(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2UNet3D_P3D:
    return CascadeShuffleNetV2UNet3D_P3D(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
    )


# ============================================================================
# MobileViT Variant (Stage 4 with MobileViT)
# ============================================================================

class CascadeShuffleNetV2UNet3D_MViT(nn.Module):
    """
    Cascade ShuffleNet V2 UNet with MobileViT at Stage 4.
    
    Stage 4의 MultiScaleDilatedDepthwise3D를 MobileViT3DBlockV3로 교체
    - 글로벌 컨텍스트 모델링 능력 향상
    - Vision Transformer의 self-attention 메커니즘 활용
    - 96^3 -> 48^3 -> 24^3 -> 12^3 구조 유지
    """
    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        norm: str = "bn",
        size: str = "s",
        include_coords: bool = True,
        # MobileViT specific parameters
        mvit_num_heads: int = 4,
        mvit_mlp_ratio: int = 2,
        mvit_patch_size: int = 2,
        mvit_num_layers: int = 2,
        mvit_attn_dropout: float = 0.0,
        mvit_ffn_dropout: float = 0.0,
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

        # Stage 4: MobileViT 블록으로 교체
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # MultiScaleDilatedDepthwise3D 대신 MobileViT3DBlockV3 사용
        self.down4_mvit = MobileViT3DBlockV3(
            channels=expanded_channels,
            hidden_dim=expanded_channels,
            num_heads=mvit_num_heads,
            mlp_ratio=mvit_mlp_ratio,
            norm=self.norm,
            patch_size=mvit_patch_size,
            num_transformer_layers=mvit_num_layers,
            attn_dropout=mvit_attn_dropout,
            ffn_dropout=mvit_ffn_dropout,
        )
        self.down4_compress = nn.Identity()

        # Upsampling blocks
        self.up1 = Up3DShuffleNetV2(
            channels["down4"],
            skip_channels=fused_channels,
            out_channels=channels["up1"],
            norm=self.norm,
            reduction=2,
            activation=activation,
        )
        self.up2 = Up3DShuffleNetV2(
            channels["up1"],
            skip_channels=channels["branch2"],
            out_channels=channels["up2"],
            norm=self.norm,
            reduction=8,
            activation=activation,
        )
        self.up3 = Up3DShuffleNetV2(
            channels["up2"],
            skip_channels=channels["stem"],
            out_channels=channels["out"],
            norm=self.norm,
            reduction=2,
            activation=activation,
        )
        self.outc = nn.Conv3d(channels["out"], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Input tensor (B, C, D, H, W)
            return_attention: If True, returns attention weights along with output
        
        Returns:
            If return_attention=False: output tensor (B, n_classes, D, H, W)
            If return_attention=True: (output, attention_dict) where attention_dict contains:
                - 'mvit_attn': List of attention weights from MobileViT transformer layers
                  Each element is (B, num_heads, num_patches, num_patches) or averaged
        """
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
        if return_attention:
            x4, mvit_attn = self.down4_mvit(x4, return_attn=True)  # MobileViT 블록에서 attention 반환
        else:
            x4 = self.down4_mvit(x4, return_attn=False)
        x4 = self.down4_compress(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x)
        
        if return_attention:
            attention_dict = {
                'mvit_attn': mvit_attn,  # List of attention weights from transformer layers
            }
            return output, attention_dict
        return output


class CascadeShuffleNetV2UNet3D_P3D_MViT(nn.Module):
    """
    Cascade ShuffleNet V2 UNet with P3D convolutions + MobileViT at Stage 4.
    
    P3D 구조: 3D conv를 2D spatial conv + 1D depth conv로 분리
    Stage 4: MobileViT3DBlockV3로 글로벌 컨텍스트 모델링
    - 메모리 효율적 (P3D)
    - 파라미터 수 감소 (P3D)
    - 글로벌 컨텍스트 모델링 (MobileViT)
    - 96^3 -> 48^3 -> 24^3 -> 12^3 구조 유지
    """
    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        norm: str = "bn",
        size: str = "s",
        include_coords: bool = True,
        # MobileViT specific parameters
        mvit_num_heads: int = 4,
        mvit_mlp_ratio: int = 2,
        mvit_patch_size: int = 2,
        mvit_num_layers: int = 2,
        mvit_attn_dropout: float = 0.0,
        mvit_ffn_dropout: float = 0.0,
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
        # P3D Stem
        self.stem = P3DStem3x3(stem_in, channels["stem"], norm=self.norm, activation=activation)

        # P3D Downsampling blocks
        self.down2 = P3DDown3DShuffleNetV2(
            channels["stem"],
            channels["branch2"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        self.down3 = P3DDown3DShuffleNetV2(
            channels["branch2"],
            channels["branch3"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=2,
            activation=activation,
        )
        self.down3_extra = _build_p3d_shufflenet_v2_extra_blocks(
            channels["branch3"], 4, self.norm, True, 2, activation
        )

        fused_channels = channels["branch3"]
        expanded_channels = channels["down4"]

        # Stage 4: MobileViT 블록으로 교체
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # MultiScaleDilatedDepthwise3D 대신 MobileViT3DBlockV3 사용
        self.down4_mvit = MobileViT3DBlockV3(
            channels=expanded_channels,
            hidden_dim=expanded_channels,
            num_heads=mvit_num_heads,
            mlp_ratio=mvit_mlp_ratio,
            norm=self.norm,
            patch_size=mvit_patch_size,
            num_transformer_layers=mvit_num_layers,
            attn_dropout=mvit_attn_dropout,
            ffn_dropout=mvit_ffn_dropout,
        )
        self.down4_compress = nn.Identity()

        # P3D Upsampling blocks
        self.up1 = P3DUp3DShuffleNetV2(
            channels["down4"],
            skip_channels=fused_channels,
            out_channels=channels["up1"],
            norm=self.norm,
            reduction=2,
            activation=activation,
        )
        self.up2 = P3DUp3DShuffleNetV2(
            channels["up1"],
            skip_channels=channels["branch2"],
            out_channels=channels["up2"],
            norm=self.norm,
            reduction=8,
            activation=activation,
        )
        self.up3 = P3DUp3DShuffleNetV2(
            channels["up2"],
            skip_channels=channels["stem"],
            out_channels=channels["out"],
            norm=self.norm,
            reduction=2,
            activation=activation,
        )
        self.outc = nn.Conv3d(channels["out"], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
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
        if return_attention:
            x4, mvit_attn_weights = self.down4_mvit(x4, return_attn=True)
            attention_dict = {'mvit_attn': mvit_attn_weights}
        else:
            x4 = self.down4_mvit(x4, return_attn=False)
            attention_dict = {}
        x4 = self.down4_compress(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        if return_attention:
            return logits, attention_dict
        return logits


def build_cascade_shufflenet_v2_unet3d_p3d_mvit(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
    mvit_num_heads: int = 4,
    mvit_mlp_ratio: int = 2,
    mvit_patch_size: int = 2,
    mvit_num_layers: int = 2,
    mvit_attn_dropout: float = 0.0,
    mvit_ffn_dropout: float = 0.0,
) -> CascadeShuffleNetV2UNet3D_P3D_MViT:
    return CascadeShuffleNetV2UNet3D_P3D_MViT(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
        mvit_num_heads=mvit_num_heads,
        mvit_mlp_ratio=mvit_mlp_ratio,
        mvit_patch_size=mvit_patch_size,
        mvit_num_layers=mvit_num_layers,
        mvit_attn_dropout=mvit_attn_dropout,
        mvit_ffn_dropout=mvit_ffn_dropout,
    )


def build_cascade_shufflenet_v2_unet3d_mvit(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
    mvit_num_heads: int = 4,
    mvit_mlp_ratio: int = 2,
    mvit_patch_size: int = 2,
    mvit_num_layers: int = 2,
    mvit_attn_dropout: float = 0.0,
    mvit_ffn_dropout: float = 0.0,
) -> CascadeShuffleNetV2UNet3D_MViT:
    return CascadeShuffleNetV2UNet3D_MViT(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
        mvit_num_heads=mvit_num_heads,
        mvit_mlp_ratio=mvit_mlp_ratio,
        mvit_patch_size=mvit_patch_size,
        mvit_num_layers=mvit_num_layers,
        mvit_attn_dropout=mvit_attn_dropout,
        mvit_ffn_dropout=mvit_ffn_dropout,
    )


__all__ = [
    "CascadeShuffleNetV2UNet3D",
    "build_cascade_shufflenet_v2_unet3d",
    "CascadeShuffleNetV2UNet3D_P3D",
    "build_cascade_shufflenet_v2_unet3d_p3d",
    "CascadeShuffleNetV2UNet3D_MViT",
    "build_cascade_shufflenet_v2_unet3d_mvit",
    "CascadeShuffleNetV2UNet3D_P3D_MViT",
    "build_cascade_shufflenet_v2_unet3d_p3d_mvit",
]




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
    ShuffleNetV2Unit3D_LKAHybrid,
)
from models.modules.lka_hybrid_modules import LKAHybridCBAM3D
from models.modules.mvit_modules import MobileViT3DBlockV3
from models.model_3d_unet import _make_norm3d, _make_activation


class DepthwiseSeparableConv3D(nn.Module):
    """
    3D Depthwise Separable Convolution
    
    Depthwise conv (각 채널별 독립 conv) + Pointwise conv (1x1x1 채널 혼합)
    채널 간 정보 교환을 위해 pointwise conv 포함
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = None,
        norm: str = "bn",
        activation: str = "relu",
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        # Depthwise conv: 각 채널별 독립 conv (groups=channels)
        self.depthwise = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=channels,  # Depthwise: 각 채널별 독립 conv
                bias=False,
            ),
            _make_norm3d(norm, channels),
            _make_activation(activation, inplace=True),
        )
        
        # Pointwise conv: 1x1x1 채널 혼합 (채널 간 정보 교환)
        self.pointwise = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_norm3d(norm, channels),
            _make_activation(activation, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # Residual connection
        x = self.depthwise(x)
        x = self.pointwise(x)
        return identity + x  # Residual connection 추가


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


def _build_shufflenet_v2_lka_extra_blocks(
    channels: int,
    num_blocks: int,
    norm: str,
    reduction: int,
    activation: str,
    drop_path_rates: list[float] | None = None,
) -> nn.Module:
    """ShuffleNetV2 + Hybrid LKA extra blocks (stride=1).

    Stage3/Stage4에서 해상도는 유지하고, 채널만 유지한 채로
    ShuffleNetV2Unit3D_LKAHybrid 블록을 반복적으로 적용합니다.
    """
    if num_blocks <= 0:
        return nn.Identity()

    layers: list[nn.Module] = []
    for idx in range(num_blocks):
        drop_rate = 0.0
        if drop_path_rates is not None and idx < len(drop_path_rates):
            drop_rate = float(drop_path_rates[idx])
        layers.append(
            ShuffleNetV2Unit3D_LKAHybrid(
                channels,
                channels,
                stride=1,
                norm=norm,
                reduction=reduction,
                activation=activation,
                drop_path=drop_rate,
            )
        )
    return nn.Sequential(*layers)


class Stem3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn", activation: str = "relu"):
        super().__init__()
        # 첫 번째 conv: in_channels -> out_channels (채널 변경, residual 불가)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )
        # 두 번째 conv: out_channels -> out_channels (채널 동일, residual 가능)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # in_channels -> out_channels
        identity = x  # Residual connection을 위한 identity 저장
        x = self.conv2(x)  # out_channels -> out_channels
        return identity + x  # Residual connection 추가


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
        # Stage 4: Sequential Dilated conv만 사용
        self.down4_dilated = MultiScaleDilatedDepthwise3D(
            channels=expanded_channels,
            dilation_rates=[1, 2, 5],
            kernel_size=3,
            norm=self.norm,
            activation=activation,
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
        x4 = self.down4_dilated(x4)  # Dilated conv로 넓은 receptive field (내부에 residual 있음)
        x4 = self.down4_compress(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


class Down3DShuffleNetV2_LKAHybrid(nn.Module):
    """Downsampling using ShuffleNetV2Unit3D_LKAHybrid (stride=2 then stride=1).

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
        self.unit1 = ShuffleNetV2Unit3D_LKAHybrid(
            in_channels,
            out_channels,
            stride=2,
            norm=norm,
            reduction=reduction,
            activation=activation,
        )
        # Second unit: stride=1 for feature refinement
        self.unit2 = ShuffleNetV2Unit3D_LKAHybrid(
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
        # 첫 번째 P3DConv: in_channels -> out_channels (채널 변경, residual 불가)
        self.conv1 = nn.Sequential(
            P3DConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )
        # 두 번째 P3DConv: out_channels -> out_channels (채널 동일, residual 가능)
        self.conv2 = nn.Sequential(
            P3DConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # in_channels -> out_channels
        identity = x  # Residual connection을 위한 identity 저장
        x = self.conv2(x)  # out_channels -> out_channels
        return identity + x  # Residual connection 추가


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

        # Stage 4: Sequential Dilated conv만 사용
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        self.down4_dilated = MultiScaleDilatedDepthwise3D(
            channels=expanded_channels,
            dilation_rates=[1, 2, 5],
            kernel_size=3,
            norm=self.norm,
            activation=activation,
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
        x4 = self.down4_dilated(x4)  # Dilated conv로 넓은 receptive field (내부에 residual 있음)
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
# Large Kernel (7x7x7 x2) Variants
# ============================================================================

class P3DDepthwiseSeparableConv3D(nn.Module):
    """
    P3D Depthwise Separable Convolution
    
    3D Depthwise Separable Conv를 P3D 방식으로 구현
    - Depthwise: P3DConv (2D spatial + 1D depth)
    - Pointwise: 1x1x1 conv (채널 혼합)
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = None,
        norm: str = "bn",
        activation: str = "relu",
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        # P3D Depthwise conv: 2D spatial + 1D depth
        self.depthwise_2d = nn.Conv2d(
            channels, channels, kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride), padding=(padding, padding),
            groups=channels, bias=False
        )
        self.depthwise_1d = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=channels, bias=False
        )
        self.dw_bn = _make_norm3d(norm, channels)
        self.dw_act = _make_activation(activation, inplace=True)
        
        # Pointwise conv: 1x1x1 채널 혼합
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, channels),
            _make_activation(activation, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # P3D Depthwise: 2D spatial conv
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        x_2d = self.depthwise_2d(x_2d)  # (B*D, C, H', W')
        _, _, H_out, W_out = x_2d.shape
        x_3d = x_2d.view(B, D, C, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()
        
        # P3D Depthwise: 1D depth conv
        x_1d = x_3d.permute(0, 3, 4, 1, 2).contiguous().view(B * H_out * W_out, C, D)
        x_1d = self.depthwise_1d(x_1d)  # (B*H*W, C, D')
        _, _, D_out = x_1d.shape
        x_3d = x_1d.view(B, H_out, W_out, C, D_out).permute(0, 3, 4, 1, 2).contiguous()
        
        # Norm & Activation
        x_3d = self.dw_bn(x_3d)
        x_3d = self.dw_act(x_3d)
        
        # Pointwise conv
        identity = x  # Residual connection
        x_3d = self.pointwise(x_3d)
        return identity + x_3d  # Residual connection


# ============================================================================
# Large Kernel (7x7x7 x2) Variants
# ============================================================================

class CascadeShuffleNetV2UNet3D_LK(nn.Module):
    """
    Cascade ShuffleNet V2 UNet with Large Kernel (7x7x7 x2) at Stage 4.
    
    Stage 4에서 7x7x7 depthwise separable conv를 2개 순차적으로 사용
    - 각 conv는 residual connection 포함
    - Dilated conv보다 비싸지만 더 dense한 receptive field
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

        # Stage 4: 7x7x7 depthwise separable conv 2개 (residual 포함)
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # 첫 번째 7x7x7 conv
        self.down4_lk1 = DepthwiseSeparableConv3D(
            channels=expanded_channels,
            kernel_size=7,
            stride=1,
            norm=self.norm,
            activation=activation,
        )
        # 두 번째 7x7x7 conv (residual connection 포함)
        self.down4_lk2 = DepthwiseSeparableConv3D(
            channels=expanded_channels,
            kernel_size=7,
            stride=1,
            norm=self.norm,
            activation=activation,
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
        x4 = self.down4_lk1(x4)  # 첫 번째 7x7x7 conv (residual 포함)
        x4 = self.down4_lk2(x4)  # 두 번째 7x7x7 conv (residual 포함)
        x4 = self.down4_compress(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


class CascadeShuffleNetV2UNet3D_LKAHybrid(nn.Module):
    """
    Cascade ShuffleNet V2 UNet with Hybrid LKA at Stage 3 and Stage 4.

    - Stage 3: Down3DShuffleNetV2_LKAHybrid 사용 + ShuffleNetV2Unit3D_LKAHybrid extra blocks
    - Stage 4: 1x1x1 conv 확장 후 ShuffleNetV2Unit3D_LKAHybrid 블록 적용
    - Decoder 및 Stem 구조는 기본 CascadeShuffleNetV2UNet3D와 동일
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
        self.stem = Stem3x3(stem_in, channels["stem"], norm=self.norm, activation=activation)

        # Stage 2: 기본 ShuffleNetV2 Down 블록 사용
        self.down2 = Down3DShuffleNetV2(
            channels["stem"],
            channels["branch2"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        # Stage 3: Hybrid LKA Down 블록 사용
        self.down3 = Down3DShuffleNetV2_LKAHybrid(
            channels["branch2"],
            channels["branch3"],
            norm=self.norm,
            reduction=4,
            activation=activation,
        )
        # Stage 3 extra blocks: Hybrid LKA ShuffleNetV2 units
        self.down3_extra = _build_shufflenet_v2_lka_extra_blocks(
            channels["branch3"],
            4,
            self.norm,
            reduction=4,
            activation=activation,
            # Stage3에서만 Stochastic Depth(Linear schedule) 적용
            drop_path_rates=[0.0, 0.05, 0.1, 0.15],
        )

        fused_channels = channels["branch3"]
        expanded_channels = channels["down4"]

        # Stage 4: 24^3 -> 12^3 다운샘플 + 채널 확장 후 Hybrid LKA 블록 2번 적용
        # Expand: 채널 확장 (해상도는 24^3 유지)
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # 첫 번째 LKA: stride=2로 24^3 -> 12^3 다운샘플 + 넓은 ERF 확보
        self.down4_lka1 = LKAHybridCBAM3D(
            channels=expanded_channels,
            reduction=4,
            norm=self.norm,
            activation=activation,
            use_residual=True,
            stride=2,
        )
        # 두 번째 LKA: stride=1로 12^3 유지, 맵 전체를 한 번 더 덮어 ERF 강화
        self.down4_lka2 = LKAHybridCBAM3D(
            channels=expanded_channels,
            reduction=4,
            norm=self.norm,
            activation=activation,
            use_residual=True,
            stride=1,
        )
        self.down4_compress = nn.Identity()

        # Upsampling blocks (기본 모델과 동일)
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

        # Stage 4: 24^3 -> 12^3 다운샘플 + LKA 2번 적용 (맵 전체를 충분히 덮도록)
        x4 = self.down4_expand(x3)
        x4 = self.down4_lka1(x4)
        x4 = self.down4_lka2(x4)
        x4 = self.down4_compress(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def build_cascade_shufflenet_v2_unet3d_lka_hybrid(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2UNet3D_LKAHybrid:
    return CascadeShuffleNetV2UNet3D_LKAHybrid(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
    )


def build_cascade_shufflenet_v2_unet3d_lk(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2UNet3D_LK:
    return CascadeShuffleNetV2UNet3D_LK(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
    )


class CascadeShuffleNetV2UNet3D_P3D_LK(nn.Module):
    """
    Cascade ShuffleNet V2 UNet with P3D convolutions + Large Kernel (7x7x7 x2) at Stage 4.
    
    P3D 구조: 3D conv를 2D spatial conv + 1D depth conv로 분리
    Stage 4: P3D 방식의 7x7x7 depthwise separable conv 2개 사용
    - 메모리 효율적 (P3D)
    - 파라미터 수 감소 (P3D)
    - 각 conv는 residual connection 포함
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

        # Stage 4: P3D 방식의 7x7x7 depthwise separable conv 2개 (residual 포함)
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # 첫 번째 P3D 7x7x7 conv
        self.down4_lk1 = P3DDepthwiseSeparableConv3D(
            channels=expanded_channels,
            kernel_size=7,
            stride=1,
            norm=self.norm,
            activation=activation,
        )
        # 두 번째 P3D 7x7x7 conv (residual connection 포함)
        self.down4_lk2 = P3DDepthwiseSeparableConv3D(
            channels=expanded_channels,
            kernel_size=7,
            stride=1,
            norm=self.norm,
            activation=activation,
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
        x4 = self.down4_lk1(x4)  # 첫 번째 P3D 7x7x7 conv (residual 포함)
        x4 = self.down4_lk2(x4)  # 두 번째 P3D 7x7x7 conv (residual 포함)
        x4 = self.down4_compress(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def build_cascade_shufflenet_v2_unet3d_p3d_lk(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2UNet3D_P3D_LK:
    return CascadeShuffleNetV2UNet3D_P3D_LK(
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
    "CascadeShuffleNetV2UNet3D_LK",
    "build_cascade_shufflenet_v2_unet3d_lk",
    "CascadeShuffleNetV2UNet3D_P3D_LK",
    "build_cascade_shufflenet_v2_unet3d_p3d_lk",
    "CascadeShuffleNetV2UNet3D_MViT",
    "build_cascade_shufflenet_v2_unet3d_mvit",
    "CascadeShuffleNetV2UNet3D_P3D_MViT",
    "build_cascade_shufflenet_v2_unet3d_p3d_mvit",
]




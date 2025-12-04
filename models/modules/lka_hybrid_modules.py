"""
LKA-inspired hybrid convolution modules for 3D

기본 VAN의 LKA는 입력 특징에서 attention map을 생성한 뒤 입력에 곱해주는 구조이지만,
여기서는 **LKA의 커널 분해 구조만 차용**하고, 실제 attention 역할은 CBAM(ChannelAttention3D)에
위임하는 **hybrid 구조**를 제공합니다.

- Dense branch  : 3x3x3 depthwise conv (local dense receptive field)
- Sparse branch : 3x3x3 depthwise conv with dilation=3 (sparse, 넓은 receptive field)
- Mixer         : 1x1x1 pointwise conv (channel mixing)

Sparse branch의 3x3x3 dilated conv(rate=3)는 단독으로 7x7x7의 유효 커널을 갖습니다
(\(k_{\mathrm{eff}} = 3 + (3-1)\times(3-1) = 7\)). 이 위에 dense 3x3x3를 쌓으면
전체 ERF는 9x9x9에 가까워지지만, 구조적으로는 LKA의 dense + sparse + mixer 패턴을 따릅니다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..model_3d_unet import _make_norm3d, _make_activation
from .cbam_modules import ChannelAttention3D


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth / DropPath 구현.

    Conv 블록 자체를 부분적으로 무작위로 꺼서(reg) 과적합을 완화합니다.
    - per-sample drop (batch 차원 기준)
    - residual branch에만 적용하는 것을 가정합니다.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # (B, 1, 1, 1, 1) 형태로 broadcast
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # scale to keep expectation
    x = x / keep_prob * random_tensor
    return x


class LKAKernel3D(nn.Module):
    """
    3D LKA-style kernel (dense + sparse + mixer), attention 없이 순수 커널만 구현.

    - Dense:  3x3x3 depthwise conv
    - Sparse: 3x3x3 depthwise conv with dilation=3
    - Mixer:  1x1x1 pointwise conv (채널 혼합)

    두 depthwise conv의 조합으로 이론적 수용 영역은 7x7x7에 해당합니다.

    Args:
        channels: 입력/출력 채널 수 (depthwise conv이므로 동일)
        norm: 정규화 타입 ('bn', 'in', 'gn')
        activation: 활성화 함수 타입 ('relu', 'hardswish', 'gelu')
    """

    def __init__(
        self,
        channels: int,
        norm: str = "bn",
        activation: str = "relu",
        stride_dense: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.norm = norm or "bn"

        # Dense branch: 3x3x3 depthwise conv (옵션으로 stride 조절 가능)
        self.conv_dense = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                stride=stride_dense,
                padding=1,
                groups=channels,  # depthwise
                bias=False,
            ),
            _make_norm3d(self.norm, channels),
            _make_activation(activation, inplace=True),
        )

        # Sparse branch: 3x3x3 depthwise conv with dilation=3 (ERF 7x7x7)
        self.conv_sparse = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
                groups=channels,  # depthwise
                bias=False,
            ),
            _make_norm3d(self.norm, channels),
            _make_activation(activation, inplace=True),
        )

        # Mixer: 1x1x1 pointwise conv (채널 혼합)
        self.mixer = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_norm3d(self.norm, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)

        Returns:
            LKA-style 커널이 적용된 출력 텐서 (B, C, D, H, W)
        """
        out = self.conv_dense(x)
        out = self.conv_sparse(out)
        out = self.mixer(out)
        return out


class LKAHybridCBAM3D(nn.Module):
    """
    3D LKA Hybrid Block with CBAM Channel Attention.

    - LKA 커널 (dense + sparse + mixer)을 통해 7x7x7 ERF를 갖는 특징을 추출
    - 그 결과에 대해 CBAM의 ChannelAttention3D를 적용하여 채널별 중요도를 재조정
    - 선택적으로 residual connection(x + F(x))을 적용 가능

    VAN의 LKA처럼 입력에 직접 attention을 곱하지 않고,
    LKA는 **커널 구조**, CBAM은 **attention 역할**을 담당하도록 분리한 구조입니다.

    Args:
        channels: 입력/출력 채널 수
        reduction: CBAM 채널 축소 비율
        norm: 정규화 타입 ('bn', 'in', 'gn')
        activation: 활성화 함수 타입 ('relu', 'hardswish', 'gelu')
        use_residual: 입력과 출력 사이에 residual connection을 사용할지 여부
        drop_path_rate: Stochastic depth (depth 앙상블) 비율
        drop_channel_rate: Spatial dropout (width 앙상블) 비율
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        norm: str = "bn",
        activation: str = "relu",
        use_residual: bool = True,
        stride: int = 1,
        drop_path_rate: float = 0.0,
        drop_channel_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual

        # stride > 1 인 경우, dense 3x3x3 depthwise conv에서 다운샘플링을 수행
        self.stride = stride

        self.lka_kernel = LKAKernel3D(
            channels=channels,
            norm=norm,
            activation=activation,
            stride_dense=stride,
        )
        self.channel_attention = ChannelAttention3D(
            channels=channels,
            reduction=reduction,
        )

        # Stochastic depth 비율 (depth 앙상블, 0이면 사용 안 함)
        self.drop_path_rate = float(drop_path_rate)
        
        # Spatial dropout (width 앙상블, CNN에 적합한 채널 단위 dropout)
        if drop_channel_rate > 0:
            self.drop_channel = nn.Dropout3d(drop_channel_rate)
        else:
            self.drop_channel = nn.Identity()

        # stride > 1인 경우 residual connection을 유지하기 위한 projection 경로
        # ShuffleNetV2에서 stride=2일 때 identity branch에 projection을 두는 것과 유사하게,
        # 여기서는 간단한 1x1x1 conv(stride=stride) + norm으로 spatial 크기를 맞춰줍니다.
        if self.use_residual and self.stride > 1:
            self.proj = nn.Sequential(
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                _make_norm3d(norm, channels),
            )
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)

        Returns:
            LKA 커널 + CBAM Channel Attention (+ optional residual)이 적용된 출력
        """
        out = self.lka_kernel(x)          # 커널 구조로 특징 추출 (7x7x7 ERF + stride에 따른 downsample)
        out = self.channel_attention(out) # 채널 어텐션 적용
        
        # Spatial dropout (width 앙상블) - CBAM 이후 적용
        # CNN에 적합한 채널 단위 dropout으로 width 앙상블 효과 제공
        out = self.drop_channel(out)

        # Projection 있는 정석 residual:
        # - stride == 1: out + x
        # - stride > 1: out + proj(x)  (spatial 크기를 맞춘 identity)
        if self.use_residual:
            identity = x
            if self.proj is not None:
                identity = self.proj(identity)
            # Stochastic depth (DropPath)를 residual branch에만 적용 (depth 앙상블)
            out = identity + drop_path(out, self.drop_path_rate, self.training)
        return out


__all__ = [
    "LKAKernel3D",
    "LKAHybridCBAM3D",
    "drop_path",
]



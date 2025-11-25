"""
Dual-Branch UNet with ShuffleNet V2
ShuffleNet V2 기반 Dual-Branch UNet 모델

- dualbranch_shufflenet_v2: ShuffleNet V2 기반 Dual-Branch UNet with Channel Attention
  - Channel Attention 사용으로 더 효과적인 특징 재보정
  - Branch별 Channel Attention: 각 모달리티별 특징 추출 후 채널 어텐션 적용
  - Decoder Channel Attention: Skip connection과 decoder feature 융합 시 어텐션 적용
"""

import torch
import torch.nn as nn

from .modules.shufflenet_modules import ShuffleNetV2Unit3D, Down3DShuffleNetV2, channel_shuffle_3d, MultiScaleDilatedDepthwise3D
from .modules.cbam_modules import CBAM3D
from .model_3d_unet import Up3D, OutConv3D, _make_norm3d, DoubleConv3D, _make_activation
from .channel_configs import get_dualbranch_channels_stage3_fused, get_activation_type
import torch.nn.functional as F


# Helper to build extra ShuffleNetV2Unit stacks
def _build_shufflenet_v2_extra_blocks(channels, num_blocks, norm, use_channel_attention, reduction, activation):
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


# ============================================================================
# Building Blocks
# ============================================================================

class Up3DShuffleNetV2(nn.Module):
    """3D Upsampling 블록 with ShuffleNet V2 (인코더와 대칭 구조)
    
    인코더의 Down3DShuffleNetV2와 대칭적으로 구성:
    - Upsampling (bilinear 또는 transpose conv)
    - Skip connection과 concat
    - ShuffleNet V2 Unit 2개 (stride=1, 인코더와 동일)
    - Channel Attention 지원 (선택적)
    
    Args:
        reduction: Channel attention의 reduction ratio (기본값: 16, up1에서는 8로 줄여서 포화 방지)
        use_cbam: Channel Attention 사용 여부 (기본값: True)
        activation: 활성화 함수 타입 ('relu', 'hardswish', 'gelu')
    """
    def __init__(self, in_channels, out_channels, bilinear=True, norm: str = 'bn', skip_channels=None, 
                 use_cbam: bool = True, reduction: int = 16, target_skip_channels: int = None, 
                 activation: str = 'relu', keep_channels: bool = False):
        super().__init__()
        self.bilinear = bilinear
        self.use_cbam = use_cbam
        
        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            if skip_channels is None:
                skip_channels = in_channels // 2
            # Skip connection 압축: target_skip_channels가 지정되고 skip_channels가 더 크면 1x1 conv로 압축
            if target_skip_channels is not None and skip_channels > target_skip_channels:
                self.skip_compress = nn.Sequential(
                    nn.Conv3d(skip_channels, target_skip_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    _make_norm3d(norm, target_skip_channels),
                    _make_activation(activation, inplace=True),
                )
                skip_channels = target_skip_channels
            else:
                self.skip_compress = None
            total_channels = in_channels + skip_channels
        else:
            # keep_channels=True일 때: ConvTranspose3d가 채널을 유지 (skip과 1:1 매칭)
            if keep_channels:
                transpose_out_channels = in_channels
            else:
                transpose_out_channels = in_channels // 2
            self.up = nn.ConvTranspose3d(in_channels, transpose_out_channels, kernel_size=2, stride=2)
            if skip_channels is None:
                skip_channels = transpose_out_channels
            # Skip connection 압축: target_skip_channels가 지정되고 skip_channels가 더 크면 1x1 conv로 압축
            if target_skip_channels is not None and skip_channels > target_skip_channels:
                self.skip_compress = nn.Sequential(
                    nn.Conv3d(skip_channels, target_skip_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    _make_norm3d(norm, target_skip_channels),
                    _make_activation(activation, inplace=True),
                )
                skip_channels = target_skip_channels
            else:
                self.skip_compress = None
            total_channels = transpose_out_channels + skip_channels
        
        # ShuffleNet V2 Units (인코더와 대칭: 2개의 stride=1 unit)
        # ShuffleNetV2Unit3D는 stride=1일 때 in_channels == out_channels여야 함
        # 따라서 unit1과 unit2는 채널을 유지하고, 중간에 1x1 conv로 채널을 조정
        # Channel Attention은 각 unit 내부의 채널 압축 직전에 적용됨
        if total_channels != out_channels:
            # 점진적 압축: total_channels -> total_channels (unit1 with CA) -> out_channels (1x1 conv) -> out_channels (unit2 with CA)
            self.unit1 = ShuffleNetV2Unit3D(total_channels, total_channels, stride=1, norm=norm,
                                            use_channel_attention=self.use_cbam, reduction=reduction, activation=activation)
            self.channel_adjust = nn.Sequential(
                nn.Conv3d(total_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
                _make_activation(activation, inplace=True),
            )
            self.unit2 = ShuffleNetV2Unit3D(out_channels, out_channels, stride=1, norm=norm,
                                            use_channel_attention=self.use_cbam, reduction=reduction, activation=activation)
        else:
            # 채널 수가 같으면 유지
            self.unit1 = ShuffleNetV2Unit3D(total_channels, total_channels, stride=1, norm=norm,
                                            use_channel_attention=self.use_cbam, reduction=reduction, activation=activation)
            self.channel_adjust = None
            self.unit2 = ShuffleNetV2Unit3D(total_channels, out_channels, stride=1, norm=norm,
                                            use_channel_attention=self.use_cbam, reduction=reduction, activation=activation)
    
    def forward(self, x1, x2):
        # Upsampling
        x1_up = self.up(x1)
        
        # 크기 맞추기
        diffZ = x2.size()[2] - x1_up.size()[2]
        diffY = x2.size()[3] - x1_up.size()[3]
        diffX = x2.size()[4] - x1_up.size()[4]
        
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
        
        # Skip connection 압축 (필요한 경우)
        if self.skip_compress is not None:
            x2 = self.skip_compress(x2)
        
        # Concat
        x = torch.cat([x2, x1_up], dim=1)
        
        # ShuffleNet V2 Units (인코더와 대칭, 점진적 압축)
        # Channel Attention은 각 unit 내부의 채널 압축 직전에 적용됨
        x = self.unit1(x)
        if self.channel_adjust is not None:
            x = self.channel_adjust(x)
        x = self.unit2(x)
        
        return x


class Stem3x3(nn.Module):
    """Stem using DoubleConv3D (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', activation: str = 'relu'):
        super().__init__()
        # DoubleConv3D는 내부적으로 ReLU를 사용하므로, 여기서는 직접 구현
        mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, mid_channels),
            _make_activation(activation, inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            _make_activation(activation, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================================
# Base Models
# ============================================================================

class DualBranchUNet3D_ShuffleNetV2_Stage3Fused(nn.Module):
    """Dual-branch UNet with ShuffleNet V2 backbone - Stage 3 fused at down4 (4-stage structure)
    
    - Stage 1-3: Dual-branch structure (each branch independently)
    - Stage 4: Single branch bottleneck (down4, Stage 3 outputs are concatenated before down4)
    - Channel Attention blocks applied at branch outputs
    - Channel Attention blocks applied in decoder for effective feature fusion
    
    Structure: stem → branch2 (dual) → branch3 (dual) → down4 (fused bottleneck) → out
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, 
                 size: str = 's', half_decoder: bool = False, fixed_decoder: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.size = size
        self.half_decoder = half_decoder
        self.fixed_decoder = fixed_decoder
        
        # Get activation type based on model size
        activation = get_activation_type(size)
        
        # Get channel configuration (Stage 3 fused at down4, 4-stage structure)
        channels = get_dualbranch_channels_stage3_fused(size, half_decoder=half_decoder, fixed_decoder=fixed_decoder)
        
        # Stage 1 stems (simple 3x3 conv)
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm, activation=activation)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm, activation=activation)
        
        # Stage 2 branches (both use ShuffleNet V2)
        # Channel Attention은 블록 내부(채널 압축 직전)에 적용
        self.branch_flair = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm,
                                               use_channel_attention=True, reduction=8, activation=activation)  # 어텐션 활성화: reduction=8
        self.branch_t1ce = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm,
                                              use_channel_attention=True, reduction=2, activation=activation)  # 포화 완화: reduction=2
        
        # Stage 3 branches (both use ShuffleNet V2)
        self.branch_flair3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm,
                                                use_channel_attention=True, reduction=2, activation=activation)  # 포화 완화: reduction=2
        self.branch_t1ce3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm,
                                               use_channel_attention=True, reduction=2, activation=activation)  # 포화 완화: reduction=2
        # Stage 3 extra blocks (추가 ShuffleNet V2 units)
        self.branch_flair3_extra = _build_shufflenet_v2_extra_blocks(channels['branch3'], 4, self.norm, True, 2, activation)
        self.branch_t1ce3_extra = _build_shufflenet_v2_extra_blocks(channels['branch3'], 4, self.norm, True, 2, activation)
        
        # Stage 4 fused branch with Inverted Bottleneck (input: branch3*2, output: down4)
        # Inverted Bottleneck: 입력 -> 2배 확장 -> down4 채널로 압축
        factor = 2 if self.bilinear else 1
        fused_channels = channels['branch3'] * 2
        expanded_channels = fused_channels * 2  # Inverted bottleneck: 2배 확장
        
        # Inverted Bottleneck 구조
        # 1. 확장: 1x1 conv로 채널 2배 확장
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # 2. Multi-Scale Dilated Depth-wise convolution (dilation=[1,2,3]로 다양한 수용 영역 포착)
        # Stage 4 해상도가 16x16x16이므로 dilation=5는 ERF가 과도하게 커짐 (17x17x17)
        self.down4_depth = MultiScaleDilatedDepthwise3D(expanded_channels, dilation_rates=[1, 2, 3], norm=self.norm, activation=activation)
        # 3. 압축: 1x1 conv로 down4 채널로 압축
        self.down4_compress = nn.Sequential(
            nn.Conv3d(expanded_channels, channels['down4'], kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(self.norm, channels['down4']),
            _make_activation(activation, inplace=True),
        )
        
        # Decoder: Up3DShuffleNetV2 사용 (인코더와 대칭: ShuffleNet V2 기반)
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        # up1.cbam은 reduction=2로 설정하여 sigmoid 포화 완화 (더 많은 MLP 파라미터로 세밀한 조정)
        # half_decoder=True 또는 fixed_decoder=True일 때는 명시적으로 정의된 디코더 채널 사용
        if fixed_decoder or half_decoder:
            # 디코더 채널이 명시적으로 정의된 경우
            up1_out = channels.get('up1', channels['branch3'] // factor)
            up2_out = channels.get('up2', channels['branch2'] // factor)
            up3_out = channels.get('up3', channels['out'])
        else:
            # bilinear=False일 때: ConvTranspose3d로 절반으로 줄인 후 skip과 concat하므로
            # 디코더 출력 채널은 인코더 stage 채널과 동일 (factor 적용 안 함)
            up1_out = channels['branch3']  # branch3 채널과 동일
            up2_out = channels['branch2']  # branch2 채널과 동일
            up3_out = channels['out']      # out 채널과 동일
        
        # half_decoder=True 또는 fixed_decoder=True일 때 skip connection을 디코더 출력 채널로 압축
        if fixed_decoder or half_decoder:
            # Skip connection을 디코더 출력 채널과 동일하게 압축
            target_skip1 = up1_out
            target_skip2 = up2_out
            target_skip3 = up3_out
        else:
            # 기존 방식: skip connection 압축 없음
            target_skip1 = target_skip2 = target_skip3 = None
        
        if self.bilinear:
            self.up1 = Up3DShuffleNetV2(channels['down4'], up1_out, self.bilinear, norm=self.norm, skip_channels=fused_channels, use_cbam=True, reduction=2, target_skip_channels=target_skip1, activation=activation)  # 포화 완화: reduction=2
            self.up2 = Up3DShuffleNetV2(up1_out, up2_out, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2, use_cbam=True, reduction=8, target_skip_channels=target_skip2, activation=activation, keep_channels=fixed_decoder)  # 어텐션 활성화: reduction=8
            self.up3 = Up3DShuffleNetV2(up2_out, up3_out, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2, use_cbam=True, reduction=2, target_skip_channels=target_skip3, activation=activation, keep_channels=fixed_decoder)  # 포화 완화: reduction=2
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3DShuffleNetV2(channels['down4'], up1_out, self.bilinear, norm=self.norm, skip_channels=fused_channels, use_cbam=True, reduction=2, target_skip_channels=target_skip1, activation=activation)  # 포화 완화: reduction=2
            self.up2 = Up3DShuffleNetV2(up1_out, up2_out, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2, use_cbam=True, reduction=8, target_skip_channels=target_skip2, activation=activation, keep_channels=fixed_decoder)  # 어텐션 활성화: reduction=8
            self.up3 = Up3DShuffleNetV2(up2_out, up3_out, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2, use_cbam=True, reduction=2, target_skip_channels=target_skip3, activation=activation, keep_channels=fixed_decoder)  # 포화 완화: reduction=2
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: Stems
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)
        
        # Stage 2: Branches (Channel Attention은 블록 내부에 적용됨)
        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)
        
        # Stage 3: Branches (Channel Attention은 블록 내부에 적용됨)
        b2_flair = self.branch_flair3(b_flair)
        b2_flair = self.branch_flair3_extra(b2_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        b2_t1ce = self.branch_t1ce3_extra(b2_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 4: Fused branch (Inverted Bottleneck)
        x4 = self.down4_expand(x3)  # 확장: fused_channels -> expanded_channels (2배)
        x4 = self.down4_depth(x4)   # Depth-wise: Multi-Scale Dilated Depthwise3D
        x4 = self.down4_compress(x4)  # 압축: expanded_channels -> down4
        
        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


# Convenience classes for DualBranchUNet3D_ShuffleNetV2_Stage3Fused
class DualBranchUNet3D_ShuffleNetV2_Stage3Fused_XS(DualBranchUNet3D_ShuffleNetV2_Stage3Fused):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class DualBranchUNet3D_ShuffleNetV2_Stage3Fused_Small(DualBranchUNet3D_ShuffleNetV2_Stage3Fused):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class DualBranchUNet3D_ShuffleNetV2_Stage3Fused_Medium(DualBranchUNet3D_ShuffleNetV2_Stage3Fused):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class DualBranchUNet3D_ShuffleNetV2_Stage3Fused_Large(DualBranchUNet3D_ShuffleNetV2_Stage3Fused):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')


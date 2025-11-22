"""
Dual-Branch UNet with ShuffleNet V1
ShuffleNet V1 기반 Dual-Branch UNet 모델

- dualbranch_shufflenet: ShuffleNet V1 기반 Dual-Branch UNet with SE blocks
"""

import torch
import torch.nn as nn

from .modules.shufflenet_modules import ShuffleNetV1Unit3D, Down3DShuffleNetV1
from .modules.se_modules import SEBlock3D
from .model_3d_unet import Up3D, OutConv3D, _make_norm3d, DoubleConv3D
from .channel_configs import get_dualbranch_channels_stage4_fused
import torch.nn.functional as F


# ============================================================================
# Building Blocks
# ============================================================================

class Up3DSE(nn.Module):
    """3D Upsampling 블록 with SE Block
    
    Up3D를 래핑하여 업샘플 후 concat한 다음 SE 블록을 적용합니다.
    SE 블록을 DoubleConv 전에 적용하여 중요한 채널에 집중합니다.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, norm: str = 'bn', skip_channels=None, use_se: bool = True):
        super().__init__()
        self.up3d = Up3D(in_channels, out_channels, bilinear, norm, skip_channels)
        self.use_se = use_se
        
        if self.use_se:
            # Up3D 내부의 DoubleConv 전에 SE를 적용해야 하므로,
            # concat 후의 채널 수를 계산해야 함
            if bilinear:
                if skip_channels is None:
                    skip_channels = in_channels // 2
                total_channels = in_channels + skip_channels
            else:
                if skip_channels is None:
                    skip_channels = in_channels // 2
                total_channels = (in_channels // 2) + skip_channels
            self.se = SEBlock3D(total_channels, reduction=16)
    
    def forward(self, x1, x2):
        x1_up = self.up3d.up(x1)
        
        # 크기 맞추기
        diffZ = x2.size()[2] - x1_up.size()[2]
        diffY = x2.size()[3] - x1_up.size()[3]
        diffX = x2.size()[4] - x1_up.size()[4]
        
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
        
        # Concat
        x = torch.cat([x2, x1_up], dim=1)
        
        # SE 블록 적용 (DoubleConv 전)
        if self.use_se:
            x = self.se(x)
        
        # DoubleConv 적용
        return self.up3d.conv(x)


class Stem3x3(nn.Module):
    """Stem using DoubleConv3D (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = DoubleConv3D(in_channels, out_channels, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================================
# Base Models
# ============================================================================

class DualBranchUNet3D_ShuffleNetV1(nn.Module):
    """Dual-branch UNet with ShuffleNet V1 backbone - Base class with configurable channel sizes
    
    - Both branches: ShuffleNet V1 units
    - Stage 5: Fused branch with ShuffleNet V1
    - SE blocks applied at branch outputs and skip connections
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, 
                 groups: int = 1, size: str = 's'):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.groups = groups
        self.size = size
        
        # Get channel configuration (Stage 4 fused, Stage 5 single branch)
        channels = get_dualbranch_channels_stage4_fused(size)
        
        # Stage 1 stems (simple 3x3 conv)
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        self.se_stem = SEBlock3D(channels['stem'] * 2, reduction=16)  # After concat of stems
        
        # Stage 2 branches (both use ShuffleNet V1)
        self.branch_flair = Down3DShuffleNetV1(channels['stem'], channels['branch2'], groups=self.groups, norm=self.norm)
        self.branch_t1ce = Down3DShuffleNetV1(channels['stem'], channels['branch2'], groups=self.groups, norm=self.norm)
        self.se_flair2 = SEBlock3D(channels['branch2'], reduction=16)
        self.se_t1ce2 = SEBlock3D(channels['branch2'], reduction=16)
        self.se_skip2 = SEBlock3D(channels['branch2'] * 2, reduction=16)  # After concat of branch2 outputs
        
        # Stage 3 branches (both use ShuffleNet V1)
        self.branch_flair3 = Down3DShuffleNetV1(channels['branch2'], channels['branch3'], groups=self.groups, norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV1(channels['branch2'], channels['branch3'], groups=self.groups, norm=self.norm)
        self.se_flair3 = SEBlock3D(channels['branch3'], reduction=16)
        self.se_t1ce3 = SEBlock3D(channels['branch3'], reduction=16)
        self.se_skip3 = SEBlock3D(channels['branch3'] * 2, reduction=16)  # After concat of branch3 outputs
        
        # Stage 4 branches (both use ShuffleNet V1)
        self.branch_flair4 = Down3DShuffleNetV1(channels['branch3'], channels['branch4'], groups=self.groups, norm=self.norm)
        self.branch_t1ce4 = Down3DShuffleNetV1(channels['branch3'], channels['branch4'], groups=self.groups, norm=self.norm)
        self.se_flair4 = SEBlock3D(channels['branch4'], reduction=16)
        self.se_t1ce4 = SEBlock3D(channels['branch4'], reduction=16)
        self.se_skip4 = SEBlock3D(channels['branch4'] * 2, reduction=16)  # After concat of branch4 outputs
        
        # Stage 5 fused branch with ShuffleNet V1 (input: branch4*2, output: down5)
        factor = 2 if self.bilinear else 1
        fused_channels = channels['branch4'] * 2
        # Use 1x1 conv to adjust channels, then ShuffleNet V1 unit for fusion (stride=1 to maintain spatial size)
        self.down5_conv = nn.Sequential(
            nn.Conv3d(fused_channels, channels['down5'], kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(self.norm, channels['down5']),
            nn.ReLU(inplace=True),
        )
        self.down5 = ShuffleNetV1Unit3D(channels['down5'], channels['down5'], stride=1, groups=self.groups, norm=self.norm)
        self.se_bottleneck = SEBlock3D(channels['down5'], reduction=16)  # After down5 output
        
        # Decoder: Up3DSE 사용 (SE 블록 포함)
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        if self.bilinear:
            self.up1 = Up3DSE(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, use_se=True)
            self.up2 = Up3DSE(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, use_se=True)
            self.up3 = Up3DSE(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, use_se=True)
            self.up4 = Up3DSE(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, use_se=True)
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3DSE(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels, use_se=True)
            self.up2 = Up3DSE(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2, use_se=True)
            self.up3 = Up3DSE(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2, use_se=True)
            self.up4 = Up3DSE(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2, use_se=True)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: Stems
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = self.se_stem(torch.cat([x1_flair, x1_t1ce], dim=1))
        
        # Stage 2: Branches
        b_flair = self.se_flair2(self.branch_flair(x1_flair))
        b_t1ce = self.se_t1ce2(self.branch_t1ce(x1_t1ce))
        x2 = self.se_skip2(torch.cat([b_flair, b_t1ce], dim=1))
        
        # Stage 3: Branches
        b2_flair = self.se_flair3(self.branch_flair3(b_flair))
        b2_t1ce = self.se_t1ce3(self.branch_t1ce3(b_t1ce))
        x3 = self.se_skip3(torch.cat([b2_flair, b2_t1ce], dim=1))
        
        # Stage 4: Branches
        b3_flair = self.se_flair4(self.branch_flair4(b2_flair))
        b3_t1ce = self.se_t1ce4(self.branch_t1ce4(b2_t1ce))
        x4 = self.se_skip4(torch.cat([b3_flair, b3_t1ce], dim=1))
        
        # Stage 5: Fused branch
        x5 = self.down5_conv(x4)
        x5 = self.se_bottleneck(self.down5(x5))
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ============================================================================
# Convenience Classes for Backward Compatibility
# ============================================================================

# DualBranchUNet3D_ShuffleNetV1
class DualBranchUNet3D_ShuffleNetV1_XS(DualBranchUNet3D_ShuffleNetV1):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, groups: int = 1):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, groups=groups, size='xs')

class DualBranchUNet3D_ShuffleNetV1_Small(DualBranchUNet3D_ShuffleNetV1):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, groups: int = 1):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, groups=groups, size='s')

class DualBranchUNet3D_ShuffleNetV1_Medium(DualBranchUNet3D_ShuffleNetV1):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, groups: int = 1):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, groups=groups, size='m')

class DualBranchUNet3D_ShuffleNetV1_Large(DualBranchUNet3D_ShuffleNetV1):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, groups: int = 1):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, groups=groups, size='l')


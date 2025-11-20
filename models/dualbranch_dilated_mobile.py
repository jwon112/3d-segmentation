"""
Dilated + MobileNet Dual-Branch UNet Models
통합된 Dilated + MobileNet 조합 모델들

- dualbranch_15: Dilated 1,2,5 (Both branches) + MobileNetV2 with SE blocks in decoder
"""

import torch
import torch.nn as nn

from .dualbranch_basic import Down3DStrideDilated
from .dualbranch_mobilenet import MobileNetV2Block3D, Down3DMobileNetV2
from .modules.mvit_modules import Down3DStrideMViT
from .modules.shufflenet_modules import channel_shuffle_3d
from .modules.se_modules import SEBlock3D
from .model_3d_unet import Up3D, OutConv3D, _make_norm3d, DoubleConv3D
from .channel_configs import get_dualbranch_channels
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


class Up3DShuffle(nn.Module):
    """3D Upsampling 블록 with Channel Shuffle and SE Block for modality fusion
    
    Upsampling 후 skip connection과 concat한 다음 shuffle을 적용하여
    두 모달리티(FLAIR, T1CE) 정보를 융합합니다.
    SE 블록을 DoubleConv 전에 적용하여 중요한 채널에 집중합니다.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, norm: str = 'bn', skip_channels=None, use_se: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            if skip_channels is None:
                skip_channels = in_channels // 2
            total_channels = in_channels + skip_channels
            self.use_se = use_se
            if self.use_se:
                self.se = SEBlock3D(total_channels, reduction=16)
            self.conv = DoubleConv3D(total_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if skip_channels is None:
                skip_channels = in_channels // 2
            total_channels = (in_channels // 2) + skip_channels
            self.use_se = use_se
            if self.use_se:
                self.se = SEBlock3D(total_channels, reduction=16)
            self.conv = DoubleConv3D(total_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 크기 맞추기
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Concat 후 shuffle로 모달리티 융합
        x = torch.cat([x2, x1], dim=1)
        x = channel_shuffle_3d(x, groups=2)
        
        # SE 블록 적용 (DoubleConv 전)
        if self.use_se:
            x = self.se(x)
        
        return self.conv(x)

class Down3DStrideDilated_1_2_3(nn.Module):
    """Downsampling with stride-2 Conv and dilated convolutions (rate 1, 2, 3) for wider ERF.
    
    Pattern: Conv(stride=2) -> Norm -> ReLU -> DilatedConv(rate=2) -> Norm -> ReLU -> DilatedConv(rate=3) -> Norm -> ReLU
    Uses dilated convolutions instead of regular conv to capture wider context.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        # Padding for dilated conv: padding = dilation * (kernel_size - 1) / 2
        # For kernel=3, dilation=2: padding = 2 * (3-1) / 2 = 2
        # For kernel=3, dilation=3: padding = 3 * (3-1) / 2 = 3
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3DStrideDilated_1_2_5(nn.Module):
    """Downsampling with stride-2 Conv and dilated convolutions (rate 1, 2, 5) for wider ERF.
    
    Pattern: Conv(stride=2) -> Norm -> ReLU -> DilatedConv(rate=1) -> Norm -> ReLU -> DilatedConv(rate=2) -> Norm -> ReLU -> DilatedConv(rate=5) -> Norm -> ReLU
    Uses dilated convolutions instead of regular conv to capture wider context.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        # Padding for dilated conv: padding = dilation * (kernel_size - 1) / 2
        # For kernel=3, dilation=1: padding = 1 * (3-1) / 2 = 1
        # For kernel=3, dilation=2: padding = 2 * (3-1) / 2 = 2
        # For kernel=3, dilation=5: padding = 5 * (3-1) / 2 = 5
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================================
# Base Models
# ============================================================================

class DualBranchUNet3D_Dilated125_Both_Mobile(nn.Module):
    """Dual-branch UNet with dilated both branches (rate 1,2,5) + MobileNetV2 backbone - Base class with configurable channel sizes
    
    - Both branches: Dilated convolutions (rate 1, 2, 5)
    - Stage 5: Fused branch with MobileViT
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, 
                 expand_ratio: float = 4.0, size: str = 's'):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.expand_ratio = expand_ratio
        self.size = size
        
        # Get channel configuration (Stage 4 fused, Stage 5 single branch)
        from .channel_configs import get_dualbranch_channels_stage4_fused
        channels = get_dualbranch_channels_stage4_fused(size)
        
        # Stage 1 stems (MobileNetV2 blocks)
        self.stem_flair = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        
        # Stage 2 branches (both use dilated)
        self.branch_flair = Down3DStrideDilated_1_2_5(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DStrideDilated_1_2_5(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3 branches (both use dilated)
        self.branch_flair3 = Down3DStrideDilated_1_2_5(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStrideDilated_1_2_5(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4 branches (both use dilated)
        self.branch_flair4 = Down3DStrideDilated_1_2_5(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DStrideDilated_1_2_5(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Stage 5 fused branch with MobileViT (input: branch4*2, output: down5)
        factor = 2 if self.bilinear else 1
        fused_channels = channels['branch4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
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
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)
        
        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)
        
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        b3_flair = self.branch_flair4(b2_flair)
        b3_t1ce = self.branch_t1ce4(b2_t1ce)
        x4 = torch.cat([b3_flair, b3_t1ce], dim=1)
        
        x5 = self.down5(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ============================================================================
# Convenience Classes for Backward Compatibility
# ============================================================================

# DualBranchUNet3D_Dilated125_Both_Mobile
class DualBranchUNet3D_Dilated125_Both_Mobile_XS(DualBranchUNet3D_Dilated125_Both_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_Dilated125_Both_Mobile_Small(DualBranchUNet3D_Dilated125_Both_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 6.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_Dilated125_Both_Mobile_Medium(DualBranchUNet3D_Dilated125_Both_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 6.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_Dilated125_Both_Mobile_Large(DualBranchUNet3D_Dilated125_Both_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 6.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')


# ============================================================================
# DualBranchUNet3D_Dilated125_Both_Mobile with Shuffle in Skip Connections
# ============================================================================

class DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle(nn.Module):
    """Dual-branch UNet with dilated both branches (rate 1,2,5) + MobileNetV2 backbone with shuffle in decoder
    
    - Both branches: Dilated convolutions (rate 1, 2, 5)
    - Stage 5: Fused branch with MobileViT
    - Encoder: 각 branch 독립적으로 처리 (dual branch의 의의 유지)
    - Skip connections: 원본 정보 유지 (shuffle 없음)
    - Decoder: Upsampling 후 concat한 다음 shuffle로 모달리티 융합 수행
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, 
                 expand_ratio: float = 4.0, size: str = 's'):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.expand_ratio = expand_ratio
        self.size = size
        
        # Get channel configuration (Stage 4 fused, Stage 5 single branch)
        from .channel_configs import get_dualbranch_channels_stage4_fused
        channels = get_dualbranch_channels_stage4_fused(size)
        
        # Stage 1 stems (MobileNetV2 blocks)
        self.stem_flair = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        
        # Stage 2 branches (both use dilated)
        self.branch_flair = Down3DStrideDilated_1_2_5(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DStrideDilated_1_2_5(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3 branches (both use dilated)
        self.branch_flair3 = Down3DStrideDilated_1_2_5(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStrideDilated_1_2_5(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4 branches (both use dilated)
        self.branch_flair4 = Down3DStrideDilated_1_2_5(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DStrideDilated_1_2_5(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Stage 5 fused branch with MobileViT (input: branch4*2, output: down5)
        factor = 2 if self.bilinear else 1
        fused_channels = channels['branch4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder: Up3DShuffle 사용 (concat 후 shuffle로 모달리티 융합, SE 블록 포함)
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        if self.bilinear:
            self.up1 = Up3DShuffle(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, use_se=True)
            self.up2 = Up3DShuffle(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, use_se=True)
            self.up3 = Up3DShuffle(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, use_se=True)
            self.up4 = Up3DShuffle(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, use_se=True)
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3DShuffle(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels, use_se=True)
            self.up2 = Up3DShuffle(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2, use_se=True)
            self.up3 = Up3DShuffle(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2, use_se=True)
            self.up4 = Up3DShuffle(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2, use_se=True)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: 각 branch 독립적으로 처리 (dual branch의 의의 유지)
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1_skip = channel_shuffle_3d(torch.cat([x1_flair, x1_t1ce], dim=1), groups=2)  # Skip connection: shuffle 적용
        
        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2_skip = channel_shuffle_3d(torch.cat([b_flair, b_t1ce], dim=1), groups=2)  # Skip connection: shuffle 적용
        
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3_skip = channel_shuffle_3d(torch.cat([b2_flair, b2_t1ce], dim=1), groups=2)  # Skip connection: shuffle 적용
        
        b3_flair = self.branch_flair4(b2_flair)
        b3_t1ce = self.branch_t1ce4(b2_t1ce)
        x4_skip = channel_shuffle_3d(torch.cat([b3_flair, b3_t1ce], dim=1), groups=2)  # Skip connection: shuffle 적용
        
        x5 = self.down5(x4_skip)
        
        # Decoder: Up3DShuffle 내부에서 concat 후 shuffle로 모달리티 융합 수행
        x = self.up1(x5, x4_skip)
        x = self.up2(x, x3_skip)
        x = self.up3(x, x2_skip)
        x = self.up4(x, x1_skip)
        
        return self.outc(x)


# Convenience Classes for DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle
class DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle_XS(DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle_Small(DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 6.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle_Medium(DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 6.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle_Large(DualBranchUNet3D_Dilated125_Both_Mobile_Shuffle):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 6.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')


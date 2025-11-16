"""
Dilated + MobileNet Dual-Branch UNet Models
통합된 Dilated + MobileNet 조합 모델들 (10, 11, 15)

- dualbranch_10: Dilated (FLAIR) + MobileNetV2 (t1ce)
- dualbranch_11: Dilated 1,2,3 (FLAIR) + MobileNetV2 (t1ce)
- dualbranch_15: Dilated 1,2,5 (Both branches) + MobileNetV2
"""

import torch
import torch.nn as nn

from .dualbranch_basic import Down3DStrideDilated
from .dualbranch_mobilenet import MobileNetV2Block3D, Down3DMobileNetV2
from .modules.mvit_modules import Down3DStrideMViT
from .model_3d_unet import Up3D, OutConv3D, _make_norm3d
from .channel_configs import get_dualbranch_channels


# ============================================================================
# Building Blocks
# ============================================================================

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

class DualBranchUNet3D_DilatedMobile(nn.Module):
    """Dual-branch UNet with dilated FLAIR branch + MobileNetV2 backbone - Base class with configurable channel sizes
    
    - FLAIR branch: Dilated convolutions (rate 2, 5)
    - t1ce branch: MobileNetV2
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
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)
        
        # Stage 1 stems (MobileNetV2 blocks)
        self.stem_flair = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        
        # Stage 2 branches
        self.branch_flair = Down3DStrideDilated(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DMobileNetV2(channels['stem'], channels['branch2'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 3 branches
        self.branch_flair3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2(channels['branch2'], channels['branch3'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 4 branches
        self.branch_flair4 = Down3DStrideDilated(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DMobileNetV2(channels['branch3'], channels['branch4'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['branch4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        if self.bilinear:
            self.up1 = Up3D(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3D(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
            self.up2 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
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


class DualBranchUNet3D_Dilated123_Mobile(nn.Module):
    """Dual-branch UNet with dilated FLAIR branch (rate 1,2,3) + MobileNetV2 backbone - Base class with configurable channel sizes
    
    - FLAIR branch: Dilated convolutions (rate 1, 2, 3)
    - t1ce branch: MobileNetV2
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
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)
        
        # Stage 1 stems (MobileNetV2 blocks)
        self.stem_flair = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        
        # Stage 2 branches
        self.branch_flair = Down3DStrideDilated_1_2_3(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DMobileNetV2(channels['stem'], channels['branch2'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 3 branches
        self.branch_flair3 = Down3DStrideDilated_1_2_3(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2(channels['branch2'], channels['branch3'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 4 branches
        self.branch_flair4 = Down3DStrideDilated_1_2_3(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DMobileNetV2(channels['branch3'], channels['branch4'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['branch4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        if self.bilinear:
            self.up1 = Up3D(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3D(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
            self.up2 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
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
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)
        
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
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['branch4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        if self.bilinear:
            self.up1 = Up3D(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3D(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
            self.up2 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
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

# DualBranchUNet3D_DilatedMobile
class DualBranchUNet3D_DilatedMobile_XS(DualBranchUNet3D_DilatedMobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_DilatedMobile_Small(DualBranchUNet3D_DilatedMobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_DilatedMobile_Medium(DualBranchUNet3D_DilatedMobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_DilatedMobile_Large(DualBranchUNet3D_DilatedMobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')

# DualBranchUNet3D_Dilated123_Mobile
class DualBranchUNet3D_Dilated123_Mobile_XS(DualBranchUNet3D_Dilated123_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_Dilated123_Mobile_Small(DualBranchUNet3D_Dilated123_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_Dilated123_Mobile_Medium(DualBranchUNet3D_Dilated123_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_Dilated123_Mobile_Large(DualBranchUNet3D_Dilated123_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')

# DualBranchUNet3D_Dilated125_Both_Mobile
class DualBranchUNet3D_Dilated125_Both_Mobile_XS(DualBranchUNet3D_Dilated125_Both_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_Dilated125_Both_Mobile_Small(DualBranchUNet3D_Dilated125_Both_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_Dilated125_Both_Mobile_Medium(DualBranchUNet3D_Dilated125_Both_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_Dilated125_Both_Mobile_Large(DualBranchUNet3D_Dilated125_Both_Mobile):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')


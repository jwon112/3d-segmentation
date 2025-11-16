"""
Basic Dual-Branch UNet Models
통합된 기본 dual-branch 모델들 (01, 02, 03)

- dualbranch_01: MaxPool 기반 Down3D
- dualbranch_02: stride downsampling (Down3DStride)
- dualbranch_03: FLAIR branch에 dilated conv 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import DoubleConv3D, Down3D, Up3D, OutConv3D, _make_norm3d
from .channel_configs import get_dualbranch_channels


# ============================================================================
# Building Blocks
# ============================================================================

class Down3DStride(nn.Module):
    """Downsampling with stride-2 Conv instead of MaxPool.
    Pattern: Conv(stride=2) -> Norm -> ReLU -> Conv -> Norm -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3DStrideDilated(nn.Module):
    """Downsampling with stride-2 Conv and dilated convolutions for wider ERF.
    
    Pattern: Conv(stride=2) -> Norm -> ReLU -> DilatedConv(rate=2) -> Norm -> ReLU -> DilatedConv(rate=5) -> Norm -> ReLU
    Uses dilated convolutions instead of regular conv to capture wider context.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        # Padding for dilated conv: padding = dilation * (kernel_size - 1) / 2
        # For kernel=3, dilation=2: padding = 2 * (3-1) / 2 = 2
        # For kernel=3, dilation=5: padding = 5 * (3-1) / 2 = 5
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
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
# Base Dual-Branch Models
# ============================================================================

class DualBranchUNet3D(nn.Module):
    """Dual-branch 3D U-Net (v0.1) - Base class with configurable channel sizes

    - Input (B, 2, D, H, W): two modalities (FLAIR, t1ce)
    - Stage1 (stem): modality-specific stems
    - Stage2-3: two branches Down3D per modality
    - Stages 4+: standard UNet encoder/decoder with skip connections
    - Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 2, "DualBranchUNet3D expects exactly 2 input channels (FLAIR, t1ce)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)

        # Stage 1: modality-specific stems
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)

        # Stage 2: modality-specific symmetrical branches
        self.branch_flair = Down3D(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3D(channels['stem'], channels['branch2'], norm=self.norm)

        # Stage 3: extend the same dual-branch pattern
        self.branch_flair3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)

        # Stages 4+: standard UNet encoder
        fused_channels = channels['branch3'] * 2
        
        if size in ['xs', 's']:
            self.down4 = Down3D(fused_channels, channels['down4'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down5 = Down3D(channels['down4'], channels['down5'] // factor, norm=self.norm)
            
            # Decoder
            # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
            if self.bilinear:
                self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
                self.up2 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm)
                self.up3 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm)
                self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
            else:
                # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
                self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down4'])
                self.up2 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
                self.up3 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
                self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
        else:  # m, l
            self.down4 = Down3D(fused_channels, channels['down4'], norm=self.norm)
            self.down5 = Down3D(channels['down4'], channels['down5'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down6 = Down3D(channels['down5'], channels['down6'] // factor, norm=self.norm)
            
            # Decoder
            # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
            if self.bilinear:
                self.up1 = Up3D(channels['down6'], channels['down5'] // factor, self.bilinear, norm=self.norm)
                self.up2 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
                self.up3 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm)
                self.up4 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm)
                self.up5 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
            else:
                # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
                self.up1 = Up3D(channels['down6'], channels['down5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down5'])
                self.up2 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down4'])
                self.up3 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
                self.up4 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
                self.up5 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
        
        self.outc = OutConv3D(channels['out'], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1 (modality-specific stems)
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)

        # Stage 2 branches: process each modality branch independently then fuse
        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)

        # Stage 3 branches: process independently then fuse
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)

        # Stages 4+ encoder
        if self.size in ['xs', 's']:
            x4 = self.down4(x3)
            x5 = self.down5(x4)
            
            # Decoder with skip connections
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:  # m, l
            x4 = self.down4(x3)
            x5 = self.down5(x4)
            x6 = self.down6(x5)
            
            # Decoder with skip connections
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        
        logits = self.outc(x)
        return logits


class DualBranchUNet3D_Stride(nn.Module):
    """Dual-branch 3D U-Net (v0.2 - stride downsampling) - Base class with configurable channel sizes

    Differences from DualBranchUNet3D:
    - Replace MaxPool-based Down3D with stride-2 convolutional downsampling.
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 2, "DualBranchUNet3D_Stride expects exactly 2 input channels (FLAIR, t1ce)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)

        # Stage 1: modality-specific stems
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)

        # Stage 2: modality-specific branches
        self.branch_flair = Down3DStride(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DStride(channels['stem'], channels['branch2'], norm=self.norm)

        # Stage 3: extend the same dual-branch pattern
        self.branch_flair3 = Down3DStride(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStride(channels['branch2'], channels['branch3'], norm=self.norm)

        # Stages 4+: encoder
        fused_channels = channels['branch3'] * 2
        
        if size in ['xs', 's']:
            self.down4 = Down3DStride(fused_channels, channels['down4'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down5 = Down3DStride(channels['down4'], channels['down5'] // factor, norm=self.norm)
            
            # Decoder
            # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
            if self.bilinear:
                self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
                self.up2 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm)
                self.up3 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm)
                self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
            else:
                # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
                self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down4'])
                self.up2 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
                self.up3 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
                self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
        else:  # m, l
            self.down4 = Down3DStride(fused_channels, channels['down4'], norm=self.norm)
            self.down5 = Down3DStride(channels['down4'], channels['down5'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down6 = Down3DStride(channels['down5'], channels['down6'] // factor, norm=self.norm)
            
            # Decoder
            # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
            if self.bilinear:
                self.up1 = Up3D(channels['down6'], channels['down5'] // factor, self.bilinear, norm=self.norm)
                self.up2 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
                self.up3 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm)
                self.up4 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm)
                self.up5 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
            else:
                # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
                self.up1 = Up3D(channels['down6'], channels['down5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down5'])
                self.up2 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down4'])
                self.up3 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
                self.up4 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
                self.up5 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
        
        self.outc = OutConv3D(channels['out'], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)

        # Stage 2 branches
        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)

        # Stage 3 branches
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)

        # Encoder
        if self.size in ['xs', 's']:
            x4 = self.down4(x3)
            x5 = self.down5(x4)
            
            # Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:  # m, l
            x4 = self.down4(x3)
            x5 = self.down5(x4)
            x6 = self.down6(x5)
            
            # Decoder
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        
        logits = self.outc(x)
        return logits


class DualBranchUNet3D_StrideDilated(nn.Module):
    """Dual-branch 3D U-Net (v0.3 - dilated conv for FLAIR) - Base class with configurable channel sizes

    Differences from DualBranchUNet3D_Stride:
    - FLAIR branch uses dilated convolutions (rate 2, rate 5) for wider ERF
    - t1ce branch uses standard stride convolutions (unchanged)
    - This allows FLAIR to capture more contextual information while t1ce focuses on local details
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 2, "DualBranchUNet3D_StrideDilated expects exactly 2 input channels (FLAIR, t1ce)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)

        # Stage 1: modality-specific stems
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)

        # Stage 2: modality-specific branches with different architectures
        # FLAIR: dilated convs for wider ERF, t1ce: standard convs
        self.branch_flair = Down3DStrideDilated(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DStride(channels['stem'], channels['branch2'], norm=self.norm)

        # Stage 3: extend the same pattern (FLAIR dilated, t1ce standard)
        self.branch_flair3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStride(channels['branch2'], channels['branch3'], norm=self.norm)

        # Stages 4+: encoder
        fused_channels = channels['branch3'] * 2
        
        if size in ['xs', 's']:
            self.down4 = Down3DStride(fused_channels, channels['down4'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down5 = Down3DStride(channels['down4'], channels['down5'] // factor, norm=self.norm)
            
            # Decoder
            # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
            if self.bilinear:
                self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
                self.up2 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm)
                self.up3 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm)
                self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
            else:
                # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
                self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down4'])
                self.up2 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
                self.up3 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
                self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
        else:  # m, l
            self.down4 = Down3DStride(fused_channels, channels['down4'], norm=self.norm)
            self.down5 = Down3DStride(channels['down4'], channels['down5'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down6 = Down3DStride(channels['down5'], channels['down6'] // factor, norm=self.norm)
            
            # Decoder
            # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
            if self.bilinear:
                self.up1 = Up3D(channels['down6'], channels['down5'] // factor, self.bilinear, norm=self.norm)
                self.up2 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
                self.up3 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm)
                self.up4 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm)
                self.up5 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
            else:
                # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
                self.up1 = Up3D(channels['down6'], channels['down5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down5'])
                self.up2 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down4'])
                self.up3 = Up3D(channels['down4'], fused_channels // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
                self.up4 = Up3D(fused_channels, channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
                self.up5 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
        
        self.outc = OutConv3D(channels['out'], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)

        # Stage 2 branches: FLAIR uses dilated conv, t1ce uses standard conv
        b_flair = self.branch_flair(x1_flair)  # wider ERF
        b_t1ce = self.branch_t1ce(x1_t1ce)    # local details
        x2 = torch.cat([b_flair, b_t1ce], dim=1)

        # Stage 3 branches (FLAIR dilated, t1ce standard)
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)

        # Encoder
        if self.size in ['xs', 's']:
            x4 = self.down4(x3)
            x5 = self.down5(x4)
            
            # Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:  # m, l
            x4 = self.down4(x3)
            x5 = self.down5(x4)
            x6 = self.down6(x5)
            
            # Decoder
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        
        logits = self.outc(x)
        return logits


# ============================================================================
# Convenience Classes for Backward Compatibility
# ============================================================================

# DualBranchUNet3D (v0.1 - MaxPool)
class DualBranchUNet3D_XS(DualBranchUNet3D):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class DualBranchUNet3D_Small(DualBranchUNet3D):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class DualBranchUNet3D_Medium(DualBranchUNet3D):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class DualBranchUNet3D_Large(DualBranchUNet3D):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')

# DualBranchUNet3D_Stride (v0.2 - stride downsampling)
class DualBranchUNet3D_Stride_XS(DualBranchUNet3D_Stride):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class DualBranchUNet3D_Stride_Small(DualBranchUNet3D_Stride):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class DualBranchUNet3D_Stride_Medium(DualBranchUNet3D_Stride):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class DualBranchUNet3D_Stride_Large(DualBranchUNet3D_Stride):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')

# DualBranchUNet3D_StrideDilated (v0.3 - dilated conv for FLAIR)
class DualBranchUNet3D_StrideDilated_XS(DualBranchUNet3D_StrideDilated):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class DualBranchUNet3D_StrideDilated_Small(DualBranchUNet3D_StrideDilated):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class DualBranchUNet3D_StrideDilated_Medium(DualBranchUNet3D_StrideDilated):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class DualBranchUNet3D_StrideDilated_Large(DualBranchUNet3D_StrideDilated):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')


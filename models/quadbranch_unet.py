"""
Quad-Branch UNet Models
4개 모달리티(T1, T1CE, T2, FLAIR)를 사용하는 Quad-Branch 3D U-Net 모델

기본 구조:
- Stage 1: 4개 모달리티별 독립적인 stem
- Stage 2-4: 4개 모달리티별 독립적인 branch
- Stage 5+: 단일 융합 분기 (4개 분기 concat 후)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import DoubleConv3D, Down3D, Up3D, OutConv3D, _make_norm3d
from .channel_configs import get_quadbranch_channels


class QuadBranchUNet3D(nn.Module):
    """Quad-Branch 3D U-Net with configurable channel sizes

    - Input (B, 4, D, H, W): four modalities (T1, T1CE, T2, FLAIR)
    - Stage 1 (stem): modality-specific stems (4 branches)
    - Stage 2-4: four branches Down3D per modality
    - Stage 5+: standard UNet encoder/decoder with skip connections
    - Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 4, "QuadBranchUNet3D expects exactly 4 input channels (T1, T1CE, T2, FLAIR)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_quadbranch_channels(size)

        # Stage 1: modality-specific stems
        self.stem_t1 = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t2 = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)

        # Stage 2: modality-specific branches
        self.branch_t1_2 = Down3D(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce_2 = Down3D(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t2_2 = Down3D(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_flair_2 = Down3D(channels['stem'], channels['branch2'], norm=self.norm)

        # Stage 3: extend the same quad-branch pattern
        self.branch_t1_3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce_3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t2_3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_flair_3 = Down3D(channels['branch2'], channels['branch3'], norm=self.norm)

        # Stage 4: extend the same quad-branch pattern
        self.branch_t1_4 = Down3D(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce_4 = Down3D(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t2_4 = Down3D(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_flair_4 = Down3D(channels['branch3'], channels['branch4'], norm=self.norm)

        # Stage 5+: standard UNet encoder (single fused branch)
        fused_channels = channels['branch4'] * 4  # 4 branches concatenated
        factor = 2 if self.bilinear else 1
        bottleneck_channels = channels['down5'] // factor
        self.down5 = Down3D(fused_channels, bottleneck_channels, norm=self.norm)
        
        # Decoder
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        if self.bilinear:
            self.up1 = Up3D(bottleneck_channels, fused_channels // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(fused_channels, channels['branch3'] * 4 // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch3'] * 4, channels['branch2'] * 4 // factor, self.bilinear, norm=self.norm)
            self.up4 = Up3D(channels['branch2'] * 4, channels['stem'] * 4 // factor, self.bilinear, norm=self.norm)
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3D(bottleneck_channels, fused_channels // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
            self.up2 = Up3D(fused_channels, channels['branch3'] * 4 // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 4)
            self.up3 = Up3D(channels['branch3'] * 4, channels['branch2'] * 4 // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 4)
            self.up4 = Up3D(channels['branch2'] * 4, channels['stem'] * 4 // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 4)
        
        self.outc = OutConv3D(channels['stem'] * 4 // factor, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, D, H, W) - [T1, T1CE, T2, FLAIR]
        
        Returns:
            logits: (B, n_classes, D, H, W)
        """
        # Stage 1 (modality-specific stems)
        x1_t1 = self.stem_t1(x[:, 0:1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1_t2 = self.stem_t2(x[:, 2:3])
        x1_flair = self.stem_flair(x[:, 3:4])
        x1 = torch.cat([x1_t1, x1_t1ce, x1_t2, x1_flair], dim=1)

        # Stage 2 branches: process each modality branch independently then fuse
        b_t1_2 = self.branch_t1_2(x1_t1)
        b_t1ce_2 = self.branch_t1ce_2(x1_t1ce)
        b_t2_2 = self.branch_t2_2(x1_t2)
        b_flair_2 = self.branch_flair_2(x1_flair)
        x2 = torch.cat([b_t1_2, b_t1ce_2, b_t2_2, b_flair_2], dim=1)

        # Stage 3 branches: process independently then fuse
        b_t1_3 = self.branch_t1_3(b_t1_2)
        b_t1ce_3 = self.branch_t1ce_3(b_t1ce_2)
        b_t2_3 = self.branch_t2_3(b_t2_2)
        b_flair_3 = self.branch_flair_3(b_flair_2)
        x3 = torch.cat([b_t1_3, b_t1ce_3, b_t2_3, b_flair_3], dim=1)

        # Stage 4 branches: process independently then fuse
        b_t1_4 = self.branch_t1_4(b_t1_3)
        b_t1ce_4 = self.branch_t1ce_4(b_t1ce_3)
        b_t2_4 = self.branch_t2_4(b_t2_3)
        b_flair_4 = self.branch_flair_4(b_flair_3)
        x4 = torch.cat([b_t1_4, b_t1ce_4, b_t2_4, b_flair_4], dim=1)

        # Stage 5 encoder (single branch)
        x5 = self.down5(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


# ============================================================================
# Convenience Classes for Backward Compatibility
# ============================================================================

class QuadBranchUNet3D_XS(QuadBranchUNet3D):
    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class QuadBranchUNet3D_Small(QuadBranchUNet3D):
    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class QuadBranchUNet3D_Medium(QuadBranchUNet3D):
    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class QuadBranchUNet3D_Large(QuadBranchUNet3D):
    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')


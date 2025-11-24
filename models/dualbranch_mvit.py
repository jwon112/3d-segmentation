"""
MobileViT Extended Dual-Branch UNet Models
통합된 MobileViT Extended 모델 (13)

- dualbranch_13: MobileViT extended to FLAIR branch Stage 3,4 + MViT Stage5
"""

import torch
import torch.nn as nn

from .dualbranch_mobile import MobileNetV2Block3D, Down3DMobileNetV2
from .modules.mvit_modules import Down3DStrideMViT
from .model_3d_unet import Up3D, OutConv3D
from .channel_configs import get_dualbranch_channels


# ============================================================================
# Base Model
# ============================================================================

class DualBranchUNet3D_MViT_Extended(nn.Module):
    """Dual-branch UNet with MobileViT extended to FLAIR branch Stage 3,4 + MViT Stage5 - Base class with configurable channel sizes
    
    - Stage 1-2: MobileNetV2 (both branches)
    - Stage 3-4: FLAIR = MobileViT, t1ce = MobileNetV2
    - Stage 5: MobileViT (fused branch)
    
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
        
        # Stage 2 branches (both MobileNetV2)
        self.branch_flair = Down3DMobileNetV2(channels['stem'], channels['branch2'], norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce = Down3DMobileNetV2(channels['stem'], channels['branch2'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 3 branches: FLAIR = MobileViT, t1ce = MobileNetV2
        self.branch_flair3 = Down3DStrideMViT(channels['branch2'], channels['branch3'], norm=self.norm, num_heads=4, mlp_ratio=2)
        self.branch_t1ce3 = Down3DMobileNetV2(channels['branch2'], channels['branch3'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 4 branches: FLAIR = MobileViT, t1ce = MobileNetV2
        self.branch_flair4 = Down3DStrideMViT(channels['branch3'], channels['down4'], norm=self.norm, num_heads=4, mlp_ratio=2)
        self.branch_t1ce4 = Down3DMobileNetV2(channels['branch3'], channels['down4'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['down4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        if self.bilinear:
            self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
            self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
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

class DualBranchUNet3D_MViT_Extended_XS(DualBranchUNet3D_MViT_Extended):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_MViT_Extended_Small(DualBranchUNet3D_MViT_Extended):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_MViT_Extended_Medium(DualBranchUNet3D_MViT_Extended):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_MViT_Extended_Large(DualBranchUNet3D_MViT_Extended):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')


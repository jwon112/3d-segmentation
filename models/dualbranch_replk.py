"""
RepLK (Reparameterizable Large Kernel) Dual-Branch UNet Models
통합된 RepLK 관련 모델들 (04, 05, 06, 07)

- dualbranch_04: RepLK (13x13x13) for FLAIR branch
- dualbranch_05: RepLK + FFN2 (expansion_ratio=2)
- dualbranch_06: RepLK + FFN2 + MViT Stage 4,5
- dualbranch_07: RepLK + FFN2 + MViT Stage 5만
"""

import torch
import torch.nn as nn

from .model_3d_unet import DoubleConv3D, Up3D, OutConv3D
from .dualbranch_basic import Down3DStride
from .channel_configs import get_dualbranch_channels
from .modules.replk_modules import (
    RepLKBlock3D,
    Transition3D,
    ConvFFN3D,
    Down3DStrideRepLK,
    Down3DStrideRepLK_FFN2,
)
from .modules.mvit_modules import Down3DStrideMViT


# ============================================================================
# Base Models
# ============================================================================

class DualBranchUNet3D_StrideLK(nn.Module):
    """Dual-branch 3D U-Net (v0.4 - RepLK for FLAIR) - Base class with configurable channel sizes

    - FLAIR branch uses RepLK (13x13x13 kernel) after stride downsampling
    - t1ce branch uses standard 3x3 convs (Down3DStride)
    - Applied at Stage 2 and Stage 3 (branching extended through Stage 3)
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 2
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)
        
        # Stage 1: modality-specific stems
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)
        
        # Stage 2 branches: FLAIR uses RepLK, t1ce standard
        self.branch_flair = Down3DStrideRepLK(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DStride(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3 branches: extend same pattern
        self.branch_flair3 = Down3DStrideRepLK(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStride(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stages 4+: encoder
        fused_channels = channels['branch3'] * 2
        if size in ['xs', 's']:
            self.down3 = Down3DStride(fused_channels, channels['down4'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down4 = Down3DStride(channels['down4'], channels['down5'] // factor, norm=self.norm)
            
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
            self.down3 = Down3DStride(fused_channels, channels['down4'], norm=self.norm)
            self.down4 = Down3DStride(channels['down4'], channels['down5'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down5 = Down3DStride(channels['down5'], channels['down6'] // factor, norm=self.norm)
            
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
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:  # m, l
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            
            # Decoder
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        
        logits = self.outc(x)
        return logits
    
    @torch.no_grad()
    def switch_to_deploy(self):
        """Switch all RepLK blocks in FLAIR branches to deploy mode (fuse branches)."""
        self.branch_flair.switch_to_deploy()
        self.branch_flair3.switch_to_deploy()


class DualBranchUNet3D_StrideLK_FFN2(nn.Module):
    """Dual-branch 3D U-Net (v0.5 - RepLK + FFN2) - Base class with configurable channel sizes

    - FLAIR branch uses RepLK + FFN2 (expansion_ratio=2)
    - t1ce branch uses standard 3x3 convs
    - Applied at Stage 2 and Stage 3
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)
        
        # Stage 1 stems
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)
        
        # Stage 2 branches
        self.branch_flair = Down3DStrideRepLK_FFN2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DStride(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3 branches
        self.branch_flair3 = Down3DStrideRepLK_FFN2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStride(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stages 4+
        fused_channels = channels['branch3'] * 2
        if size in ['xs', 's']:
            self.down3 = Down3DStride(fused_channels, channels['down4'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down4 = Down3DStride(channels['down4'], channels['down5'] // factor, norm=self.norm)
            
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
            self.down3 = Down3DStride(fused_channels, channels['down4'], norm=self.norm)
            self.down4 = Down3DStride(channels['down4'], channels['down5'], norm=self.norm)
            factor = 2 if self.bilinear else 1
            self.down5 = Down3DStride(channels['down5'], channels['down6'] // factor, norm=self.norm)
            
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
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)
        
        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)
        
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        if self.size in ['xs', 's']:
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:  # m, l
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        
        return self.outc(x)
    
    @torch.no_grad()
    def switch_to_deploy(self):
        self.branch_flair.switch_to_deploy()
        self.branch_flair3.switch_to_deploy()


class DualBranchUNet3D_StrideLK_FFN2_MViT(nn.Module):
    """Dual-branch 3D U-Net (v0.6 - RepLK + FFN2 + MViT Stage 4,5) - Base class with configurable channel sizes

    - Stage 2/3: FLAIR = RepLK+FFN2, t1ce = stride path
    - Stage 4 and 5: Single fused branch with MobileViT (no re-splitting)
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)
        
        # Stage 1
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)
        
        # Stage 2 (RepLK + FFN2)
        self.branch_flair = nn.Sequential(
            Transition3D(channels['stem'], channels['branch2'], norm=self.norm),
            RepLKBlock3D(channels['branch2'], norm=self.norm, dw_ratio=1.5),
            ConvFFN3D(channels['branch2'], expansion_ratio=2, norm=self.norm),
        )
        self.branch_t1ce = Down3DStride(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3 (RepLK + FFN2)
        self.branch_flair3 = nn.Sequential(
            Transition3D(channels['branch2'], channels['branch3'], norm=self.norm),
            RepLKBlock3D(channels['branch3'], norm=self.norm, dw_ratio=1.5),
            ConvFFN3D(channels['branch3'], expansion_ratio=2, norm=self.norm),
        )
        self.branch_t1ce3 = Down3DStride(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4 and 5: Single fused branch with MobileViT (no re-splitting)
        fused_channels = channels['branch3'] * 2
        factor = 2 if self.bilinear else 1
        if size in ['xs', 's']:
            self.down3 = Down3DStrideMViT(fused_channels, channels['down4'], norm=self.norm, num_heads=4, mlp_ratio=2)
            self.down4 = Down3DStrideMViT(channels['down4'], channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
            
            # Decoder (standard)
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
            self.down3 = Down3DStrideMViT(fused_channels, channels['down4'], norm=self.norm, num_heads=4, mlp_ratio=2)
            self.down4 = Down3DStrideMViT(channels['down4'], channels['down5'], norm=self.norm, num_heads=4, mlp_ratio=2)
            self.down5 = Down3DStrideMViT(channels['down5'], channels['down6'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
            
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
        # Stage 1 stems
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)
        
        # Stage 2
        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)
        
        # Stage 3
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 4 and 5 on fused features (single branch)
        if self.size in ['xs', 's']:
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:  # m, l
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            
            # Decoder
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        
        return self.outc(x)
    
    @torch.no_grad()
    def switch_to_deploy(self):
        # RepLK reparam in earlier stages
        for m in [self.branch_flair[1], self.branch_flair3[1]]:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


class DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5(nn.Module):
    """Dual-branch 3D U-Net (v0.7 - RepLK + FFN2 + MViT Stage 5만) - Base class with configurable channel sizes

    - Stage 2/3: FLAIR = RepLK+FFN2, t1ce = stride path, then concat
    - Stage 4: keep branches (FLAIR RepLK+FFN2, t1ce stride), then concat
    - Stage 5: single fused branch with MobileViT3D
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_dualbranch_channels(size)
        
        # Stage 1 stems
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)
        
        # Stage 2 branches
        self.branch_flair = Down3DStrideRepLK_FFN2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DStride(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3 branches
        self.branch_flair3 = Down3DStrideRepLK_FFN2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStride(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4 branches (still dual-branch)
        self.branch_flair4 = Down3DStrideRepLK_FFN2(channels['branch3'], channels['down4'], norm=self.norm)
        self.branch_t1ce4 = Down3DStride(channels['branch3'], channels['down4'], norm=self.norm)
        
        # Stage 5: single fused branch with MobileViT
        fused_channels = channels['down4'] * 2
        factor = 2 if self.bilinear else 1
        if size in ['xs', 's']:
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
        else:  # m, l
            self.down5 = Down3DStrideMViT(fused_channels, channels['down5'], norm=self.norm, num_heads=4, mlp_ratio=2)
            self.down6 = Down3DStrideMViT(channels['down5'], channels['down6'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
            
            # Decoder
            # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
            if self.bilinear:
                self.up1 = Up3D(channels['down6'], channels['down5'] // factor, self.bilinear, norm=self.norm)
                self.up2 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
                self.up3 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
                self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
                self.up5 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
            else:
                # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
                self.up1 = Up3D(channels['down6'], channels['down5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down5'])
                self.up2 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['down4'])
                self.up3 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
                self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
                self.up5 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
        
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)
        
        # Stage 2
        b_flair = self.branch_flair(x1_flair)
        b_t1ce = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)
        
        # Stage 3
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 4 (still dual-branch)
        b3_flair = self.branch_flair4(b2_flair)
        b3_t1ce = self.branch_t1ce4(b2_t1ce)
        x4 = torch.cat([b3_flair, b3_t1ce], dim=1)
        
        # Stage 5 (single MViT branch)
        if self.size in ['xs', 's']:
            x5 = self.down5(x4)
            
            # Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:  # m, l
            x5 = self.down5(x4)
            x6 = self.down6(x5)
            
            # Decoder
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        
        return self.outc(x)
    
    @torch.no_grad()
    def switch_to_deploy(self):
        # RepLK reparam in branches
        for m in [self.branch_flair.replk, self.branch_flair3.replk, self.branch_flair4.replk]:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


# ============================================================================
# Convenience Classes for Backward Compatibility
# ============================================================================

# DualBranchUNet3D_StrideLK
class DualBranchUNet3D_StrideLK_XS(DualBranchUNet3D_StrideLK):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class DualBranchUNet3D_StrideLK_Small(DualBranchUNet3D_StrideLK):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class DualBranchUNet3D_StrideLK_Medium(DualBranchUNet3D_StrideLK):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class DualBranchUNet3D_StrideLK_Large(DualBranchUNet3D_StrideLK):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')

# DualBranchUNet3D_StrideLK_FFN2
class DualBranchUNet3D_StrideLK_FFN2_XS(DualBranchUNet3D_StrideLK_FFN2):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class DualBranchUNet3D_StrideLK_FFN2_Small(DualBranchUNet3D_StrideLK_FFN2):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class DualBranchUNet3D_StrideLK_FFN2_Medium(DualBranchUNet3D_StrideLK_FFN2):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class DualBranchUNet3D_StrideLK_FFN2_Large(DualBranchUNet3D_StrideLK_FFN2):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')

# DualBranchUNet3D_StrideLK_FFN2_MViT
class DualBranchUNet3D_StrideLK_FFN2_MViT_XS(DualBranchUNet3D_StrideLK_FFN2_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class DualBranchUNet3D_StrideLK_FFN2_MViT_Small(DualBranchUNet3D_StrideLK_FFN2_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class DualBranchUNet3D_StrideLK_FFN2_MViT_Medium(DualBranchUNet3D_StrideLK_FFN2_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class DualBranchUNet3D_StrideLK_FFN2_MViT_Large(DualBranchUNet3D_StrideLK_FFN2_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')

# DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5
class DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_XS(DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')

class DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Small(DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')

class DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Medium(DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')

class DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Large(DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')


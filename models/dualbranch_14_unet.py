import torch
import torch.nn as nn

from .dualbranch_mobilenet import MobileNetV2Block3D
from .dualbranch_basic import Down3DStrideDilated
from .model_3d_unet import Up3D, OutConv3D, DoubleConv3D, _make_norm3d
from .channel_configs import get_dualbranch_channels
from .modules.mvit_modules import Down3DStrideMViT
from .modules.ghostnet_modules import GhostBottleneck3D, Down3DGhostNet
from .modules.shufflenet_modules import (
    ShuffleNetV2Unit3D,
    ShuffleNetV2Unit3D_Dilated,
    ShuffleNetV2Unit3D_LK,
    Down3DShuffleNetV2,
    Down3DShuffleNetV2_Dilated,
    Down3DShuffleNetV2_LK,
)
from .modules.convnext_modules import ConvNeXtBlock3D, Down3DConvNeXt


# ============================================================================
# Backbone Blocks for PAM Comparison Experiments
# ============================================================================

class Down3DMobileNetV2_Expand2(nn.Module):
    """Downsampling using MobileNetV2 inverted residual block (stride=2, expand_ratio=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = MobileNetV2Block3D(in_channels, out_channels, stride=2, expand_ratio=2.0, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================================
# Dual-Branch Models with Different Backbones
# ============================================================================

class DualBranchUNet3D_MobileNetV2_Expand2(nn.Module):
    """Dual-branch UNet with MobileNetV2 (expand_ratio=2) for both branches - Base class with configurable channel sizes
    
    Stage 1-4: Dual-branch with MobileNetV2 (expand_ratio=2)
    Stage 5: Fused branch with MobileViT
    
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
        
        # Stage 1 stems (MobileNetV2 blocks, expand_ratio=2)
        self.stem_flair = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=2.0, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, channels['stem'], stride=1, expand_ratio=2.0, norm=self.norm)
        
        # Stage 2-4 branches (both MobileNetV2, expand_ratio=2)
        self.branch_flair = Down3DMobileNetV2_Expand2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DMobileNetV2_Expand2(channels['stem'], channels['branch2'], norm=self.norm)
        
        self.branch_flair3 = Down3DMobileNetV2_Expand2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2_Expand2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        self.branch_flair4 = Down3DMobileNetV2_Expand2(channels['branch3'], channels['down4'], norm=self.norm)
        self.branch_t1ce4 = Down3DMobileNetV2_Expand2(channels['branch3'], channels['down4'], norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['down4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
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


class DualBranchUNet3D_GhostNet(nn.Module):
    """Dual-branch UNet with GhostNet for both branches - Base class with configurable channel sizes
    
    Stage 1-4: Dual-branch with GhostNet
    Stage 5: Fused branch with MobileViT
    
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
        
        # Stage 1 stems (GhostNet blocks)
        self.stem_flair = GhostBottleneck3D(1, channels['stem'], stride=1, norm=self.norm)
        self.stem_t1ce = GhostBottleneck3D(1, channels['stem'], stride=1, norm=self.norm)
        
        # Stage 2-4 branches (both GhostNet)
        self.branch_flair = Down3DGhostNet(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DGhostNet(channels['stem'], channels['branch2'], norm=self.norm)
        
        self.branch_flair3 = Down3DGhostNet(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DGhostNet(channels['branch2'], channels['branch3'], norm=self.norm)
        
        self.branch_flair4 = Down3DGhostNet(channels['branch3'], channels['down4'], norm=self.norm)
        self.branch_t1ce4 = Down3DGhostNet(channels['branch3'], channels['down4'], norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['down4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
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


class DualBranchUNet3D_Dilated(nn.Module):
    """Dual-branch UNet with Dilated Conv (rate 1,2,5) for both branches - Base class with configurable channel sizes
    
    Stage 1-4: Dual-branch with Dilated Conv
    Stage 5: Fused branch with MobileViT
    
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
        
        # Stage 1 stems (DoubleConv)
        self.stem_flair = DoubleConv3D(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, channels['stem'], norm=self.norm)
        
        # Stage 2-4 branches (both Dilated Conv)
        self.branch_flair = Down3DStrideDilated(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DStrideDilated(channels['stem'], channels['branch2'], norm=self.norm)
        
        self.branch_flair3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        
        self.branch_flair4 = Down3DStrideDilated(channels['branch3'], channels['down4'], norm=self.norm)
        self.branch_t1ce4 = Down3DStrideDilated(channels['branch3'], channels['down4'], norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['down4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
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


class DualBranchUNet3D_ConvNeXt(nn.Module):
    """Dual-branch UNet with ConvNeXt for both branches - Base class with configurable channel sizes
    
    Stage 1-4: Dual-branch with ConvNeXt
    Stage 5: Fused branch with MobileViT
    
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
        
        # Stage 1 stems (ConvNeXt blocks)
        # Note: ConvNeXt 원래는 4x4 stride=4로 패치화하지만, segmentation에서는 크기 유지를 위해 3x3 사용
        self.stem_flair = nn.Sequential(
            nn.Conv3d(1, channels['stem'], kernel_size=3, stride=1, padding=1, bias=True),
            ConvNeXtBlock3D(channels['stem'], norm=self.norm),
        )
        self.stem_t1ce = nn.Sequential(
            nn.Conv3d(1, channels['stem'], kernel_size=3, stride=1, padding=1, bias=True),
            ConvNeXtBlock3D(channels['stem'], norm=self.norm),
        )
        
        # Stage 2-4 branches (both ConvNeXt)
        self.branch_flair = Down3DConvNeXt(channels['stem'], channels['branch2'], norm=self.norm, num_blocks=2)
        self.branch_t1ce = Down3DConvNeXt(channels['stem'], channels['branch2'], norm=self.norm, num_blocks=2)
        
        self.branch_flair3 = Down3DConvNeXt(channels['branch2'], channels['branch3'], norm=self.norm, num_blocks=2)
        self.branch_t1ce3 = Down3DConvNeXt(channels['branch2'], channels['branch3'], norm=self.norm, num_blocks=2)
        
        self.branch_flair4 = Down3DConvNeXt(channels['branch3'], channels['down4'], norm=self.norm, num_blocks=2)
        self.branch_t1ce4 = Down3DConvNeXt(channels['branch3'], channels['down4'], norm=self.norm, num_blocks=2)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['down4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
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


class DualBranchUNet3D_ShuffleNetV2(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 for both branches - Base class with configurable channel sizes
    
    Stage 1-4: Dual-branch with ShuffleNetV2
    Stage 5: Fused branch with MobileViT
    
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
        
        # Stage 1 stems (ShuffleNetV2: initial conv + ShuffleNetV2 unit stride=1)
        # Initial conv to get channels, then split for ShuffleNetV2
        self.stem_flair = nn.Sequential(
            nn.Conv3d(1, channels['stem'], kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, channels['stem']),
            nn.ReLU(inplace=True),
            ShuffleNetV2Unit3D(channels['stem'], channels['stem'], stride=1, norm=norm),
        )
        self.stem_t1ce = nn.Sequential(
            nn.Conv3d(1, channels['stem'], kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, channels['stem']),
            nn.ReLU(inplace=True),
            ShuffleNetV2Unit3D(channels['stem'], channels['stem'], stride=1, norm=norm),
        )
        
        # Stage 2-4 branches (both ShuffleNetV2)
        self.branch_flair = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        
        self.branch_flair3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        self.branch_flair4 = Down3DShuffleNetV2(channels['branch3'], channels['down4'], norm=self.norm)
        self.branch_t1ce4 = Down3DShuffleNetV2(channels['branch3'], channels['down4'], norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['down4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
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


class DualBranchUNet3D_ShuffleNetV2_Dilated(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 Dilated for both branches - Base class with configurable channel sizes
    
    Stage 1-4: Dual-branch with ShuffleNetV2 Dilated (rate [1,2,5])
    Stage 5: Fused branch with MobileViT
    
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
        
        # Stage 1 stems (ShuffleNetV2 Dilated: initial conv + ShuffleNetV2 Dilated unit stride=1)
        self.stem_flair = nn.Sequential(
            nn.Conv3d(1, channels['stem'], kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, channels['stem']),
            nn.ReLU(inplace=True),
            ShuffleNetV2Unit3D_Dilated(channels['stem'], channels['stem'], stride=1, norm=norm, dilation_rates=[1, 2, 5]),
        )
        self.stem_t1ce = nn.Sequential(
            nn.Conv3d(1, channels['stem'], kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, channels['stem']),
            nn.ReLU(inplace=True),
            ShuffleNetV2Unit3D_Dilated(channels['stem'], channels['stem'], stride=1, norm=norm, dilation_rates=[1, 2, 5]),
        )
        
        # Stage 2-4 branches (both ShuffleNetV2 Dilated)
        self.branch_flair = Down3DShuffleNetV2_Dilated(channels['stem'], channels['branch2'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce = Down3DShuffleNetV2_Dilated(channels['stem'], channels['branch2'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        self.branch_flair3 = Down3DShuffleNetV2_Dilated(channels['branch2'], channels['branch3'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce3 = Down3DShuffleNetV2_Dilated(channels['branch2'], channels['branch3'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        self.branch_flair4 = Down3DShuffleNetV2_Dilated(channels['branch3'], channels['down4'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce4 = Down3DShuffleNetV2_Dilated(channels['branch3'], channels['down4'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['down4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
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


class DualBranchUNet3D_ShuffleNetV2_LK(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 Large Kernel (7x7x7) for both branches - Base class with configurable channel sizes
    
    Stage 1-4: Dual-branch with ShuffleNetV2 Large Kernel
    Stage 5: Fused branch with MobileViT
    
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
        
        # Stage 1 stems (ShuffleNetV2 LK: initial conv + ShuffleNetV2 LK unit stride=1)
        self.stem_flair = nn.Sequential(
            nn.Conv3d(1, channels['stem'], kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, channels['stem']),
            nn.ReLU(inplace=True),
            ShuffleNetV2Unit3D_LK(channels['stem'], channels['stem'], stride=1, norm=norm),
        )
        self.stem_t1ce = nn.Sequential(
            nn.Conv3d(1, channels['stem'], kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, channels['stem']),
            nn.ReLU(inplace=True),
            ShuffleNetV2Unit3D_LK(channels['stem'], channels['stem'], stride=1, norm=norm),
        )
        
        # Stage 2-4 branches (both ShuffleNetV2 LK)
        self.branch_flair = Down3DShuffleNetV2_LK(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DShuffleNetV2_LK(channels['stem'], channels['branch2'], norm=self.norm)
        
        self.branch_flair3 = Down3DShuffleNetV2_LK(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2_LK(channels['branch2'], channels['branch3'], norm=self.norm)
        
        self.branch_flair4 = Down3DShuffleNetV2_LK(channels['branch3'], channels['down4'], norm=self.norm)
        self.branch_t1ce4 = Down3DShuffleNetV2_LK(channels['branch3'], channels['down4'], norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        fused_channels = channels['down4'] * 2
        self.down5 = Down3DStrideMViT(fused_channels, channels['down5'] // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(channels['down5'], channels['down4'] // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(channels['down4'], channels['branch3'] // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(channels['branch2'], channels['out'], self.bilinear, norm=self.norm)
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


# Convenience classes for backward compatibility
class DualBranchUNet3D_MobileNetV2_Expand2_Small(DualBranchUNet3D_MobileNetV2_Expand2):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels, n_classes, norm, bilinear, size='s')

class DualBranchUNet3D_GhostNet_Small(DualBranchUNet3D_GhostNet):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels, n_classes, norm, bilinear, size='s')

class DualBranchUNet3D_Dilated_Small(DualBranchUNet3D_Dilated):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels, n_classes, norm, bilinear, size='s')

class DualBranchUNet3D_ConvNeXt_Small(DualBranchUNet3D_ConvNeXt):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels, n_classes, norm, bilinear, size='s')

class DualBranchUNet3D_ShuffleNetV2_Small(DualBranchUNet3D_ShuffleNetV2):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels, n_classes, norm, bilinear, size='s')

class DualBranchUNet3D_ShuffleNetV2_Dilated_Small(DualBranchUNet3D_ShuffleNetV2_Dilated):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels, n_classes, norm, bilinear, size='s')

class DualBranchUNet3D_ShuffleNetV2_LK_Small(DualBranchUNet3D_ShuffleNetV2_LK):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels, n_classes, norm, bilinear, size='s')


__all__ = [
    # Base classes with size parameter
    'DualBranchUNet3D_MobileNetV2_Expand2',
    'DualBranchUNet3D_GhostNet',
    'DualBranchUNet3D_Dilated',
    'DualBranchUNet3D_ConvNeXt',
    'DualBranchUNet3D_ShuffleNetV2',
    'DualBranchUNet3D_ShuffleNetV2_Dilated',
    'DualBranchUNet3D_ShuffleNetV2_LK',
    # Convenience classes (backward compatibility)
    'DualBranchUNet3D_MobileNetV2_Expand2_Small',
    'DualBranchUNet3D_GhostNet_Small',
    'DualBranchUNet3D_Dilated_Small',
    'DualBranchUNet3D_ConvNeXt_Small',
    'DualBranchUNet3D_ShuffleNetV2_Small',
    'DualBranchUNet3D_ShuffleNetV2_Dilated_Small',
    'DualBranchUNet3D_ShuffleNetV2_LK_Small',
]


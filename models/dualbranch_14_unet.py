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
from .modules.cross_attention_3d import BidirectionalCrossAttention3D


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
    
    Stage 1-3: Dual-branch with MobileNetV2 (expand_ratio=2) (all stages maintain dual-branch structure)
    Stage 3 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1-3 branches (all MobileNetV2, expand_ratio=2, maintaining dual-branch structure)
        # Stage 1: 128×128×128 -> 64×64×64
        self.branch_flair1 = Down3DMobileNetV2_Expand2(1, channels['branch1'], norm=self.norm)
        self.branch_t1ce1 = Down3DMobileNetV2_Expand2(1, channels['branch1'], norm=self.norm)
        
        # Stage 2: 64×64×64 -> 32×32×32
        self.branch_flair2 = Down3DMobileNetV2_Expand2(channels['branch1'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DMobileNetV2_Expand2(channels['branch1'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair3 = Down3DMobileNetV2_Expand2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2_Expand2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch3'], num_heads=8, norm=self.norm)
        
        # Decoder (3 stages: up1, up2, up3)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm)
        else:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch1'] * 2)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: 128×128×128 -> 64×64×64
        b1_flair = self.branch_flair1(x[:, :1])
        b1_t1ce = self.branch_t1ce1(x[:, 1:2])
        x1 = torch.cat([b1_flair, b1_t1ce], dim=1)
        
        # Stage 2: 64×64×64 -> 32×32×32
        b2_flair = self.branch_flair2(b1_flair)
        b2_t1ce = self.branch_t1ce2(b1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 3: 32×32×32 -> 16×16×16
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        
        # Cross Attention fusion at bottleneck
        x3 = self.cross_attn(b3_flair, b3_t1ce)
        
        # Decoder
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x[:, :2])
        return self.outc(x)


class DualBranchUNet3D_GhostNet(nn.Module):
    """Dual-branch UNet with GhostNet for both branches - Base class with configurable channel sizes
    
    Stage 1-3: Dual-branch with GhostNet (all stages maintain dual-branch structure)
    Stage 3 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1-3 branches (all GhostNet, maintaining dual-branch structure)
        # Stage 1: 128×128×128 -> 64×64×64
        self.branch_flair1 = Down3DGhostNet(1, channels['branch1'], norm=self.norm)
        self.branch_t1ce1 = Down3DGhostNet(1, channels['branch1'], norm=self.norm)
        
        # Stage 2: 64×64×64 -> 32×32×32
        self.branch_flair2 = Down3DGhostNet(channels['branch1'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DGhostNet(channels['branch1'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair3 = Down3DGhostNet(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DGhostNet(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch3'], num_heads=8, norm=self.norm)
        
        # Decoder (3 stages: up1, up2, up3)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm)
        else:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch1'] * 2)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: 128×128×128 -> 64×64×64
        b1_flair = self.branch_flair1(x[:, :1])
        b1_t1ce = self.branch_t1ce1(x[:, 1:2])
        x1 = torch.cat([b1_flair, b1_t1ce], dim=1)
        
        # Stage 2: 64×64×64 -> 32×32×32
        b2_flair = self.branch_flair2(b1_flair)
        b2_t1ce = self.branch_t1ce2(b1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 3: 32×32×32 -> 16×16×16
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        
        # Cross Attention fusion at bottleneck
        x3 = self.cross_attn(b3_flair, b3_t1ce)
        
        # Decoder
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x[:, :2])
        return self.outc(x)


class DualBranchUNet3D_Dilated(nn.Module):
    """Dual-branch UNet with Dilated Conv (rate 1,2,5) for both branches - Base class with configurable channel sizes
    
    Stage 1-3: Dual-branch with Dilated Conv (all stages maintain dual-branch structure)
    Stage 3 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1-3 branches (all Dilated Conv, maintaining dual-branch structure)
        # Stage 1: 128×128×128 -> 64×64×64
        self.branch_flair1 = Down3DStrideDilated(1, channels['branch1'], norm=self.norm)
        self.branch_t1ce1 = Down3DStrideDilated(1, channels['branch1'], norm=self.norm)
        
        # Stage 2: 64×64×64 -> 32×32×32
        self.branch_flair2 = Down3DStrideDilated(channels['branch1'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DStrideDilated(channels['branch1'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch3'], num_heads=8, norm=self.norm)
        
        # Decoder (3 stages: up1, up2, up3)
        # skip_channels: 각 stage에서 concat된 skip connection의 채널 수
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm)
        else:
            # bilinear=False일 때는 skip connection 채널 수를 명시적으로 지정
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch1'] * 2)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)  # Input channels: 1 per modality
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: 128×128×128 -> 64×64×64
        b1_flair = self.branch_flair1(x[:, :1])
        b1_t1ce = self.branch_t1ce1(x[:, 1:2])
        x1 = torch.cat([b1_flair, b1_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 2: 64×64×64 -> 32×32×32
        b2_flair = self.branch_flair2(b1_flair)
        b2_t1ce = self.branch_t1ce2(b1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 3: 32×32×32 -> 16×16×16
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        
        # Cross Attention fusion at bottleneck
        x3 = self.cross_attn(b3_flair, b3_t1ce)  # (B, C, 16, 16, 16)
        
        # Decoder
        x = self.up1(x3, x2)  # 16×16×16 -> 32×32×32
        x = self.up2(x, x1)   # 32×32×32 -> 64×64×64
        x = self.up3(x, x[:, :2])  # 64×64×64 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ConvNeXt(nn.Module):
    """Dual-branch UNet with ConvNeXt for both branches - Base class with configurable channel sizes
    
    Stage 1-3: Dual-branch with ConvNeXt (all stages maintain dual-branch structure)
    Stage 3 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1-3 branches (all ConvNeXt, maintaining dual-branch structure)
        # Stage 1: 128×128×128 -> 64×64×64
        self.branch_flair1 = Down3DConvNeXt(1, channels['branch1'], norm=self.norm, num_blocks=2)
        self.branch_t1ce1 = Down3DConvNeXt(1, channels['branch1'], norm=self.norm, num_blocks=2)
        
        # Stage 2: 64×64×64 -> 32×32×32
        self.branch_flair2 = Down3DConvNeXt(channels['branch1'], channels['branch2'], norm=self.norm, num_blocks=2)
        self.branch_t1ce2 = Down3DConvNeXt(channels['branch1'], channels['branch2'], norm=self.norm, num_blocks=2)
        
        # Stage 3: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair3 = Down3DConvNeXt(channels['branch2'], channels['branch3'], norm=self.norm, num_blocks=2)
        self.branch_t1ce3 = Down3DConvNeXt(channels['branch2'], channels['branch3'], norm=self.norm, num_blocks=2)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch3'], num_heads=8, norm=self.norm)
        
        # Decoder (3 stages: up1, up2, up3)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm)
        else:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch1'] * 2)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: 128×128×128 -> 64×64×64
        b1_flair = self.branch_flair1(x[:, :1])
        b1_t1ce = self.branch_t1ce1(x[:, 1:2])
        x1 = torch.cat([b1_flair, b1_t1ce], dim=1)
        
        # Stage 2: 64×64×64 -> 32×32×32
        b2_flair = self.branch_flair2(b1_flair)
        b2_t1ce = self.branch_t1ce2(b1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 3: 32×32×32 -> 16×16×16
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        
        # Cross Attention fusion at bottleneck
        x3 = self.cross_attn(b3_flair, b3_t1ce)
        
        # Decoder
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x[:, :2])
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 for both branches - Base class with configurable channel sizes
    
    Stage 1-3: Dual-branch with ShuffleNetV2 (all stages maintain dual-branch structure)
    Stage 3 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1-3 branches (all ShuffleNetV2, maintaining dual-branch structure)
        # Stage 1: 128×128×128 -> 64×64×64
        self.branch_flair1 = Down3DShuffleNetV2(1, channels['branch1'], norm=self.norm)
        self.branch_t1ce1 = Down3DShuffleNetV2(1, channels['branch1'], norm=self.norm)
        
        # Stage 2: 64×64×64 -> 32×32×32
        self.branch_flair2 = Down3DShuffleNetV2(channels['branch1'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DShuffleNetV2(channels['branch1'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch3'], num_heads=8, norm=self.norm)
        
        # Decoder (3 stages: up1, up2, up3)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm)
        else:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch1'] * 2)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: 128×128×128 -> 64×64×64
        b1_flair = self.branch_flair1(x[:, :1])
        b1_t1ce = self.branch_t1ce1(x[:, 1:2])
        x1 = torch.cat([b1_flair, b1_t1ce], dim=1)
        
        # Stage 2: 64×64×64 -> 32×32×32
        b2_flair = self.branch_flair2(b1_flair)
        b2_t1ce = self.branch_t1ce2(b1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 3: 32×32×32 -> 16×16×16
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        
        # Cross Attention fusion at bottleneck
        x3 = self.cross_attn(b3_flair, b3_t1ce)
        
        # Decoder
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x[:, :2])
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2_Dilated(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 Dilated for both branches - Base class with configurable channel sizes
    
    Stage 1-3: Dual-branch with ShuffleNetV2 Dilated (rate [1,2,5]) (all stages maintain dual-branch structure)
    Stage 3 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1-3 branches (all ShuffleNetV2 Dilated, maintaining dual-branch structure)
        # Stage 1: 128×128×128 -> 64×64×64
        self.branch_flair1 = Down3DShuffleNetV2_Dilated(1, channels['branch1'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce1 = Down3DShuffleNetV2_Dilated(1, channels['branch1'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Stage 2: 64×64×64 -> 32×32×32
        self.branch_flair2 = Down3DShuffleNetV2_Dilated(channels['branch1'], channels['branch2'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce2 = Down3DShuffleNetV2_Dilated(channels['branch1'], channels['branch2'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Stage 3: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair3 = Down3DShuffleNetV2_Dilated(channels['branch2'], channels['branch3'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce3 = Down3DShuffleNetV2_Dilated(channels['branch2'], channels['branch3'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch3'], num_heads=8, norm=self.norm)
        
        # Decoder (3 stages: up1, up2, up3)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm)
        else:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch1'] * 2)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: 128×128×128 -> 64×64×64
        b1_flair = self.branch_flair1(x[:, :1])
        b1_t1ce = self.branch_t1ce1(x[:, 1:2])
        x1 = torch.cat([b1_flair, b1_t1ce], dim=1)
        
        # Stage 2: 64×64×64 -> 32×32×32
        b2_flair = self.branch_flair2(b1_flair)
        b2_t1ce = self.branch_t1ce2(b1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 3: 32×32×32 -> 16×16×16
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        
        # Cross Attention fusion at bottleneck
        x3 = self.cross_attn(b3_flair, b3_t1ce)
        
        # Decoder
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x[:, :2])
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2_LK(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 Large Kernel (7x7x7) for both branches - Base class with configurable channel sizes
    
    Stage 1-3: Dual-branch with ShuffleNetV2 Large Kernel (all stages maintain dual-branch structure)
    Stage 3 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1-3 branches (all ShuffleNetV2 LK, maintaining dual-branch structure)
        # Stage 1: 128×128×128 -> 64×64×64
        self.branch_flair1 = Down3DShuffleNetV2_LK(1, channels['branch1'], norm=self.norm)
        self.branch_t1ce1 = Down3DShuffleNetV2_LK(1, channels['branch1'], norm=self.norm)
        
        # Stage 2: 64×64×64 -> 32×32×32
        self.branch_flair2 = Down3DShuffleNetV2_LK(channels['branch1'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DShuffleNetV2_LK(channels['branch1'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair3 = Down3DShuffleNetV2_LK(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2_LK(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch3'], num_heads=8, norm=self.norm)
        
        # Decoder (3 stages: up1, up2, up3)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm)
        else:
            self.up1 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up2 = Up3D(channels['branch2'], channels['branch1'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch1'] * 2)
            self.up3 = Up3D(channels['branch1'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: 128×128×128 -> 64×64×64
        b1_flair = self.branch_flair1(x[:, :1])
        b1_t1ce = self.branch_t1ce1(x[:, 1:2])
        x1 = torch.cat([b1_flair, b1_t1ce], dim=1)
        
        # Stage 2: 64×64×64 -> 32×32×32
        b2_flair = self.branch_flair2(b1_flair)
        b2_t1ce = self.branch_t1ce2(b1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)
        
        # Stage 3: 32×32×32 -> 16×16×16
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        
        # Cross Attention fusion at bottleneck
        x3 = self.cross_attn(b3_flair, b3_t1ce)
        
        # Decoder
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x[:, :2])
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


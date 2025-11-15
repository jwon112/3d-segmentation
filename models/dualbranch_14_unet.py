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

class StemMobileNetV2_Expand2(nn.Module):
    """Stem using MobileNetV2 inverted residual block (stride=1, expand_ratio=2, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = MobileNetV2Block3D(in_channels, out_channels, stride=1, expand_ratio=2.0, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3DMobileNetV2_Expand2(nn.Module):
    """Downsampling using MobileNetV2 inverted residual block (stride=2, expand_ratio=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = MobileNetV2Block3D(in_channels, out_channels, stride=2, expand_ratio=2.0, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Stem versions (stride=1, no downsampling)
class StemGhostNet(nn.Module):
    """Stem using GhostNet bottleneck (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = GhostBottleneck3D(in_channels, out_channels, stride=1, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StemDilated(nn.Module):
    """Stem using dilated convolutions (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, dilation=2, padding=4, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, dilation=5, padding=10, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StemConvNeXt(nn.Module):
    """Stem using ConvNeXt blocks (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', num_blocks: int = 2):
        super().__init__()
        # Channel projection (no downsampling)
        self.proj = nn.Sequential(
            _make_norm3d(norm, in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=True),
        )
        # ConvNeXt blocks
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock3D(out_channels, norm=norm) for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        return x


class StemShuffleNetV2(nn.Module):
    """Stem using ShuffleNetV2 unit (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        # First, project to out_channels if needed
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()
        self.block = ShuffleNetV2Unit3D(out_channels, out_channels, stride=1, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.block(x)
        return x


class StemShuffleNetV2_Dilated(nn.Module):
    """Stem using ShuffleNetV2 dilated unit (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()
        self.block = ShuffleNetV2Unit3D_Dilated(out_channels, out_channels, stride=1, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.block(x)
        return x


class StemShuffleNetV2_LK(nn.Module):
    """Stem using ShuffleNetV2 large kernel unit (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()
        self.block = ShuffleNetV2Unit3D_LK(out_channels, out_channels, stride=1, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.block(x)
        return x


# ============================================================================
# Dual-Branch Models with Different Backbones
# ============================================================================

class DualBranchUNet3D_MobileNetV2_Expand2(nn.Module):
    """Dual-branch UNet with MobileNetV2 (expand_ratio=2) for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with MobileNetV2 (expand_ratio=2) (all stages maintain dual-branch structure)
    Stage 4 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        self.stem_flair = StemMobileNetV2_Expand2(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = StemMobileNetV2_Expand2(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DMobileNetV2_Expand2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DMobileNetV2_Expand2(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DMobileNetV2_Expand2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2_Expand2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair4 = Down3DMobileNetV2_Expand2(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DMobileNetV2_Expand2(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch4'], num_heads=8, norm=self.norm)
        
        # Decoder (4 stages: up1, up2, up3, up4)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        
        # Cross Attention fusion at bottleneck
        x4 = self.cross_attn(b4_flair, b4_t1ce)  # (B, C, 16, 16, 16)
        
        # Decoder
        x = self.up1(x4, x3)  # 16×16×16 -> 32×32×32
        x = self.up2(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up3(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up4(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_GhostNet(nn.Module):
    """Dual-branch UNet with GhostNet for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with GhostNet (all stages maintain dual-branch structure)
    Stage 4 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        self.stem_flair = StemGhostNet(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = StemGhostNet(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DGhostNet(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DGhostNet(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DGhostNet(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DGhostNet(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair4 = Down3DGhostNet(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DGhostNet(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch4'], num_heads=8, norm=self.norm)
        
        # Decoder (4 stages: up1, up2, up3, up4)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        
        # Cross Attention fusion at bottleneck
        x4 = self.cross_attn(b4_flair, b4_t1ce)  # (B, C, 16, 16, 16)
        
        # Decoder
        x = self.up1(x4, x3)  # 16×16×16 -> 32×32×32
        x = self.up2(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up3(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up4(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_Dilated(nn.Module):
    """Dual-branch UNet with Dilated Conv (rate 1,2,5) for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with Dilated Conv (all stages maintain dual-branch structure)
    Stage 4 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        self.stem_flair = StemDilated(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = StemDilated(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DStrideDilated(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DStrideDilated(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair4 = Down3DStrideDilated(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DStrideDilated(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch4'], num_heads=8, norm=self.norm)
        
        # Decoder (4 stages: up1, up2, up3, up4)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        
        # Cross Attention fusion at bottleneck
        x4 = self.cross_attn(b4_flair, b4_t1ce)  # (B, C, 16, 16, 16)
        
        # Decoder
        x = self.up1(x4, x3)  # 16×16×16 -> 32×32×32
        x = self.up2(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up3(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up4(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ConvNeXt(nn.Module):
    """Dual-branch UNet with ConvNeXt for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with ConvNeXt (all stages maintain dual-branch structure)
    Stage 4 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        self.stem_flair = StemConvNeXt(1, channels['stem'], norm=self.norm, num_blocks=2)
        self.stem_t1ce = StemConvNeXt(1, channels['stem'], norm=self.norm, num_blocks=2)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DConvNeXt(channels['stem'], channels['branch2'], norm=self.norm, num_blocks=2)
        self.branch_t1ce2 = Down3DConvNeXt(channels['stem'], channels['branch2'], norm=self.norm, num_blocks=2)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DConvNeXt(channels['branch2'], channels['branch3'], norm=self.norm, num_blocks=2)
        self.branch_t1ce3 = Down3DConvNeXt(channels['branch2'], channels['branch3'], norm=self.norm, num_blocks=2)
        
        # Stage 4: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair4 = Down3DConvNeXt(channels['branch3'], channels['branch4'], norm=self.norm, num_blocks=2)
        self.branch_t1ce4 = Down3DConvNeXt(channels['branch3'], channels['branch4'], norm=self.norm, num_blocks=2)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch4'], num_heads=8, norm=self.norm)
        
        # Decoder (4 stages: up1, up2, up3, up4)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        
        # Cross Attention fusion at bottleneck
        x4 = self.cross_attn(b4_flair, b4_t1ce)  # (B, C, 16, 16, 16)
        
        # Decoder
        x = self.up1(x4, x3)  # 16×16×16 -> 32×32×32
        x = self.up2(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up3(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up4(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with ShuffleNetV2 (all stages maintain dual-branch structure)
    Stage 4 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        self.stem_flair = StemShuffleNetV2(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = StemShuffleNetV2(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair4 = Down3DShuffleNetV2(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DShuffleNetV2(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch4'], num_heads=8, norm=self.norm)
        
        # Decoder (4 stages: up1, up2, up3, up4)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        
        # Cross Attention fusion at bottleneck
        x4 = self.cross_attn(b4_flair, b4_t1ce)  # (B, C, 16, 16, 16)
        
        # Decoder
        x = self.up1(x4, x3)  # 16×16×16 -> 32×32×32
        x = self.up2(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up3(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up4(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2_Dilated(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 Dilated for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with ShuffleNetV2 Dilated (rate [1,2,5]) (all stages maintain dual-branch structure)
    Stage 4 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        self.stem_flair = StemShuffleNetV2_Dilated(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = StemShuffleNetV2_Dilated(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DShuffleNetV2_Dilated(channels['stem'], channels['branch2'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce2 = Down3DShuffleNetV2_Dilated(channels['stem'], channels['branch2'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DShuffleNetV2_Dilated(channels['branch2'], channels['branch3'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce3 = Down3DShuffleNetV2_Dilated(channels['branch2'], channels['branch3'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Stage 4: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair4 = Down3DShuffleNetV2_Dilated(channels['branch3'], channels['branch4'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce4 = Down3DShuffleNetV2_Dilated(channels['branch3'], channels['branch4'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch4'], num_heads=8, norm=self.norm)
        
        # Decoder (4 stages: up1, up2, up3, up4)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        
        # Cross Attention fusion at bottleneck
        x4 = self.cross_attn(b4_flair, b4_t1ce)  # (B, C, 16, 16, 16)
        
        # Decoder
        x = self.up1(x4, x3)  # 16×16×16 -> 32×32×32
        x = self.up2(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up3(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up4(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2_LK(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 Large Kernel (7x7x7) for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with ShuffleNetV2 Large Kernel (all stages maintain dual-branch structure)
    Stage 4 output is fused via Cross Attention before decoder
    
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
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        self.stem_flair = StemShuffleNetV2_LK(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = StemShuffleNetV2_LK(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DShuffleNetV2_LK(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DShuffleNetV2_LK(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DShuffleNetV2_LK(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2_LK(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16 (bottleneck)
        self.branch_flair4 = Down3DShuffleNetV2_LK(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DShuffleNetV2_LK(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck
        self.cross_attn = BidirectionalCrossAttention3D(channels['branch4'], num_heads=8, norm=self.norm)
        
        # Decoder (4 stages: up1, up2, up3, up4)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up2 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up3 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up4 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        
        # Cross Attention fusion at bottleneck
        x4 = self.cross_attn(b4_flair, b4_t1ce)  # (B, C, 16, 16, 16)
        
        # Decoder
        x = self.up1(x4, x3)  # 16×16×16 -> 32×32×32
        x = self.up2(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up3(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up4(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
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


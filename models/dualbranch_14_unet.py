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
from .modules.cross_attention_3d import BidirectionalCrossAttentionTransformer3D


# ============================================================================
# Backbone Blocks for PAM Comparison Experiments
# ============================================================================

# Stem: DoubleConv3D (모든 모델에서 공통 사용)
class Stem3x3(nn.Module):
    """Stem using DoubleConv3D (stride=1, no downsampling)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = DoubleConv3D(in_channels, out_channels, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with MobileNetV2 (expand_ratio=2) (all stages maintain dual-branch structure)
    Stage 5: Dual-branch with MobileViT (each branch independently)
    Stage 5 output is fused via Cross Attention
    Stage 6: Fused branch with MobileViT (single branch)
    
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
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DMobileNetV2_Expand2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DMobileNetV2_Expand2(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DMobileNetV2_Expand2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2_Expand2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16
        self.branch_flair4 = Down3DMobileNetV2_Expand2(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DMobileNetV2_Expand2(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        self.branch_flair5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        self.branch_t1ce5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Cross Attention for feature fusion at bottleneck (Stage 5 output)
        self.cross_attn = BidirectionalCrossAttentionTransformer3D(
            channels=channels['branch5'],
            num_heads=8,
            norm=self.norm,
            patch_size=2,
            num_transformer_layers=2,
        )
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with ShuffleNetV2)
        self.down6 = Down3DShuffleNetV2(channels['branch5'], channels['down6'], norm=self.norm)
        
        # Decoder (6 stages: up1, up2, up3, up4, up5, up6)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up6
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up5
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        b5_flair = self.branch_flair5(b4_flair)
        b5_t1ce = self.branch_t1ce5(b4_t1ce)
        
        # Cross Attention fusion at bottleneck (Stage 5 output)
        x5 = self.cross_attn(b5_flair, b5_t1ce)  # (B, C, 8, 8, 8)
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with ShuffleNetV2)
        x6 = self.down6(x5)  # (B, C, 4, 4, 4)
        
        # Decoder
        x = self.up1(x6, x5)  # 4×4×4 -> 8×8×8
        x = self.up2(x, x4)   # 8×8×8 -> 16×16×16
        x = self.up3(x, x3)   # 16×16×16 -> 32×32×32
        x = self.up4(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up5(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up6(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_GhostNet(nn.Module):
    """Dual-branch UNet with GhostNet for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with GhostNet (all stages maintain dual-branch structure)
    Stage 5: Dual-branch with MobileViT (each branch independently)
    Stage 5 output is fused via Cross Attention
    Stage 6: Fused branch with MobileViT (single branch)
    
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
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DGhostNet(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DGhostNet(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DGhostNet(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DGhostNet(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16
        self.branch_flair4 = Down3DGhostNet(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DGhostNet(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        self.branch_flair5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        self.branch_t1ce5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Cross Attention for feature fusion at bottleneck (Stage 5 output)
        self.cross_attn = BidirectionalCrossAttentionTransformer3D(
            channels=channels['branch5'],
            num_heads=8,
            norm=self.norm,
            patch_size=2,
            num_transformer_layers=2,
        )
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        self.down6 = Down3DStrideMViT(channels['branch5'], channels['down6'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder (6 stages: up1, up2, up3, up4, up5, up6)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up6
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up5
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        b5_flair = self.branch_flair5(b4_flair)
        b5_t1ce = self.branch_t1ce5(b4_t1ce)
        
        # Cross Attention fusion at bottleneck (Stage 5 output)
        x5 = self.cross_attn(b5_flair, b5_t1ce)  # (B, C, 8, 8, 8)
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        x6 = self.down6(x5)  # (B, C, 4, 4, 4)
        
        # Decoder
        x = self.up1(x6, x5)  # 4×4×4 -> 8×8×8
        x = self.up2(x, x4)   # 8×8×8 -> 16×16×16
        x = self.up3(x, x3)   # 16×16×16 -> 32×32×32
        x = self.up4(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up5(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up6(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_Dilated(nn.Module):
    """Dual-branch UNet with Dilated Conv (rate 1,2,5) for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with Dilated Conv (all stages maintain dual-branch structure)
    Stage 5: Dual-branch with MobileViT (each branch independently)
    Stage 5 output is fused via Cross Attention
    Stage 6: Fused branch with MobileViT (single branch)
    
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
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DStrideDilated(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DStrideDilated(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DStrideDilated(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16
        self.branch_flair4 = Down3DStrideDilated(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DStrideDilated(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        self.branch_flair5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        self.branch_t1ce5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Cross Attention for feature fusion at bottleneck (Stage 5 output)
        self.cross_attn = BidirectionalCrossAttentionTransformer3D(
            channels=channels['branch5'],
            num_heads=8,
            norm=self.norm,
            patch_size=2,
            num_transformer_layers=2,
        )
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        self.down6 = Down3DStrideMViT(channels['branch5'], channels['down6'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder (6 stages: up1, up2, up3, up4, up5, up6)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up5
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        b5_flair = self.branch_flair5(b4_flair)
        b5_t1ce = self.branch_t1ce5(b4_t1ce)
        
        # Cross Attention fusion at bottleneck (Stage 5 output)
        x5 = self.cross_attn(b5_flair, b5_t1ce)  # (B, C, 8, 8, 8)
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        x6 = self.down6(x5)  # (B, C, 4, 4, 4)
        
        # Decoder
        x = self.up1(x6, x5)  # 4×4×4 -> 8×8×8
        x = self.up2(x, x4)   # 8×8×8 -> 16×16×16
        x = self.up3(x, x3)   # 16×16×16 -> 32×32×32
        x = self.up4(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up5(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up6(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ConvNeXt(nn.Module):
    """Dual-branch UNet with ConvNeXt for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with ConvNeXt (all stages maintain dual-branch structure)
    Stage 5: Dual-branch with MobileViT (each branch independently)
    Stage 5 output is fused via Cross Attention
    Stage 6: Fused branch with MobileViT (single branch)
    
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
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DConvNeXt(channels['stem'], channels['branch2'], norm=self.norm, num_blocks=2)
        self.branch_t1ce2 = Down3DConvNeXt(channels['stem'], channels['branch2'], norm=self.norm, num_blocks=2)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DConvNeXt(channels['branch2'], channels['branch3'], norm=self.norm, num_blocks=2)
        self.branch_t1ce3 = Down3DConvNeXt(channels['branch2'], channels['branch3'], norm=self.norm, num_blocks=2)
        
        # Stage 4: 32×32×32 -> 16×16×16
        self.branch_flair4 = Down3DConvNeXt(channels['branch3'], channels['branch4'], norm=self.norm, num_blocks=2)
        self.branch_t1ce4 = Down3DConvNeXt(channels['branch3'], channels['branch4'], norm=self.norm, num_blocks=2)
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        self.branch_flair5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        self.branch_t1ce5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Cross Attention for feature fusion at bottleneck (Stage 5 output)
        self.cross_attn = BidirectionalCrossAttentionTransformer3D(
            channels=channels['branch5'],
            num_heads=8,
            norm=self.norm,
            patch_size=2,
            num_transformer_layers=2,
        )
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        self.down6 = Down3DStrideMViT(channels['branch5'], channels['down6'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder (6 stages: up1, up2, up3, up4, up5, up6)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up5
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        b5_flair = self.branch_flair5(b4_flair)
        b5_t1ce = self.branch_t1ce5(b4_t1ce)
        
        # Cross Attention fusion at bottleneck (Stage 5 output)
        x5 = self.cross_attn(b5_flair, b5_t1ce)  # (B, C, 8, 8, 8)
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        x6 = self.down6(x5)  # (B, C, 4, 4, 4)
        
        # Decoder
        x = self.up1(x6, x5)  # 4×4×4 -> 8×8×8
        x = self.up2(x, x4)   # 8×8×8 -> 16×16×16
        x = self.up3(x, x3)   # 16×16×16 -> 32×32×32
        x = self.up4(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up5(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up6(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-5: Dual-branch with ShuffleNetV2 (all stages maintain dual-branch structure)
    Stage 5 output is fused via concatenation + MobileViT Stage 6 fused branch
    
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
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16
        self.branch_flair4 = Down3DShuffleNetV2(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DShuffleNetV2(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with ShuffleNetV2)
        self.branch_flair5 = Down3DShuffleNetV2(channels['branch4'], channels['branch5'], norm=self.norm)
        self.branch_t1ce5 = Down3DShuffleNetV2(channels['branch4'], channels['branch5'], norm=self.norm)
        
        # Decoder (5 stages: up1, up2, up3, up4, up5)
        fused_channels = channels['branch5'] * 2
        self.down6 = Down3DShuffleNetV2(fused_channels, channels['down6'], norm=self.norm)
        
        # Decoder (6 stages: up1, up2, up3, up4, up5, up6)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=fused_channels)
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up5
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with ShuffleNetV2)
        b5_flair = self.branch_flair5(b4_flair)
        b5_t1ce = self.branch_t1ce5(b4_t1ce)
        
        # Concatenation fusion at bottleneck (Stage 5 output)
        x5 = torch.cat([b5_flair, b5_t1ce], dim=1)  # (B, 2*C, 8, 8, 8)
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with ShuffleNetV2)
        x6 = self.down6(x5)  # (B, channels['down6'], 4, 4, 4)
        
        # Decoder
        x = self.up1(x6, x5)  # 4×4×4 -> 8×8×8
        x = self.up2(x, x4)   # 8×8×8 -> 16×16×16
        x = self.up3(x, x3)   # 16×16×16 -> 32×32×32
        x = self.up4(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up5(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up6(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2_CrossAttn(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 for both branches with Cross Attention fusion - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-4: Dual-branch with ShuffleNetV2
    Stage 5: Dual-branch with MobileViT (each branch independently)
    Stage 5 output is fused via Cross Attention
    Stage 6: Fused branch with MobileViT (single branch)
    
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
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16
        self.branch_flair4 = Down3DShuffleNetV2(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DShuffleNetV2(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        self.branch_flair5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        self.branch_t1ce5 = Down3DStrideMViT(channels['branch4'], channels['branch5'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Cross Attention for feature fusion at bottleneck (Stage 5 output)
        self.cross_attn = BidirectionalCrossAttentionTransformer3D(
            channels=channels['branch5'],
            num_heads=8,
            norm=self.norm,
            patch_size=2,
            num_transformer_layers=2,
        )
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        self.down6 = Down3DStrideMViT(channels['branch5'], channels['down6'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder (6 stages: up1, up2, up3, up4, up5, up6)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up5
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with MobileViT)
        b5_flair = self.branch_flair5(b4_flair)
        b5_t1ce = self.branch_t1ce5(b4_t1ce)
        
        # Cross Attention fusion at bottleneck (Stage 5 output)
        x5 = self.cross_attn(b5_flair, b5_t1ce)  # (B, C, 8, 8, 8)
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        x6 = self.down6(x5)  # (B, C, 4, 4, 4)
        
        # Decoder
        x = self.up1(x6, x5)  # 4×4×4 -> 8×8×8
        x = self.up2(x, x4)   # 8×8×8 -> 16×16×16
        x = self.up3(x, x3)   # 16×16×16 -> 32×32×32
        x = self.up4(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up5(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up6(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2_Dilated(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 Dilated for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-5: Dual-branch with ShuffleNetV2 Dilated (rate [1,2,5]) (all stages maintain dual-branch structure)
    Stage 5 output is fused via Cross Attention
    Stage 6: Fused branch with MobileViT (single branch)
    
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
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DShuffleNetV2_Dilated(channels['stem'], channels['branch2'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce2 = Down3DShuffleNetV2_Dilated(channels['stem'], channels['branch2'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DShuffleNetV2_Dilated(channels['branch2'], channels['branch3'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce3 = Down3DShuffleNetV2_Dilated(channels['branch2'], channels['branch3'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Stage 4: 32×32×32 -> 16×16×16
        self.branch_flair4 = Down3DShuffleNetV2_Dilated(channels['branch3'], channels['branch4'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce4 = Down3DShuffleNetV2_Dilated(channels['branch3'], channels['branch4'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with ShuffleNetV2 Dilated)
        self.branch_flair5 = Down3DShuffleNetV2_Dilated(channels['branch4'], channels['branch5'], norm=self.norm, dilation_rates=[1, 2, 5])
        self.branch_t1ce5 = Down3DShuffleNetV2_Dilated(channels['branch4'], channels['branch5'], norm=self.norm, dilation_rates=[1, 2, 5])
        
        # Cross Attention for feature fusion at bottleneck (Stage 5 output)
        self.cross_attn = BidirectionalCrossAttentionTransformer3D(
            channels=channels['branch5'],
            num_heads=8,
            norm=self.norm,
            patch_size=2,
            num_transformer_layers=2,
        )
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        self.down6 = Down3DStrideMViT(channels['branch5'], channels['down6'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder (6 stages: up1, up2, up3, up4, up5, up6)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up5
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with ShuffleNetV2 Dilated)
        b5_flair = self.branch_flair5(b4_flair)
        b5_t1ce = self.branch_t1ce5(b4_t1ce)
        
        # Cross Attention fusion at bottleneck (Stage 5 output)
        x5 = self.cross_attn(b5_flair, b5_t1ce)  # (B, C, 8, 8, 8)
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        x6 = self.down6(x5)  # (B, C, 4, 4, 4)
        
        # Decoder
        x = self.up1(x6, x5)  # 4×4×4 -> 8×8×8
        x = self.up2(x, x4)   # 8×8×8 -> 16×16×16
        x = self.up3(x, x3)   # 16×16×16 -> 32×32×32
        x = self.up4(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up5(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up6(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
        return self.outc(x)


class DualBranchUNet3D_ShuffleNetV2_LK(nn.Module):
    """Dual-branch UNet with ShuffleNetV2 Large Kernel (7x7x7) for both branches - Base class with configurable channel sizes
    
    Stage 1: Stem (no downsampling, feature extraction only)
    Stage 2-5: Dual-branch with ShuffleNetV2 Large Kernel (all stages maintain dual-branch structure)
    Stage 5 output is fused via Cross Attention
    Stage 6: Fused branch with MobileViT (single branch)
    
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
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)
        
        # Stage 2: 128×128×128 -> 64×64×64
        self.branch_flair2 = Down3DShuffleNetV2_LK(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DShuffleNetV2_LK(channels['stem'], channels['branch2'], norm=self.norm)
        
        # Stage 3: 64×64×64 -> 32×32×32
        self.branch_flair3 = Down3DShuffleNetV2_LK(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2_LK(channels['branch2'], channels['branch3'], norm=self.norm)
        
        # Stage 4: 32×32×32 -> 16×16×16
        self.branch_flair4 = Down3DShuffleNetV2_LK(channels['branch3'], channels['branch4'], norm=self.norm)
        self.branch_t1ce4 = Down3DShuffleNetV2_LK(channels['branch3'], channels['branch4'], norm=self.norm)
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with ShuffleNetV2 Large Kernel)
        self.branch_flair5 = Down3DShuffleNetV2_LK(channels['branch4'], channels['branch5'], norm=self.norm)
        self.branch_t1ce5 = Down3DShuffleNetV2_LK(channels['branch4'], channels['branch5'], norm=self.norm)
        
        # Cross Attention for feature fusion at bottleneck (Stage 5 output)
        self.cross_attn = BidirectionalCrossAttentionTransformer3D(
            channels=channels['branch5'],
            num_heads=8,
            norm=self.norm,
            patch_size=2,
            num_transformer_layers=2,
        )
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        self.down6 = Down3DStrideMViT(channels['branch5'], channels['down6'], norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder (6 stages: up1, up2, up3, up4, up5, up6)
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        else:
            self.up1 = Up3D(channels['down6'], channels['branch5'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch5'])
            self.up2 = Up3D(channels['branch5'], channels['branch4'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch4'] * 2)
            self.up3 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch3'] * 2)
            self.up4 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['branch2'] * 2)
            self.up5 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm, skip_channels=channels['stem'] * 2)
            self.up6 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=1 * 2)
        self.outc = OutConv3D(channels['out'], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for skip connection
        x_input = x
        
        # Stage 1: Stem (no downsampling, 128×128×128 -> 128×128×128)
        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)  # Skip connection for up5
        
        # Stage 2: 128×128×128 -> 64×64×64
        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)  # Skip connection for up4
        
        # Stage 3: 64×64×64 -> 32×32×32
        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)  # Skip connection for up3
        
        # Stage 4: 32×32×32 -> 16×16×16
        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)  # Skip connection for up2
        
        # Stage 5: 16×16×16 -> 8×8×8 (dual-branch with ShuffleNetV2 Large Kernel)
        b5_flair = self.branch_flair5(b4_flair)
        b5_t1ce = self.branch_t1ce5(b4_t1ce)
        
        # Cross Attention fusion at bottleneck (Stage 5 output)
        x5 = self.cross_attn(b5_flair, b5_t1ce)  # (B, C, 8, 8, 8)
        
        # Stage 6: 8×8×8 -> 4×4×4 (fused branch with MobileViT)
        x6 = self.down6(x5)  # (B, C, 4, 4, 4)
        
        # Decoder
        x = self.up1(x6, x5)  # 4×4×4 -> 8×8×8
        x = self.up2(x, x4)   # 8×8×8 -> 16×16×16
        x = self.up3(x, x3)   # 16×16×16 -> 32×32×32
        x = self.up4(x, x2)   # 32×32×32 -> 64×64×64
        x = self.up5(x, x1)   # 64×64×64 -> 128×128×128
        x = self.up6(x, x_input[:, :2])  # 128×128×128 -> 128×128×128 (skip from input)
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

class DualBranchUNet3D_ShuffleNetV2_CrossAttn_Small(DualBranchUNet3D_ShuffleNetV2_CrossAttn):
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
    'DualBranchUNet3D_ShuffleNetV2_CrossAttn',
    'DualBranchUNet3D_ShuffleNetV2_Dilated',
    'DualBranchUNet3D_ShuffleNetV2_LK',
    # Convenience classes (backward compatibility)
    'DualBranchUNet3D_MobileNetV2_Expand2_Small',
    'DualBranchUNet3D_GhostNet_Small',
    'DualBranchUNet3D_Dilated_Small',
    'DualBranchUNet3D_ConvNeXt_Small',
    'DualBranchUNet3D_ShuffleNetV2_Small',
    'DualBranchUNet3D_ShuffleNetV2_CrossAttn_Small',
    'DualBranchUNet3D_ShuffleNetV2_Dilated_Small',
    'DualBranchUNet3D_ShuffleNetV2_LK_Small',
]


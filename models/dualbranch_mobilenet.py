"""
MobileNetV2 Dual-Branch UNet Models
통합된 MobileNetV2 관련 모델들 (08, 09, 12)

- dualbranch_08: RepLK (FLAIR) + MobileNetV2 (t1ce) + MViT Stage5
- dualbranch_09: RepLK 7x7 (FLAIR) + MobileNetV2 (t1ce) + MViT Stage5
- dualbranch_12: MobileNetV2 Both + MViT Stage5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import Up3D, OutConv3D, _make_norm3d
from .channel_configs import get_dualbranch_channels
from .modules.replk_modules import (
    Transition3D,
    ConvFFN3D,
    Down3DStrideRepLK_FFN2,
    _fold_bn_to_conv,
    _conv_bn_3d,
    _conv_bn_relu_3d,
)
from .modules.mvit_modules import Down3DStrideMViT


# ============================================================================
# Building Blocks
# ============================================================================

class MobileNetV2Block3D(nn.Module):
    """3D MobileNetV2 inverted residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: float = 4.0, norm: str = 'bn'):
        super().__init__()
        assert stride in [1, 2], "Stride must be 1 or 2 for MobileNetV2 blocks."
        self.stride = stride
        self.norm = norm or 'bn'
        hidden_dim = int(round(in_channels * expand_ratio))
        hidden_dim = max(1, hidden_dim)
        self.use_res_connect = (self.stride == 1) and (in_channels == out_channels)

        layers = []
        if hidden_dim != in_channels:
            layers.append(nn.Conv3d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(_make_norm3d(self.norm, hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias=False)
        )
        layers.append(_make_norm3d(self.norm, hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv3d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(_make_norm3d(self.norm, out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res_connect:
            return x + out
        return out


class Down3DMobileNetV2(nn.Module):
    """Downsampling using MobileNetV2 inverted residual block (stride=2)."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', expand_ratio: float = 4.0):
        super().__init__()
        self.block = MobileNetV2Block3D(in_channels, out_channels, stride=2, expand_ratio=expand_ratio, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReparamLargeKernelConv3D_7x7(nn.Module):
    """3D depthwise large-kernel conv (7x7x7) + optional small-kernel branch with BN, re-parameterizable to single conv."""
    def __init__(self, channels: int, kernel_size: int = 7, small_kernel: int = 3, norm: str = 'bn'):
        super().__init__()
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.norm = norm or 'bn'
        # Large kernel depthwise conv + BN
        self.lkb = _conv_bn_3d(channels, channels, kernel_size, 1, padding, groups=channels, norm=self.norm)
        # Optional small kernel depthwise conv + BN
        if small_kernel is not None and small_kernel > 0:
            sp = small_kernel // 2
            self.skb = _conv_bn_3d(channels, channels, small_kernel, 1, sp, groups=channels, norm=self.norm)
        else:
            self.skb = None
        # After merge
        self.lkb_reparam: nn.Conv3d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lkb_reparam is not None:
            return self.lkb_reparam(x)
        out = self.lkb(x)
        if self.skb is not None:
            out = out + self.skb(x)
        return out

    @torch.no_grad()
    def get_equivalent_kernel_bias(self):
        # fold BN for lkb
        k_l, b_l = _fold_bn_to_conv(self.lkb.conv, self.lkb.bn)
        k_eq, b_eq = k_l, b_l
        if self.skb is not None:
            k_s, b_s = _fold_bn_to_conv(self.skb.conv, self.skb.bn)
            pad = (self.kernel_size - self.small_kernel) // 2
            k_s_padded = F.pad(k_s, (pad, pad, pad, pad, pad, pad))
            k_eq = k_eq + k_s_padded
            b_eq = b_eq + b_s
        return k_eq, b_eq

    @torch.no_grad()
    def merge_kernel(self):
        if self.lkb_reparam is not None:
            return
        k_eq, b_eq = self.get_equivalent_kernel_bias()
        ch = k_eq.size(0)
        # ensure new conv is created on the same device as folded weights
        device = k_eq.device
        self.lkb_reparam = nn.Conv3d(ch, ch, kernel_size=self.kernel_size, stride=1,
                                      padding=self.kernel_size // 2, groups=ch, bias=True).to(device)
        self.lkb_reparam.weight.data.copy_(k_eq)
        self.lkb_reparam.bias.data.copy_(b_eq)
        # remove train branches
        del self.lkb
        if self.skb is not None:
            del self.skb
        self.lkb = None
        self.skb = None


class RepLKBlock3D_7x7(nn.Module):
    """RepLK block for 3D with 7x7x7 kernel: Pre-BN -> PW(1x1, expand) -> DW Large-Kernel -> ReLU -> PW(1x1, project) -> Residual."""
    def __init__(self, channels: int, norm: str = 'bn', dw_ratio: float = 1.5):
        super().__init__()
        self.norm = norm or 'bn'
        self.pre_bn = _make_norm3d(self.norm, channels)
        expanded = max(1, int(round(channels * dw_ratio)))
        # pointwise expand then contract (inverted bottleneck)
        self.pw1 = _conv_bn_relu_3d(channels, expanded, kernel_size=1, stride=1, padding=0, groups=1, norm=self.norm)
        self.lk = ReparamLargeKernelConv3D_7x7(expanded, kernel_size=7, small_kernel=3, norm=self.norm)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = _conv_bn_3d(expanded, channels, kernel_size=1, stride=1, padding=0, groups=1, norm=self.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_bn(x)
        out = self.pw1(out)
        out = self.lk(out)
        out = self.act(out)
        out = self.pw2(out)
        return x + out

    @torch.no_grad()
    def switch_to_deploy(self):
        # merge only large-kernel branch; PW convs stay with fused BN handled by framework
        if hasattr(self.lk, 'merge_kernel'):
            self.lk.merge_kernel()


class Down3DStrideRepLK_FFN2_7x7(nn.Module):
    """Transition (PW 1x1 -> DW 3x3 stride=2) -> RepLKBlock3D_7x7 -> ConvFFN3D(expansion_ratio=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.conv_down = Transition3D(in_channels, out_channels, norm=norm)
        self.replk = RepLKBlock3D_7x7(out_channels, norm=norm, dw_ratio=1.5)
        self.ffn = ConvFFN3D(out_channels, expansion_ratio=2, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = self.replk(x)
        x = self.ffn(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        self.replk.switch_to_deploy()


# ============================================================================
# Base Models
# ============================================================================

class DualBranchUNet3D_LK_MobileNet_MViT(nn.Module):
    """Dual-branch UNet with RepLK (FLAIR) + MobileNetV2 (t1ce) + MViT Stage5 - Base class with configurable channel sizes
    
    - FLAIR branch: RepLK (13x13x13) + FFN2
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
        self.branch_flair = Down3DStrideRepLK_FFN2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DMobileNetV2(channels['stem'], channels['branch2'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 3 branches
        self.branch_flair3 = Down3DStrideRepLK_FFN2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2(channels['branch2'], channels['branch3'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 4 branches
        self.branch_flair4 = Down3DStrideRepLK_FFN2(channels['branch3'], channels['down4'], norm=self.norm)
        self.branch_t1ce4 = Down3DMobileNetV2(channels['branch3'], channels['down4'], norm=self.norm, expand_ratio=self.expand_ratio)
        
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
    
    @torch.no_grad()
    def switch_to_deploy(self):
        for m in [self.branch_flair.replk, self.branch_flair3.replk, self.branch_flair4.replk]:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


class DualBranchUNet3D_LK7x7_MobileNet_MViT(nn.Module):
    """Dual-branch UNet with RepLK 7x7x7 (FLAIR) + MobileNetV2 (t1ce) + MViT Stage5 - Base class with configurable channel sizes
    
    - FLAIR branch: RepLK 7x7x7 + FFN2
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
        self.branch_flair = Down3DStrideRepLK_FFN2_7x7(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce = Down3DMobileNetV2(channels['stem'], channels['branch2'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 3 branches
        self.branch_flair3 = Down3DStrideRepLK_FFN2_7x7(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2(channels['branch2'], channels['branch3'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 4 branches
        self.branch_flair4 = Down3DStrideRepLK_FFN2_7x7(channels['branch3'], channels['down4'], norm=self.norm)
        self.branch_t1ce4 = Down3DMobileNetV2(channels['branch3'], channels['down4'], norm=self.norm, expand_ratio=self.expand_ratio)
        
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
    
    @torch.no_grad()
    def switch_to_deploy(self):
        for m in [self.branch_flair.replk, self.branch_flair3.replk, self.branch_flair4.replk]:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


class DualBranchUNet3D_MobileNet_MViT(nn.Module):
    """Dual-branch UNet with MobileNetV2 for both FLAIR and t1ce branches + MViT Stage5 - Base class with configurable channel sizes
    
    - Both branches: MobileNetV2
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
        
        # Stage 2 branches (both MobileNetV2)
        self.branch_flair = Down3DMobileNetV2(channels['stem'], channels['branch2'], norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce = Down3DMobileNetV2(channels['stem'], channels['branch2'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 3 branches (both MobileNetV2)
        self.branch_flair3 = Down3DMobileNetV2(channels['branch2'], channels['branch3'], norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce3 = Down3DMobileNetV2(channels['branch2'], channels['branch3'], norm=self.norm, expand_ratio=self.expand_ratio)
        
        # Stage 4 branches (both MobileNetV2)
        self.branch_flair4 = Down3DMobileNetV2(channels['branch3'], channels['down4'], norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce4 = Down3DMobileNetV2(channels['branch3'], channels['down4'], norm=self.norm, expand_ratio=self.expand_ratio)
        
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


# ============================================================================
# Convenience Classes for Backward Compatibility
# ============================================================================

# DualBranchUNet3D_LK_MobileNet_MViT
class DualBranchUNet3D_LK_MobileNet_MViT_XS(DualBranchUNet3D_LK_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_LK_MobileNet_MViT_Small(DualBranchUNet3D_LK_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_LK_MobileNet_MViT_Medium(DualBranchUNet3D_LK_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_LK_MobileNet_MViT_Large(DualBranchUNet3D_LK_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')

# DualBranchUNet3D_LK7x7_MobileNet_MViT
class DualBranchUNet3D_LK7x7_MobileNet_MViT_XS(DualBranchUNet3D_LK7x7_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_LK7x7_MobileNet_MViT_Small(DualBranchUNet3D_LK7x7_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_LK7x7_MobileNet_MViT_Medium(DualBranchUNet3D_LK7x7_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_LK7x7_MobileNet_MViT_Large(DualBranchUNet3D_LK7x7_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')

# DualBranchUNet3D_MobileNet_MViT
class DualBranchUNet3D_MobileNet_MViT_XS(DualBranchUNet3D_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='xs')

class DualBranchUNet3D_MobileNet_MViT_Small(DualBranchUNet3D_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='s')

class DualBranchUNet3D_MobileNet_MViT_Medium(DualBranchUNet3D_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='m')

class DualBranchUNet3D_MobileNet_MViT_Large(DualBranchUNet3D_MobileNet_MViT):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, expand_ratio=expand_ratio, size='l')


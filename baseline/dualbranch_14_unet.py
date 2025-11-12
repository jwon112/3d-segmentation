import torch
import torch.nn as nn

from .dualbranch_08_unet import MobileNetV2Block3D
from .dualbranch_06_unet import Down3DStrideMViT
from .dualbranch_03_unet import Down3DStrideDilated
from .model_3d_unet import Up3D, OutConv3D, DoubleConv3D, _make_norm3d


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


class GhostModule3D(nn.Module):
    """3D Ghost Module: cheap operations to generate more feature maps."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, 
                 ratio: int = 2, norm: str = 'bn'):
        super().__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            _make_norm3d(norm, init_channels),
            nn.ReLU(inplace=True) if ratio == 2 else nn.ReLU(inplace=True),
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, 
                     groups=init_channels, bias=False),
            _make_norm3d(norm, new_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :, :]


class GhostBottleneck3D(nn.Module):
    """3D Ghost Bottleneck block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = 'bn'):
        super().__init__()
        self.stride = stride
        
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels),
            )
        else:
            self.downsample = None
        
        mid_channels = out_channels // 2
        self.ghost1 = GhostModule3D(in_channels, mid_channels, stride=stride, norm=norm)
        self.ghost2 = GhostModule3D(mid_channels, out_channels, stride=1, norm=norm)
        self.use_res_connect = (stride == 1) and (in_channels == out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        
        out = self.ghost1(x)
        out = self.ghost2(out)
        
        if self.use_res_connect:
            return out + residual
        return out


class Down3DGhostNet(nn.Module):
    """Downsampling using GhostNet bottleneck (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = GhostBottleneck3D(in_channels, out_channels, stride=2, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv3D(nn.Module):
    """3D Depth-wise Separable Convolution block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, norm: str = 'bn'):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, 
                     groups=in_channels, bias=False),
            _make_norm3d(norm, in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Down3DDepthwiseSeparable(nn.Module):
    """Downsampling using Depth-wise Separable Convolution (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv3D(in_channels, out_channels, kernel_size=3, stride=2, 
                                    padding=1, norm=norm),
            DepthwiseSeparableConv3D(out_channels, out_channels, kernel_size=3, stride=1, 
                                    padding=1, norm=norm),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNeXtBlock3D(nn.Module):
    """3D ConvNeXt block: depthwise conv + layer scale + stochastic depth."""
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6, norm: str = 'bn'):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = _make_norm3d(norm, dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # Stochastic depth can be added if needed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1, 1) * x
        
        x = input + self.drop_path(x)
        return x


class Down3DConvNeXt(nn.Module):
    """Downsampling using ConvNeXt block (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', num_blocks: int = 2):
        super().__init__()
        # Downsampling layer
        self.downsample = nn.Sequential(
            _make_norm3d(norm, in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
        )
        # ConvNeXt blocks
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock3D(out_channels, norm=norm) for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# ============================================================================
# Dual-Branch Models with Different Backbones
# ============================================================================

class DualBranchUNet3D_MobileNetV2_Expand2_Small(nn.Module):
    """Dual-branch UNet with MobileNetV2 (expand_ratio=2) for both branches (Small).
    
    Stage 1-4: Dual-branch with MobileNetV2 (expand_ratio=2)
    Stage 5: Fused branch with MobileViT
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        
        # Stage 1 stems (MobileNetV2 blocks, expand_ratio=2)
        self.stem_flair = MobileNetV2Block3D(1, 16, stride=1, expand_ratio=2.0, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, 16, stride=1, expand_ratio=2.0, norm=self.norm)
        
        # Stage 2-4 branches (both MobileNetV2, expand_ratio=2)
        self.branch_flair = Down3DMobileNetV2_Expand2(16, 32, norm=self.norm)
        self.branch_t1ce = Down3DMobileNetV2_Expand2(16, 32, norm=self.norm)
        
        self.branch_flair3 = Down3DMobileNetV2_Expand2(32, 64, norm=self.norm)
        self.branch_t1ce3 = Down3DMobileNetV2_Expand2(32, 64, norm=self.norm)
        
        self.branch_flair4 = Down3DMobileNetV2_Expand2(64, 128, norm=self.norm)
        self.branch_t1ce4 = Down3DMobileNetV2_Expand2(64, 128, norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        self.down5 = Down3DStrideMViT(128 + 128, 512 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)
    
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


class DualBranchUNet3D_GhostNet_Small(nn.Module):
    """Dual-branch UNet with GhostNet for both branches (Small).
    
    Stage 1-4: Dual-branch with GhostNet
    Stage 5: Fused branch with MobileViT
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        
        # Stage 1 stems (GhostNet blocks)
        self.stem_flair = GhostBottleneck3D(1, 16, stride=1, norm=self.norm)
        self.stem_t1ce = GhostBottleneck3D(1, 16, stride=1, norm=self.norm)
        
        # Stage 2-4 branches (both GhostNet)
        self.branch_flair = Down3DGhostNet(16, 32, norm=self.norm)
        self.branch_t1ce = Down3DGhostNet(16, 32, norm=self.norm)
        
        self.branch_flair3 = Down3DGhostNet(32, 64, norm=self.norm)
        self.branch_t1ce3 = Down3DGhostNet(32, 64, norm=self.norm)
        
        self.branch_flair4 = Down3DGhostNet(64, 128, norm=self.norm)
        self.branch_t1ce4 = Down3DGhostNet(64, 128, norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        self.down5 = Down3DStrideMViT(128 + 128, 512 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)
    
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


class DualBranchUNet3D_DepthwiseSeparable_Small(nn.Module):
    """Dual-branch UNet with Depth-wise Separable Conv for both branches (Small).
    
    Stage 1-4: Dual-branch with Depth-wise Separable Conv
    Stage 5: Fused branch with MobileViT
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        
        # Stage 1 stems (Depth-wise Separable Conv)
        self.stem_flair = DepthwiseSeparableConv3D(1, 16, kernel_size=3, stride=1, padding=1, norm=self.norm)
        self.stem_t1ce = DepthwiseSeparableConv3D(1, 16, kernel_size=3, stride=1, padding=1, norm=self.norm)
        
        # Stage 2-4 branches (both Depth-wise Separable Conv)
        self.branch_flair = Down3DDepthwiseSeparable(16, 32, norm=self.norm)
        self.branch_t1ce = Down3DDepthwiseSeparable(16, 32, norm=self.norm)
        
        self.branch_flair3 = Down3DDepthwiseSeparable(32, 64, norm=self.norm)
        self.branch_t1ce3 = Down3DDepthwiseSeparable(32, 64, norm=self.norm)
        
        self.branch_flair4 = Down3DDepthwiseSeparable(64, 128, norm=self.norm)
        self.branch_t1ce4 = Down3DDepthwiseSeparable(64, 128, norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        self.down5 = Down3DStrideMViT(128 + 128, 512 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)
    
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


class DualBranchUNet3D_Dilated_Small(nn.Module):
    """Dual-branch UNet with Dilated Conv (rate 1,2,5) for both branches (Small).
    
    Stage 1-4: Dual-branch with Dilated Conv
    Stage 5: Fused branch with MobileViT
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        
        # Stage 1 stems (DoubleConv)
        self.stem_flair = DoubleConv3D(1, 16, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 16, norm=self.norm)
        
        # Stage 2-4 branches (both Dilated Conv)
        self.branch_flair = Down3DStrideDilated(16, 32, norm=self.norm)
        self.branch_t1ce = Down3DStrideDilated(16, 32, norm=self.norm)
        
        self.branch_flair3 = Down3DStrideDilated(32, 64, norm=self.norm)
        self.branch_t1ce3 = Down3DStrideDilated(32, 64, norm=self.norm)
        
        self.branch_flair4 = Down3DStrideDilated(64, 128, norm=self.norm)
        self.branch_t1ce4 = Down3DStrideDilated(64, 128, norm=self.norm)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        self.down5 = Down3DStrideMViT(128 + 128, 512 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)
    
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


class DualBranchUNet3D_ConvNeXt_Small(nn.Module):
    """Dual-branch UNet with ConvNeXt for both branches (Small).
    
    Stage 1-4: Dual-branch with ConvNeXt
    Stage 5: Fused branch with MobileViT
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        
        # Stage 1 stems (ConvNeXt blocks)
        self.stem_flair = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=1, padding=1, bias=True),
            ConvNeXtBlock3D(16, norm=self.norm),
        )
        self.stem_t1ce = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=1, padding=1, bias=True),
            ConvNeXtBlock3D(16, norm=self.norm),
        )
        
        # Stage 2-4 branches (both ConvNeXt)
        self.branch_flair = Down3DConvNeXt(16, 32, norm=self.norm, num_blocks=2)
        self.branch_t1ce = Down3DConvNeXt(16, 32, norm=self.norm, num_blocks=2)
        
        self.branch_flair3 = Down3DConvNeXt(32, 64, norm=self.norm, num_blocks=2)
        self.branch_t1ce3 = Down3DConvNeXt(32, 64, norm=self.norm, num_blocks=2)
        
        self.branch_flair4 = Down3DConvNeXt(64, 128, norm=self.norm, num_blocks=2)
        self.branch_t1ce4 = Down3DConvNeXt(64, 128, norm=self.norm, num_blocks=2)
        
        # Stage 5 fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        self.down5 = Down3DStrideMViT(128 + 128, 512 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)
        
        # Decoder
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)
    
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


__all__ = [
    'DualBranchUNet3D_MobileNetV2_Expand2_Small',
    'DualBranchUNet3D_GhostNet_Small',
    'DualBranchUNet3D_DepthwiseSeparable_Small',
    'DualBranchUNet3D_Dilated_Small',
    'DualBranchUNet3D_ConvNeXt_Small',
]


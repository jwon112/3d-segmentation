import torch
import torch.nn as nn

from .dualbranch_04_unet import Transition3D, RepLKBlock3D, ConvFFN3D
from .dualbranch_05_unet import Down3DStride, Down3DStrideRepLK_FFN2
from .dualbranch_06_unet import Down3DStrideMViT
from .model_3d_unet import Up3D, OutConv3D, DoubleConv3D


class DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Small(nn.Module):
    """Dual-branch until Stage 4, then fuse and apply MobileViT3D only at Stage 5 (Small).

    - Stage 2/3: FLAIR = RepLK+FFN2, t1ce = stride path, then concat
    - Stage 4: keep branches (FLAIR RepLK+FFN2, t1ce stride), then concat
    - Stage 5: single fused branch with MobileViT3D
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear

        # Stage 1 stems
        self.stem_flair = DoubleConv3D(1, 16, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 16, norm=self.norm)

        # Stage 2 branches
        self.branch_flair = Down3DStrideRepLK_FFN2(16, 32, norm=self.norm)
        self.branch_t1ce  = Down3DStride(16, 32, norm=self.norm)

        # Stage 3 branches
        self.branch_flair3 = Down3DStrideRepLK_FFN2(32, 64, norm=self.norm)
        self.branch_t1ce3  = Down3DStride(32, 64, norm=self.norm)

        # Stage 4 branches (still dual-branch)
        self.branch_flair4 = Down3DStrideRepLK_FFN2(64, 128, norm=self.norm)
        self.branch_t1ce4  = Down3DStride(64, 128, norm=self.norm)

        # Stage 5: single fused branch with MobileViT
        factor = 2 if self.bilinear else 1
        self.down5 = Down3DStrideMViT(128 + 128, 512 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)

        # Decoder
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)

        # Stage 2
        b_flair = self.branch_flair(x1_flair)
        b_t1ce  = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)

        # Stage 3
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce  = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)

        # Stage 4 (still dual-branch)
        b3_flair = self.branch_flair4(b2_flair)
        b3_t1ce  = self.branch_t1ce4(b2_t1ce)
        x4 = torch.cat([b3_flair, b3_t1ce], dim=1)

        # Stage 5 (single MViT branch)
        x5 = self.down5(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        # RepLK reparam in branches
        for m in [self.branch_flair.replk, self.branch_flair3.replk, self.branch_flair4.replk]:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


class DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Medium(nn.Module):
    """Dual-branch until Stage 4, then MobileViT at Stage 5 (Medium)."""
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear

        # Stage 1
        self.stem_flair = DoubleConv3D(1, 32, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 32, norm=self.norm)

        # Stage 2/3 dual-branch
        self.branch_flair = Down3DStrideRepLK_FFN2(32, 64, norm=self.norm)
        self.branch_t1ce  = Down3DStride(32, 64, norm=self.norm)
        self.branch_flair3 = Down3DStrideRepLK_FFN2(64, 128, norm=self.norm)
        self.branch_t1ce3  = Down3DStride(64, 128, norm=self.norm)

        # Stage 4 dual-branch
        self.branch_flair4 = Down3DStrideRepLK_FFN2(128, 256, norm=self.norm)
        self.branch_t1ce4  = Down3DStride(128, 256, norm=self.norm)

        # Stage 5 single MViT
        factor = 2 if self.bilinear else 1
        self.down5 = Down3DStrideMViT(256 + 256, 1024 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)

        # Decoder
        self.up1 = Up3D(1024, 512 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(128, 64, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)

        b_flair = self.branch_flair(x1_flair)
        b_t1ce  = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)

        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce  = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)

        b3_flair = self.branch_flair4(b2_flair)
        b3_t1ce  = self.branch_t1ce4(b2_t1ce)
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



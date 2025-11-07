import torch
import torch.nn as nn

from .dualbranch_08_unet import MobileNetV2Block3D, Down3DMobileNetV2
from .dualbranch_06_unet import Down3DStrideMViT
from .model_3d_unet import Up3D, OutConv3D


class DualBranchUNet3D_MobileNet_MViT_Small(nn.Module):
    """Dual-branch UNet with MobileNetV2 for both FLAIR and t1ce branches + MViT Stage5 (Small)."""

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.expand_ratio = expand_ratio

        # Stage 1 stems (MobileNetV2 blocks)
        self.stem_flair = MobileNetV2Block3D(1, 16, stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, 16, stride=1, expand_ratio=self.expand_ratio, norm=self.norm)

        # Stage 2 branches (both MobileNetV2)
        self.branch_flair = Down3DMobileNetV2(16, 32, norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce = Down3DMobileNetV2(16, 32, norm=self.norm, expand_ratio=self.expand_ratio)

        # Stage 3 branches (both MobileNetV2)
        self.branch_flair3 = Down3DMobileNetV2(32, 64, norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce3 = Down3DMobileNetV2(32, 64, norm=self.norm, expand_ratio=self.expand_ratio)

        # Stage 4 branches (both MobileNetV2)
        self.branch_flair4 = Down3DMobileNetV2(64, 128, norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce4 = Down3DMobileNetV2(64, 128, norm=self.norm, expand_ratio=self.expand_ratio)

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


class DualBranchUNet3D_MobileNet_MViT_Medium(nn.Module):
    """Dual-branch UNet with MobileNetV2 for both FLAIR and t1ce branches + MViT Stage5 (Medium)."""

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.expand_ratio = expand_ratio

        # Stage 1 stems (MobileNetV2 blocks)
        self.stem_flair = MobileNetV2Block3D(1, 32, stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, 32, stride=1, expand_ratio=self.expand_ratio, norm=self.norm)

        # Stage 2 branches (both MobileNetV2)
        self.branch_flair = Down3DMobileNetV2(32, 64, norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce = Down3DMobileNetV2(32, 64, norm=self.norm, expand_ratio=self.expand_ratio)

        # Stage 3 branches (both MobileNetV2)
        self.branch_flair3 = Down3DMobileNetV2(64, 128, norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce3 = Down3DMobileNetV2(64, 128, norm=self.norm, expand_ratio=self.expand_ratio)

        # Stage 4 branches (both MobileNetV2)
        self.branch_flair4 = Down3DMobileNetV2(128, 256, norm=self.norm, expand_ratio=self.expand_ratio)
        self.branch_t1ce4 = Down3DMobileNetV2(128, 256, norm=self.norm, expand_ratio=self.expand_ratio)

        # Stage 5 fused branch with MobileViT
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
    'DualBranchUNet3D_MobileNet_MViT_Small',
    'DualBranchUNet3D_MobileNet_MViT_Medium',
]


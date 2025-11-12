import torch
import torch.nn as nn

from .dualbranch_08_unet import MobileNetV2Block3D, Down3DMobileNetV2
from .dualbranch_06_unet import Down3DStrideMViT
from .model_3d_unet import Up3D, OutConv3D, _make_norm3d


class Down3DStrideDilated_1_2_5(nn.Module):
    """Downsampling with stride-2 Conv and dilated convolutions (rate 1, 2, 5) for wider ERF.
    
    Pattern: Conv(stride=2) -> Norm -> ReLU -> DilatedConv(rate=1) -> Norm -> ReLU -> DilatedConv(rate=2) -> Norm -> ReLU -> DilatedConv(rate=5) -> Norm -> ReLU
    Uses dilated convolutions instead of regular conv to capture wider context.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        # Padding for dilated conv: padding = dilation * (kernel_size - 1) / 2
        # For kernel=3, dilation=1: padding = 1 * (3-1) / 2 = 1
        # For kernel=3, dilation=2: padding = 2 * (3-1) / 2 = 2
        # For kernel=3, dilation=5: padding = 5 * (3-1) / 2 = 5
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualBranchUNet3D_Dilated125_Both_Mobile_Small(nn.Module):
    """Dual-branch UNet with dilated both branches (rate 1,2,5) + MobileNetV2 backbone + MViT (Small).
    
    Both T1CE and FLAIR branches use dilated convolutions (rate 1, 2, 5).
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, expand_ratio: float = 4.0):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.expand_ratio = expand_ratio

        # Stage 1 stems (MobileNetV2 blocks)
        self.stem_flair = MobileNetV2Block3D(1, 16, stride=1, expand_ratio=self.expand_ratio, norm=self.norm)
        self.stem_t1ce = MobileNetV2Block3D(1, 16, stride=1, expand_ratio=self.expand_ratio, norm=self.norm)

        # Stage 2 branches (both use dilated)
        self.branch_flair = Down3DStrideDilated_1_2_5(16, 32, norm=self.norm)
        self.branch_t1ce = Down3DStrideDilated_1_2_5(16, 32, norm=self.norm)

        # Stage 3 branches (both use dilated)
        self.branch_flair3 = Down3DStrideDilated_1_2_5(32, 64, norm=self.norm)
        self.branch_t1ce3 = Down3DStrideDilated_1_2_5(32, 64, norm=self.norm)

        # Stage 4 branches (both use dilated)
        self.branch_flair4 = Down3DStrideDilated_1_2_5(64, 128, norm=self.norm)
        self.branch_t1ce4 = Down3DStrideDilated_1_2_5(64, 128, norm=self.norm)

        factor = 2 if self.bilinear else 1
        self.down5 = Down3DStrideMViT(128 + 128, 512 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)

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
    'DualBranchUNet3D_Dilated125_Both_Mobile_Small',
]


import torch
import torch.nn as nn

# Reuse blocks from dualbranch_04 (RepLK transition/block/ffn)
from .dualbranch_04_unet import (
    Transition3D,
    RepLKBlock3D,
    ConvFFN3D,
    Down3DStride,
    Up3D,
    OutConv3D,
    DoubleConv3D,
)


class Down3DStrideRepLK_FFN2(nn.Module):
    """Transition (PW 1x1 -> DW 3x3 stride=2) -> RepLKBlock3D(dw_ratio=1.5) -> ConvFFN3D(expansion_ratio=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.conv_down = Transition3D(in_channels, out_channels, norm=norm)
        self.replk = RepLKBlock3D(out_channels, norm=norm, dw_ratio=1.5)
        self.ffn = ConvFFN3D(out_channels, expansion_ratio=2, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = self.replk(x)
        x = self.ffn(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        self.replk.switch_to_deploy()


class DualBranchUNet3D_StrideLK_FFN2_Small(nn.Module):
    """dualbranch_04 변형: FFN ratio=2 (Small)."""
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
        self.branch_t1ce = Down3DStride(16, 32, norm=self.norm)

        # Stage 3 branches
        self.branch_flair3 = Down3DStrideRepLK_FFN2(32, 64, norm=self.norm)
        self.branch_t1ce3 = Down3DStride(32, 64, norm=self.norm)

        # Stages 4+
        self.down3 = Down3DStride(128, 256, norm=self.norm)
        factor = 2 if self.bilinear else 1
        self.down4 = Down3DStride(256, 512 // factor, norm=self.norm)

        # Decoder
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)

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

        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        self.branch_flair.switch_to_deploy()
        self.branch_flair3.switch_to_deploy()


class DualBranchUNet3D_StrideLK_FFN2_Medium(nn.Module):
    """dualbranch_04 변형: FFN ratio=2 (Medium)."""
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear

        # Stage 1 stems
        self.stem_flair = DoubleConv3D(1, 32, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 32, norm=self.norm)

        # Stage 2 branches
        self.branch_flair = Down3DStrideRepLK_FFN2(32, 64, norm=self.norm)
        self.branch_t1ce = Down3DStride(32, 64, norm=self.norm)

        # Stage 3 branches
        self.branch_flair3 = Down3DStrideRepLK_FFN2(64, 128, norm=self.norm)
        self.branch_t1ce3 = Down3DStride(64, 128, norm=self.norm)

        # Stages 4+
        self.down3 = Down3DStride(256, 512, norm=self.norm)
        factor = 2 if self.bilinear else 1
        self.down4 = Down3DStride(512, 1024 // factor, norm=self.norm)

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

        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        self.branch_flair.switch_to_deploy()
        self.branch_flair3.switch_to_deploy()



import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse common UNet building blocks
from .model_3d_unet import DoubleConv3D, Up3D, OutConv3D, _make_norm3d


class Down3DStride(nn.Module):
    """Downsampling with stride-2 Conv instead of MaxPool.
    Pattern: Conv(stride=2) -> Norm -> ReLU -> Conv -> Norm -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3DStrideDilated(nn.Module):
    """Downsampling with stride-2 Conv and dilated convolutions for wider ERF.
    
    Pattern: Conv(stride=2) -> Norm -> ReLU -> DilatedConv(rate=2) -> Norm -> ReLU -> DilatedConv(rate=5) -> Norm -> ReLU
    Uses dilated convolutions instead of regular conv to capture wider context.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        # Padding for dilated conv: padding = dilation * (kernel_size - 1) / 2
        # For kernel=5, dilation=2: padding = 2 * (5-1) / 2 = 4
        # For kernel=5, dilation=5: padding = 5 * (5-1) / 2 = 10
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
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


class DualBranchUNet3D_StrideDilated_Small(nn.Module):
    """Dual-branch 3D U-Net (v0.3 - dilated conv for FLAIR) - Small channel version

    Differences from dualbranch_02_unet:
    - FLAIR branch uses dilated convolutions (rate 2, rate 3) for wider ERF
    - t1ce branch uses standard 3x3 convolutions (unchanged)
    - This allows FLAIR to capture more contextual information while t1ce focuses on local details

    Width matches UNet3D_Small for fair comparison.
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2, "DualBranchUNet3D_StrideDilated_Small expects exactly 2 input channels (FLAIR, t1ce)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear

        # Stage 1: modality-specific stems (1->16 each), then concat -> 32ch
        self.stem_flair = DoubleConv3D(1, 16, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 16, norm=self.norm)

        # Stage 2: modality-specific branches with different architectures
        # FLAIR: dilated convs for wider ERF, t1ce: standard convs
        self.branch_flair = Down3DStrideDilated(16, 32, norm=self.norm)
        self.branch_t1ce = Down3DStride(16, 32, norm=self.norm)

        # Stage 3: extend the same pattern (FLAIR dilated, t1ce standard)
        self.branch_flair3 = Down3DStrideDilated(32, 64, norm=self.norm)
        self.branch_t1ce3 = Down3DStride(32, 64, norm=self.norm)

        # Stages 4+: encoder (128->256->512)
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
        # Stage 1
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)  # (B,32,...)

        # Stage 2 branches: FLAIR uses dilated conv, t1ce uses standard conv
        b_flair = self.branch_flair(x1_flair)  # (B,32,.../2) - wider ERF
        b_t1ce = self.branch_t1ce(x1_t1ce)    # (B,32,.../2) - local details
        x2 = torch.cat([b_flair, b_t1ce], dim=1)  # (B,64,.../2)

        # Stage 3 branches (FLAIR dilated, t1ce standard)
        b2_flair = self.branch_flair3(b_flair)  # (B,64,.../4)
        b2_t1ce  = self.branch_t1ce3(b_t1ce)   # (B,64,.../4)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)  # (B,128,.../4)

        # Encoder
        x4 = self.down3(x3)   # 256
        x5 = self.down4(x4)   # 512

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DualBranchUNet3D_StrideDilated_Medium(nn.Module):
    """Dual-branch 3D U-Net (v0.3 - dilated conv for FLAIR) - Medium channel version

    Differences from dualbranch_02_unet:
    - FLAIR branch uses dilated convolutions (rate 2, rate 3) for wider ERF
    - t1ce branch uses standard 3x3 convolutions (unchanged)
    - This allows FLAIR to capture more contextual information while t1ce focuses on local details

    Width matches UNet3D_Medium (64, 128, 256, 512, 1024) for fair comparison.
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2, "DualBranchUNet3D_StrideDilated_Medium expects exactly 2 input channels (FLAIR, t1ce)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear

        # Stage 1: modality-specific stems (1->32 each), then concat -> 64ch
        self.stem_flair = DoubleConv3D(1, 32, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 32, norm=self.norm)

        # Stage 2: modality-specific branches with different architectures
        # FLAIR: dilated convs for wider ERF, t1ce: standard convs
        self.branch_flair = Down3DStrideDilated(32, 64, norm=self.norm)
        self.branch_t1ce = Down3DStride(32, 64, norm=self.norm)

        # Stage 3: extend the same pattern (FLAIR dilated, t1ce standard)
        self.branch_flair3 = Down3DStrideDilated(64, 128, norm=self.norm)
        self.branch_t1ce3 = Down3DStride(64, 128, norm=self.norm)

        # Stages 4+: encoder (256->512->1024)
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
        # Stage 1
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)  # (B,64,...)

        # Stage 2 branches: FLAIR uses dilated conv, t1ce uses standard conv
        b_flair = self.branch_flair(x1_flair)  # (B,64,.../2) - wider ERF
        b_t1ce = self.branch_t1ce(x1_t1ce)    # (B,64,.../2) - local details
        x2 = torch.cat([b_flair, b_t1ce], dim=1)  # (B,128,.../2)

        # Stage 3 branches (FLAIR dilated, t1ce standard)
        b2_flair = self.branch_flair3(b_flair)  # (B,128,.../4)
        b2_t1ce  = self.branch_t1ce3(b_t1ce)   # (B,128,.../4)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)  # (B,256,.../4)

        # Encoder (UNet3D_Medium: 256->512->1024)
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualBranchUNet3D_StrideDilated_Small(n_channels=2, n_classes=4).to(device)
    inp = torch.randn(1, 2, 64, 64, 64).to(device)
    with torch.no_grad():
        out = model(inp)
    print(f"Input: {tuple(inp.shape)} -> Output: {tuple(out.shape)}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


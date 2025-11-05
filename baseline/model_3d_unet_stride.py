import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import _make_norm3d, DoubleConv3D, Up3D, OutConv3D


class Down3DStride(nn.Module):
    """3D Downsampling block using stride-2 convolution (no MaxPool).

    Structure: Conv3d(stride=2) -> Norm -> ReLU -> Conv3d -> Norm -> ReLU
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


class UNet3D_Stride_Small(nn.Module):
    """UNet3D Small variant using stride-2 downsampling instead of MaxPool.

    Channel widths match UNet3D_Small for fair comparison.
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')

        # Encoder
        self.enc1 = DoubleConv3D(n_channels, 32, norm=self.norm)
        self.enc2 = Down3DStride(32, 64, norm=self.norm)
        self.enc3 = Down3DStride(64, 128, norm=self.norm)
        self.enc4 = Down3DStride(128, 256, norm=self.norm)

        # Bottleneck
        self.bottleneck = DoubleConv3D(256, 512, norm=self.norm)

        # Decoder (keep bilinear=False to match channel math with transposed conv)
        self.dec4 = Up3D(512, 256, bilinear=False, norm=self.norm)
        self.dec3 = Up3D(256, 128, bilinear=False, norm=self.norm)
        self.dec2 = Up3D(128, 64, bilinear=False, norm=self.norm)
        self.dec1 = Up3D(64, 32, bilinear=False, norm=self.norm)

        # Output
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Output
        logits = self.outc(d1)
        return logits


class UNet3D_Stride_Medium(nn.Module):
    """UNet3D Medium variant using stride-2 downsampling instead of MaxPool.

    Channel widths match UNet3D_Medium (64, 128, 256, 512, 1024) for fair comparison.
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear

        # Encoder (UNet3D standard channels)
        self.inc = DoubleConv3D(n_channels, 64, norm=self.norm)
        self.down1 = Down3DStride(64, 128, norm=self.norm)
        self.down2 = Down3DStride(128, 256, norm=self.norm)
        self.down3 = Down3DStride(256, 512, norm=self.norm)
        factor = 2 if bilinear else 1
        self.down4 = Down3DStride(512, 1024 // factor, norm=self.norm)

        # Decoder
        self.up1 = Up3D(1024, 512 // factor, bilinear, norm=self.norm)
        self.up2 = Up3D(512, 256 // factor, bilinear, norm=self.norm)
        self.up3 = Up3D(256, 128 // factor, bilinear, norm=self.norm)
        self.up4 = Up3D(128, 64, bilinear, norm=self.norm)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D_Stride_Small(n_channels=4, n_classes=4).to(device)
    x = torch.randn(1, 4, 64, 64, 64).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



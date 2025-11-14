import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import _make_norm3d, DoubleConv3D, Up3D, OutConv3D
from .channel_configs import get_unet_channels


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


class UNet3D_Stride(nn.Module):
    """UNet3D variant using stride-2 downsampling instead of MaxPool.
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False, size: str = 's'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear
        self.size = size
        
        # Get channel configuration
        channels = get_unet_channels(size)
        enc_channels = channels['enc']

        # Encoder
        self.enc1 = DoubleConv3D(n_channels, enc_channels[0], norm=self.norm)
        self.enc2 = Down3DStride(enc_channels[0], enc_channels[1], norm=self.norm)
        self.enc3 = Down3DStride(enc_channels[1], enc_channels[2], norm=self.norm)
        self.enc4 = Down3DStride(enc_channels[2], enc_channels[3], norm=self.norm)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv3D(enc_channels[3], channels['bottleneck'] // factor, norm=self.norm)

        # Decoder
        self.dec4 = Up3D(channels['bottleneck'], enc_channels[3] // factor, bilinear, norm=self.norm)
        self.dec3 = Up3D(enc_channels[3], enc_channels[2] // factor, bilinear, norm=self.norm)
        self.dec2 = Up3D(enc_channels[2], enc_channels[1] // factor, bilinear, norm=self.norm)
        self.dec1 = Up3D(enc_channels[1], enc_channels[0] // factor, bilinear, norm=self.norm)

        # Output
        self.outc = OutConv3D(enc_channels[0] // factor, n_classes)

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


# Convenience classes for backward compatibility
class UNet3D_Stride_XS(UNet3D_Stride):
    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')


class UNet3D_Stride_Small(UNet3D_Stride):
    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')


class UNet3D_Stride_Medium(UNet3D_Stride):
    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')


class UNet3D_Stride_Large(UNet3D_Stride):
    def __init__(self, n_channels: int = 4, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')


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



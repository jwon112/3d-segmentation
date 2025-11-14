import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_norm3d(norm: str, num_features: int) -> nn.Module:
    norm = (norm or 'bn').lower()
    if norm in ('in', 'instancenorm', 'instance'):
        return nn.InstanceNorm3d(num_features, affine=True, track_running_stats=False)
    if norm in ('gn', 'groupnorm', 'group'):
        # 기본 그룹 수 8 (채널 수가 8의 배수가 아닐 경우 4로 폴백)
        num_groups = 8 if num_features % 8 == 0 else (4 if num_features % 4 == 0 else 1)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    # 기본값: BatchNorm3d
    return nn.BatchNorm3d(num_features)


class DoubleConv3D(nn.Module):
    """3D Double Convolution 블록"""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm: str = 'bn'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """3D Downsampling 블록"""
    def __init__(self, in_channels, out_channels, norm: str = 'bn'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, norm=norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """3D Upsampling 블록"""
    def __init__(self, in_channels, out_channels, bilinear=True, norm: str = 'bn', skip_channels=None):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # bilinear=False일 때는 upsampling 후 skip connection과 concat하므로
            # total_channels = (in_channels // 2) + skip_channels
            # skip_channels가 None이면 기본값으로 in_channels // 2를 사용 (기존 동작 유지)
            if skip_channels is None:
                skip_channels = in_channels // 2
            total_channels = (in_channels // 2) + skip_channels
            self.conv = DoubleConv3D(total_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 크기 맞추기
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    """3D Output Convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

from .channel_configs import get_unet_channels


class UNet3D(nn.Module):
    """3D U-Net with configurable channel sizes
    
    Channel widths are configurable via size parameter ('xs', 's', 'm', 'l')
    """
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear=False, size: str = 's'):
        super(UNet3D, self).__init__()
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
        self.enc2 = Down3D(enc_channels[0], enc_channels[1], norm=self.norm)
        self.enc3 = Down3D(enc_channels[1], enc_channels[2], norm=self.norm)
        self.enc4 = Down3D(enc_channels[2], enc_channels[3], norm=self.norm)
        
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

    def forward(self, x):
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
class UNet3D_XS(UNet3D):
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear=False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')


class UNet3D_Small(UNet3D):
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear=False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')


class UNet3D_Medium(UNet3D):
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear=False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')


class UNet3D_Large(UNet3D):
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'bn', bilinear=False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='l')

# Losses and metrics moved to ml/losses.py and ml/metrics.py

if __name__ == "__main__":
    # 모델 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small 모델 사용 (메모리 효율성)
    model = UNet3D_Small(n_channels=4, n_classes=4).to(device)
    
    # 더미 입력 생성
    batch_size = 1
    input_tensor = torch.randn(batch_size, 4, 64, 64, 64).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

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


class HardSwish(nn.Module):
    """Hard-Swish 활성화 함수
    
    MobileNetV3에서 사용된 활성화 함수로, ReLU보다 더 부드러운 비선형성을 제공합니다.
    x * ReLU6(x + 3) / 6
    
    Reference:
        Searching for MobileNetV3 (Howard et al., ICCV 2019)
    """
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


def _make_activation(activation: str = 'relu', inplace: bool = True) -> nn.Module:
    """활성화 함수 생성 헬퍼 함수
    
    Args:
        activation: 활성화 함수 타입 ('relu', 'hardswish', 'hswish', 'gelu')
        inplace: inplace 연산 사용 여부
    
    Returns:
        활성화 함수 모듈
    """
    activation = (activation or 'relu').lower()
    if activation in ('hardswish', 'hswish'):
        return HardSwish(inplace=inplace)
    elif activation in ('gelu',):
        return nn.GELU()
    elif activation in ('relu',):
        return nn.ReLU(inplace=inplace)
    else:
        raise ValueError(f"Unknown activation: {activation}")


class DoubleConv3D(nn.Module):
    """3D Double Convolution 블록
    
    nnUNet 스타일: LeakyReLU 활성화 함수, conv_bias=True 사용
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, norm: str = 'bn'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            _make_norm3d(norm, mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            _make_norm3d(norm, out_channels),
            nn.LeakyReLU(inplace=True)
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
            # bilinear=True일 때도 skip connection을 concat하므로 채널 수를 고려해야 함
            # skip_channels가 None이면 기본값으로 in_channels // 2를 사용
            if skip_channels is None:
                skip_channels = in_channels // 2
            total_channels = in_channels + skip_channels
            self.conv = DoubleConv3D(total_channels, out_channels, in_channels // 2, norm=norm)
        else:
            # nnUNet PlainConvUNet 스타일:
            # ConvTranspose3d 출력 채널 수는 해당 decoder stage의 feature 수(out_channels)와 동일
            #   - 예: bottleneck 320 -> upsample 256, skip 256 → concat 512 채널
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            # bilinear=False일 때는 upsampling 후 skip connection과 concat
            # total_channels = out_channels + skip_channels
            # skip_channels가 None이면 기본값으로 out_channels를 사용
            if skip_channels is None:
                skip_channels = out_channels
            total_channels = out_channels + skip_channels
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
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'in', bilinear=False, size: str = 's', deep_supervision: bool = True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'in')
        self.bilinear = bilinear
        self.size = size
        self.deep_supervision = deep_supervision
        
        # Get channel configuration
        channels = get_unet_channels(size)
        enc_channels = channels['enc']
        
        # Encoder (4 stages: [32, 64, 128, 256])
        # nnUNet과 동일하게 encoder는 4개 stage만 사용
        self.enc1 = DoubleConv3D(n_channels, enc_channels[0], norm=self.norm)
        self.enc2 = Down3D(enc_channels[0], enc_channels[1], norm=self.norm)
        self.enc3 = Down3D(enc_channels[1], enc_channels[2], norm=self.norm)
        self.enc4 = Down3D(enc_channels[2], enc_channels[3], norm=self.norm)
        
        # Bottleneck (nnUNet의 마지막 encoder stage와 동일한 채널 수)
        # nnUNet에서는 encoder의 마지막 stage가 bottleneck 역할
        # 우리는 별도 bottleneck 블록 사용하지만 채널 수는 동일하게 유지
        factor = 2 if bilinear else 1
        bottleneck_channel = channels['bottleneck'] // factor
        self.bottleneck = DoubleConv3D(enc_channels[3], bottleneck_channel, norm=self.norm)
        
        # Decoder
        # nnUNet 스타일: decoder의 in_channels는 skip connection 채널과 일치해야 함
        # dec4: bottleneck 출력과 enc4 skip connection 결합
        #   - bottleneck 출력: bottleneck_channel
        #   - enc4 skip: enc_channels[3] = 256
        #   - bilinear=False일 때: ConvTranspose3d로 절반으로 줄이므로 bottleneck_channel // 2 + enc_channels[3]
        #   - bilinear=True일 때: Upsample이므로 bottleneck_channel + enc_channels[3]
        if bilinear:
            # bilinear=True: Upsample 사용, skip과 concat
            self.dec4 = Up3D(bottleneck_channel, enc_channels[3] // factor, bilinear, norm=self.norm, skip_channels=enc_channels[3])
            self.dec3 = Up3D(enc_channels[3] // factor, enc_channels[2] // factor, bilinear, norm=self.norm, skip_channels=enc_channels[2])
            self.dec2 = Up3D(enc_channels[2] // factor, enc_channels[1] // factor, bilinear, norm=self.norm, skip_channels=enc_channels[1])
            self.dec1 = Up3D(enc_channels[1] // factor, enc_channels[0] // factor, bilinear, norm=self.norm, skip_channels=enc_channels[0])
        else:
            # bilinear=False: ConvTranspose3d 사용
            # dec4: bottleneck_channel -> bottleneck_channel // 2 (upsample) + enc_channels[3] (skip) = bottleneck_channel // 2 + enc_channels[3]
            # 하지만 bottleneck_channel이 이미 enc_channels[3]과 다를 수 있으므로, skip_channels를 명시적으로 지정
            self.dec4 = Up3D(bottleneck_channel, enc_channels[3] // factor, bilinear, norm=self.norm, skip_channels=enc_channels[3])
            self.dec3 = Up3D(enc_channels[3] // factor, enc_channels[2] // factor, bilinear, norm=self.norm, skip_channels=enc_channels[2])
            self.dec2 = Up3D(enc_channels[2] // factor, enc_channels[1] // factor, bilinear, norm=self.norm, skip_channels=enc_channels[1])
            self.dec1 = Up3D(enc_channels[1] // factor, enc_channels[0] // factor, bilinear, norm=self.norm, skip_channels=enc_channels[0])
        
        # Output
        self.outc = OutConv3D(enc_channels[0] // factor, n_classes)
        
        # Deep Supervision outputs (각 디코더 레벨에서 출력)
        if deep_supervision:
            self.ds4 = OutConv3D(enc_channels[3] // factor, n_classes)
            self.ds3 = OutConv3D(enc_channels[2] // factor, n_classes)
            self.ds2 = OutConv3D(enc_channels[1] // factor, n_classes)

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
        
        # Main output
        main_output = self.outc(d1)
        
        # Deep Supervision outputs
        if self.deep_supervision:
            ds4 = self.ds4(d4)
            ds3 = self.ds3(d3)
            ds2 = self.ds2(d2)
            # 깊은 레벨일수록 낮은 가중치를 위해 역순으로 반환 (main이 가장 중요)
            return [main_output, ds2, ds3, ds4]
        else:
            return main_output


# Convenience classes for backward compatibility
class UNet3D_XS(UNet3D):
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'in', bilinear=False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='xs')


class UNet3D_Small(UNet3D):
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'in', bilinear=False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='s')


class UNet3D_Medium(UNet3D):
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'in', bilinear=False):
        super().__init__(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=bilinear, size='m')


class UNet3D_Large(UNet3D):
    def __init__(self, n_channels=4, n_classes=4, norm: str = 'in', bilinear=False):
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

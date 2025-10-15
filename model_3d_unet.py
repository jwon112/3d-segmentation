import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """3D Double Convolution 블록"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """3D Downsampling 블록"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """3D Upsampling 블록"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # bilinear=False일 때는 upsampling 후 skip connection과 concat하므로
            # in_channels = (in_channels // 2) + skip_channels
            # 여기서는 skip_channels = in_channels // 2이므로
            # total_channels = (in_channels // 2) + (in_channels // 2) = in_channels
            self.conv = DoubleConv3D(in_channels, out_channels)

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

class UNet3D(nn.Module):
    """3D U-Net 모델"""
    def __init__(self, n_channels=4, n_classes=4, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024 // factor)
        
        self.up1 = Up3D(1024, 512 // factor, bilinear)
        self.up2 = Up3D(512, 256 // factor, bilinear)
        self.up3 = Up3D(256, 128 // factor, bilinear)
        self.up4 = Up3D(128, 64, bilinear)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x):
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

class UNet3D_Simplified(nn.Module):
    """간소화된 3D U-Net (메모리 효율성을 위해)"""
    def __init__(self, n_channels=4, n_classes=4):
        super(UNet3D_Simplified, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.enc1 = DoubleConv3D(n_channels, 32)
        self.enc2 = Down3D(32, 64)
        self.enc3 = Down3D(64, 128)
        self.enc4 = Down3D(128, 256)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(256, 512)
        
        # Decoder - bilinear=False로 설정하여 채널 수 문제 해결
        self.dec4 = Up3D(512, 256, bilinear=False)
        self.dec3 = Up3D(256, 128, bilinear=False)
        self.dec2 = Up3D(128, 64, bilinear=False)
        self.dec1 = Up3D(64, 32, bilinear=False)
        
        # Output
        self.outc = OutConv3D(32, n_classes)

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

def dice_loss(pred, target, smooth=1e-5):
    """Dice Loss 계산"""
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target, alpha=0.5):
    """Cross Entropy + Dice Loss"""
    ce_loss = F.cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return alpha * ce_loss + (1 - alpha) * dice

def calculate_dice_score(pred, target, smooth=1e-5):
    """Dice Score 계산"""
    pred = torch.argmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.max().item() + 1).permute(0, 4, 1, 2, 3).float()
    pred_one_hot = F.one_hot(pred, num_classes=pred.max().item() + 1).permute(0, 4, 1, 2, 3).float()
    
    intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3, 4))
    union = pred_one_hot.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean(dim=0)  # 클래스별 평균

if __name__ == "__main__":
    # 모델 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 간소화된 모델 사용 (메모리 효율성)
    model = UNet3D_Simplified(n_channels=4, n_classes=4).to(device)
    
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

"""
모달리티 비교 실험을 위한 UNet 모델들

4가지 시나리오:
1. UNet3D_2Modal_Small: 단일 분기, 2채널 (t1ce, flair) concat
2. UNet3D_4Modal_Small: 단일 분기, 4채널 (t1, t1ce, t2, flair) concat
3. QuadBranchUNet3D_4Modal_Small: 4개 분기 (t1, t1ce, t2, flair 각각 분기)
4. DualBranchUNet3D_2Modal_Small: 2개 분기 (t1ce, flair) - dualbranch_02_unet과 동일
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import DoubleConv3D, Down3D, Up3D, OutConv3D, _make_norm3d


class UNet3D_2Modal_Small(nn.Module):
    """단일 분기 UNet - 2개 모달리티 (t1ce, flair) 채널 concat
    
    Input: (B, 2, D, H, W) - [t1ce, flair]
    """
    def __init__(self, n_classes=4, norm: str = 'bn'):
        super(UNet3D_2Modal_Small, self).__init__()
        self.n_channels = 2  # 고정: t1ce, flair
        self.n_classes = n_classes
        self.norm = (norm or 'bn')

        # Encoder
        self.enc1 = DoubleConv3D(2, 32, norm=self.norm)  # 2채널 입력
        self.enc2 = Down3D(32, 64, norm=self.norm)
        self.enc3 = Down3D(64, 128, norm=self.norm)
        self.enc4 = Down3D(128, 256, norm=self.norm)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(256, 512, norm=self.norm)
        
        # Decoder
        self.dec4 = Up3D(512, 256, bilinear=False, norm=self.norm)
        self.dec3 = Up3D(256, 128, bilinear=False, norm=self.norm)
        self.dec2 = Up3D(128, 64, bilinear=False, norm=self.norm)
        self.dec1 = Up3D(64, 32, bilinear=False, norm=self.norm)
        
        # Output
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x):
        # x: (B, 2, D, H, W) - [t1ce, flair]
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


class UNet3D_4Modal_Small(nn.Module):
    """단일 분기 UNet - 4개 모달리티 (t1, t1ce, t2, flair) 채널 concat
    
    Input: (B, 4, D, H, W) - [t1, t1ce, t2, flair]
    """
    def __init__(self, n_classes=4, norm: str = 'bn'):
        super(UNet3D_4Modal_Small, self).__init__()
        self.n_channels = 4  # 고정: t1, t1ce, t2, flair
        self.n_classes = n_classes
        self.norm = (norm or 'bn')

        # Encoder
        self.enc1 = DoubleConv3D(4, 32, norm=self.norm)  # 4채널 입력
        self.enc2 = Down3D(32, 64, norm=self.norm)
        self.enc3 = Down3D(64, 128, norm=self.norm)
        self.enc4 = Down3D(128, 256, norm=self.norm)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(256, 512, norm=self.norm)
        
        # Decoder
        self.dec4 = Up3D(512, 256, bilinear=False, norm=self.norm)
        self.dec3 = Up3D(256, 128, bilinear=False, norm=self.norm)
        self.dec2 = Up3D(128, 64, bilinear=False, norm=self.norm)
        self.dec1 = Up3D(64, 32, bilinear=False, norm=self.norm)
        
        # Output
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x):
        # x: (B, 4, D, H, W) - [t1, t1ce, t2, flair]
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


class QuadBranchUNet3D_4Modal_Small(nn.Module):
    """4개 분기 UNet - 4개 모달리티 각각 분기 처리 (채널 수 절반으로 조정)
    
    Input: (B, 4, D, H, W) - [t1, t1ce, t2, flair]
    - Stage 1: 각 모달리티별 stem (1->8 each), concat -> 32ch (다른 모델과 동일)
    - Stage 2: 각 모달리티별 branch (8->16 each), concat -> 64ch (다른 모델과 동일)
    - Stage 3: 각 모달리티별 branch (16->32 each), concat -> 128ch (다른 모델과 동일)
    - Stage 4+: 단일 융합 분기
    """
    def __init__(self, n_classes=4, norm: str = 'bn', bilinear: bool = False):
        super(QuadBranchUNet3D_4Modal_Small, self).__init__()
        self.n_channels = 4  # 고정: t1, t1ce, t2, flair
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear

        # Stage 1: modality-specific stems (1->8 each), then concat -> 32ch (다른 모델과 동일)
        self.stem_t1 = DoubleConv3D(1, 8, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 8, norm=self.norm)
        self.stem_t2 = DoubleConv3D(1, 8, norm=self.norm)
        self.stem_flair = DoubleConv3D(1, 8, norm=self.norm)

        # Stage 2: modality-specific branches (8->16 each), then concat -> 64ch (다른 모델과 동일)
        self.branch_t1 = Down3D(8, 16, norm=self.norm)
        self.branch_t1ce = Down3D(8, 16, norm=self.norm)
        self.branch_t2 = Down3D(8, 16, norm=self.norm)
        self.branch_flair = Down3D(8, 16, norm=self.norm)

        # Stage 3: modality-specific branches (16->32 each), then concat -> 128ch (다른 모델과 동일)
        self.branch_t1_3 = Down3D(16, 32, norm=self.norm)
        self.branch_t1ce_3 = Down3D(16, 32, norm=self.norm)
        self.branch_t2_3 = Down3D(16, 32, norm=self.norm)
        self.branch_flair_3 = Down3D(16, 32, norm=self.norm)

        # Stage 4+: 단일 융합 분기 (다른 모델과 동일)
        self.down4 = Down3D(128, 256, norm=self.norm)
        factor = 2 if self.bilinear else 1
        self.down5 = Down3D(256, 512 // factor, norm=self.norm)

        # Decoder (다른 모델과 동일)
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, D, H, W) - [t1, t1ce, t2, flair]
        
        # Stage 1: modality-specific stems
        x1_t1 = self.stem_t1(x[:, 0:1])      # (B, 8, ...)
        x1_t1ce = self.stem_t1ce(x[:, 1:2])  # (B, 8, ...)
        x1_t2 = self.stem_t2(x[:, 2:3])      # (B, 8, ...)
        x1_flair = self.stem_flair(x[:, 3:4]) # (B, 8, ...)
        x1 = torch.cat([x1_t1, x1_t1ce, x1_t2, x1_flair], dim=1)  # (B, 32, ...)

        # Stage 2: modality-specific branches
        b_t1 = self.branch_t1(x1_t1)      # (B, 16, .../2)
        b_t1ce = self.branch_t1ce(x1_t1ce)  # (B, 16, .../2)
        b_t2 = self.branch_t2(x1_t2)      # (B, 16, .../2)
        b_flair = self.branch_flair(x1_flair)  # (B, 16, .../2)
        x2 = torch.cat([b_t1, b_t1ce, b_t2, b_flair], dim=1)  # (B, 64, .../2)

        # Stage 3: modality-specific branches
        b2_t1 = self.branch_t1_3(b_t1)      # (B, 32, .../4)
        b2_t1ce = self.branch_t1ce_3(b_t1ce)  # (B, 32, .../4)
        b2_t2 = self.branch_t2_3(b_t2)      # (B, 32, .../4)
        b2_flair = self.branch_flair_3(b_flair)  # (B, 32, .../4)
        x3 = torch.cat([b2_t1, b2_t1ce, b2_t2, b2_flair], dim=1)  # (B, 128, .../4)

        # Stage 4+: 단일 융합 분기
        x4 = self.down4(x3)   # (B, 256, .../8)
        x5 = self.down5(x4)   # (B, 512, .../16)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# DualBranchUNet3D_2Modal_Small은 dualbranch_02_unet.py의 DualBranchUNet3D_Stride_Small과 동일
# 별도로 만들 필요 없이 기존 모델 재사용


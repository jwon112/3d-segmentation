"""
모달리티 비교 실험을 위한 UNet 모델들

시나리오:
1. UNet3D_2Modal_Small: 단일 분기, 2채널 (t1ce, flair) concat
2. UNet3D_4Modal_Small: 단일 분기, 4채널 (t1, t1ce, t2, flair) concat
3. DualBranchUNet3D_2Modal_Small: 2개 분기 (t1ce, flair) - 별도 dualbranch 모듈 참고
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_3d_unet import DoubleConv3D, Down3D, Up3D, OutConv3D, _make_norm3d


class ModalityAttention3D(nn.Module):
    """모달리티별 어텐션 (SE-Net 스타일)
    
    각 모달리티 브랜치의 출력에 가중치를 부여하여 모달리티별 기여도를 학습.
    Global Average Pooling + FC를 통해 모달리티별 어텐션 가중치를 생성.
    
    Returns:
        weighted_features: 가중치가 적용된 모달리티별 feature list
        attention_weights: [B, num_modalities] 형태의 어텐션 가중치
    """
    def __init__(self, num_modalities=4, channels_per_modality=32, reduction=4):
        super(ModalityAttention3D, self).__init__()
        self.num_modalities = num_modalities
        self.channels_per_modality = channels_per_modality
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # FC layers for attention weights
        # 입력: 각 모달리티의 GAP feature (num_modalities * channels_per_modality)
        # 출력: 각 모달리티별 가중치 (num_modalities)
        bottleneck_dim = (channels_per_modality * num_modalities) // reduction
        self.fc = nn.Sequential(
            nn.Linear(channels_per_modality * num_modalities, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, num_modalities),
            nn.Sigmoid()  # 0~1 사이의 가중치
        )
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: list of [B, C, D, H, W] tensors (각 모달리티별 feature)
        
        Returns:
            weighted_features: 가중치가 적용된 feature list
            attention_weights: [B, num_modalities] 형태의 어텐션 가중치
        """
        # 각 모달리티별로 Global Average Pooling
        gap_features = []
        for feat in modality_features:
            # [B, C, D, H, W] -> [B, C, 1, 1, 1] -> [B, C]
            gap_feat = self.gap(feat).squeeze(-1).squeeze(-1).squeeze(-1)
            gap_features.append(gap_feat)
        
        # Concat all modality features: [B, C*num_modalities]
        concat_gap = torch.cat(gap_features, dim=1)
        
        # Get attention weights: [B, num_modalities]
        attention_weights = self.fc(concat_gap)
        
        # Apply attention to each modality feature
        weighted_features = []
        for i, feat in enumerate(modality_features):
            # [B, num_modalities] -> [B, 1, 1, 1, 1] for broadcasting
            weight = attention_weights[:, i].view(-1, 1, 1, 1, 1)
            weighted_features.append(feat * weight)
        
        return weighted_features, attention_weights


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


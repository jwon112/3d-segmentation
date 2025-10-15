#!/usr/bin/env python3
"""종합 분석 테스트 스크립트"""

import torch
from data_loader_kaggle import get_data_loaders_kaggle
from model_3d_unet import UNet3D_Simplified
from train import Trainer

def main():
    print("종합 분석 테스트 시작...")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 로더 생성
    print("데이터 로딩 중...")
    train_loader, val_loader = get_data_loaders_kaggle(
        data_dir='data',
        batch_size=1,
        max_samples=5
    )
    
    # 모델 생성
    print("모델 생성 중...")
    model = UNet3D_Simplified(n_channels=4, n_classes=4).to(device)
    
    # 설정
    config = {
        'batch_size': 1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 10,
        'save_interval': 5,
        'early_stopping': True,
        'patience': 3
    }
    
    # 트레이너 생성
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # 종합 분석 실행
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS")
    print("="*60)
    trainer.comprehensive_analysis()

if __name__ == "__main__":
    main()


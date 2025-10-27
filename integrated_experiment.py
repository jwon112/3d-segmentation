#!/usr/bin/env python3
"""
3D Segmentation Integrated Experiment System
Baseline 모델들을 훈련하고 결과를 비교 분석하는 통합 시스템

Usage:
    python integrated_experiment.py --epochs 10 --seeds 24
    python integrated_experiment.py --datasets brats2021 --epochs 10
    python integrated_experiment.py --data_path /path/to/data --epochs 50 --seeds 24 42 123

Dataset Selection:
    --datasets: 사용할 데이터셋 목록
        - brats2021: BraTS2021 데이터셋 (기본값)
        - auto: 자동 감지 (사용 가능한 데이터셋)

Path Configuration:
    --data_path: 데이터셋 루트 디렉토리 (기본값: 'data')
    환경 변경 시:
        - 로컬: --data_path ./data
        - 서버: --data_path /data/project/brats
        - 절대 경로: --data_path D:/Projects/data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import baseline models
from baseline import (
    UNet3D_Simplified, 
    UNETR_Simplified, 
    SwinUNETR_Simplified,
    combined_loss,
    calculate_dice_score
)

# Import data loader and utilities
from data_loader import get_data_loaders

# Import visualization
from visualization import create_comprehensive_analysis, create_interactive_3d_plot

def set_seed(seed):
    """랜덤 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_flops(model, input_size=(1, 4, 64, 64, 64)):
    """모델의 FLOPs 계산"""
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return 0
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return 0

def get_model(model_name, n_channels=4, n_classes=4):
    """모델 생성 함수"""
    if model_name == 'unet3d':
        return UNet3D_Simplified(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'unetr':
        return UNETR_Simplified(
            img_size=(64, 64, 64), 
            patch_size=(8, 8, 8),
            in_channels=n_channels, 
            out_channels=n_classes
        )
    elif model_name == 'swin_unetr':
        return SwinUNETR_Simplified(
            img_size=(64, 64, 64), 
            patch_size=(4, 4, 4),
            in_channels=n_channels, 
            out_channels=n_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, train_loader, val_loader, test_loader, epochs=10, lr=0.001, device='cuda', model_name='model', seed=24):
    """모델 훈련 함수"""
    model = model.to(device)
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_losses = []
    val_dices = []
    test_dices = []
    epoch_results = []
    
    best_val_dice = 0.0
    best_epoch = 0
    
    # 체크포인트 저장 경로
    os.makedirs("baseline_results", exist_ok=True)
    ckpt_path = f"baseline_results/{model_name}_seed_{seed}_best.pth"
    
    for epoch in range(epochs):
        # Training
        model.train()
        tr_loss = tr_dice_sum = n_tr = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            dice_scores = calculate_dice_score(logits.detach(), labels)
            mean_dice = dice_scores.mean()
            bsz = inputs.size(0)
            tr_loss += loss.item() * bsz
            tr_dice_sum += mean_dice.item() * bsz
            n_tr += bsz
        
        tr_loss /= max(1, n_tr)
        tr_dice = tr_dice_sum / max(1, n_tr)
        train_losses.append(tr_loss)
        
        # Validation
        model.eval()
        va_loss = va_dice_sum = n_va = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Val   {epoch+1}/{epochs}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                dice_scores = calculate_dice_score(logits, labels)
                mean_dice = dice_scores.mean()
                bsz = inputs.size(0)
                va_loss += loss.item() * bsz
                va_dice_sum += mean_dice.item() * bsz
                n_va += bsz
        
        va_loss /= max(1, n_va)
        va_dice = va_dice_sum / max(1, n_va)
        val_dices.append(va_dice)
        
        # Test
        test_dice = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Test  ", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                dice_scores = calculate_dice_score(logits, labels)
                mean_dice = dice_scores.mean()
                test_dice += mean_dice.item()
        
        test_dice /= len(test_loader)
        test_dices.append(test_dice)
        
        # Learning rate scheduling
        scheduler.step(va_loss)
        
        # Best model tracking 및 체크포인트 저장
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")
        
        # Epoch 결과 저장
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': tr_loss,
            'val_dice': va_dice,
            'test_dice': test_dice
        })
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} | Val Loss {va_loss:.4f} Dice {va_dice:.4f} | Test Dice {test_dice:.4f}")
    
    return train_losses, val_dices, test_dices, epoch_results, best_epoch, best_val_dice

def evaluate_model(model, test_loader, device='cuda'):
    """모델 평가 함수"""
    model.eval()
    test_dice = 0.0
    precision_scores = []
    recall_scores = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            
            # Dice score 계산
            dice_scores = calculate_dice_score(logits, labels)
            mean_dice = dice_scores.mean()
            test_dice += mean_dice.item()
            
            # Precision, Recall 계산 (클래스별)
            pred = torch.argmax(logits, dim=1)
            pred_np = pred.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for class_id in range(4):
                pred_class = (pred_np == class_id)
                true_class = (labels_np == class_id)
                
                if true_class.sum() > 0:
                    precision = (pred_class & true_class).sum() / pred_class.sum() if pred_class.sum() > 0 else 0
                    recall = (pred_class & true_class).sum() / true_class.sum()
                else:
                    precision = recall = 0
                    
                precision_scores.append(precision)
                recall_scores.append(recall)
    
    test_dice /= len(test_loader)
    
    # Background 제외한 평균 (클래스 1, 2, 3만)
    avg_precision = np.mean(precision_scores[1::4])  # 클래스별로 평균
    avg_recall = np.mean(recall_scores[1::4])
    
    return {
        'dice': test_dice,
        'precision': avg_precision,
        'recall': avg_recall
    }

def run_integrated_experiment(data_path, epochs=10, batch_size=1, seeds=[24], models=None, datasets=None):
    """3D Segmentation 통합 실험 실행
    
    Args:
        data_path: 데이터셋 루트 디렉토리 경로 (기본: 'data')
        epochs: 훈련 에포크 수
        batch_size: 배치 크기
        seeds: 실험 시드 리스트
        models: 사용할 모델 리스트 (기본: ['unet3d', 'unetr', 'swin_unetr'])
        datasets: 사용할 데이터셋 리스트 (기본: ['brats2021'])
                  지원: 'brats2021', 'auto' (자동 선택)
    """
    
    # 실험 결과 저장 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"baseline_results/integrated_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 사용 가능한 모델들
    if models is None:
        available_models = ['unet3d', 'unetr', 'swin_unetr']
    else:
        available_models = [m for m in models if m in ['unet3d', 'unetr', 'swin_unetr']]
    
    # 사용 가능한 데이터셋들
    if datasets is None:
        available_datasets = ['brats2021']  # 기본값
    else:
        available_datasets = []
        for ds in datasets:
            if ds == 'auto':
                # 자동 감지: 데이터셋 존재 여부 확인
                brats2021_path = os.path.join(data_path, 'BraTS2021_Training_Data')
                
                if os.path.exists(brats2021_path):
                    available_datasets.append('brats2021')
            elif ds in ['brats2021']:
                available_datasets.append(ds)
    
    # 결과 저장용
    all_results = []
    all_epochs_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 각 데이터셋별로 실험
    for dataset_name in available_datasets:
        print(f"\n{'#'*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'#'*80}")
        
        # 데이터셋별 디렉토리 구조 설정
        dataset_dir = os.path.join(data_path, 'BraTS2021_Training_Data')
        
        if not os.path.exists(dataset_dir):
            print(f"Warning: Dataset {dataset_name} not found at {dataset_dir}. Skipping...")
            continue
        
        # 각 시드별로 실험
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Training 3D Segmentation Models - Dataset: {dataset_name.upper()}, Seed: {seed}")
            print(f"{'='*60}")
            
            set_seed(seed)
            
            # 데이터 로더 생성 (3D 데이터)
            train_loader, val_loader, test_loader = get_data_loaders(
                data_dir=data_path,
                batch_size=batch_size,
                num_workers=0,  # Windows에서 안정성을 위해 0으로 설정
                max_samples=10,  # 메모리 효율성을 위해 제한
                dim='3d'  # 3D 데이터 사용
            )
            
            # 각 모델별로 실험
            for model_name in available_models:
                try:
                    print(f"\nTraining {model_name.upper()}...")
                    
                    # 모델 생성
                    model = get_model(model_name, n_channels=4, n_classes=4)
                    
                    # 모델 정보 출력
                    print(f"\n=== {model_name.upper()} Model Information ===")
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"Total parameters: {total_params:,}")
                    print(f"Trainable parameters: {trainable_params:,}")
                    
                    # 모델 크기 계산
                    param_size = 0
                    buffer_size = 0
                    for param in model.parameters():
                        param_size += param.nelement() * param.element_size()
                    for buffer in model.buffers():
                        buffer_size += buffer.nelement() * buffer.element_size()
                    model_size_mb = (param_size + buffer_size) / 1024 / 1024
                    print(f"Model size: {model_size_mb:.2f} MB")
                    
                    # FLOPs 계산
                    flops = calculate_flops(model, input_size=(1, 4, 64, 64, 64))
                    print(f"FLOPs: {flops:,}")
                    print("=" * 50)
                    
                    # 훈련
                    train_losses, val_dices, test_dices, epoch_results, best_epoch, best_val_dice = train_model(
                        model, train_loader, val_loader, test_loader, epochs, device=device, model_name=model_name, seed=seed
                    )
                    
                    # 평가
                    metrics = evaluate_model(model, test_loader, device)
                    
                    # 결과 저장
                    result = {
                        'dataset': dataset_name,
                        'seed': seed,
                        'model_name': model_name,
                        'total_params': total_params,
                        'flops': flops,
                        'test_dice': metrics['dice'],
                        'val_dice': best_val_dice,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'best_epoch': best_epoch
                    }
                    all_results.append(result)
                    
                    # 모든 epoch 결과 저장
                    for epoch_result in epoch_results:
                        epoch_data = {
                            'dataset': dataset_name,
                            'seed': seed,
                            'model_name': model_name,
                            'epoch': epoch_result['epoch'],
                            'train_loss': epoch_result['train_loss'],
                            'val_dice': epoch_result['val_dice'],
                            'test_dice': epoch_result['test_dice']
                        }
                        all_epochs_results.append(epoch_data)
                    
                    print(f"Final Val Dice: {best_val_dice:.4f} (epoch {best_epoch}) | Test Dice: {metrics['dice']:.4f} | Test Prec {metrics['precision']:.4f} Rec {metrics['recall']:.4f}")
                    
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    continue
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(all_results)
    
    # CSV로 저장
    csv_path = os.path.join(results_dir, "integrated_experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # 모든 epoch 결과 저장
    if all_epochs_results:
        epochs_df = pd.DataFrame(all_epochs_results)
        epochs_csv_path = os.path.join(results_dir, "all_epochs_results.csv")
        epochs_df.to_csv(epochs_csv_path, index=False)
        print(f"All epochs results saved to: {epochs_csv_path}")
    
    # 모델별 성능 비교 분석
    print("\nCreating model comparison analysis...")
    
    # 모델별 평균 성능
    model_comparison = results_df.groupby('model_name').agg({
        'test_dice': ['mean', 'std'],
        'val_dice': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(4)
    
    comparison_path = os.path.join(results_dir, "model_comparison.csv")
    model_comparison.to_csv(comparison_path)
    print(f"Model comparison saved to: {comparison_path}")
    
    # 시각화 생성
    print("\nCreating visualization charts...")
    create_comprehensive_analysis(results_df, epochs_df, results_dir)
    create_interactive_3d_plot(results_df, results_dir)
    
    # 결과 출력
    print("\n" + "="*80)
    print("3D SEGMENTATION INTEGRATED EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    print("\n--- Model Performance Summary ---")
    for model_name in available_models:
        model_results = results_df[results_df['model_name'] == model_name]
        if not model_results.empty:
            print(f"\n{model_name.upper()} Model:")
            for _, row in model_results.iterrows():
                print(f"  Seed {row['seed']:3d} | Test Dice: {row['test_dice']:.4f} | Val Dice: {row['val_dice']:.4f} | Params: {row['total_params']:,}")
    
    # 모델별 평균 성능
    print("\n--- Model-wise Average Performance ---")
    for model_name in available_models:
        model_results = results_df[results_df['model_name'] == model_name]
        if not model_results.empty:
            avg_dice = model_results['test_dice'].mean()
            avg_val_dice = model_results['val_dice'].mean()
            avg_params = model_results['total_params'].mean()
            print(f"{model_name.upper():12}: Test Dice {avg_dice:.4f} | Val Dice {avg_val_dice:.4f} | Avg Params {avg_params:,.0f}")
    
    return results_dir, results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Segmentation Integrated Experiment System')
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Path to BraTS dataset root (default: data/)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seeds', nargs='+', type=int, default=[24], 
                       help='Random seeds for experiments')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                       help='Specific models to train (default: unet3d,unetr,swin_unetr)')
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                       help='Datasets to use: brats2021, auto (default: brats2021)')
    
    args = parser.parse_args()
    
    print("Starting 3D Segmentation Integrated Experiment System")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seeds: {args.seeds}")
    print(f"Models: {args.models if args.models else 'unet3d,unetr,swin_unetr'}")
    print(f"Datasets: {args.datasets if args.datasets else 'brats2021 (auto-detected)'}")
    print(f"Results will be saved in: baseline_results/ folder")
    
    try:
        results_dir, results_df = run_integrated_experiment(
            args.data_path, args.epochs, args.batch_size, args.seeds, args.models, args.datasets
        )
        
        if results_dir and results_df is not None:
            print(f"\n3D Segmentation integrated experiment completed successfully!")
            print(f"Results saved in: {results_dir}")
        else:
            print(f"\nExperiment failed.")
        
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()

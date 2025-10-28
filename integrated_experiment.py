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

# Global input size configuration
# 2D models use 256x256 for stable down/upsampling (avoid odd sizes)
INPUT_SIZE_2D = (256, 256)
# 3D models keep dataset-native depth; adjust if needed
INPUT_SIZE_3D = (240, 240, 155)

# Distributed training helpers
def setup_distributed():
    import torch.distributed as dist
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1

def cleanup_distributed():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int):
    return rank == 0

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import baseline models
from baseline import (
    UNet3D_Simplified, 
    UNETR_Simplified, 
    SwinUNETR_Simplified,
    MobileUNETR,
)
from losses import combined_loss
from metrics import calculate_dice_score

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
        # unwrap DDP if needed
        real_model = model.module if hasattr(model, 'module') else model
        device = next(real_model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        flops, params = profile(real_model, inputs=(dummy_input,), verbose=False)
        return flops
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return 0
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return 0

def get_model(model_name, n_channels=4, n_classes=4, dim='3d', patch_size=None):
    """모델 생성 함수
    
    Args:
        model_name: 모델 이름
        n_channels: 입력 채널 수
        n_classes: 출력 클래스 수
        dim: '2d' 또는 '3d'
        patch_size: 하이퍼파라미터 (None이면 모델별 기본값 사용)
    """
    # 2D 입력인 경우 3D로 확장 (unsqueeze depth dimension)
    if model_name == 'unet3d':
        if dim == '2d':
            # 2D 데이터는 depth 차원 추가가 필요
            pass
        return UNet3D_Simplified(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'unetr':
        # dim에 따라 img_size 설정
        if dim == '2d':
            img_size = INPUT_SIZE_2D
            # patch_size가 지정되지 않으면 기본값 사용 (2D)
            if patch_size is None:
                patch_size = (16, 16)  # 논문 권장값 (16, 16, 16)의 2D 버전
        else:
            img_size = (240, 240, 155)
            # patch_size가 지정되지 않으면 기본값 사용 (3D)
            if patch_size is None:
                patch_size = (16, 16, 16)  # 논문 권장값
        return UNETR_Simplified(
            img_size=img_size, 
            patch_size=patch_size,
            in_channels=n_channels, 
            out_channels=n_classes
        )
    elif model_name == 'swin_unetr':
        # dim에 따라 img_size 설정
        if dim == '2d':
            img_size = INPUT_SIZE_2D
            # patch_size가 지정되지 않으면 기본값 사용 (2D)
            if patch_size is None:
                patch_size = (4, 4)
        else:
            img_size = (240, 240, 155)
            # patch_size가 지정되지 않으면 기본값 사용 (3D)
            if patch_size is None:
                patch_size = (4, 4, 4)
        return SwinUNETR_Simplified(
            img_size=img_size, 
            patch_size=patch_size,
            in_channels=n_channels, 
            out_channels=n_classes
        )
    elif model_name == 'mobile_unetr':
        # MobileUNETR는 2D 전용 모델
        if dim != '2d':
            raise ValueError("MobileUNETR is only supported for 2D data (dim='2d')")
        
        # img_size 설정 (2D만) - 전역 설정 사용
        img_size = INPUT_SIZE_2D
        if patch_size is None:
            patch_size = (16, 16)  # 2D에서 권장값
        
        return MobileUNETR(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=n_channels,
            out_channels=n_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, train_loader, val_loader, test_loader, epochs=10, lr=0.001, device='cuda', model_name='model', seed=24, train_sampler=None, rank: int = 0):
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
        if train_sampler is not None:
            # ensure different shuffles per epoch
            train_sampler.set_epoch(epoch)
        model.train()
        tr_loss = tr_dice_sum = n_tr = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # MobileUNETR는 2D 입력을 그대로 사용 (depth 차원 추가 안함)
            # 다른 모델들은 3D 입력 필요 (depth 차원 추가)
            if model_name != 'mobile_unetr' and len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(2)  # Add depth dimension (B, C, H, W) -> (B, C, 1, H, W)
                labels = labels.unsqueeze(2)
            
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
                
                # MobileUNETR는 2D 입력을 그대로 사용
                if model_name != 'mobile_unetr' and len(inputs.shape) == 4:
                    inputs = inputs.unsqueeze(2)
                    labels = labels.unsqueeze(2)
                
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
                
                # MobileUNETR는 2D 입력을 그대로 사용
                if model_name != 'mobile_unetr' and len(inputs.shape) == 4:
                    inputs = inputs.unsqueeze(2)
                    labels = labels.unsqueeze(2)
                
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
        
        if is_main_process(rank):
            print(f"Epoch {epoch+1}/{epochs} | Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} | Val Loss {va_loss:.4f} Dice {va_dice:.4f} | Test Dice {test_dice:.4f}")
    
    return train_losses, val_dices, test_dices, epoch_results, best_epoch, best_val_dice

def evaluate_model(model, test_loader, device='cuda', model_name: str = 'model', distributed: bool = False, world_size: int = 1):
    """모델 평가 함수"""
    model.eval()
    test_dice = 0.0
    precision_scores = []
    recall_scores = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 2D/3D 분기: 2D 모델은 그대로, 3D 모델은 depth 차원 추가
            if model_name != 'mobile_unetr' and len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(2)
                labels = labels.unsqueeze(2)
            
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
    # Reduce across processes if distributed
    if distributed and world_size > 1:
        import torch.distributed as dist
        td = torch.tensor([test_dice], device=device)
        dist.all_reduce(td, op=dist.ReduceOp.SUM)
        test_dice = (td / world_size).item()
    
    # Background 제외한 평균 (클래스 1, 2, 3만)
    avg_precision = np.mean(precision_scores[1::4])  # 클래스별로 평균
    avg_recall = np.mean(recall_scores[1::4])
    
    return {
        'dice': test_dice,
        'precision': avg_precision,
        'recall': avg_recall
    }

def run_integrated_experiment(data_path, epochs=10, batch_size=1, seeds=[24], models=None, datasets=None, dim='2d'):
    """3D Segmentation 통합 실험 실행
    
    Args:
        data_path: 데이터셋 루트 디렉토리 경로 (기본: 'data')
        epochs: 훈련 에포크 수
        batch_size: 배치 크기
        seeds: 실험 시드 리스트
        models: 사용할 모델 리스트 (기본: ['unet3d', 'unetr', 'swin_unetr', 'mobile_unetr'])
        datasets: 사용할 데이터셋 리스트 (기본: ['brats2021'])
                  지원: 'brats2021', 'auto' (자동 선택)
        dim: 데이터 차원 '2d' 또는 '3d' (기본: '2d')
    """
    
    # 실험 결과 저장 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"baseline_results/integrated_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 사용 가능한 모델들
    if models is None:
        available_models = ['unet3d', 'unetr', 'swin_unetr', 'mobile_unetr']
    else:
        available_models = [m for m in models if m in ['unet3d', 'unetr', 'swin_unetr', 'mobile_unetr']]
    
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
    
    # Distributed setup
    distributed, rank, local_rank, world_size = setup_distributed()
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process(rank):
            print(f"\nUsing DDP with world_size={world_size}, local_rank={local_rank}")
    else:
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
            
            # 데이터 로더 생성
            train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = get_data_loaders(
                data_dir=data_path,
                batch_size=batch_size,
                num_workers=0,  # Windows에서 안정성을 위해 0으로 설정 (서버에서는 늘려도 됨)
                max_samples=None,  # 전체 데이터 사용
                dim=dim,  # 2D 또는 3D
                distributed=distributed,
                world_size=world_size,
                rank=rank
            )
            
            # 각 모델별로 실험
            for model_name in available_models:
                try:
                    print(f"\nTraining {model_name.upper()}...")
                    
                    # 모델 생성 (T1CE, FLAIR만 사용하므로 2 channels)
                    model = get_model(model_name, n_channels=2, n_classes=4, dim=dim)
                    # DDP wrap
                    if distributed:
                        from torch.nn.parallel import DistributedDataParallel as DDP
                        model = model.to(device)
                        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
                    
                    # 모델 정보 출력
                    print(f"\n=== {model_name.upper()} Model Information ===")
                    real_model = model.module if hasattr(model, 'module') else model
                    total_params = sum(p.numel() for p in real_model.parameters())
                    trainable_params = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
                    print(f"Total parameters: {total_params:,}")
                    print(f"Trainable parameters: {trainable_params:,}")
                    
                    # 모델 크기 계산 (real_model 사용)
                    param_size = 0
                    buffer_size = 0
                    for param in real_model.parameters():
                        param_size += param.nelement() * param.element_size()
                    for buffer in real_model.buffers():
                        buffer_size += buffer.nelement() * buffer.element_size()
                    model_size_mb = (param_size + buffer_size) / 1024 / 1024
                    if is_main_process(rank):
                        print(f"Model size: {model_size_mb:.2f} MB")
                        print("=" * 50)
                    
                    # 훈련
                    train_losses, val_dices, test_dices, epoch_results, best_epoch, best_val_dice = train_model(
                        model, train_loader, val_loader, test_loader, epochs, device=device, model_name=model_name, seed=seed,
                        train_sampler=train_sampler, rank=rank
                    )
                    
                    # FLOPs 계산 (모델이 device에 있는 상태에서)
                    if dim == '2d':
                        flops = calculate_flops(model, input_size=(1, 2, *INPUT_SIZE_2D))
                    else:
                        flops = calculate_flops(model, input_size=(1, 2, *INPUT_SIZE_3D))
                    if is_main_process(rank):
                        print(f"FLOPs: {flops:,}")
                    
                    # 평가
                    metrics = evaluate_model(model, test_loader, device, model_name, distributed=distributed, world_size=world_size)
                    
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
                    if is_main_process(rank):
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
                        if is_main_process(rank):
                            all_epochs_results.append(epoch_data)
                    
                    if is_main_process(rank):
                        print(f"Final Val Dice: {best_val_dice:.4f} (epoch {best_epoch}) | Test Dice: {metrics['dice']:.4f} | Test Prec {metrics['precision']:.4f} Rec {metrics['recall']:.4f}")
                    
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    import traceback
                    print("Full traceback:")
                    traceback.print_exc()
                    continue
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(all_results)
    
    # 결과가 비어있는 경우 처리
    if results_df.empty:
        print("Warning: No results were collected. All experiments failed.")
        return results_dir, results_df
    
    # CSV로 저장
    if is_main_process(0) or not distributed:
        csv_path = os.path.join(results_dir, "integrated_experiment_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    # 모든 epoch 결과 저장
    epochs_df = None
    if all_epochs_results:
        epochs_df = pd.DataFrame(all_epochs_results)
        if is_main_process(0) or not distributed:
            epochs_csv_path = os.path.join(results_dir, "all_epochs_results.csv")
            epochs_df.to_csv(epochs_csv_path, index=False)
            print(f"All epochs results saved to: {epochs_csv_path}")
    
    # 모델별 성능 비교 분석
    if is_main_process(0) or not distributed:
        print("\nCreating model comparison analysis...")
    
    # 모델별 평균 성능
    try:
        model_comparison = results_df.groupby('model_name').agg({
            'test_dice': ['mean', 'std'],
            'val_dice': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }).round(4)
        if is_main_process(0) or not distributed:
            comparison_path = os.path.join(results_dir, "model_comparison.csv")
            model_comparison.to_csv(comparison_path)
            print(f"Model comparison saved to: {comparison_path}")
    except KeyError as e:
        print(f"Warning: Could not create model comparison: {e}")
    
    # 시각화 생성
    if is_main_process(0) or not distributed:
        print("\nCreating visualization charts...")
        try:
            create_comprehensive_analysis(results_df, epochs_df, results_dir)
            create_interactive_3d_plot(results_df, results_dir)
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    # 결과 출력
    if (is_main_process(0) or not distributed) and not results_df.empty:
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
    
    # cleanup
    if distributed:
        cleanup_distributed()
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
    parser.add_argument('--dim', type=str, default='2d', choices=['2d', '3d'],
                       help='Data dimension: 2d or 3d (default: 2d)')
    
    args = parser.parse_args()
    
    print("Starting 3D Segmentation Integrated Experiment System")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seeds: {args.seeds}")
    print(f"Models: {args.models if args.models else 'unet3d,unetr,swin_unetr,mobile_unetr'}")
    print(f"Datasets: {args.datasets if args.datasets else 'brats2021 (auto-detected)'}")
    print(f"Dimension: {args.dim}")
    print(f"Results will be saved in: baseline_results/ folder")
    
    try:
        results_dir, results_df = run_integrated_experiment(
            args.data_path, args.epochs, args.batch_size, args.seeds, args.models, args.datasets, args.dim
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

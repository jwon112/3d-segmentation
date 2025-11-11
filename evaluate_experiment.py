#!/usr/bin/env python3
"""
평가 전용 스크립트 - 저장된 체크포인트를 로드하여 평가만 수행

Usage:
    python evaluate_experiment.py --results_dir baseline_results/integrated_experiment_results_YYYYMMDD_HHMMSS --models unet3d_2modal_s unet3d_4modal_s
    python evaluate_experiment.py --results_dir baseline_results/integrated_experiment_results_YYYYMMDD_HHMMSS --seeds 24 42
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from integrated_experiment
from integrated_experiment import (
    get_model, evaluate_model, calculate_flops, 
    setup_distributed, cleanup_distributed, is_main_process,
    INPUT_SIZE_2D, INPUT_SIZE_3D, sliding_window_inference_3d
)
from data_loader import get_data_loaders
from visualization import create_comprehensive_analysis, create_interactive_3d_plot

def load_checkpoint_and_evaluate(results_dir, model_name, seed, data_path, dim='3d', 
                                 dataset_version='brats2018', batch_size=1, num_workers=0,
                                 device='cuda', distributed=False, rank=0, world_size=1):
    """저장된 체크포인트를 로드하여 평가만 수행"""
    
    # 모델에 따라 use_4modalities 및 n_channels 결정
    use_4modalities = model_name in ['unet3d_4modal_s', 'quadbranch_4modal_unet_s', 'quadbranch_4modal_attention_unet_s']
    if use_4modalities:
        n_channels = 4
    else:
        n_channels = 2
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader, _, _, _ = get_data_loaders(
        data_dir=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=None,
        dim=dim,
        dataset_version=dataset_version,
        seed=seed,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        use_4modalities=use_4modalities
    )
    
    # 모델 생성
    model = get_model(model_name, n_channels=n_channels, n_classes=4, dim=dim)
    
    # DDP wrap if distributed
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = model.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        model = model.to(device)
    
    # 체크포인트 로드
    ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_best.pth")
    if not os.path.exists(ckpt_path):
        if is_main_process(rank):
            print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping...")
        return None
    
    real_model = model.module if hasattr(model, 'module') else model
    state = torch.load(ckpt_path, map_location=device)
    real_model.load_state_dict(state, strict=False)
    
    if is_main_process(rank):
        print(f"Loaded checkpoint from {ckpt_path}")
    
    # RepLK 모델의 경우 deploy 모드로 전환
    if model_name in ['dualbranch_04_unet_s', 'dualbranch_04_unet_m', 'dualbranch_05_unet_s', 'dualbranch_05_unet_m', 
                      'dualbranch_08_unet_s', 'dualbranch_08_unet_m', 'dualbranch_09_unet_s', 'dualbranch_09_unet_m']:
        if hasattr(real_model, 'switch_to_deploy'):
            if is_main_process(rank):
                print(f"Switching RepLK blocks to deploy mode...")
            real_model.switch_to_deploy()
    
    # 파라미터 및 FLOPs 계산
    total_params = sum(p.numel() for p in real_model.parameters())
    if dim == '2d':
        flops = calculate_flops(model, input_size=(1, n_channels, *INPUT_SIZE_2D))
    else:
        flops = calculate_flops(model, input_size=(1, n_channels, *INPUT_SIZE_3D))
    
    if is_main_process(rank):
        print(f"Parameters: {total_params:,}")
        print(f"FLOPs: {flops:,}")
    
    # Test set 평가
    metrics = evaluate_model(
        model, test_loader, device, model_name, 
        distributed=distributed, world_size=world_size,
        sw_patch_size=(128, 128, 128), sw_overlap=0.10, 
        results_dir=results_dir
    )
    
    # 결과 반환
    result = {
        'dataset': dataset_version,
        'seed': seed,
        'model_name': model_name,
        'total_params': total_params,
        'flops': flops,
        'test_dice': metrics['dice'],
        'test_wt': metrics.get('wt', None),
        'test_tc': metrics.get('tc', None),
        'test_et': metrics.get('et', None),
        'val_dice': None,  # 평가 전용이므로 val_dice는 없음
        'val_wt': None,
        'val_tc': None,
        'val_et': None,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'best_epoch': None  # 평가 전용이므로 best_epoch는 없음
    }
    
    if is_main_process(rank):
        print(f"Test Dice: {metrics['dice']:.4f} (WT {metrics['wt']:.4f} | TC {metrics['tc']:.4f} | ET {metrics['et']:.4f})")
        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    
    return result


def run_evaluation(results_dir, data_path, models=None, seeds=None, dim='3d', 
                   dataset_version='brats2018', batch_size=1, num_workers=0):
    """평가 실행"""
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return None, pd.DataFrame()
    
    # Distributed setup
    distributed, rank, local_rank, world_size = setup_distributed()
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process(rank):
            print(f"\nUsing DDP with world_size={world_size}, local_rank={local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
    
    # 사용 가능한 체크포인트 찾기
    all_checkpoints = []
    for file in os.listdir(results_dir):
        if file.endswith('_best.pth'):
            # 파일명 형식: {model_name}_seed_{seed}_best.pth
            parts = file.replace('_best.pth', '').split('_seed_')
            if len(parts) == 2:
                model_name = parts[0]
                seed = int(parts[1])
                all_checkpoints.append((model_name, seed))
    
    if not all_checkpoints:
        print(f"Error: No checkpoint files found in {results_dir}")
        return None, pd.DataFrame()
    
    # 필터링
    if models:
        all_checkpoints = [(m, s) for m, s in all_checkpoints if m in models]
    if seeds:
        all_checkpoints = [(m, s) for m, s in all_checkpoints if s in seeds]
    
    if not all_checkpoints:
        print(f"Error: No matching checkpoints found after filtering")
        return None, pd.DataFrame()
    
    if is_main_process(rank):
        print(f"\nFound {len(all_checkpoints)} checkpoint(s) to evaluate:")
        for model_name, seed in all_checkpoints:
            print(f"  - {model_name} (seed {seed})")
    
    # 평가 실행
    all_results = []
    for model_name, seed in all_checkpoints:
        try:
            if is_main_process(rank):
                print(f"\n{'='*60}")
                print(f"Evaluating {model_name.upper()} (seed {seed})...")
                print(f"{'='*60}")
            
            result = load_checkpoint_and_evaluate(
                results_dir=results_dir,
                model_name=model_name,
                seed=seed,
                data_path=data_path,
                dim=dim,
                dataset_version=dataset_version,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                distributed=distributed,
                rank=rank,
                world_size=world_size
            )
            
            if result and is_main_process(rank):
                all_results.append(result)
                
        except Exception as e:
            if is_main_process(rank):
                print(f"Error evaluating {model_name} (seed {seed}): {e}")
                import traceback
                traceback.print_exc()
            continue
    
    # 결과를 DataFrame으로 변환
    if not all_results:
        print("Warning: No results were collected.")
        return results_dir, pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    
    # CSV로 저장 (기존 파일이 있으면 백업)
    if is_main_process(rank) or not distributed:
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        if os.path.exists(csv_path):
            backup_path = os.path.join(results_dir, f"evaluation_results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.rename(csv_path, backup_path)
            print(f"Backed up existing results to {backup_path}")
        
        results_df.to_csv(csv_path, index=False)
        print(f"\nEvaluation results saved to: {csv_path}")
    
    # 시각화 생성
    if is_main_process(rank) or not distributed:
        print("\nCreating visualization charts...")
        try:
            create_comprehensive_analysis(results_df, None, results_dir)
            create_interactive_3d_plot(results_df, results_dir)
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    # 결과 출력
    if (is_main_process(rank) or not distributed) and not results_df.empty:
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        for _, row in results_df.iterrows():
            print(f"{row['model_name']:30} (seed {row['seed']:3d}) | "
                  f"Test Dice: {row['test_dice']:.4f} | "
                  f"WT: {row['test_wt']:.4f} | TC: {row['test_tc']:.4f} | ET: {row['test_et']:.4f}")
    
    # cleanup
    if distributed:
        cleanup_distributed()
    
    return results_dir, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate saved checkpoints')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Results directory containing checkpoint files (e.g., baseline_results/integrated_experiment_results_YYYYMMDD_HHMMSS)')
    parser.add_argument('--data_path', type=str, default='/home/work/3D_/BT/',
                       help='Common path to BraTS dataset root (default: /home/work/3D_/BT/)')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                       help='Specific models to evaluate (default: all found checkpoints)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help='Specific seeds to evaluate (default: all found seeds)')
    parser.add_argument('--dim', type=str, default='3d', choices=['2d', '3d'],
                       help='Data dimension: 2d or 3d (default: 3d)')
    parser.add_argument('--dataset_version', type=str, default='brats2018', choices=['brats2021', 'brats2018'],
                       help='Dataset version: brats2021 or brats2018 (default: brats2018)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers (default: 0 for evaluation)')
    
    args = parser.parse_args()
    
    print("Starting evaluation of saved checkpoints...")
    print(f"Results directory: {args.results_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Models: {args.models if args.models else 'All found checkpoints'}")
    print(f"Seeds: {args.seeds if args.seeds else 'All found seeds'}")
    print(f"Dimension: {args.dim}")
    print(f"Dataset version: {args.dataset_version}")
    
    try:
        results_dir, results_df = run_evaluation(
            results_dir=args.results_dir,
            data_path=args.data_path,
            models=args.models,
            seeds=args.seeds,
            dim=args.dim,
            dataset_version=args.dataset_version,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        if results_dir and results_df is not None and not results_df.empty:
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved in: {results_dir}")
        else:
            print(f"\nEvaluation completed with no results.")
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()


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
import argparse
import torch.multiprocessing as mp

# Import experiment runner
from experiment_runner import run_integrated_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Segmentation Integrated Experiment System')
    parser.add_argument('--data_path', type=str, default='/home/work/3D_/BT/', 
                       help='Common path to BraTS dataset root (default: /home/work/3D_/BT/). Server: /home/work/3D_/BT/, Local: C:\\Users\\user\\Desktop\\성균관대\\3d_segmentation\\data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seeds', nargs='+', type=int, default=[24], 
                       help='Random seeds for experiments')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                       help='Specific models to train (default: unet3d_s,unet3d_m,unet3d_stride_s,unet3d_stride_m,unetr,swin_unetr,mobile_unetr,mobile_unetr_3d,dualbranch_01_unet_s,dualbranch_01_unet_m,dualbranch_02_unet_s,dualbranch_02_unet_m,dualbranch_03_unet_s,dualbranch_03_unet_m)')
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                       help='Datasets to use: brats2021, auto (default: brats2021)')
    parser.add_argument('--dim', type=str, default='2d', choices=['2d', '3d'],
                       help='Data dimension: 2d or 3d (default: 2d)')
    parser.add_argument('--use_pretrained', action='store_true', default=False,
                       help='Use pretrained weights (default: False, scratch training)')
    parser.add_argument('--use_nnunet_loss', action='store_true', default=True,
                       help='Use nnU-Net style loss (Soft Dice with Squared Pred, Dice 70%% + CE 30%%) (default: True)')
    parser.add_argument('--use_standard_loss', action='store_true', default=False,
                       help='Use standard combined loss (Dice 50%% + CE 50%%) instead of nnU-Net style')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of DataLoader workers for training (val/test use 0 by default). Default: 2 for 2GB /dev/shm')
    parser.add_argument('--dataset_version', type=str, default='brats2018', choices=['brats2021', 'brats2018'],
                       help='Dataset version: brats2021 or brats2018 (default: brats2018)')
    parser.add_argument('--sharing_strategy', type=str, default='file_descriptor', choices=['file_system', 'file_descriptor'],
                       help='PyTorch tensor sharing strategy for DataLoader workers. file_system avoids /dev/shm pressure.')
    parser.add_argument('--use_5fold', action='store_true', default=False,
                       help='Use 5-fold cross-validation instead of simple train/val/test split')
    parser.add_argument('--use_mri_augmentation', action='store_true', default=False,
                       help='Apply MRI-specific data augmentations to training patches (default: False)')
    
    args = parser.parse_args()
    
    # --use_standard_loss가 True이면 nnU-Net loss 비활성화
    use_nnunet_loss = args.use_nnunet_loss and not args.use_standard_loss
    
    print("Starting 3D Segmentation Integrated Experiment System")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seeds: {args.seeds}")
    print(f"Models: {args.models if args.models else 'unet3d_s,unet3d_m,unet3d_stride_s,unet3d_stride_m,unetr,swin_unetr,mobile_unetr,mobile_unetr_3d,dualbranch_01_unet_s,dualbranch_01_unet_m,dualbranch_02_unet_s,dualbranch_02_unet_m,dualbranch_03_unet_s,dualbranch_03_unet_m'}")
    print(f"Datasets: {args.datasets if args.datasets else 'brats2021 (auto-detected)'}")
    print(f"Dataset version: {args.dataset_version}")
    print(f"Dimension: {args.dim}")
    print(f"Loss function: {'nnU-Net style (Soft Dice Squared + Dice 70%%/CE 30%%)' if use_nnunet_loss else 'Standard (Dice 50%%/CE 50%%)'}")
    print(f"MRI Augmentation: {'Enabled' if args.use_mri_augmentation else 'Disabled'}")
    print(f"Results will be saved in: baseline_results/ folder")
    if args.use_5fold:
        print(f"Using 5-fold cross-validation")
    
    # Configure PyTorch sharing strategy to avoid /dev/shm pressure if requested
    try:
        if args.sharing_strategy:
            os.environ.setdefault('PYTORCH_SHARING_STRATEGY', args.sharing_strategy)
            mp.set_sharing_strategy(args.sharing_strategy)
            print(f"Using PyTorch sharing strategy: {mp.get_sharing_strategy()}")
    except Exception as _e:
        print(f"Warning: Failed to set sharing strategy: {_e}")

    try:
        results_dir, results_df = run_integrated_experiment(
            args.data_path, args.epochs, args.batch_size, args.seeds, args.models, args.datasets, args.dim,
            args.use_pretrained, use_nnunet_loss, args.num_workers, args.dataset_version, args.use_5fold,
            use_mri_augmentation=args.use_mri_augmentation
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

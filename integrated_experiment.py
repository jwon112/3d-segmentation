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
from utils.runner import run_integrated_experiment


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
    parser.add_argument('--dataset_version', type=str, default='brats2018',
                       choices=['brats2017', 'brats2018', 'brats2019', 'brats2020', 'brats2021', 'brats2023', 'brats2024'],
                       help='Dataset version (default: brats2018)')
    parser.add_argument('--sharing_strategy', type=str, default='file_descriptor', choices=['file_system', 'file_descriptor'],
                       help='PyTorch tensor sharing strategy for DataLoader workers. file_system avoids /dev/shm pressure.')
    parser.add_argument('--use_5fold', action='store_true', default=False,
                       help='Use 5-fold cross-validation instead of simple train/val/test split')
    parser.add_argument('--use_mri_augmentation', action='store_true', default=False,
                       help='Apply MRI-specific data augmentations to training patches (default: False)')
    parser.add_argument('--anisotropy_augmentation', action='store_true', default=False,
                       help='Apply depth anisotropy resize augmentation (train only, default: False)')
    parser.add_argument('--use_cascade_pipeline', action='store_true', default=False,
                       help='Enable cascade ROI→Seg evaluation using a pre-trained ROI detector')
    parser.add_argument('--roi_model_name', type=str, default='roi_mobileunetr3d_tiny',
                       help='ROI detection model architecture name (used for cascade models and cascade pipeline evaluation)')
    parser.add_argument('--roi_weight_path', type=str, default=None,
                       help='Path to pre-trained ROI detector weights (.pth). If not specified, uses default path: models/weights/cascade/roi_model/{roi_model_name}/seed_{seed}/weights/best.pth')
    parser.add_argument('--roi_resize', type=int, nargs=3, default=[64, 64, 64],
                       help='ROI detector input size (DxHxW) (default: 64 64 64)')
    parser.add_argument('--cascade_crop_size', type=int, nargs=3, default=[96, 96, 96],
                       help='Segmentation crop size (DxHxW) used during cascade inference (default: 96 96 96)')
    parser.add_argument('--crops_per_center', type=int, default=1,
                       help='Number of crops per center during inference (1=single crop, 2=2x2x2=8 crops, 3=3x3x3=27 crops) (default: 1)')
    parser.add_argument('--crop_overlap', type=float, default=0.5,
                       help='Overlap ratio between crops (0.0 ~ 1.0) (default: 0.5)')
    parser.add_argument('--use_crop_blending', action='store_true', default=True,
                       help='Use cosine blending for multi-crop merging (default: True, False uses voxel-wise max)')
    parser.add_argument('--train_crops_per_center', type=int, default=1,
                       help='Number of crops per center during training (1=single crop, 2=2x2x2=8 crops, 3=3x3x3=27 crops). Each epoch randomly samples one crop from multiple positions. (default: 1)')
    parser.add_argument('--train_crop_overlap', type=float, default=0.5,
                       help='Overlap ratio between crops during training (0.0 ~ 1.0). If not specified, uses --crop_overlap value. (default: 0.5)')
    parser.add_argument('--coord_type', type=str, default='none',
                       choices=['none', 'simple', 'hybrid'],
                       help='Coordinate encoding type: none (no coords, 4 channels), simple (3 coord channels, 7 total), hybrid (9 coord channels, 13 total). Default: none')
    parser.add_argument('--use_4modalities', action='store_true', default=False,
                       help='Use 4 modalities (T1, T1CE, T2, FLAIR) instead of 2 (T1CE, FLAIR). Default: False (uses 2 modalities)')
    parser.add_argument('--preprocessed_base_dir', type=str, default='/home/work/3D_/processed_data',
                       help='Base directory containing preprocessed H5 files (default: /home/work/3D_/processed_data). If None, uses default project data/ directory')
    
    args = parser.parse_args()
    
    # ROI weight 경로 자동 생성 함수
    def get_default_roi_weight_path(roi_model_name: str, seed: int) -> str:
        """ROI 모델 이름과 seed로 기본 weight 경로 생성"""
        return f"models/weights/cascade/roi_model/{roi_model_name}/seed_{seed}/weights/best.pth"
    
    # ROI weight 경로 자동 생성 (지정되지 않은 경우)
    if args.use_cascade_pipeline and not args.roi_weight_path:
        # 첫 번째 seed 사용 (여러 seed인 경우 첫 번째 것 사용)
        default_path = get_default_roi_weight_path(args.roi_model_name, args.seeds[0])
        if os.path.exists(default_path):
            args.roi_weight_path = default_path
            print(f"Using default ROI weight path: {args.roi_weight_path}")
        else:
            raise FileNotFoundError(
                f"ROI weight file not found. Please specify --roi_weight_path or ensure the default path exists: {default_path}"
            )
    
    if args.use_cascade_pipeline and args.roi_weight_path and not os.path.exists(args.roi_weight_path):
        raise FileNotFoundError(f"ROI weight file not found: {args.roi_weight_path}")
    
    # --use_standard_loss가 True이면 nnU-Net loss 비활성화
    use_nnunet_loss = args.use_nnunet_loss and not args.use_standard_loss
    
    # 기본 전처리 디렉토리 설정 (이미 argparse default로 설정되어 있음)
    
    print("Starting 3D Segmentation Integrated Experiment System")
    print(f"Data path: {args.data_path}")
    print(f"Preprocessed base dir: {args.preprocessed_base_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seeds: {args.seeds}")
    print(f"Models: {args.models if args.models else 'unet3d_s,unet3d_m,unet3d_stride_s,unet3d_stride_m,unetr,swin_unetr,mobile_unetr,mobile_unetr_3d,dualbranch_01_unet_s,dualbranch_01_unet_m,dualbranch_02_unet_s,dualbranch_02_unet_m,dualbranch_03_unet_s,dualbranch_03_unet_m'}")
    print(f"Datasets: {args.datasets if args.datasets else 'brats2021 (auto-detected)'}")
    print(f"Dataset version: {args.dataset_version}")
    print(f"Dimension: {args.dim}")
    print(f"Loss function: {'nnU-Net style (Soft Dice Squared + Dice 70%%/CE 30%%)' if use_nnunet_loss else 'Standard (Dice 50%%/CE 50%%)'}")
    print(f"MRI Augmentation: {'Enabled' if args.use_mri_augmentation else 'Disabled'}")
    print(f"Anisotropy Augmentation: {'Enabled' if args.anisotropy_augmentation else 'Disabled'}")
    print(f"Modalities: {'4 (T1, T1CE, T2, FLAIR)' if args.use_4modalities else '2 (T1CE, FLAIR)'}")
    print(f"Coordinate encoding type: {args.coord_type} ({'no coords' if args.coord_type == 'none' else '3 channels' if args.coord_type == 'simple' else '9 channels'})")
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

    # Cascade 모델 학습 시에도 ROI 모델 정보를 전달
    cascade_cfg = None
    if args.use_cascade_pipeline:
        cascade_cfg = {
            'roi_model_name': args.roi_model_name,
            'roi_weight_path': args.roi_weight_path,
            'roi_resize': tuple(args.roi_resize),
            'crop_size': tuple(args.cascade_crop_size),
            'coord_type': args.coord_type,
            'crops_per_center': args.crops_per_center,
            'crop_overlap': args.crop_overlap,
            'use_blending': args.use_crop_blending,
        }
    
    # Cascade 모델인 경우 ROI 모델 정보를 전달 (학습 시에도 메타데이터로 저장)
    cascade_model_cfg = None
    if args.models:
        # Cascade 모델이 있는지 확인
        cascade_models = [m for m in args.models if m.startswith('cascade_')]
        if cascade_models:
            cascade_model_cfg = {
                'roi_model_name': args.roi_model_name,
                'roi_resize': tuple(args.roi_resize),
                'crop_size': tuple(args.cascade_crop_size),
            }

    try:
        results_dir, results_df = run_integrated_experiment(
            args.data_path, args.epochs, args.batch_size, args.seeds, args.models, args.datasets, args.dim,
            args.use_pretrained, use_nnunet_loss, args.num_workers, args.dataset_version, args.use_5fold,
            use_mri_augmentation=args.use_mri_augmentation,
            anisotropy_augment=args.anisotropy_augmentation,
            cascade_infer_cfg=cascade_cfg if args.use_cascade_pipeline else None,
            cascade_model_cfg=cascade_model_cfg,
            train_crops_per_center=args.train_crops_per_center,
            train_crop_overlap=args.train_crop_overlap,
            coord_type=args.coord_type,
            use_4modalities=args.use_4modalities,
            preprocessed_base_dir=args.preprocessed_base_dir,
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

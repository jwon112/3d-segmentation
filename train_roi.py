#!/usr/bin/env python3
"""
ROI Detector Training Script

학습 완료 후 모델 아티팩트를 다음 구조로 저장합니다.
models/weights/cascade/roi_model/<model_name>/seed_<seed>/
    ├─ weights/best.pth
    ├─ metrics.csv (seed별 성능 누적)
    └─ config.json (선택적으로 마지막 실행 설정 기록)
"""

import os
import json
import argparse
import torch
import torch.distributed as dist
import pandas as pd

from dataloaders import get_roi_data_loaders
from utils.experiment_utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    set_seed,
    get_roi_model,
)
from utils.runner import train_roi_model
from metrics import calculate_dice_score
from utils.experiment_config import get_roi_model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train ROI detection model")
    parser.add_argument('--data_path', type=str, default='/home/work/3D_/BT/',
                        help='Dataset root directory')
    parser.add_argument('--dataset_version', type=str, default='brats2018', choices=['brats2021', 'brats2018'],
                        help='Dataset version')
    parser.add_argument('--output_dir', type=str, default='models/weights/cascade/roi_model',
                        help='Directory to store ROI artifacts')
    parser.add_argument('--roi_model_name', type=str, default='roi_mobileunetr3d_tiny',
                        help='ROI detector architecture name')
    parser.add_argument('--roi_weight_name', type=str, default='best.pth',
                        help='Filename for the best checkpoint')
    parser.add_argument('--seeds', nargs='+', type=int, default=[24],
                        help='Random seeds for experiments')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of patients (debugging)')
    parser.add_argument('--use_5fold', action='store_true', default=False, help='Use 5-fold splits')
    parser.add_argument('--fold_idx', type=int, default=None, help='Fold index when --use_5fold is set')
    parser.add_argument('--roi_resize', type=int, nargs=3, default=[64, 64, 64],
                        help='ROI input size (DxHxW)')
    parser.add_argument('--disable_coords', action='store_true', default=False,
                        help='Do not append CoordConv channels (use pure MRI channels)')
    parser.add_argument('--sharing_strategy', type=str, default='file_descriptor', choices=['file_system', 'file_descriptor'],
                        help='PyTorch sharing strategy for dataloaders')
    parser.add_argument('--use_mri_augmentation', action='store_true', default=False,
                        help='Apply MRI-style intensity augmentations to ROI crops')
    parser.add_argument('--anisotropy_augmentation', action='store_true', default=False,
                        help='Apply depth anisotropy resize augmentation (train only)')
    parser.add_argument('--use_4modalities', action='store_true', default=False,
                        help='Use 4 modalities (T1, T1CE, T2, FLAIR) instead of 2 (T1CE, FLAIR)')
    return parser.parse_args()


def evaluate_roi_model(model, data_loader, device, distributed=False, world_size=None):
    """Compute WT Dice on the ROI test loader."""
    model.eval()
    total_dice = 0.0
    count = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            dice_scores = calculate_dice_score(logits.detach().cpu(), labels.detach().cpu(), num_classes=2)
            if dice_scores.numel() >= 2:
                total_dice += dice_scores[1].item()
                count += 1.0

    if distributed and dist.is_available() and dist.is_initialized():
        tensor = torch.tensor([total_dice, count], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_dice, count = tensor.tolist()

    return float(total_dice / count) if count > 0 else 0.0


def save_metrics_row(metrics_path, row):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(metrics_path, index=False)


def main():
    args = parse_args()
    os.environ.setdefault('PYTORCH_SHARING_STRATEGY', args.sharing_strategy)

    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    include_coords = not args.disable_coords

    if is_main_process(rank):
        print("Starting ROI training")
        print(f"ROI model: {args.roi_model_name}")
        print(f"Seeds: {args.seeds}")
        print(f"Output dir: {args.output_dir}")

    for seed in args.seeds:
        set_seed(seed)
        loaders = get_roi_data_loaders(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            dataset_version=args.dataset_version,
            seed=seed,
            use_5fold=args.use_5fold,
            fold_idx=args.fold_idx,
            roi_resize=tuple(args.roi_resize),
            include_coords=include_coords,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            use_mri_augmentation=args.use_mri_augmentation,
            anisotropy_augment=args.anisotropy_augmentation,
            use_4modalities=args.use_4modalities,
        )

        roi_model_cfg = get_roi_model_config(args.roi_model_name)
        # ROI 모델 입력 채널 수 계산
        n_modalities = 4 if args.use_4modalities else 2
        n_coord_channels = 3 if include_coords else 0  # simple coords만 사용 (ROI 모델은 hybrid coords 사용 안 함)
        n_channels = n_modalities + n_coord_channels
        
        model = get_roi_model(
            args.roi_model_name,
            n_channels=n_channels,
            n_classes=2,
            roi_model_cfg=roi_model_cfg,
        )

        seed_dir = os.path.join(args.output_dir, args.roi_model_name, f"seed_{seed}")
        weights_dir = os.path.join(seed_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        ckpt_path = os.path.join(weights_dir, args.roi_weight_name)

        stats = train_roi_model(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            epochs=args.epochs,
            device=device,
            lr=args.lr,
            ckpt_path=ckpt_path,
            results_dir=seed_dir,
            model_name=args.roi_model_name,
            train_sampler=loaders.get('train_sampler'),
            rank=rank,
            include_coords=include_coords,
            use_4modalities=args.use_4modalities,
        )

        test_dice = evaluate_roi_model(
            model,
            loaders['test'],
            device=device,
            distributed=distributed,
            world_size=world_size,
        )

        if is_main_process(rank):
            metrics_path = os.path.join(seed_dir, "metrics.csv")
            row = {
                'seed': seed,
                'best_epoch': stats['best_epoch'],
                'best_val_dice': stats['best_val_dice'],
                'test_wt_dice': test_dice,
                'weight_path': ckpt_path,
            }
            save_metrics_row(metrics_path, row)

            config_path = os.path.join(seed_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            print(f"[Seed {seed}] WT Dice (test): {test_dice:.4f} | artifacts -> {seed_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()


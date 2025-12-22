#!/usr/bin/env python3
"""
5-Fold Cross-Validation 데이터셋 분할 준비 스크립트

전처리된 H5 파일들을 5-fold로 분할하여 fold별 디렉토리 구조를 생성합니다.
심볼릭 링크를 사용하여 저장 공간을 절약합니다.

Usage:
    python prepare_5fold_splits.py --data_dir /path/to/data --dataset_version brats2021
    python prepare_5fold_splits.py --data_dir /path/to/data --dataset_version brats2021 --seed 42
"""

import os
import json
import random
import argparse
from pathlib import Path


def get_project_root():
    """프로젝트 루트 디렉토리 찾기"""
    return Path(__file__).parent.absolute()


def prepare_5fold_splits(
    data_dir,
    dataset_version='brats2021',
    seed=24,
    preprocessed_dir=None,
    output_base_dir=None
):
    """
    5-fold 분할을 미리 준비하여 fold별 디렉토리 구조 생성
    
    Args:
        data_dir: 원본 데이터 디렉토리
        dataset_version: 'brats2021' 또는 'brats2018'
        seed: 분할 시드
        preprocessed_dir: 전처리된 H5 파일이 있는 디렉토리
        output_base_dir: fold별 디렉토리를 생성할 기본 경로
    
    Returns:
        output_base_dir: 생성된 fold 디렉토리의 기본 경로
    """
    # 샘플 수집 (이름으로 정렬하여 순서 고정)
    if dataset_version == 'brats2021':
        brats_dir = os.path.join(data_dir, 'BRATS2021', 'BraTS2021_Training_Data')
        if not os.path.exists(brats_dir):
            raise FileNotFoundError(f"BraTS2021 dataset not found at {brats_dir}")
        
        patient_dirs = []
        for patient_dir in sorted(os.listdir(brats_dir)):
            patient_path = os.path.join(brats_dir, patient_dir)
            if os.path.isdir(patient_path):
                patient_dirs.append(patient_path)
    elif dataset_version == 'brats2018':
        brats_dir = os.path.join(data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
        if not os.path.exists(brats_dir):
            raise FileNotFoundError(f"BraTS2018 dataset not found at {brats_dir}")
        
        patient_dirs = []
        hgg_dir = os.path.join(brats_dir, 'HGG')
        lgg_dir = os.path.join(brats_dir, 'LGG')
        
        if os.path.exists(hgg_dir):
            for patient_dir in sorted(os.listdir(hgg_dir)):
                patient_path = os.path.join(hgg_dir, patient_dir)
                if os.path.isdir(patient_path):
                    patient_dirs.append(patient_path)
        
        if os.path.exists(lgg_dir):
            for patient_dir in sorted(os.listdir(lgg_dir)):
                patient_path = os.path.join(lgg_dir, patient_dir)
                if os.path.isdir(patient_path):
                    patient_dirs.append(patient_path)
    else:
        raise ValueError(f"Unknown dataset_version: {dataset_version}")
    
    if not patient_dirs:
        raise ValueError(f"No patient data found in {brats_dir}")
    
    # 샘플 이름으로 정렬 (파일 시스템 순서와 무관)
    patient_names = [os.path.basename(p) for p in patient_dirs]
    sorted_indices = sorted(range(len(patient_names)), key=lambda i: patient_names[i])
    sorted_patient_dirs = [patient_dirs[i] for i in sorted_indices]
    sorted_patient_names = [patient_names[i] for i in sorted_indices]
    
    print(f"Found {len(sorted_patient_dirs)} patient directories")
    
    # 5-fold 분할
    random.seed(seed)
    indices = list(range(len(sorted_patient_dirs)))
    random.shuffle(indices)
    
    fold_size = len(indices) // 5
    fold_sizes = [fold_size] * 5
    remainder = len(indices) % 5
    for i in range(remainder):
        fold_sizes[i] += 1
    
    fold_starts = [0]
    for size in fold_sizes:
        fold_starts.append(fold_starts[-1] + size)
    
    # 출력 디렉토리 설정
    if output_base_dir is None:
        project_root = get_project_root()
        output_base_dir = project_root / 'data' / f'{dataset_version.upper()}_5fold_splits'
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 전처리된 파일 디렉토리
    if preprocessed_dir is None:
        project_root = get_project_root()
        preprocessed_dir = project_root / 'data' / f'{dataset_version.upper()}_preprocessed'
    preprocessed_dir = Path(preprocessed_dir)
    
    if not preprocessed_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed directory not found at {preprocessed_dir}\n"
            f"Please run preprocess_brats.py first to create preprocessed H5 files."
        )
    
    print(f"Preprocessed directory: {preprocessed_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Seed: {seed}")
    print()
    
    # 각 fold별로 디렉토리 생성 및 심볼릭 링크
    split_info = {}
    for fold_idx in range(5):
        test_start = fold_starts[fold_idx]
        test_end = fold_starts[fold_idx + 1]
        test_indices = indices[test_start:test_end]
        
        val_fold_idx = (fold_idx + 1) % 5
        val_start = fold_starts[val_fold_idx]
        val_end = fold_starts[val_fold_idx + 1]
        val_indices = indices[val_start:val_end]
        
        train_indices = []
        for i in range(5):
            if i in (fold_idx, val_fold_idx):
                continue
            train_start = fold_starts[i]
            train_end = fold_starts[i + 1]
            train_indices.extend(indices[train_start:train_end])
        
        # Fold 디렉토리 생성
        fold_dir = output_base_dir / f'fold_{fold_idx}'
        train_dir = fold_dir / 'train'
        val_dir = fold_dir / 'val'
        test_dir = fold_dir / 'test'
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 심볼릭 링크 생성
        def create_links(indices_list, target_dir, split_name):
            created = 0
            missing = 0
            for idx in indices_list:
                patient_name = sorted_patient_names[idx]
                source = preprocessed_dir / f"{patient_name}.h5"
                target = target_dir / f"{patient_name}.h5"
                
                if source.exists():
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    try:
                        target.symlink_to(source.absolute())
                        created += 1
                    except OSError as e:
                        # Windows에서는 관리자 권한이 필요할 수 있음
                        print(f"Warning: Failed to create symlink for {patient_name}: {e}")
                        # Windows에서는 복사로 대체
                        import shutil
                        shutil.copy2(source, target)
                        created += 1
                else:
                    missing += 1
                    print(f"Warning: Preprocessed file not found: {source}")
            return created, missing
        
        train_created, train_missing = create_links(train_indices, train_dir, 'train')
        val_created, val_missing = create_links(val_indices, val_dir, 'val')
        test_created, test_missing = create_links(test_indices, test_dir, 'test')
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_indices)} samples ({train_created} created, {train_missing} missing)")
        print(f"  Val:   {len(val_indices)} samples ({val_created} created, {val_missing} missing)")
        print(f"  Test:  {len(test_indices)} samples ({test_created} created, {test_missing} missing)")
        
        # 분할 정보 저장
        split_info[f'fold_{fold_idx}'] = {
            'train': [sorted_patient_names[i] for i in train_indices],
            'val': [sorted_patient_names[i] for i in val_indices],
            'test': [sorted_patient_names[i] for i in test_indices],
        }
    
    # 분할 정보를 JSON으로 저장
    split_info_path = output_base_dir / 'split_info.json'
    with open(split_info_path, 'w') as f:
        json.dump({
            'seed': seed,
            'dataset_version': dataset_version,
            'total_samples': len(sorted_patient_dirs),
            'splits': split_info
        }, f, indent=2)
    
    print()
    print(f"5-fold splits prepared in {output_base_dir}")
    print(f"Split info saved to {split_info_path}")
    
    return str(output_base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare 5-fold cross-validation splits')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory containing BRATS dataset')
    parser.add_argument('--dataset_version', type=str, default='brats2021',
                       choices=['brats2021', 'brats2018'],
                       help='Dataset version (default: brats2021)')
    parser.add_argument('--seed', type=int, default=24,
                       help='Random seed for splitting (default: 24)')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory containing preprocessed H5 files (default: data/{DATASET}_preprocessed)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for fold splits (default: data/{DATASET}_5fold_splits)')
    
    args = parser.parse_args()
    
    prepare_5fold_splits(
        data_dir=args.data_dir,
        dataset_version=args.dataset_version,
        seed=args.seed,
        preprocessed_dir=args.preprocessed_dir,
        output_base_dir=args.output_dir,
    )


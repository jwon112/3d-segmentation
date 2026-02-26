#!/usr/bin/env python3
"""디버깅: 실제 로드된 라벨 값 확인"""

import torch
import sys
sys.path.insert(0, '3d-segmentation')

from dataloaders import get_data_loaders

# BRATS2024 데이터 로더 생성
train_loader, val_loader, test_loader, _, _, _ = get_data_loaders(
    data_dir='/home/work/3D_/BT/',
    batch_size=1,
    num_workers=0,
    max_samples=1,
    dim='3d',
    dataset_version='brats2024',
    seed=42,
    distributed=False,
    world_size=1,
    rank=0,
    use_4modalities=False,
    use_5fold=False,
    fold_idx=None,
    fold_split_dir='/home/work/3D_/processed_data/BRATS2024_5fold_splits',
    preprocessed_dir=None,
)

print("Checking validation loader...")
for idx, batch_data in enumerate(val_loader):
    if len(batch_data) == 3:
        inputs, labels, _ = batch_data
    else:
        inputs, labels = batch_data
    
    print(f"\nBatch {idx}:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Label shape: {labels.shape}")
    print(f"  Label unique values: {torch.unique(labels)}")
    print(f"  Label min: {labels.min()}, max: {labels.max()}")
    
    # 각 클래스별 복셀 수
    for cls in range(int(labels.max().item()) + 1):
        count = (labels == cls).sum().item()
        print(f"    Class {cls}: {count} voxels")
    
    if idx >= 2:  # 3개 샘플만 확인
        break

print("\n" + "="*60)
print("If labels max is 3, then RC/ET mapping issue!")
print("If labels max is 4, then model might be 4 classes!")
print("="*60)


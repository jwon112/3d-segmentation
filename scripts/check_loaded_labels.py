#!/usr/bin/env python3
"""실제 학습 중인 데이터의 라벨 분포 확인"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders import get_data_loaders

# BRATS2024 데이터 로더 생성
train_loader, val_loader, test_loader, _, _, _ = get_data_loaders(
    data_dir='/home/work/3D_/BT/',
    batch_size=1,
    num_workers=0,
    max_samples=None,
    dim='3d',
    dataset_version='brats2024',
    seed=42,
    distributed=False,
    world_size=1,
    rank=0,
    use_4modalities=False,
    use_5fold=False,
    fold_idx=None,
    fold_split_dir=None,
    preprocessed_dir='/home/work/3D_/processed_data/BRATS2024',
)

print("Checking validation loader labels...")
label_3_counts = []
label_4_counts = []
total_samples = 0

for idx, batch_data in enumerate(val_loader):
    if len(batch_data) == 3:
        inputs, labels, _ = batch_data
    else:
        inputs, labels = batch_data
    
    unique_labels = torch.unique(labels).tolist()
    count_3 = (labels == 3).sum().item()
    count_4 = (labels == 4).sum().item()
    
    label_3_counts.append(count_3)
    label_4_counts.append(count_4)
    total_samples += 1
    
    if idx < 5:  # 처음 5개 샘플 상세 출력
        print(f"\nSample {idx+1}:")
        print(f"  Unique labels: {unique_labels}")
        print(f"  Label 3 (assumed RC) voxels: {count_3:,}")
        print(f"  Label 4 (assumed ET) voxels: {count_4:,}")
        if count_3 > 0 and count_4 > 0:
            ratio = count_3 / count_4 if count_4 > 0 else float('inf')
            print(f"  Ratio (3/4): {ratio:.2f}")

print("\n" + "="*60)
print(f"Summary over {total_samples} samples:")
if label_3_counts:
    print(f"  Average label 3 (RC) voxels: {sum(label_3_counts)/len(label_3_counts):,.0f}")
    print(f"  Average label 4 (ET) voxels: {sum(label_4_counts)/len(label_4_counts):,.0f}")
    print(f"  Samples with label 3: {sum(1 for c in label_3_counts if c > 0)}/{total_samples}")
    print(f"  Samples with label 4: {sum(1 for c in label_4_counts if c > 0)}/{total_samples}")
print("="*60)
print("\nNote:")
print("  - If label 3 is more common: Current mapping (RC=3, ET=4) is likely correct")
print("  - If label 4 is more common: May need to swap (RC=4, ET=3)")
print("  - Check BRATS2024 official documentation for correct mapping")


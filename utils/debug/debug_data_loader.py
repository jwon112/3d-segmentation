#!/usr/bin/env python3
"""
데이터 로더 검증 스크립트

검증 항목:
1. 데이터 로더가 올바른 라벨 범위(0-3)를 반환하는지 확인
2. 라벨 분포 (클래스별 픽셀 수) 확인
3. 정규화 후 입력 데이터 범위 확인 (NaN/Inf 체크)
4. 학습/검증 데이터셋의 라벨 분포 비교
5. BratsPatchDataset3D의 패치 샘플링이 올바른지 확인
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import argparse

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataloaders import get_data_loaders, BratsDataset3D, BratsPatchDataset3D, BratsDataset2D
import torch.nn.functional as F


def check_label_range(labels, name="Labels"):
    """라벨 범위 확인"""
    print(f"\n{'='*60}")
    print(f"{name} - 라벨 범위 확인")
    print(f"{'='*60}")
    
    if isinstance(labels, torch.Tensor):
        min_label = labels.min().item()
        max_label = labels.max().item()
        unique_labels = torch.unique(labels).tolist()
        total_pixels = labels.numel()
    else:
        min_label = labels.min()
        max_label = labels.max()
        unique_labels = np.unique(labels).tolist()
        total_pixels = labels.size
    
    print(f"최소값: {min_label}")
    print(f"최대값: {max_label}")
    print(f"고유 라벨: {unique_labels}")
    print(f"총 픽셀 수: {total_pixels}")
    
    # 기대 범위는 0-3
    if min_label < 0 or max_label > 3:
        print(f"⚠️  경고: 라벨 범위가 예상 범위(0-3)를 벗어났습니다!")
        return False
    
    # 4가 라벨에 있는지 확인 (매핑 전 상태)
    if 4 in unique_labels:
        print(f"⚠️  경고: 라벨 4가 여전히 존재합니다. 매핑이 제대로 되지 않았습니다!")
        return False
    
    print("✓ 라벨 범위 검증 통과")
    return True


def check_label_distribution(labels, name="Labels"):
    """라벨 분포 확인 (클래스별 픽셀 수)"""
    print(f"\n{'='*60}")
    print(f"{name} - 라벨 분포")
    print(f"{'='*60}")
    
    if isinstance(labels, torch.Tensor):
        labels_np = labels.numpy()
    else:
        labels_np = labels
    
    total_pixels = labels_np.size
    print(f"총 픽셀 수: {total_pixels}")
    
    for class_id in range(4):
        count = np.sum(labels_np == class_id)
        percentage = (count / total_pixels) * 100
        print(f"클래스 {class_id}: {count:,} 픽셀 ({percentage:.2f}%)")
    
    # 포그라운드 비율
    fg_pixels = np.sum(labels_np > 0)
    fg_ratio = (fg_pixels / total_pixels) * 100
    print(f"\n포그라운드 (>0): {fg_pixels:,} 픽셀 ({fg_ratio:.2f}%)")
    print(f"배경 (0): {total_pixels - fg_pixels:,} 픽셀 ({100 - fg_ratio:.2f}%)")
    
    return {
        'class_counts': [np.sum(labels_np == c) for c in range(4)],
        'fg_ratio': fg_ratio / 100
    }


def check_normalization(images, name="Images"):
    """정규화 후 입력 데이터 범위 확인"""
    print(f"\n{'='*60}")
    print(f"{name} - 정규화 검증")
    print(f"{'='*60}")
    
    if isinstance(images, torch.Tensor):
        images_np = images.numpy()
    else:
        images_np = images
    
    print(f"Shape: {images_np.shape}")
    print(f"데이터 타입: {images_np.dtype}")
    
    # NaN/Inf 체크
    nan_count = np.isnan(images_np).sum()
    inf_count = np.isinf(images_np).sum()
    
    if nan_count > 0:
        print(f"⚠️  경고: NaN이 {nan_count}개 발견되었습니다!")
        return False
    
    if inf_count > 0:
        print(f"⚠️  경고: Inf가 {inf_count}개 발견되었습니다!")
        return False
    
    # 통계 정보
    for c in range(images_np.shape[0] if len(images_np.shape) > 3 else 1):
        if len(images_np.shape) == 4:  # (C, H, W, D)
            channel_data = images_np[c]
        elif len(images_np.shape) == 5:  # (B, C, H, W, D)
            channel_data = images_np[0, c]
        elif len(images_np.shape) == 3:  # (C, H, W)
            channel_data = images_np[c]
        else:
            channel_data = images_np
        
        # 비영점 픽셀만 확인 (정규화는 비영점 기준)
        nz_mask = channel_data != 0
        if nz_mask.sum() > 0:
            nz_data = channel_data[nz_mask]
            print(f"\n채널 {c} (비영점 픽셀):")
            print(f"  최소값: {nz_data.min():.4f}")
            print(f"  최대값: {nz_data.max():.4f}")
            print(f"  평균: {nz_data.mean():.4f}")
            print(f"  표준편차: {nz_data.std():.4f}")
            print(f"  비영점 픽셀 수: {nz_mask.sum():,}")
        
        # 전체 데이터 범위
        print(f"채널 {c} (전체):")
        print(f"  최소값: {channel_data.min():.4f}")
        print(f"  최대값: {channel_data.max():.4f}")
    
    print("✓ 정규화 검증 통과 (NaN/Inf 없음)")
    return True


def check_dataset_samples(dataset, dataset_name, num_samples=5):
    """데이터셋 샘플 검증"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} 데이터셋 검증")
    print(f"{'='*60}")
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    label_distributions = []
    
    for i in range(min(num_samples, len(dataset))):
        print(f"\n--- 샘플 {i+1} ---")
        try:
            # dataset이 (image, mask) 또는 (image, mask, fg_coords_dict) 반환 가능
            loaded_data = dataset[i]
            if len(loaded_data) == 3:
                image, mask, _ = loaded_data  # fg_coords_dict는 debug에서는 사용 안 함
            else:
                image, mask = loaded_data
            
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            
            # 라벨 범위 확인
            if not check_label_range(mask, f"Sample {i+1} Mask"):
                return False
            
            # 라벨 분포 확인
            dist = check_label_distribution(mask, f"Sample {i+1} Mask")
            label_distributions.append(dist)
            
            # 정규화 확인
            if not check_normalization(image, f"Sample {i+1} Image"):
                return False
            
        except Exception as e:
            print(f"⚠️  샘플 {i+1} 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 평균 분포 계산
    if label_distributions:
        print(f"\n{'='*60}")
        print(f"{dataset_name} 평균 라벨 분포")
        print(f"{'='*60}")
        avg_fg_ratio = np.mean([d['fg_ratio'] for d in label_distributions])
        print(f"평균 포그라운드 비율: {avg_fg_ratio:.4f} ({avg_fg_ratio*100:.2f}%)")
        
        for class_id in range(4):
            avg_count = np.mean([d['class_counts'][class_id] for d in label_distributions])
            print(f"클래스 {class_id} 평균 픽셀 수: {avg_count:,.0f}")
    
    return True


def check_patch_dataset(base_dataset, patch_size=(128, 128, 128), 
                        samples_per_volume=16):
    """BratsPatchDataset3D 검증 (nnU-Net 스타일)"""
    print(f"\n{'='*60}")
    print("BratsPatchDataset3D 패치 샘플링 검증 (nnU-Net 스타일)")
    print(f"{'='*60}")
    
    patch_dataset = BratsPatchDataset3D(
        base_dataset=base_dataset,
        patch_size=patch_size,
        samples_per_volume=samples_per_volume,
    )
    
    print(f"패치 데이터셋 크기: {len(patch_dataset)}")
    print(f"기본 데이터셋 크기: {len(base_dataset)}")
    print(f"볼륨당 샘플 수: {samples_per_volume}")
    print(f"샘플링 전략: nnU-Net 스타일 (1/3 포그라운드 오버샘플링, 2/3 완전 랜덤)")
    
    # 여러 패치 샘플 확인
    num_patches_to_check = min(10, len(patch_dataset))
    fg_ratios = []
    
    for i in range(num_patches_to_check):
        try:
            image_patch, mask_patch = patch_dataset[i]
            
            assert image_patch.shape[1:] == patch_size, f"패치 크기 불일치: {image_patch.shape[1:]} vs {patch_size}"
            assert mask_patch.shape == patch_size, f"마스크 크기 불일치: {mask_patch.shape} vs {patch_size}"
            
            # 포그라운드 비율 확인
            fg_ratio = (mask_patch > 0).float().mean().item()
            fg_ratios.append(fg_ratio)
            
            if i < 3:  # 처음 3개만 상세 출력
                print(f"\n패치 {i+1}:")
                print(f"  Image shape: {image_patch.shape}")
                print(f"  Mask shape: {mask_patch.shape}")
                print(f"  포그라운드 비율: {fg_ratio:.4f}")
                check_label_range(mask_patch, f"Patch {i+1}")
                
        except Exception as e:
            print(f"⚠️  패치 {i+1} 검증 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 포그라운드 비율 통계
    if fg_ratios:
        print(f"\n{'='*60}")
        print("패치 포그라운드 비율 통계")
        print(f"{'='*60}")
        print(f"평균: {np.mean(fg_ratios):.4f}")
        print(f"최소: {np.min(fg_ratios):.4f}")
        print(f"최대: {np.max(fg_ratios):.4f}")
        print(f"표준편차: {np.std(fg_ratios):.4f}")
        
        # nnU-Net 스타일 검증: 1/3은 포그라운드 오버샘플링이므로 포그라운드가 있을 가능성이 높음
        fg_patches = [r for r in fg_ratios if r > 0]
        print(f"포그라운드 포함 패치 비율: {len(fg_patches)}/{len(fg_ratios)} ({len(fg_patches)/len(fg_ratios)*100:.1f}%)")
        if fg_patches:
            print(f"포그라운드 포함 패치의 평균 비율: {np.mean(fg_patches):.4f}")
    
    print("✓ 패치 샘플링 검증 통과")
    return True


def compare_train_val_distributions(train_loader, val_loader):
    """학습/검증 데이터셋의 라벨 분포 비교"""
    print(f"\n{'='*60}")
    print("학습/검증 라벨 분포 비교")
    print(f"{'='*60}")
    
    # 학습 데이터 분포 수집
    train_distributions = []
    for i, (image, mask) in enumerate(train_loader):
        if i >= 10:  # 처음 10개만
            break
        dist = check_label_distribution(mask[0], f"Train Batch {i}")
        train_distributions.append(dist)
    
    # 검증 데이터 분포 수집
    val_distributions = []
    for i, (image, mask) in enumerate(val_loader):
        if i >= 10:  # 처음 10개만
            break
        dist = check_label_distribution(mask[0], f"Val Batch {i}")
        val_distributions.append(dist)
    
    if train_distributions and val_distributions:
        train_avg_fg = np.mean([d['fg_ratio'] for d in train_distributions])
        val_avg_fg = np.mean([d['fg_ratio'] for d in val_distributions])
        
        print(f"\n평균 포그라운드 비율:")
        print(f"  학습: {train_avg_fg:.4f} ({train_avg_fg*100:.2f}%)")
        print(f"  검증: {val_avg_fg:.4f} ({val_avg_fg*100:.2f}%)")
        print(f"  차이: {abs(train_avg_fg - val_avg_fg):.4f}")
        
        if abs(train_avg_fg - val_avg_fg) > 0.1:
            print(f"⚠️  경고: 학습/검증 포그라운드 비율 차이가 큽니다!")
            return False
    
    return True


def main():
    """메인 검증 함수"""
    parser = argparse.ArgumentParser(description='데이터 로더 검증')
    parser.add_argument('--data_dir', type=str, 
                        default=os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021'),
                        help='BraTS 데이터셋 루트 디렉토리 (기본값: 환경변수 BRATS_DATA_DIR 또는 /home/work/3D_/BT/BRATS2021)')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    print("="*60)
    print("데이터 로더 종합 검증")
    print("="*60)
    print(f"데이터 디렉토리: {data_dir}")
    
    # 1. 3D 데이터셋 기본 검증
    print("\n[1단계] 3D 데이터셋 기본 검증")
    base_dataset_3d = BratsDataset3D(data_dir, split='train', max_samples=5)
    if not check_dataset_samples(base_dataset_3d, "BratsDataset3D", num_samples=min(3, len(base_dataset_3d))):
        print("❌ 3D 데이터셋 검증 실패")
        return
    
    # 2. 패치 데이터셋 검증
    print("\n[2단계] 패치 데이터셋 검증")
    if not check_patch_dataset(base_dataset_3d, patch_size=(128, 128, 128), 
                               samples_per_volume=16):
        print("❌ 패치 데이터셋 검증 실패")
        return
    
    # 3. 데이터 로더 검증
    print("\n[3단계] 데이터 로더 검증")
    train_loader, val_loader, test_loader, _, _, _ = get_data_loaders(
        data_dir, batch_size=1, num_workers=0, max_samples=10, dim='3d'
    )
    
    print(f"\n로더 크기:")
    print(f"  학습: {len(train_loader)} 배치")
    print(f"  검증: {len(val_loader)} 배치")
    print(f"  테스트: {len(test_loader)} 배치")
    
    # 학습 로더 샘플 검증
    print("\n[4단계] 학습 로더 샘플 검증")
    for i, (image, mask) in enumerate(train_loader):
        if i >= 3:
            break
        print(f"\n학습 배치 {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        check_label_range(mask[0], f"Train Batch {i+1}")
        check_label_distribution(mask[0], f"Train Batch {i+1}")
    
    # 검증 로더 샘플 검증
    print("\n[5단계] 검증 로더 샘플 검증")
    for i, (image, mask) in enumerate(val_loader):
        if i >= 3:
            break
        print(f"\n검증 배치 {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        check_label_range(mask[0], f"Val Batch {i+1}")
        check_label_distribution(mask[0], f"Val Batch {i+1}")
    
    # 학습/검증 분포 비교
    print("\n[6단계] 학습/검증 분포 비교")
    compare_train_val_distributions(train_loader, val_loader)
    
    print("\n" + "="*60)
    print("✅ 데이터 로더 검증 완료")
    print("="*60)


if __name__ == "__main__":
    main()


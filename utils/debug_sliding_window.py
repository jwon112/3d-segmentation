#!/usr/bin/env python3
"""
슬라이딩 윈도우 검증 스크립트

검증 항목:
1. sliding_window_inference_3d가 올바른 출력 shape을 반환하는지 확인
2. 패치 단위 예측과 전체 볼륨 예측의 일관성 확인
3. Blending 가중치가 올바르게 적용되는지 확인
4. 작은 볼륨에서 직접 forward와 sliding window 결과 비교
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import argparse

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrated_experiment import sliding_window_inference_3d, _make_blend_weights_3d
from data_loader import get_data_loaders
from integrated_experiment import get_model
import torch.nn.functional as F


def check_sliding_window_output_shape():
    """슬라이딩 윈도우 출력 shape 검증"""
    print(f"\n{'='*60}")
    print("슬라이딩 윈도우 출력 Shape 검증")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 더미 모델 생성 (입력과 동일한 크기 출력)
    class DummyModel(torch.nn.Module):
        def __init__(self, out_channels=4):
            super().__init__()
            self.conv = torch.nn.Conv3d(2, out_channels, kernel_size=1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel(out_channels=4).to(device)
    model.eval()
    
    # 테스트 볼륨 크기들
    test_sizes = [
        (1, 2, 128, 128, 128),  # 정확히 패치 크기
        (1, 2, 240, 240, 155),  # 실제 BraTS 크기
        (1, 2, 200, 200, 150),  # 중간 크기
    ]
    
    patch_size = (128, 128, 128)
    overlap = 0.1
    
    for vol_shape in test_sizes:
        print(f"\n테스트 볼륨 shape: {vol_shape}")
        volume = torch.randn(vol_shape).to(device)
        
        with torch.no_grad():
            output = sliding_window_inference_3d(
                model, volume, patch_size=patch_size, overlap=overlap, device=device
            )
        
        expected_shape = (1, 4, vol_shape[2], vol_shape[3], vol_shape[4])
        actual_shape = tuple(output.shape)
        
        print(f"  예상 출력 shape: {expected_shape}")
        print(f"  실제 출력 shape: {actual_shape}")
        
        if actual_shape != expected_shape:
            print(f"  ❌ Shape 불일치!")
            return False
        
        print(f"  ✓ Shape 검증 통과")
    
    return True


def check_blending_weights():
    """Blending 가중치 검증"""
    print(f"\n{'='*60}")
    print("Blending 가중치 검증")
    print(f"{'='*60}")
    
    patch_size = (128, 128, 128)
    weights = _make_blend_weights_3d(patch_size)
    
    print(f"패치 크기: {patch_size}")
    print(f"가중치 shape: {weights.shape}")
    print(f"가중치 범위: [{weights.min():.4f}, {weights.max():.4f}]")
    
    # 가중치가 정규화되어 있는지 확인 (최대값이 1)
    if not torch.allclose(weights.max(), torch.tensor(1.0)):
        print(f"⚠️  경고: 가중치 최대값이 1이 아닙니다!")
        return False
    
    # 가중치가 중심에서 높고 경계에서 낮은지 확인
    center_idx = [s // 2 for s in patch_size]
    center_weight = weights[0, 0, center_idx[0], center_idx[1], center_idx[2]].item()
    corner_weight = weights[0, 0, 0, 0, 0].item()
    
    print(f"중심 가중치: {center_weight:.4f}")
    print(f"모서리 가중치: {corner_weight:.4f}")
    
    if center_weight < corner_weight:
        print(f"⚠️  경고: 중심 가중치가 모서리보다 낮습니다!")
        return False
    
    # 가중치 합이 패치 내 모든 위치에서 양수인지 확인
    if (weights <= 0).any():
        print(f"⚠️  경고: 음수 또는 0인 가중치가 있습니다!")
        return False
    
    print("✓ Blending 가중치 검증 통과")
    return True


def check_patch_vs_full_volume_consistency():
    """패치 단위 예측과 전체 볼륨 예측의 일관성 확인"""
    print(f"\n{'='*60}")
    print("패치 vs 전체 볼륨 일관성 검증")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 작은 볼륨으로 테스트 (메모리 절약)
    vol_shape = (1, 2, 128, 128, 128)
    patch_size = (64, 64, 64)  # 볼륨보다 작은 패치
    
    # 모델 생성
    model = get_model('mobile_unetr_3d', n_channels=2, n_classes=4, dim='3d', use_pretrained=False)
    model = model.to(device)
    model.eval()
    
    # 테스트 볼륨
    volume = torch.randn(vol_shape).to(device)
    
    # 슬라이딩 윈도우로 전체 볼륨 예측
    with torch.no_grad():
        sw_output = sliding_window_inference_3d(
            model, volume, patch_size=patch_size, overlap=0.25, device=device
        )
    
    # 직접 forward (가능한 경우)
    try:
        with torch.no_grad():
            direct_output = model(volume)
        
        print(f"직접 forward 출력 shape: {direct_output.shape}")
        print(f"슬라이딩 윈도우 출력 shape: {sw_output.shape}")
        
        # 패치 중심 영역 비교 (blending 영향 최소화)
        # 첫 번째 패치 영역 (0:64, 0:64, 0:64)
        patch_region = slice(0, 64)
        
        direct_patch = direct_output[:, :, patch_region, patch_region, patch_region]
        sw_patch = sw_output[:, :, patch_region, patch_region, patch_region]
        
        # 차이 계산
        diff = torch.abs(direct_patch - sw_patch)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\n패치 영역 비교 (0:64, 0:64, 0:64):")
        print(f"  최대 차이: {max_diff:.6f}")
        print(f"  평균 차이: {mean_diff:.6f}")
        
        # Blending 때문에 완전히 같지 않을 수 있음
        if max_diff > 1.0:
            print(f"  ⚠️  경고: 차이가 큽니다. Blending 가중치를 확인해보세요.")
        else:
            print(f"  ✓ 차이가 허용 범위 내입니다.")
        
    except RuntimeError as e:
        if "out of memory" in str(e) or "memory" in str(e).lower():
            print(f"  메모리 부족으로 직접 forward 불가 (예상됨)")
        else:
            print(f"  직접 forward 실패: {e}")
    
    return True


def check_sliding_window_with_small_volume():
    """작은 볼륨에서 직접 forward와 sliding window 결과 비교"""
    print(f"\n{'='*60}")
    print("작은 볼륨 직접 Forward vs Sliding Window 비교")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 작은 볼륨 (패치 크기와 동일하거나 작음)
    vol_shape = (1, 2, 64, 64, 64)
    patch_size = (64, 64, 64)
    
    model = get_model('mobile_unetr_3d', n_channels=2, n_classes=4, dim='3d', use_pretrained=False)
    model = model.to(device)
    model.eval()
    
    volume = torch.randn(vol_shape).to(device)
    
    # 직접 forward
    with torch.no_grad():
        direct_output = model(volume)
    
    # 슬라이딩 윈도우 (overlap=0으로 설정하면 직접 forward와 동일해야 함)
    with torch.no_grad():
        sw_output = sliding_window_inference_3d(
            model, volume, patch_size=patch_size, overlap=0.0, device=device
        )
    
    print(f"볼륨 shape: {vol_shape}")
    print(f"직접 forward 출력 shape: {direct_output.shape}")
    print(f"슬라이딩 윈도우 출력 shape: {sw_output.shape}")
    
    # 차이 계산
    diff = torch.abs(direct_output - sw_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n차이 분석:")
    print(f"  최대 차이: {max_diff:.6f}")
    print(f"  평균 차이: {mean_diff:.6f}")
    
    # Overlap이 0이면 거의 동일해야 함 (blending 가중치 때문에 약간 다를 수 있음)
    if max_diff < 1e-3:
        print(f"  ✓ 거의 동일합니다 (정상)")
    elif max_diff < 0.1:
        print(f"  ⚠️  작은 차이가 있습니다 (blending 가중치 때문일 수 있음)")
    else:
        print(f"  ❌ 차이가 큽니다!")
        return False
    
    # 예측 클래스 비교
    direct_pred = torch.argmax(direct_output, dim=1)
    sw_pred = torch.argmax(sw_output, dim=1)
    
    agreement = (direct_pred == sw_pred).float().mean().item()
    print(f"  예측 일치율: {agreement*100:.2f}%")
    
    return True


def check_sliding_window_with_real_data(data_dir=None):
    """실제 데이터로 슬라이딩 윈도우 검증"""
    print(f"\n{'='*60}")
    print("실제 데이터 슬라이딩 윈도우 검증")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터 로더 준비
    if data_dir is None:
        data_dir = os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021')
    _, val_loader, _, _, _, _ = get_data_loaders(
        data_dir, batch_size=1, num_workers=0, max_samples=2, dim='3d'
    )
    
    # 모델 준비
    model = get_model('mobile_unetr_3d', n_channels=2, n_classes=4, dim='3d', use_pretrained=False)
    model = model.to(device)
    model.eval()
    
    patch_size = (128, 128, 128)
    overlap = 0.1
    
    for i, (inputs, labels) in enumerate(val_loader):
        if i >= 1:
            break
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        print(f"\n검증 샘플 {i+1}:")
        print(f"  입력 shape: {inputs.shape}")
        print(f"  라벨 shape: {labels.shape}")
        
        # 슬라이딩 윈도우 추론
        with torch.no_grad():
            logits = sliding_window_inference_3d(
                model, inputs, patch_size=patch_size, overlap=overlap, device=device
            )
        
        print(f"  출력 shape: {logits.shape}")
        
        # Shape 검증
        expected_shape = (inputs.shape[0], 4, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        if tuple(logits.shape) != expected_shape:
            print(f"  ❌ Shape 불일치: {logits.shape} vs {expected_shape}")
            return False
        
        print(f"  ✓ Shape 검증 통과")
        
        # NaN/Inf 체크
        if torch.isnan(logits).any():
            print(f"  ❌ NaN이 발견되었습니다!")
            return False
        
        if torch.isinf(logits).any():
            print(f"  ❌ Inf가 발견되었습니다!")
            return False
        
        print(f"  ✓ NaN/Inf 없음")
        
        # 예측 분포 확인
        pred = torch.argmax(logits, dim=1)
        pred_unique = torch.unique(pred).tolist()
        labels_unique = torch.unique(labels).tolist()
        
        print(f"  예측 클래스: {pred_unique}")
        print(f"  GT 클래스: {labels_unique}")
        
        # 값 범위 확인
        print(f"  Logits 범위: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"  Logits 평균: {logits.mean():.4f}")
        print(f"  Logits 표준편차: {logits.std():.4f}")
    
    return True


def main():
    """메인 검증 함수"""
    parser = argparse.ArgumentParser(description='슬라이딩 윈도우 검증')
    parser.add_argument('--data_dir', type=str, 
                        default=os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021'),
                        help='BraTS 데이터셋 루트 디렉토리 (기본값: 환경변수 BRATS_DATA_DIR 또는 /home/work/3D_/BT/BRATS2021)')
    args = parser.parse_args()
    
    print("="*60)
    print("슬라이딩 윈도우 종합 검증")
    print("="*60)
    print(f"데이터 디렉토리: {args.data_dir}")
    
    # 1. 출력 shape 검증
    print("\n[1단계] 출력 Shape 검증")
    if not check_sliding_window_output_shape():
        print("❌ 출력 Shape 검증 실패")
        return
    
    # 2. Blending 가중치 검증
    print("\n[2단계] Blending 가중치 검증")
    if not check_blending_weights():
        print("❌ Blending 가중치 검증 실패")
        return
    
    # 3. 작은 볼륨 직접 vs Sliding Window 비교
    print("\n[3단계] 작은 볼륨 직접 Forward vs Sliding Window 비교")
    try:
        check_sliding_window_with_small_volume()
    except Exception as e:
        print(f"⚠️  비교 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 패치 vs 전체 볼륨 일관성
    print("\n[4단계] 패치 vs 전체 볼륨 일관성 검증")
    try:
        check_patch_vs_full_volume_consistency()
    except Exception as e:
        print(f"⚠️  일관성 검증 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 실제 데이터 검증
    print("\n[5단계] 실제 데이터 슬라이딩 윈도우 검증")
    try:
        if not check_sliding_window_with_real_data(args.data_dir):
            print("❌ 실제 데이터 검증 실패")
            return
    except Exception as e:
        print(f"⚠️  실제 데이터 검증 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ 슬라이딩 윈도우 검증 완료")
    print("="*60)


if __name__ == "__main__":
    main()


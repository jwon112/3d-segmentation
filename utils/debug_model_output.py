#!/usr/bin/env python3
"""
모델 출력 검증 스크립트

검증 항목:
1. 모델 출력 shape 확인 (예상: (B, 4, H, W, D))
2. Logits의 값 범위 확인 (NaN/Inf 체크)
3. Softmax 적용 후 클래스별 확률 분포 확인
4. Argmax 후 예측 분포 확인
5. 동일 입력에 대한 학습/검증 모드 차이 확인 (dropout, BN 등)
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import get_data_loaders
from integrated_experiment import get_model
import torch.nn.functional as F


def check_output_shape(logits, expected_shape, name="Logits"):
    """모델 출력 shape 확인"""
    print(f"\n{'='*60}")
    print(f"{name} - Shape 검증")
    print(f"{'='*60}")
    
    actual_shape = tuple(logits.shape)
    print(f"실제 shape: {actual_shape}")
    print(f"예상 shape: {expected_shape}")
    
    if actual_shape != expected_shape:
        print(f"⚠️  경고: Shape이 예상과 다릅니다!")
        # B, C 차원만 확인
        if len(actual_shape) == len(expected_shape):
            if actual_shape[0] != expected_shape[0]:
                print(f"  배치 크기 불일치: {actual_shape[0]} vs {expected_shape[0]}")
            if actual_shape[1] != expected_shape[1]:
                print(f"  채널 수 불일치: {actual_shape[1]} vs {expected_shape[1]}")
        return False
    
    print("✓ Shape 검증 통과")
    return True


def check_logits_values(logits, name="Logits"):
    """Logits의 값 범위 및 NaN/Inf 체크"""
    print(f"\n{'='*60}")
    print(f"{name} - 값 범위 검증")
    print(f"{'='*60}")
    
    logits_np = logits.detach().cpu().numpy()
    
    # NaN/Inf 체크
    nan_count = np.isnan(logits_np).sum()
    inf_count = np.isinf(logits_np).sum()
    
    if nan_count > 0:
        print(f"❌ 오류: NaN이 {nan_count}개 발견되었습니다!")
        return False
    
    if inf_count > 0:
        print(f"❌ 오류: Inf가 {inf_count}개 발견되었습니다!")
        return False
    
    # 통계 정보
    print(f"Shape: {logits_np.shape}")
    print(f"최소값: {logits_np.min():.4f}")
    print(f"최대값: {logits_np.max():.4f}")
    print(f"평균: {logits_np.mean():.4f}")
    print(f"표준편차: {logits_np.std():.4f}")
    
    # 클래스별 통계
    print(f"\n클래스별 통계:")
    for c in range(logits_np.shape[1]):
        class_logits = logits_np[:, c, ...]
        print(f"  클래스 {c}: min={class_logits.min():.4f}, max={class_logits.max():.4f}, "
              f"mean={class_logits.mean():.4f}, std={class_logits.std():.4f}")
    
    print("✓ 값 범위 검증 통과 (NaN/Inf 없음)")
    return True


def check_softmax_distribution(logits, name="Logits"):
    """Softmax 적용 후 클래스별 확률 분포 확인"""
    print(f"\n{'='*60}")
    print(f"{name} - Softmax 확률 분포")
    print(f"{'='*60}")
    
    probs = F.softmax(logits, dim=1)
    probs_np = probs.detach().cpu().numpy()
    
    # 확률 합이 1인지 확인
    prob_sums = probs_np.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-5):
        print(f"⚠️  경고: 클래스별 확률 합이 1이 아닙니다!")
        print(f"  최소 합: {prob_sums.min():.6f}")
        print(f"  최대 합: {prob_sums.max():.6f}")
        return False
    
    # 클래스별 평균 확률
    print(f"클래스별 평균 확률:")
    for c in range(probs_np.shape[1]):
        class_probs = probs_np[:, c, ...]
        mean_prob = class_probs.mean()
        max_prob = class_probs.max()
        print(f"  클래스 {c}: 평균={mean_prob:.4f}, 최대={max_prob:.4f}")
    
    # 공간 위치별 최대 확률 클래스 분포
    pred_classes = np.argmax(probs_np, axis=1)
    for c in range(probs_np.shape[1]):
        count = np.sum(pred_classes == c)
        percentage = (count / pred_classes.size) * 100
        print(f"  예측 클래스 {c}: {count:,} 픽셀 ({percentage:.2f}%)")
    
    print("✓ Softmax 확률 분포 검증 통과")
    return True


def check_argmax_prediction(logits, labels=None, name="Prediction"):
    """Argmax 후 예측 분포 확인"""
    print(f"\n{'='*60}")
    print(f"{name} - Argmax 예측 분포")
    print(f"{'='*60}")
    
    pred = torch.argmax(logits, dim=1)
    pred_np = pred.detach().cpu().numpy()
    
    print(f"예측 shape: {pred_np.shape}")
    print(f"예측 범위: {pred_np.min()} ~ {pred_np.max()}")
    print(f"고유 예측 클래스: {np.unique(pred_np).tolist()}")
    
    # 클래스별 픽셀 수
    print(f"\n클래스별 예측 픽셀 수:")
    total_pixels = pred_np.size
    for c in range(4):
        count = np.sum(pred_np == c)
        percentage = (count / total_pixels) * 100
        print(f"  클래스 {c}: {count:,} 픽셀 ({percentage:.2f}%)")
    
    # GT와 비교 (있는 경우)
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        print(f"\nGT 클래스별 픽셀 수:")
        for c in range(4):
            count = np.sum(labels_np == c)
            percentage = (count / labels_np.size) * 100
            print(f"  클래스 {c}: {count:,} 픽셀 ({percentage:.2f}%)")
        
        # 배경만 예측하는지 확인
        bg_only = (pred_np == 0).all()
        if bg_only:
            print(f"\n⚠️  경고: 모든 예측이 배경(0)입니다!")
            
            fg_in_gt = (labels_np > 0).any()
            if fg_in_gt:
                print(f"  GT에는 포그라운드가 있지만, 모델이 모두 배경으로 예측했습니다!")
                return False
    
    print("✓ Argmax 예측 분포 검증 통과")
    return True


def check_train_eval_mode_difference(model, inputs, name="Model"):
    """학습/검증 모드 차이 확인"""
    print(f"\n{'='*60}")
    print(f"{name} - 학습/검증 모드 차이")
    print(f"{'='*60}")
    
    model.train()
    with torch.no_grad():
        logits_train = model(inputs)
    
    model.eval()
    with torch.no_grad():
        logits_eval = model(inputs)
    
    # 차이 계산
    diff = torch.abs(logits_train - logits_eval)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"최대 차이: {max_diff:.6f}")
    print(f"평균 차이: {mean_diff:.6f}")
    
    if max_diff > 0.1:
        print(f"⚠️  학습/검증 모드 간 차이가 큽니다 (최대 {max_diff:.6f})")
        print(f"  Dropout이나 BatchNorm이 활성화되어 있을 수 있습니다.")
    
    # 클래스별 예측 비교
    pred_train = torch.argmax(logits_train, dim=1)
    pred_eval = torch.argmax(logits_eval, dim=1)
    agreement = (pred_train == pred_eval).float().mean().item()
    print(f"예측 일치율: {agreement*100:.2f}%")
    
    return True


def test_model_forward(model, inputs, labels, model_name="Model"):
    """모델 forward pass 종합 검증"""
    print(f"\n{'='*60}")
    print(f"{model_name} - Forward Pass 종합 검증")
    print(f"{'='*60}")
    
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    
    # 1. Shape 검증
    expected_shape = (inputs.shape[0], 4, *inputs.shape[2:])
    if not check_output_shape(logits, expected_shape, f"{model_name} Output"):
        return False
    
    # 2. 값 범위 검증
    if not check_logits_values(logits, f"{model_name} Logits"):
        return False
    
    # 3. Softmax 분포 검증
    if not check_softmax_distribution(logits, f"{model_name} Probs"):
        return False
    
    # 4. Argmax 예측 검증
    if not check_argmax_prediction(logits, labels, f"{model_name} Prediction"):
        return False
    
    # 5. 학습/검증 모드 차이 확인
    check_train_eval_mode_difference(model, inputs, model_name)
    
    return True


def main():
    """메인 검증 함수"""
    print("="*60)
    print("모델 출력 종합 검증")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    
    # 데이터 로더 준비
    data_dir = 'data'
    train_loader, val_loader, _, _, _, _ = get_data_loaders(
        data_dir, batch_size=1, num_workers=0, max_samples=5, dim='3d'
    )
    
    # 테스트용 입력 데이터
    print("\n[1단계] 테스트 데이터 준비")
    inputs, labels = next(iter(train_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    print(f"입력 shape: {inputs.shape}")
    print(f"라벨 shape: {labels.shape}")
    print(f"입력 범위: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"라벨 범위: [{labels.min()}, {labels.max()}]")
    
    # 여러 모델 테스트
    models_to_test = ['mobile_unetr_3d', 'unet3d']
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"[2단계] {model_name} 모델 검증")
        print(f"{'='*60}")
        
        try:
            # 모델 생성
            model = get_model(model_name, n_channels=2, n_classes=4, dim='3d', use_pretrained=False)
            model = model.to(device)
            
            # Forward pass 검증
            if not test_model_forward(model, inputs, labels, model_name):
                print(f"❌ {model_name} 모델 검증 실패")
                continue
            
            # 여러 배치 테스트
            print(f"\n[3단계] {model_name} - 여러 배치 테스트")
            model.eval()
            with torch.no_grad():
                for i, (batch_inputs, batch_labels) in enumerate(train_loader):
                    if i >= 3:
                        break
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    logits = model(batch_inputs)
                    print(f"\n배치 {i+1}:")
                    print(f"  입력 shape: {batch_inputs.shape}")
                    print(f"  출력 shape: {logits.shape}")
                    check_argmax_prediction(logits, batch_labels, f"Batch {i+1}")
            
        except Exception as e:
            print(f"❌ {model_name} 모델 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("✅ 모델 출력 검증 완료")
    print("="*60)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Dice 계산 함수 검증 스크립트

검증 항목:
1. calculate_dice_score 함수가 올바른 num_classes를 사용하는지 확인
2. 클래스별 Dice 점수가 올바르게 계산되는지 수동 검증
3. 예측이 모두 0일 때와 실제 포그라운드가 있을 때의 Dice 계산 비교
4. 배경 제외 평균(dice_scores[1:].mean()) 계산 검증
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import argparse

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.metrics import calculate_dice_score
import torch.nn.functional as F


def manual_dice_calculation(pred, target, num_classes, smooth=1e-5):
    """수동 Dice 계산 (검증용)"""
    dice_scores = []
    
    for c in range(num_classes):
        pred_class = (pred == c)
        target_class = (target == c)
        
        intersection = (pred_class & target_class).sum().float()
        union = pred_class.sum().float() + target_class.sum().float()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    
    return np.array(dice_scores)


def check_num_classes_calculation(logits, labels):
    """num_classes 계산 로직 검증"""
    print(f"\n{'='*60}")
    print("num_classes 계산 로직 검증")
    print(f"{'='*60}")
    
    pred = torch.argmax(logits, dim=1)
    
    # 현재 구현의 num_classes 계산
    pred_max = pred.max().item()
    target_max = labels.max().item()
    num_classes_impl = max(pred_max + 1, target_max + 1)
    
    print(f"예측 최대값: {pred_max}")
    print(f"타겟 최대값: {target_max}")
    print(f"계산된 num_classes: {num_classes_impl}")
    
    # 실제 필요한 클래스 수는 4 (0, 1, 2, 3)
    if num_classes_impl < 4:
        print(f"⚠️  경고: num_classes가 4보다 작습니다!")
        print(f"  예측이 모두 0이면 num_classes=1이 될 수 있습니다.")
        return False
    
    if num_classes_impl > 4:
        print(f"⚠️  경고: num_classes가 4보다 큽니다!")
        print(f"  예측이나 타겟에 4 이상의 값이 있을 수 있습니다.")
    
    # 예측과 타겟에 실제로 나타나는 클래스 확인
    pred_unique = torch.unique(pred).tolist()
    target_unique = torch.unique(labels).tolist()
    all_unique = sorted(set(pred_unique + target_unique))
    
    print(f"예측에 나타나는 클래스: {pred_unique}")
    print(f"타겟에 나타나는 클래스: {target_unique}")
    print(f"모든 클래스: {all_unique}")
    
    return True


def test_dice_calculation_simple():
    """간단한 케이스로 Dice 계산 검증"""
    print(f"\n{'='*60}")
    print("간단한 케이스 Dice 계산 검증")
    print(f"{'='*60}")
    
    # 테스트 케이스 1: 완벽한 예측
    print("\n[테스트 1] 완벽한 예측")
    target = torch.zeros((1, 5, 5), dtype=torch.long)
    target[0, 1:4, 1:4] = 1  # 클래스 1 영역
    
    logits = torch.zeros((1, 4, 5, 5))
    logits[0, 1, 1:4, 1:4] = 10  # 클래스 1로 예측
    logits[0, 0, :, :] = 5  # 나머지는 배경
    
    dice_scores = calculate_dice_score(logits, target)
    print(f"Dice 점수: {dice_scores.tolist()}")
    print(f"클래스 0 (배경): {dice_scores[0]:.4f} (예상: ~1.0)")
    print(f"클래스 1: {dice_scores[1]:.4f} (예상: ~1.0)")
    
    # 수동 계산과 비교
    pred = torch.argmax(logits, dim=1)
    manual_dice = manual_dice_calculation(pred[0], target[0], num_classes=4)
    print(f"수동 계산: {manual_dice}")
    
    if not np.allclose(dice_scores.numpy(), manual_dice, atol=1e-4):
        print("⚠️  경고: 함수 계산과 수동 계산이 일치하지 않습니다!")
        return False
    
    # 테스트 케이스 2: 모든 예측이 배경
    print("\n[테스트 2] 모든 예측이 배경")
    target = torch.zeros((1, 5, 5), dtype=torch.long)
    target[0, 1:4, 1:4] = 1  # GT에는 클래스 1이 있음
    
    logits = torch.zeros((1, 4, 5, 5))
    logits[0, 0, :, :] = 10  # 모두 배경으로 예측
    
    dice_scores = calculate_dice_score(logits, target)
    print(f"Dice 점수: {dice_scores.tolist()}")
    print(f"클래스 0 (배경): {dice_scores[0]:.4f}")
    print(f"클래스 1: {dice_scores[1]:.4f} (예상: 0.0 또는 매우 작은 값)")
    
    pred = torch.argmax(logits, dim=1)
    manual_dice = manual_dice_calculation(pred[0], target[0], num_classes=4)
    print(f"수동 계산: {manual_dice}")
    
    if not np.allclose(dice_scores.numpy(), manual_dice, atol=1e-4):
        print("⚠️  경고: 함수 계산과 수동 계산이 일치하지 않습니다!")
        return False
    
    # 배경 제외 평균 확인
    mean_dice_fg = dice_scores[1:].mean()
    print(f"배경 제외 평균 Dice: {mean_dice_fg:.4f}")
    if mean_dice_fg > 0.01:
        print("⚠️  경고: 포그라운드가 예측되지 않았는데 Dice가 0이 아닙니다!")
        return False
    
    # 테스트 케이스 3: 부분적 정확도
    print("\n[테스트 3] 부분적 정확도")
    target = torch.zeros((1, 10, 10), dtype=torch.long)
    target[0, 2:8, 2:8] = 1  # 6x6 클래스 1 영역
    
    logits = torch.zeros((1, 4, 10, 10))
    logits[0, 1, 3:7, 3:7] = 10  # 4x4 클래스 1로 예측 (교집합: 4x4)
    logits[0, 0, :, :] = 5
    
    dice_scores = calculate_dice_score(logits, target)
    print(f"Dice 점수: {dice_scores.tolist()}")
    
    pred = torch.argmax(logits, dim=1)
    manual_dice = manual_dice_calculation(pred[0], target[0], num_classes=4)
    print(f"수동 계산: {manual_dice}")
    
    # 교집합: 4x4=16, 합집합: 4x4 + 6x6 = 16+36=52
    # Dice = 2*16 / (16+36) = 32/52 ≈ 0.615
    expected_dice_1 = 2.0 * 16 / (16 + 36)
    print(f"예상 클래스 1 Dice: {expected_dice_1:.4f}")
    print(f"실제 클래스 1 Dice: {dice_scores[1]:.4f}")
    
    if abs(dice_scores[1].item() - expected_dice_1) > 0.01:
        print("⚠️  경고: 예상 Dice와 실제 Dice가 크게 다릅니다!")
        return False
    
    print("✓ 간단한 케이스 검증 통과")
    return True


def test_dice_with_real_data(data_dir=None):
    """실제 데이터로 Dice 계산 검증"""
    print(f"\n{'='*60}")
    print("실제 데이터 Dice 계산 검증")
    print(f"{'='*60}")
    
    # 데이터 로더 준비
    from data_loader import get_data_loaders
    
    if data_dir is None:
        data_dir = os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021')
    train_loader, val_loader, _, _, _, _ = get_data_loaders(
        data_dir, batch_size=1, num_workers=0, max_samples=3, dim='3d'
    )
    
    # 모델 준비 (더미 모델)
    from integrated_experiment import get_model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model('mobile_unetr_3d', n_channels=2, n_classes=4, dim='3d', use_pretrained=False)
    model = model.to(device)
    model.eval()
    
    # 학습 데이터 테스트
    print("\n[학습 데이터 테스트]")
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= 2:
            break
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            logits = model(inputs)
        
        print(f"\n배치 {i+1}:")
        print(f"  입력 shape: {inputs.shape}")
        print(f"  라벨 shape: {labels.shape}")
        print(f"  출력 shape: {logits.shape}")
        
        # Dice 계산
        dice_scores = calculate_dice_score(logits, labels)
        print(f"  Dice 점수: {dice_scores.tolist()}")
        print(f"  클래스별 Dice:")
        for c in range(4):
            print(f"    클래스 {c}: {dice_scores[c]:.4f}")
        
        # 배경 제외 평균
        mean_dice_fg = dice_scores[1:].mean()
        print(f"  배경 제외 평균 Dice: {mean_dice_fg:.4f}")
        
        # num_classes 검증
        check_num_classes_calculation(logits, labels)
        
        # 예측 분포 확인
        pred = torch.argmax(logits, dim=1)
        pred_np = pred[0].cpu().numpy()
        labels_np = labels[0].cpu().numpy()
        
        print(f"  예측 클래스: {np.unique(pred_np).tolist()}")
        print(f"  GT 클래스: {np.unique(labels_np).tolist()}")
        
        for c in range(4):
            pred_count = np.sum(pred_np == c)
            gt_count = np.sum(labels_np == c)
            print(f"    클래스 {c}: 예측={pred_count:,}, GT={gt_count:,}")
    
    # 검증 데이터 테스트
    print("\n[검증 데이터 테스트]")
    for i, (inputs, labels) in enumerate(val_loader):
        if i >= 2:
            break
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            logits = model(inputs)
        
        print(f"\n배치 {i+1}:")
        dice_scores = calculate_dice_score(logits, labels)
        print(f"  Dice 점수: {dice_scores.tolist()}")
        
        mean_dice_fg = dice_scores[1:].mean()
        print(f"  배경 제외 평균 Dice: {mean_dice_fg:.4f}")
        
        # 예측이 모두 배경인지 확인
        pred = torch.argmax(logits, dim=1)
        pred_np = pred[0].cpu().numpy()
        labels_np = labels[0].cpu().numpy()
        
        pred_unique = np.unique(pred_np).tolist()
        gt_unique = np.unique(labels_np).tolist()
        
        print(f"  예측 클래스: {pred_unique}")
        print(f"  GT 클래스: {gt_unique}")
        
        if pred_unique == [0] and 0 not in gt_unique or (len(gt_unique) > 1):
            print(f"  ⚠️  경고: 예측이 모두 배경인데 GT에는 포그라운드가 있습니다!")
            print(f"    이 경우 배경 제외 평균 Dice는 0에 가까워야 합니다.")
            if mean_dice_fg > 0.01:
                print(f"    하지만 평균 Dice가 {mean_dice_fg:.4f}로 계산되었습니다.")
    
    return True


def test_background_exclusion():
    """배경 제외 평균 계산 검증"""
    print(f"\n{'='*60}")
    print("배경 제외 평균 계산 검증")
    print(f"{'='*60}")
    
    # 시나리오: 배경 Dice는 높지만 포그라운드 Dice는 낮음
    target = torch.zeros((1, 10, 10), dtype=torch.long)
    target[0, 2:8, 2:8] = 1  # GT에 클래스 1이 있음
    
    logits = torch.zeros((1, 4, 10, 10))
    logits[0, 0, :, :] = 10  # 모두 배경으로 예측
    
    dice_scores = calculate_dice_score(logits, target)
    print(f"전체 Dice 점수: {dice_scores.tolist()}")
    
    # 배경 Dice
    dice_bg = dice_scores[0].item()
    print(f"배경 Dice: {dice_bg:.4f}")
    
    # 포그라운드 Dice들
    dice_fg = dice_scores[1:].tolist()
    print(f"포그라운드 Dice: {dice_fg}")
    
    # 배경 제외 평균
    mean_dice_fg = dice_scores[1:].mean().item()
    print(f"배경 제외 평균 Dice: {mean_dice_fg:.4f}")
    
    # 배경 Dice는 높을 수 있지만, 포그라운드가 예측되지 않으면
    # 배경 제외 평균은 0에 가까워야 함
    if dice_bg > 0.9 and mean_dice_fg < 0.1:
        print("✓ 예상대로 동작: 배경은 정확하지만 포그라운드는 예측되지 않음")
        print("  배경 제외 평균이 낮게 나오는 것이 올바릅니다.")
    else:
        print("⚠️  경고: 예상과 다른 결과")
    
    return True


def main():
    """메인 검증 함수"""
    parser = argparse.ArgumentParser(description='Dice 계산 검증')
    parser.add_argument('--data_dir', type=str, 
                        default=os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021'),
                        help='BraTS 데이터셋 루트 디렉토리 (기본값: 환경변수 BRATS_DATA_DIR 또는 /home/work/3D_/BT/BRATS2021)')
    args = parser.parse_args()
    
    print("="*60)
    print("Dice 계산 함수 종합 검증")
    print("="*60)
    print(f"데이터 디렉토리: {args.data_dir}")
    
    # 1. 간단한 케이스 검증
    print("\n[1단계] 간단한 케이스 검증")
    if not test_dice_calculation_simple():
        print("❌ 간단한 케이스 검증 실패")
        return
    
    # 2. 배경 제외 평균 검증
    print("\n[2단계] 배경 제외 평균 계산 검증")
    test_background_exclusion()
    
    # 3. 실제 데이터 검증
    print("\n[3단계] 실제 데이터 검증")
    try:
        test_dice_with_real_data(args.data_dir)
    except Exception as e:
        print(f"⚠️  실제 데이터 검증 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Dice 계산 함수 검증 완료")
    print("="*60)


if __name__ == "__main__":
    main()


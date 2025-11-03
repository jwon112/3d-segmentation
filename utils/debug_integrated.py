#!/usr/bin/env python3
"""
전체 파이프라인 통합 테스트 스크립트

검증 항목:
1. 단일 샘플에 대해 학습과 검증을 모두 실행하여 각 단계별 출력 확인
2. 작은 데이터셋(1-2개 샘플)으로 전체 파이프라인 테스트
3. 고정된 seed로 재현 가능한 결과 확인
4. 학습과 검증에서 예측 분포 차이 확인
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import random
import argparse

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import get_data_loaders
from integrated_experiment import get_model, train_model
from metrics.metrics import calculate_dice_score
from losses.losses import combined_loss
from integrated_experiment import sliding_window_inference_3d
import torch.nn.functional as F


def set_seed(seed=42):
    """재현성을 위한 seed 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_single_sample_pipeline(data_dir=None, model_name='mobile_unetr_3d'):
    """단일 샘플로 전체 파이프라인 테스트"""
    print(f"\n{'='*60}")
    print("단일 샘플 파이프라인 테스트")
    print(f"{'='*60}")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    
    # 데이터 로더 준비 (최소한의 샘플)
    if data_dir is None:
        data_dir = os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021')
    train_loader, val_loader, test_loader, train_sampler, _, _ = get_data_loaders(
        data_dir, batch_size=1, num_workers=0, max_samples=3, dim='3d'
    )
    
    print(f"학습 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")
    print(f"테스트 배치 수: {len(test_loader)}")
    
    # 모델 준비
    model = get_model(model_name, n_channels=2, n_classes=4, dim='3d', use_pretrained=False)
    model = model.to(device)
    
    print(f"\n모델: {model_name}")
    
    # 1. 학습 전 상태 확인
    print(f"\n[1단계] 학습 전 모델 출력 검증")
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(val_loader))
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        print(f"검증 입력 shape: {inputs.shape}")
        print(f"검증 라벨 shape: {labels.shape}")
        
        # 슬라이딩 윈도우 추론
        logits = sliding_window_inference_3d(
            model, inputs, patch_size=(128, 128, 128), overlap=0.1, device=device, model_name=model_name
        )
        
        print(f"출력 logits shape: {logits.shape}")
        
        # Dice 계산
        dice_scores = calculate_dice_score(logits, labels)
        print(f"클래스별 Dice: {dice_scores.tolist()}")
        mean_dice_fg = dice_scores[1:].mean()
        print(f"배경 제외 평균 Dice: {mean_dice_fg:.4f}")
        
        # 예측 분포
        pred = torch.argmax(logits, dim=1)
        pred_unique = torch.unique(pred).tolist()
        labels_unique = torch.unique(labels).tolist()
        print(f"예측 클래스: {pred_unique}")
        print(f"GT 클래스: {labels_unique}")
        
        pred_counts = [(pred == c).sum().item() for c in range(4)]
        gt_counts = [(labels == c).sum().item() for c in range(4)]
        print(f"예측 픽셀 수: {pred_counts}")
        print(f"GT 픽셀 수: {gt_counts}")
    
    # 2. 학습 스텝 확인
    print(f"\n[2단계] 학습 스텝 검증")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = combined_loss
    
    for step, (train_inputs, train_labels) in enumerate(train_loader):
        if step >= 3:  # 3스텝만
            break
        
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)
        
        print(f"\n학습 스텝 {step+1}:")
        print(f"  입력 shape: {train_inputs.shape}")
        print(f"  라벨 shape: {train_labels.shape}")
        
        optimizer.zero_grad()
        logits = model(train_inputs)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        print(f"  Loss: {loss.item():.4f}")
        
        # Dice 계산
        dice_scores = calculate_dice_score(logits.detach(), train_labels)
        mean_dice_fg = dice_scores[1:].mean()
        print(f"  배경 제외 평균 Dice: {mean_dice_fg:.4f}")
        
        # 예측 분포
        pred = torch.argmax(logits, dim=1)
        pred_unique = torch.unique(pred).tolist()
        print(f"  예측 클래스: {pred_unique}")
        
        pred_counts = [(pred == c).sum().item() for c in range(4)]
        train_gt_counts = [(train_labels == c).sum().item() for c in range(4)]
        print(f"  예측 픽셀 수: {pred_counts}")
        print(f"  GT 픽셀 수: {train_gt_counts}")
    
    # 3. 학습 후 검증 상태 확인
    print(f"\n[3단계] 학습 후 검증 출력 검증")
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(val_loader))
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        logits = sliding_window_inference_3d(
            model, inputs, patch_size=(128, 128, 128), overlap=0.1, device=device, model_name=model_name
        )
        
        dice_scores = calculate_dice_score(logits, labels)
        print(f"클래스별 Dice: {dice_scores.tolist()}")
        mean_dice_fg = dice_scores[1:].mean()
        print(f"배경 제외 평균 Dice: {mean_dice_fg:.4f}")
        
        # 예측 분포
        pred = torch.argmax(logits, dim=1)
        pred_unique = torch.unique(pred).tolist()
        labels_unique = torch.unique(labels).tolist()
        print(f"예측 클래스: {pred_unique}")
        print(f"GT 클래스: {labels_unique}")
        
        pred_counts = [(pred == c).sum().item() for c in range(4)]
        gt_counts = [(labels == c).sum().item() for c in range(4)]
        print(f"예측 픽셀 수: {pred_counts}")
        print(f"GT 픽셀 수: {gt_counts}")
        
        # 문제 진단
        if pred_unique == [0] and len(labels_unique) > 1:
            print(f"\n⚠️  문제 발견: 예측이 모두 배경인데 GT에는 포그라운드가 있습니다!")
            print(f"  이는 모델이 배경으로 붕괴(collapse)했음을 의미합니다.")
            
            # 학습 데이터와 검증 데이터의 분포 비교
            print(f"\n  진단: 학습/검증 분포 차이 확인")
            train_inputs_sample, train_labels_sample = next(iter(train_loader))
            train_labels_sample = train_labels_sample.to(device)
            
            train_fg_ratio = (train_labels_sample > 0).float().mean().item()
            val_fg_ratio = (labels > 0).float().mean().item()
            
            print(f"    학습 데이터 포그라운드 비율: {train_fg_ratio:.4f}")
            print(f"    검증 데이터 포그라운드 비율: {val_fg_ratio:.4f}")
            
            if train_fg_ratio > 0.5 and val_fg_ratio < 0.1:
                print(f"    → 학습 데이터는 포그라운드가 많지만, 검증 데이터는 배경이 지배적입니다.")
                print(f"    → 모델이 포그라운드에만 학습되어 배경이 많은 검증 데이터에서 실패했습니다.")
    
    # 4. 추가 검증: 슬라이딩 윈도우 vs 직접 forward, BatchNorm 동작 확인
    print(f"\n[4단계] 상세 검증: 슬라이딩 윈도우 vs 직접 forward, BatchNorm 동작")
    
    # 4-1. 슬라이딩 윈도우 vs 직접 forward 비교 (패치 단위)
    print(f"\n[4-1] 슬라이딩 윈도우 vs 직접 forward 비교")
    with torch.no_grad():
        # 검증 데이터에서 하나의 패치 추출 (128x128x128)
        val_inputs, val_labels = next(iter(val_loader))
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        
        # 패치 크기로 크롭 (처음 128x128x128 영역)
        patch_inputs = val_inputs[:, :, :128, :128, :128]
        patch_labels = val_labels[:, :128, :128, :128]
        
        # 직접 forward (eval 모드)
        model.eval()
        direct_logits = model(patch_inputs)
        direct_pred = torch.argmax(direct_logits, dim=1)
        direct_pred_counts = [(direct_pred == c).sum().item() for c in range(4)]
        direct_dice = calculate_dice_score(direct_logits, patch_labels)
        direct_mean_dice_fg = direct_dice[1:].mean()
        
        print(f"  직접 forward (패치 단위):")
        print(f"    예측 클래스: {torch.unique(direct_pred).tolist()}")
        print(f"    예측 픽셀 수: {direct_pred_counts}")
        print(f"    배경 제외 평균 Dice: {direct_mean_dice_fg:.4f}")
        
        # 슬라이딩 윈도우 (동일한 패치에 대해)
        sw_logits = sliding_window_inference_3d(
            model, patch_inputs, patch_size=(128, 128, 128), overlap=0.1, device=device, model_name=model_name
        )
        sw_pred = torch.argmax(sw_logits, dim=1)
        sw_pred_counts = [(sw_pred == c).sum().item() for c in range(4)]
        sw_dice = calculate_dice_score(sw_logits, patch_labels)
        sw_mean_dice_fg = sw_dice[1:].mean()
        
        print(f"  슬라이딩 윈도우 (동일 패치):")
        print(f"    예측 클래스: {torch.unique(sw_pred).tolist()}")
        print(f"    예측 픽셀 수: {sw_pred_counts}")
        print(f"    배경 제외 평균 Dice: {sw_mean_dice_fg:.4f}")
        
        # 차이 분석
        logits_diff = torch.abs(direct_logits - sw_logits).mean().item()
        pred_agreement = (direct_pred == sw_pred).float().mean().item() * 100
        print(f"  차이 분석:")
        print(f"    Logits 평균 절대 차이: {logits_diff:.6f}")
        print(f"    예측 일치율: {pred_agreement:.2f}%")
        
        if logits_diff > 0.1:
            print(f"    ⚠️  슬라이딩 윈도우와 직접 forward 간 차이가 큽니다!")
    
    # 4-2. BatchNorm train/eval 모드 차이 확인
    print(f"\n[4-2] BatchNorm train/eval 모드 차이 확인")
    with torch.no_grad():
        test_inputs = patch_inputs  # 위에서 사용한 패치 재사용
        
        # Train 모드
        model.train()
        train_logits = model(test_inputs)
        train_pred = torch.argmax(train_logits, dim=1)
        train_pred_counts = [(train_pred == c).sum().item() for c in range(4)]
        train_dice = calculate_dice_score(train_logits, patch_labels)
        train_mean_dice_fg = train_dice[1:].mean()
        
        # Eval 모드
        model.eval()
        eval_logits = model(test_inputs)
        eval_pred = torch.argmax(eval_logits, dim=1)
        eval_pred_counts = [(eval_pred == c).sum().item() for c in range(4)]
        eval_dice = calculate_dice_score(eval_logits, patch_labels)
        eval_mean_dice_fg = eval_dice[1:].mean()
        
        print(f"  Train 모드:")
        print(f"    예측 클래스: {torch.unique(train_pred).tolist()}")
        print(f"    예측 픽셀 수: {train_pred_counts}")
        print(f"    배경 제외 평균 Dice: {train_mean_dice_fg:.4f}")
        
        print(f"  Eval 모드:")
        print(f"    예측 클래스: {torch.unique(eval_pred).tolist()}")
        print(f"    예측 픽셀 수: {eval_pred_counts}")
        print(f"    배경 제외 평균 Dice: {eval_mean_dice_fg:.4f}")
        
        # 차이 분석
        mode_logits_diff = torch.abs(train_logits - eval_logits)
        max_diff = mode_logits_diff.max().item()
        mean_diff = mode_logits_diff.mean().item()
        mode_pred_agreement = (train_pred == eval_pred).float().mean().item() * 100
        
        print(f"  차이 분석:")
        print(f"    Logits 최대 차이: {max_diff:.6f}")
        print(f"    Logits 평균 차이: {mean_diff:.6f}")
        print(f"    예측 일치율: {mode_pred_agreement:.2f}%")
        
        if max_diff > 1.0:
            print(f"    ⚠️  Train/Eval 모드 간 차이가 매우 큽니다! (최대 {max_diff:.2f})")
            print(f"    → BatchNorm의 running stats가 문제일 수 있습니다.")
        elif mode_pred_agreement < 50:
            print(f"    ⚠️  Train/Eval 모드 예측 일치율이 낮습니다! ({mode_pred_agreement:.2f}%)")
    
    # 4-3. BatchNorm running stats 확인
    print(f"\n[4-3] BatchNorm running stats 상태 확인")
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            bn_layers.append((name, module))
    
    if bn_layers:
        print(f"  발견된 BatchNorm 레이어 수: {len(bn_layers)}")
        # 처음 3개만 출력 (너무 많을 수 있음)
        for i, (name, bn) in enumerate(bn_layers[:3]):
            running_mean = bn.running_mean.detach().cpu().numpy()
            running_var = bn.running_var.detach().cpu().numpy()
            print(f"  {name}:")
            print(f"    running_mean 범위: [{running_mean.min():.4f}, {running_mean.max():.4f}]")
            print(f"    running_var 범위: [{running_var.min():.4f}, {running_var.max():.4f}]")
            if running_var.max() < 1e-5:
                print(f"    ⚠️  running_var가 매우 작습니다! (최대 {running_var.max():.6e})")
    else:
        print(f"  BatchNorm 레이어를 찾을 수 없습니다.")
    
    return True


def test_full_training_loop(data_dir=None, model_name='mobile_unetr_3d'):
    """전체 학습 루프 테스트 (1 epoch만)"""
    print(f"\n{'='*60}")
    print("전체 학습 루프 테스트 (1 Epoch)")
    print(f"{'='*60}")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터 로더 준비
    if data_dir is None:
        data_dir = os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021')
    train_loader, val_loader, test_loader, train_sampler, _, _ = get_data_loaders(
        data_dir, batch_size=1, num_workers=0, max_samples=5, dim='3d'
    )
    
    # 모델 준비
    model = get_model(model_name, n_channels=2, n_classes=4, dim='3d', use_pretrained=False)
    model = model.to(device)
    
    print(f"모델: {model_name}")
    print(f"학습 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")
    
    # 1 epoch 학습
    try:
        train_losses, val_dices, epoch_results, best_epoch, best_val_dice = train_model(
            model, train_loader, val_loader, test_loader,
            epochs=1, device=device, model_name=model_name, seed=42,
            train_sampler=train_sampler, rank=0,
            sw_patch_size=(128, 128, 128), sw_overlap=0.1, dim='3d'
        )
        
        print(f"\n학습 결과:")
        print(f"  학습 Loss: {train_losses[-1]:.4f}")
        print(f"  검증 Dice: {val_dices[-1]:.4f}")
        print(f"  최고 검증 Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
        
        # 결과 분석
        if val_dices[-1] < 0.01:
            print(f"\n⚠️  경고: 검증 Dice가 매우 낮습니다 ({val_dices[-1]:.4f})")
            print(f"  모델이 배경으로 붕괴했을 가능성이 높습니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 학습 루프 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_val_distribution_difference(data_dir=None):
    """학습/검증 분포 차이 분석"""
    print(f"\n{'='*60}")
    print("학습/검증 분포 차이 분석")
    print(f"{'='*60}")
    
    if data_dir is None:
        data_dir = os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021')
    train_loader, val_loader, _, _, _, _ = get_data_loaders(
        data_dir, batch_size=1, num_workers=0, max_samples=10, dim='3d'
    )
    
    # 학습 데이터 분포 수집
    train_fg_ratios = []
    train_class_distributions = []
    
    print("학습 데이터 분포 수집 중...")
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= 20:
            break
        labels_np = labels[0].numpy()
        fg_ratio = (labels_np > 0).sum() / labels_np.size
        train_fg_ratios.append(fg_ratio)
        
        class_counts = [np.sum(labels_np == c) for c in range(4)]
        train_class_distributions.append(class_counts)
    
    # 검증 데이터 분포 수집
    val_fg_ratios = []
    val_class_distributions = []
    
    print("검증 데이터 분포 수집 중...")
    for i, (inputs, labels) in enumerate(val_loader):
        if i >= 10:
            break
        labels_np = labels[0].numpy()
        fg_ratio = (labels_np > 0).sum() / labels_np.size
        val_fg_ratios.append(fg_ratio)
        
        class_counts = [np.sum(labels_np == c) for c in range(4)]
        val_class_distributions.append(class_counts)
    
    # 통계 출력
    print(f"\n학습 데이터 통계:")
    print(f"  샘플 수: {len(train_fg_ratios)}")
    print(f"  평균 포그라운드 비율: {np.mean(train_fg_ratios):.4f} ({np.mean(train_fg_ratios)*100:.2f}%)")
    print(f"  최소 포그라운드 비율: {np.min(train_fg_ratios):.4f}")
    print(f"  최대 포그라운드 비율: {np.max(train_fg_ratios):.4f}")
    
    avg_train_class_counts = np.mean(train_class_distributions, axis=0)
    print(f"  평균 클래스별 픽셀 수: {avg_train_class_counts}")
    
    print(f"\n검증 데이터 통계:")
    print(f"  샘플 수: {len(val_fg_ratios)}")
    print(f"  평균 포그라운드 비율: {np.mean(val_fg_ratios):.4f} ({np.mean(val_fg_ratios)*100:.2f}%)")
    print(f"  최소 포그라운드 비율: {np.min(val_fg_ratios):.4f}")
    print(f"  최대 포그라운드 비율: {np.max(val_fg_ratios):.4f}")
    
    avg_val_class_counts = np.mean(val_class_distributions, axis=0)
    print(f"  평균 클래스별 픽셀 수: {avg_val_class_counts}")
    
    # 차이 분석
    fg_diff = abs(np.mean(train_fg_ratios) - np.mean(val_fg_ratios))
    print(f"\n분포 차이:")
    print(f"  포그라운드 비율 차이: {fg_diff:.4f}")
    
    if fg_diff > 0.3:
        print(f"  ⚠️  경고: 학습/검증 포그라운드 비율 차이가 큽니다!")
        print(f"    이는 모델이 학습 데이터 분포에만 과적합되었을 가능성이 있습니다.")
    
    return True


def main():
    """메인 검증 함수"""
    parser = argparse.ArgumentParser(description='전체 파이프라인 통합 테스트')
    parser.add_argument('--data_dir', type=str, 
                        default=os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021'),
                        help='BraTS 데이터셋 루트 디렉토리 (기본값: 환경변수 BRATS_DATA_DIR 또는 /home/work/3D_/BT/BRATS2021)')
    parser.add_argument('--model_name', type=str, default='mobile_unetr_3d',
                        choices=['unet3d', 'unetr', 'swin_unetr', 'mobile_unetr', 'mobile_unetr_3d'],
                        help='테스트할 모델 이름 (기본값: mobile_unetr_3d)')
    args = parser.parse_args()
    
    print("="*60)
    print("전체 파이프라인 통합 테스트")
    print("="*60)
    print(f"데이터 디렉토리: {args.data_dir}")
    print(f"모델: {args.model_name}")
    
    # 1. 단일 샘플 파이프라인 테스트
    print("\n[1단계] 단일 샘플 파이프라인 테스트")
    try:
        test_single_sample_pipeline(args.data_dir, model_name=args.model_name)
    except Exception as e:
        print(f"❌ 단일 샘플 파이프라인 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 학습/검증 분포 차이 분석
    print("\n[2단계] 학습/검증 분포 차이 분석")
    try:
        test_train_val_distribution_difference(args.data_dir)
    except Exception as e:
        print(f"⚠️  분포 차이 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 전체 학습 루프 테스트 (선택적, 시간이 오래 걸림)
    print("\n[3단계] 전체 학습 루프 테스트 (1 Epoch)")
    try:
        test_full_training_loop(args.data_dir, model_name=args.model_name)
    except Exception as e:
        print(f"⚠️  전체 학습 루프 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ 전체 파이프라인 통합 테스트 완료")
    print("="*60)
    print("\n권장 사항:")
    print("1. 각 단계별 디버그 스크립트를 개별 실행하여 상세한 문제 확인")
    print("2. utils/debug_data_loader.py: 데이터 로더 검증")
    print("3. utils/debug_model_output.py: 모델 출력 검증")
    print("4. utils/debug_dice_calculation.py: Dice 계산 검증")
    print("5. utils/debug_sliding_window.py: 슬라이딩 윈도우 검증")


if __name__ == "__main__":
    main()


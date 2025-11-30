"""
Model Training
메인 모델 학습 관련 함수
"""

import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from utils.experiment_utils import (
    set_seed, is_main_process, sliding_window_inference_3d
)
from losses import combined_loss, combined_loss_nnunet_style
from metrics import calculate_wt_tc_et_dice


def _extract_hybrid_stats(model):
    real_model = model.module if hasattr(model, 'module') else model
    if not getattr(real_model, 'log_hybrid_stats', False):
        return None
    if not hasattr(real_model, 'get_hybrid_stats'):
        return None
    stats = real_model.get_hybrid_stats()
    if not stats:
        return None
    return stats


def log_hybrid_stats_epoch(model, epoch: int, rank: int):
    stats = _extract_hybrid_stats(model)
    if not stats or not is_main_process(rank):
        return
    latest_entries = []
    for key in sorted(stats.keys()):
        values = stats[key]
        if not values:
            continue
        latest_entries.append(f"{key}={values[-1]:.6f}")
    if latest_entries:
        print(f"[HybridStats][Epoch {epoch}] " + " | ".join(latest_entries))


def save_hybrid_stats_to_csv(model, results_dir: str, model_name: str, seed: int, rank: int):
    stats = _extract_hybrid_stats(model)
    if not stats or not is_main_process(rank):
        return
    rows = []
    for key, values in stats.items():
        for idx, value in enumerate(values):
            rows.append({
                'key': key,
                'step': idx,
                'value': value
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"hybrid_stats_{model_name}_seed_{seed}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[HybridStats] Saved stats to {csv_path}")


def train_model(model, train_loader, val_loader, test_loader, epochs=10, lr=0.001, device='cuda', model_name='model', seed=24, train_sampler=None, rank: int = 0,
                sw_patch_size=(128, 128, 128), sw_overlap=0.5, dim='3d', use_nnunet_loss=True, results_dir=None, ckpt_path=None, train_crops_per_center=1):
    """모델 훈련 함수
    
    Args:
        use_nnunet_loss: If True, use nnU-Net style loss (Soft Dice with Squared Pred, Dice 70% + CE 30%)
                        If False, use standard combined loss (Dice 50% + CE 50%)
        results_dir: 실험 결과 저장 디렉토리 (체크포인트 저장 경로)
        ckpt_path: 체크포인트 저장 경로 (None이면 자동 생성)
    """
    # 훈련 시작 전 시드 재고정 (완전한 재현성 보장)
    set_seed(seed)
    
    model = model.to(device)
    # nnU-Net style loss: Soft Dice with Squared Prediction, Dice 70% + CE 30%
    # Standard loss: Dice 50% + CE 50%
    criterion = combined_loss_nnunet_style if use_nnunet_loss else combined_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ReduceLROnPlateau: 검증 성능이 개선되지 않을 때 학습률 감소 (nnU-Net 스타일)
    # mode='min': validation loss가 낮을수록 좋음
    # factor=0.5: 학습률을 0.5배로 감소
    # patience=3: 3 epoch 동안 개선 없으면 감소 (nnU-Net 표준)
    # Note: verbose 파라미터는 일부 PyTorch 버전에서 지원되지 않으므로 제거
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_losses = []
    val_dices = []
    epoch_results = []
    
    best_val_dice = 0.0
    best_epoch = 0
    best_val_wt = best_val_tc = best_val_et = 0.0
    
    # 체크포인트 저장 경로 (실험 결과 폴더 내부)
    if results_dir is None:
        results_dir = "baseline_results"
    os.makedirs(results_dir, exist_ok=True)
    if ckpt_path is None:
        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_best.pth")
    
    # BatchNorm Warmup: 초기 running stats를 실제 데이터 분포로 업데이트
    # 검증 모드에서 잘못된 running stats 사용으로 인한 문제 해결
    # Multi-crop 모드에서는 메모리 부족 방지를 위해 warmup 건너뛰기 또는 최소화
    skip_warmup = (train_crops_per_center > 1)  # Multi-crop 모드에서는 warmup 건너뛰기
    if not skip_warmup:
        if is_main_process(rank):
            print("\n[Warmup] Initializing BatchNorm running statistics...")
        model.train()  # train 모드로 설정 (running stats 업데이트됨)
        warmup_batches = 20
        with torch.no_grad():  # gradient 계산 불필요, 메모리 절약
            for i, (inputs, labels) in enumerate(train_loader):
                if i >= warmup_batches:
                    break
                
                # 기존 방식 (단일 crop)
                inputs = inputs.to(device)
                
                # 모델 입력 shape 조정 (일부 모델은 depth 차원 추가 필요)
                if model_name not in ['mobile_unetr', 'mobile_unetr_3d'] and len(inputs.shape) == 4:
                    inputs = inputs.unsqueeze(2)
                
                _ = model(inputs)  # forward만 수행하여 running stats 업데이트
                # 각 forward마다: running_mean = 0.9 * running_mean + 0.1 * batch_mean
                # 점진적으로 실제 데이터 분포로 수렴
        if is_main_process(rank):
            print(f"[Warmup] Processed {warmup_batches} batches. Running stats initialized.\n")
    else:
        if is_main_process(rank):
            print("\n[Warmup] Skipped (multi-crop mode to save memory).\n")
    
    for epoch in range(epochs):
        # Training
        if train_sampler is not None:
            # ensure different shuffles per epoch
            train_sampler.set_epoch(epoch)
        model.train()
        tr_loss = tr_dice_sum = n_tr = 0.0
        
        # 프로파일링: 첫 10 step만 타이밍 측정
        profile_steps = 10
        load_times, fwd_times, bwd_times = [], [], []
        torch.cuda.synchronize()
        
        # 데이터셋/로더 길이 확인 (첫 epoch만)
        if epoch == 0 and is_main_process(rank):
            print(f"\n[Debug] Dataset info:")
            print(f"  Dataset length: {len(train_loader.dataset)}")
            if train_sampler is not None:
                print(f"  Sampler length: {len(train_sampler)}")
            print(f"  Loader length: {len(train_loader)}")
            print(f"  Batch size: {train_loader.batch_size}")
            if train_crops_per_center > 1:
                print(f"  Multi-crop mode: {train_crops_per_center} crops per center ({train_crops_per_center**3} total crops per sample)")
        
        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)):
            if step < profile_steps:
                torch.cuda.synchronize()
                t_start = time.time()
            
            # 각 crop이 별도의 샘플로 취급되므로 일반적인 단일 crop 처리
            inputs, labels = inputs.to(device), labels.to(device)
            
            # MobileUNETR 2D는 2D 입력을 그대로 사용 (depth 차원 추가 안함)
            # mobile_unetr_3d는 3D 입력을 그대로 사용
            # 다른 모델들은 3D 입력 필요 (depth 차원 추가)
            if model_name not in ['mobile_unetr', 'mobile_unetr_3d'] and len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(2)  # Add depth dimension (B, C, H, W) -> (B, C, 1, H, W)
                labels = labels.unsqueeze(2)
            
            if step < profile_steps:
                t_load = time.time()
                load_times.append(t_load - t_start)
            
            optimizer.zero_grad()
            # 학습 단계에서는 슬라이딩 윈도우를 사용하지 않음 (단일 패치 forward)
            logits = model(inputs)
            
            if step < profile_steps:
                torch.cuda.synchronize()
                t_fwd = time.time()
                fwd_times.append(t_fwd - t_load)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            if step < profile_steps:
                torch.cuda.synchronize()
                t_bwd = time.time()
                bwd_times.append(t_bwd - t_fwd)
            
            # BraTS composite Dice (WT, TC, ET)
            dice_scores = calculate_wt_tc_et_dice(logits.detach(), labels)
            # 평균 Dice (WT/TC/ET 평균)
            mean_dice = dice_scores.mean()
            bsz = inputs.size(0)
            tr_loss += loss.item() * bsz
            tr_dice_sum += mean_dice.item() * bsz
            n_tr += bsz
        
        tr_loss /= max(1, n_tr)
        tr_dice = tr_dice_sum / max(1, n_tr)
        train_losses.append(tr_loss)
        
        # 프로파일링 결과 출력 (첫 epoch만)
        if epoch == 0 and n_tr > 0:
            avg_load = np.mean(load_times)
            avg_fwd = np.mean(fwd_times)
            avg_bwd = np.mean(bwd_times)
            if is_main_process(rank):
                print(f"\n[Profile] Avg load: {avg_load:.3f}s, fwd: {avg_fwd:.3f}s, bwd: {avg_bwd:.3f}s, total: {avg_load+avg_fwd+avg_bwd:.3f}s/step")
        
        # Validation (all ranks, simpler/robust)
        model.eval()
        va_loss = va_dice_sum = n_va = 0.0
        va_wt_sum = va_tc_sum = va_et_sum = 0.0
        with torch.no_grad():
            debug_printed = False
            all_sample_dices = []  # 디버깅: 모든 샘플의 Dice 수집
            for idx, (inputs, labels) in enumerate(tqdm(val_loader, desc=f"Val   {epoch+1}/{epochs}", leave=False)):
                inputs, labels = inputs.to(device), labels.to(device)

                # MobileUNETR 2D는 2D 입력을 그대로 사용
                # mobile_unetr_3d는 3D 입력을 그대로 사용
                if model_name not in ['mobile_unetr', 'mobile_unetr_3d'] and len(inputs.shape) == 4:
                    inputs = inputs.unsqueeze(2)
                    labels = labels.unsqueeze(2)

                # 3D 검증: 슬라이딩 윈도우 추론 (학습 아님)
                # 모든 3D 모델은 전체 볼륨을 처리하기 위해 슬라이딩 윈도우 사용
                if dim == '3d' and inputs.dim() == 5 and inputs.size(0) == 1:
                    logits = sliding_window_inference_3d(
                        model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name
                    )
                else:
                    logits = model(inputs)
                loss = criterion(logits, labels)
                dice_scores = calculate_wt_tc_et_dice(logits, labels)
                # WT/TC/ET 평균
                mean_dice = dice_scores.mean()
                all_sample_dices.append(mean_dice.item())  # 디버깅
                
                if not debug_printed:
                    pred_arg = torch.argmax(logits, dim=1)
                    pred_counts = [int((pred_arg == c).sum().item()) for c in range(4)]
                    gt_counts = [int((labels == c).sum().item()) for c in range(4)]
                    if is_main_process(rank):
                        try:
                            dv = dice_scores.detach().cpu().tolist()
                        except Exception:
                            dv = []
                        print(f"Val sample {idx+1} stats | pred counts: {pred_counts} | gt counts: {gt_counts}")
                        print(f"Val sample {idx+1} WT/TC/ET dice: {dice_scores.detach().cpu().tolist()}")
                        print(f"Val sample {idx+1} mean_dice (fg only): {mean_dice.item():.10f}")
                    debug_printed = True
                bsz = inputs.size(0)
                va_loss += loss.item() * bsz
                va_dice_sum += mean_dice.item() * bsz
                va_wt_sum += float(dice_scores[0].item()) * bsz
                va_tc_sum += float(dice_scores[1].item()) * bsz
                va_et_sum += float(dice_scores[2].item()) * bsz
                n_va += bsz
            
            # 디버깅: 모든 샘플의 Dice 통계 출력
            if is_main_process(rank) and len(all_sample_dices) > 0:
                all_dices_arr = np.array(all_sample_dices)
                print(f"\n[Val Epoch {epoch+1}] All samples Dice stats:")
                print(f"  샘플 수: {len(all_sample_dices)}")
                print(f"  평균: {all_dices_arr.mean():.10f}")
                print(f"  최소: {all_dices_arr.min():.10f}")
                print(f"  최대: {all_dices_arr.max():.10f}")
                print(f"  표준편차: {all_dices_arr.std():.10f}")
                print(f"  0.0317과의 차이: {abs(all_dices_arr.mean() - 0.0317):.10f}")
        
        va_loss /= max(1, n_va)
        va_dice = va_dice_sum / max(1, n_va)
        va_wt = va_wt_sum / max(1, n_va)
        va_tc = va_tc_sum / max(1, n_va)
        va_et = va_et_sum / max(1, n_va)
        val_dices.append(va_dice)
        
        # Learning rate scheduling (ReduceLROnPlateau는 validation metric 필요)
        # nnU-Net 스타일: validation loss를 모니터링
        scheduler.step(va_loss)
        
        # Best model tracking 및 체크포인트 저장 (rank 0만)
        # Test set은 최종에만 평가하므로, epoch 중에는 평가하지 않음
        checkpoint_saved = False
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            best_val_wt = va_wt
            best_val_tc = va_tc
            best_val_et = va_et
            best_epoch = epoch + 1
            checkpoint_saved = True
            if is_main_process(rank):
                # DDP 모델의 경우 module을 통해 접근
                model_to_save = model.module if hasattr(model, 'module') else model
                # Clean thop profiling buffers if any
                for m in model_to_save.modules():
                    for bname in ('total_ops', 'total_params'):
                        if hasattr(m, bname):
                            try:
                                delattr(m, bname)
                            except Exception:
                                pass
                        if isinstance(getattr(m, '_buffers', None), dict) and bname in m._buffers:
                            m._buffers.pop(bname, None)
                torch.save(model_to_save.state_dict(), ckpt_path)
                print(f"[Epoch {epoch+1}] Saved best checkpoint (Val Dice: {va_dice:.4f}) to {ckpt_path}")
        
        # Epoch 결과 저장 (test_dice는 최종 평가 시에만 설정됨)
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': tr_loss,
            'train_dice': tr_dice,
            'val_loss': va_loss,
            'val_dice': va_dice,
            'val_wt': va_wt,
            'val_tc': va_tc,
            'val_et': va_et,
            'test_dice': None  # 최종 평가 시에만 설정
        })
        
        if is_main_process(rank):
            checkpoint_msg = " [BEST]" if checkpoint_saved else ""
            print(f"Epoch {epoch+1}/{epochs} | Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} | Val Loss {va_loss:.4f} Dice {va_dice:.4f} (WT {va_wt:.4f} | TC {va_tc:.4f} | ET {va_et:.4f}){checkpoint_msg}")
        log_hybrid_stats_epoch(model, epoch + 1, rank)
    
    save_hybrid_stats_to_csv(model, results_dir, model_name, seed, rank)
    return train_losses, val_dices, epoch_results, best_epoch, best_val_dice, best_val_wt, best_val_tc, best_val_et


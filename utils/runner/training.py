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
from losses import combined_loss, combined_loss_nnunet_style, DeepSupervisionWrapper
from metrics import calculate_wt_tc_et_dice
from utils.lr_scheduler import PolyLRScheduler


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


def train_model(model, train_loader, val_loader, test_loader, epochs=10, lr=0.01, device='cuda', model_name='model', seed=24, train_sampler=None, rank: int = 0,
                sw_patch_size=(128, 128, 128), sw_overlap=0.5, dim='3d', use_nnunet_loss=True, results_dir=None, ckpt_path=None, train_crops_per_center=1, dataset_version='brats2021',
                num_iterations_per_epoch=250, num_val_iterations_per_epoch=50):
    """모델 훈련 함수
    
    Args:
        use_nnunet_loss: If True, use nnU-Net style loss (Soft Dice with Squared Pred, Dice 70% + CE 30%)
                        If False, use standard combined loss (Dice 50% + CE 50%)
        results_dir: 실험 결과 저장 디렉토리 (체크포인트 저장 경로)
                     이미 존재하는 경로면 자동으로 재개, 새로 생성되면 새로 시작
        ckpt_path: 체크포인트 저장 경로 (None이면 자동 생성)
    """
    # 훈련 시작 전 시드 재고정 (완전한 재현성 보장)
    set_seed(seed)
    
    model = model.to(device)
    # nnU-Net style loss: Soft Dice with Squared Prediction, Dice 70% + CE 30%
    # Standard loss: Dice 50% + CE 50%
    base_loss = combined_loss_nnunet_style if use_nnunet_loss else combined_loss
    
    # Deep Supervision 지원 여부 확인 (모델이 리스트를 반환하는지 확인)
    # 첫 번째 forward로 확인 (더미 입력 사용)
    with torch.no_grad():
        dummy_input = torch.randn(1, 4, 32, 32, 32).to(device)
        dummy_output = model(dummy_input)
        use_deep_supervision = isinstance(dummy_output, (list, tuple))
    
    # Deep Supervision이 활성화된 경우 wrapper 사용
    if use_deep_supervision:
        # nnUNet 방식: 동적으로 가중치 계산
        # weights = [1, 0.5, 0.25, 0.125, ...] 형태로 지수적으로 감소
        num_outputs = len(dummy_output)
        import numpy as np
        weights = np.array([1.0 / (2 ** i) for i in range(num_outputs)], dtype=np.float32)
        weights[-1] = 0  # 마지막 weight는 0 (가장 낮은 해상도 출력 제외)
        weights = weights / weights.sum()  # 정규화하여 합이 1이 되도록
        weights = weights.tolist()
        
        criterion = DeepSupervisionWrapper(base_loss, weight_factors=weights)
        if is_main_process(rank):
            print(f"[Deep Supervision] Enabled with {num_outputs} outputs, weights: {weights}")
    else:
        criterion = base_loss
        if is_main_process(rank):
            print(f"[Deep Supervision] Disabled (single output)")
    
    # nnUNet 기본 설정: SGD with momentum=0.99, nesterov=True, weight_decay=3e-5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    
    # Learning rate scheduler: PolyLR (nnUNet 스타일)
    # nnUNet 방식: 고정 iteration per epoch 사용
    max_steps = epochs * num_iterations_per_epoch
    scheduler = PolyLRScheduler(optimizer, initial_lr=lr, max_steps=max_steps, exponent=0.9)
    if is_main_process(rank):
        print(f"[PolyLR] Using Polynomial LR Scheduler (max_steps={max_steps}, exponent=0.9)")
    
    train_losses = []
    val_dices = []
    epoch_results = []
    
    best_val_dice = 0.0
    best_epoch = 0
    best_val_wt = best_val_tc = best_val_et = best_val_rc = 0.0
    epochs_without_improvement = 0  # Early stopping을 위한 카운터
    early_stopping_patience = 50  # 50 epoch 동안 개선 없으면 중단
    is_brats2024 = (dataset_version == 'brats2024')
    
    # Cascade/ROI 기반 학습 로직은 더 이상 사용하지 않음 (nnUNet 스타일 단일 파이프라인만 유지)
    
    # 체크포인트 저장 경로 (실험 결과 폴더 내부)
    if results_dir is None:
        results_dir = "experiment_result"
    os.makedirs(results_dir, exist_ok=True)
    if ckpt_path is None:
        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_best.pth")
    
    # Latest checkpoint 경로 (재개용)
    latest_ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_latest.pth")
    if use_5fold and fold_idx is not None:
        latest_ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_fold_{fold_idx}_latest.pth")
    
    # 재개 로직: latest checkpoint가 있으면 자동으로 재개
    # results_dir가 이미 존재하는 경로로 주어졌다면 재개 모드로 간주
    start_epoch = 0
    if os.path.exists(latest_ckpt_path):
        if is_main_process(rank):
            print(f"[Resume] Found checkpoint: {latest_ckpt_path}")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            
            # DDP 모델 처리: 모든 프로세스에서 로드
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(checkpoint['state_dict'])
            
            # 모든 프로세스에서 optimizer, scheduler 로드
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            start_epoch = checkpoint.get('epoch', 0)
            best_val_dice = checkpoint.get('best_val_dice', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            best_val_wt = checkpoint.get('best_val_wt', 0.0)
            best_val_tc = checkpoint.get('best_val_tc', 0.0)
            best_val_et = checkpoint.get('best_val_et', 0.0)
            if is_brats2024:
                best_val_rc = checkpoint.get('best_val_rc', 0.0)
            # Early stopping 상태 복원 (중요: 재개 시 이어서 작동)
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            
            if is_main_process(rank):
                print(f"[Resume] Resumed from epoch {start_epoch}/{epochs}")
                print(f"[Resume] Best val dice: {best_val_dice:.4f} (epoch {best_epoch})")
                print(f"[Resume] Early stopping: {epochs_without_improvement}/{early_stopping_patience} epochs without improvement")
            
            # DDP 동기화: 모든 프로세스가 체크포인트 로드 완료 대기
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as e:
            if is_main_process(rank):
                print(f"[Resume] Warning: Failed to load checkpoint: {e}")
                print(f"[Resume] Starting from scratch...")
            start_epoch = 0
            epochs_without_improvement = 0  # 재개 실패 시 초기화
    
    # Epoch 범위 체크: 재개 시 epoch가 총 epoch보다 크거나 같으면 새로 시작
    if start_epoch >= epochs:
        if is_main_process(rank):
            print(f"[Resume] Warning: Checkpoint epoch {start_epoch} >= total epochs {epochs}. Starting from scratch.")
        start_epoch = 0
    
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
            for i, batch_data in enumerate(train_loader):
                # 포그라운드 좌표가 포함될 수 있으므로 처리
                if len(batch_data) == 3:
                    inputs, labels, _ = batch_data  # fg_coords_dict 무시
                else:
                    inputs, labels = batch_data
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
    
    for epoch in range(start_epoch, epochs):
        # 각 epoch 시작 시 seed 재설정하여 재현성 보장 (Stochastic depth 등 랜덤 연산 포함)
        # base_seed + epoch을 사용하여 각 epoch마다 다른 seed를 가지지만, 같은 seed로 시작하면 같은 순서로 재현 가능
        epoch_seed = seed + epoch
        set_seed(epoch_seed)
        
        # Training
        if train_sampler is not None:
            # ensure different shuffles per epoch
            train_sampler.set_epoch(epoch)
        model.train()
        tr_loss = tr_dice_sum = n_tr = 0.0
        
        # 프로파일링: 전체 에포크 타이밍 측정
        profile_steps = len(train_loader)  # 전체 step 측정
        wait_times, load_times, fwd_times, bwd_times = [], [], [], []
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
        
        # Iterator를 명시적으로 사용하여 배치 대기 시간 측정
        # nnUNet 방식: 고정 iteration per epoch 사용
        train_iter = iter(train_loader)
        for step in tqdm(range(num_iterations_per_epoch), desc=f"Train {epoch+1}/{epochs}", leave=False):
            # 배치를 받기 전 시간 측정 (대기 시간 포함)
            torch.cuda.synchronize()
            t_wait_start = time.time()
            
            inputs, labels = next(train_iter)
            
            # 배치를 받은 후 시간 측정
            t_wait_end = time.time()
            wait_times.append(t_wait_end - t_wait_start)
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
            
            t_load = time.time()
            load_times.append(t_load - t_start)
            
            optimizer.zero_grad()
            # 학습 단계에서는 슬라이딩 윈도우를 사용하지 않음 (단일 패치 forward)
            logits = model(inputs)
            
            # PolyLR 스케줄러: 매 step마다 호출
            scheduler.step()
            
            torch.cuda.synchronize()
            t_fwd = time.time()
            fwd_times.append(t_fwd - t_load)
            
            # Deep Supervision wrapper가 적용되었지만 실제 출력이 Tensor인 경우 처리
            if isinstance(criterion, DeepSupervisionWrapper) and not isinstance(logits, (list, tuple)):
                # wrapper를 사용하지 않고 base_loss 직접 사용
                loss = base_loss(logits, labels)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            t_bwd = time.time()
            bwd_times.append(t_bwd - t_fwd)
            
            # Deep Supervision이 활성화된 경우 메인 출력만 사용
            if isinstance(logits, (list, tuple)):
                logits_for_dice = logits[0].detach()  # 메인 출력만 사용
            else:
                logits_for_dice = logits.detach()
            
            # BraTS composite Dice (WT, TC, ET, RC for BRATS2024)
            dice_scores = calculate_wt_tc_et_dice(logits_for_dice, labels, dataset_version=dataset_version)
            # 평균 Dice (WT/TC/ET 평균, BRATS2024는 RC 포함)
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
            avg_wait = np.mean(wait_times) if wait_times else 0.0
            avg_load = np.mean(load_times)
            avg_fwd = np.mean(fwd_times)
            avg_bwd = np.mean(bwd_times)
            if is_main_process(rank):
                print(f"\n[Profile] Avg wait: {avg_wait:.3f}s, load: {avg_load:.3f}s, fwd: {avg_fwd:.3f}s, bwd: {avg_bwd:.3f}s")
                print(f"[Profile] Total per step: {avg_wait+avg_load+avg_fwd+avg_bwd:.3f}s (wait+load+fwd+bwd)")
                if wait_times:
                    max_wait = np.max(wait_times)
                    min_wait = np.min(wait_times)
                    print(f"[Profile] Wait time range: {min_wait:.3f}s ~ {max_wait:.3f}s")
                
                # 캐시 통계 출력 (BratsPatchDataset3D인 경우)
                try:
                    dataset = train_loader.dataset
                    if hasattr(dataset, 'get_cache_stats'):
                        cache_stats = dataset.get_cache_stats()
                        print(f"[Profile] Cache: hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
                              f"hit_rate={cache_stats['hit_rate']:.1f}%, "
                              f"size={cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
                except Exception:
                    pass
        
        # Validation (all ranks, simpler/robust)
        model.eval()
        va_loss = va_dice_sum = n_va = 0.0
        va_wt_sum = va_tc_sum = va_et_sum = va_rc_sum = 0.0
        
        # nnUNet 스타일 단일 검증 경로만 사용
        with torch.no_grad():
            debug_printed = False
            all_sample_dices = []  # 디버깅: 모든 샘플의 Dice 수집
            # nnUNet 방식: 고정 iteration per epoch 사용
            val_iter = iter(val_loader)
            for idx in tqdm(range(num_val_iterations_per_epoch), desc=f"Val   {epoch+1}/{epochs}", leave=False):
                try:
                    batch_data = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    batch_data = next(val_iter)
                # 포그라운드 좌표가 포함될 수 있으므로 처리
                if len(batch_data) == 3:
                    inputs, labels, _ = batch_data  # fg_coords_dict 무시
                else:
                    inputs, labels = batch_data
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
                # Deep Supervision wrapper가 적용되었지만 실제 출력이 Tensor인 경우 처리
                if isinstance(criterion, DeepSupervisionWrapper) and not isinstance(logits, (list, tuple)):
                    # wrapper를 사용하지 않고 base_loss 직접 사용
                    loss = base_loss(logits, labels)
                else:
                    loss = criterion(logits, labels)
                # Deep Supervision이 활성화된 경우 메인 출력만 사용
                if isinstance(logits, (list, tuple)):
                    logits_for_dice = logits[0]  # 메인 출력만 사용
                else:
                    logits_for_dice = logits
                dice_scores = calculate_wt_tc_et_dice(logits_for_dice, labels, dataset_version=dataset_version)
                # WT/TC/ET 평균 (BRATS2024는 RC 포함)
                mean_dice = dice_scores.mean()
                all_sample_dices.append(mean_dice.item())  # 디버깅
                
                if not debug_printed:
                    pred_arg = torch.argmax(logits, dim=1)
                    n_classes = 5 if is_brats2024 else 4
                    pred_counts = [int((pred_arg == c).sum().item()) for c in range(n_classes)]
                    gt_counts = [int((labels == c).sum().item()) for c in range(n_classes)]
                    if is_main_process(rank):
                        try:
                            dv = dice_scores.detach().cpu().tolist()
                        except Exception:
                            dv = []
                        dice_str = "WT/TC/ET/RC" if is_brats2024 else "WT/TC/ET"
                        print(f"Val sample {idx+1} stats | pred counts: {pred_counts} | gt counts: {gt_counts}")
                        print(f"Val sample {idx+1} {dice_str} dice: {dice_scores.detach().cpu().tolist()}")
                        print(f"Val sample {idx+1} mean_dice (fg only): {mean_dice.item():.10f}")
                    debug_printed = True
                bsz = inputs.size(0)
                va_loss += loss.item() * bsz
                va_dice_sum += mean_dice.item() * bsz
                va_wt_sum += float(dice_scores[0].item()) * bsz
                va_tc_sum += float(dice_scores[1].item()) * bsz
                va_et_sum += float(dice_scores[2].item()) * bsz
                if is_brats2024 and len(dice_scores) >= 4:
                    va_rc_sum += float(dice_scores[3].item()) * bsz
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
        if is_brats2024:
            va_rc = va_rc_sum / max(1, n_va)
        else:
            va_rc = 0.0
        val_dices.append(va_dice)
        
        # Learning rate scheduling
        # PolyLR은 매 step마다 이미 호출되므로 epoch 끝에서는 호출하지 않음
        
        # Best model tracking 및 체크포인트 저장 (rank 0만)
        # Test set은 최종에만 평가하므로, epoch 중에는 평가하지 않음
        checkpoint_saved = False
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            best_val_wt = va_wt
            best_val_tc = va_tc
            best_val_et = va_et
            if is_brats2024:
                best_val_rc = va_rc
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # 개선됨 - 카운터 리셋
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
                # Best checkpoint: 모델 가중치만 저장 (평가용, 기존 호환성 유지)
                torch.save(model_to_save.state_dict(), ckpt_path)
                print(f"[Epoch {epoch+1}] Saved best checkpoint (Val Dice: {va_dice:.4f}) to {ckpt_path}")
        else:
            epochs_without_improvement += 1  # 개선 없음
        
        # Latest checkpoint: 매 epoch마다 저장 (재개용)
        if is_main_process(rank):
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                'best_val_wt': best_val_wt,
                'best_val_tc': best_val_tc,
                'best_val_et': best_val_et,
                'epochs_without_improvement': epochs_without_improvement,
            }
            if is_brats2024:
                checkpoint['best_val_rc'] = best_val_rc
            torch.save(checkpoint, latest_ckpt_path)
        
        # Early stopping 체크
        if epochs_without_improvement >= early_stopping_patience:
            if is_main_process(rank):
                print(f"\n[Early Stopping] No improvement for {early_stopping_patience} epochs. Stopping training.")
                print(f"Best validation dice: {best_val_dice:.4f} at epoch {best_epoch}")
            break
        
        # Epoch 결과 저장 (test_dice는 최종 평가 시에만 설정됨)
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': tr_loss,
            'train_dice': tr_dice,
            'val_loss': va_loss,
            'val_dice': va_dice,
            'val_wt': va_wt,
            'val_tc': va_tc,
            'val_et': va_et,
            'test_dice': None  # 최종 평가 시에만 설정
        }
        if is_brats2024:
            epoch_result['val_rc'] = va_rc
        epoch_results.append(epoch_result)
        
        if is_main_process(rank):
            checkpoint_msg = " [BEST]" if checkpoint_saved else ""
            print(f"[Training Debug] Epoch {epoch+1} - va_loss before final print: {va_loss}")
            if is_brats2024:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} | Val Loss {va_loss:.4f} Dice {va_dice:.4f} (WT {va_wt:.4f} | TC {va_tc:.4f} | ET {va_et:.4f} | RC {va_rc:.4f}){checkpoint_msg}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} | Val Loss {va_loss:.4f} Dice {va_dice:.4f} (WT {va_wt:.4f} | TC {va_tc:.4f} | ET {va_et:.4f}){checkpoint_msg}")
        log_hybrid_stats_epoch(model, epoch + 1, rank)
    
    save_hybrid_stats_to_csv(model, results_dir, model_name, seed, rank)
    if is_brats2024:
        return train_losses, val_dices, epoch_results, best_epoch, best_val_dice, best_val_wt, best_val_tc, best_val_et, best_val_rc
    else:
        return train_losses, val_dices, epoch_results, best_epoch, best_val_dice, best_val_wt, best_val_tc, best_val_et


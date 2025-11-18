#!/usr/bin/env python3
"""
Experiment Runner
실험 실행 관련 함수들: train_model, evaluate_model, run_integrated_experiment
"""

import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

# Import utilities
from utils.experiment_utils import (
    setup_distributed, cleanup_distributed, is_main_process,
    set_seed, sliding_window_inference_3d, calculate_flops, calculate_pam, calculate_inference_latency, get_model,
    INPUT_SIZE_2D, INPUT_SIZE_3D
)

# Import losses and metrics
from losses import combined_loss, combined_loss_nnunet_style
from metrics import calculate_wt_tc_et_dice

# Import data loader and visualization
from data_loader import get_data_loaders
from visualization import create_comprehensive_analysis, create_interactive_3d_plot

# Import Grad-CAM utilities
from utils.gradcam_utils import generate_gradcam_for_model


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
                sw_patch_size=(128, 128, 128), sw_overlap=0.5, dim='3d', use_nnunet_loss=True, results_dir=None, ckpt_path=None):
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
    if is_main_process(rank):
        print("\n[Warmup] Initializing BatchNorm running statistics...")
    model.train()  # train 모드로 설정 (running stats 업데이트됨)
    warmup_batches = 20
    with torch.no_grad():  # gradient 계산 불필요, 메모리 절약
        for i, (inputs, labels) in enumerate(train_loader):
            if i >= warmup_batches:
                break
            inputs = inputs.to(device)
            
            # 모델 입력 shape 조정 (일부 모델은 depth 차원 추가 필요)
            if model_name not in ['mobile_unetr', 'mobile_unetr_3d'] and len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(2)
            
            _ = model(inputs)  # forward만 수행하여 running stats 업데이트
            # 각 forward마다: running_mean = 0.9 * running_mean + 0.1 * batch_mean
            # 점진적으로 실제 데이터 분포로 수렴
    if is_main_process(rank):
        print(f"[Warmup] Processed {warmup_batches} batches. Running stats initialized.\n")
    
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
        
        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)):
            if step < profile_steps:
                torch.cuda.synchronize()
                t_start = time.time()
            
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
        
        # Learning rate scheduling
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


def evaluate_model(model, test_loader, device='cuda', model_name: str = 'model', distributed: bool = False, world_size: int = 1,
                   sw_patch_size=(128, 128, 128), sw_overlap=0.25, results_dir: str = None):
    """모델 평가 함수"""
    model.eval()
    test_dice = 0.0
    test_wt_sum = test_tc_sum = test_et_sum = 0.0
    n_te = 0
    precision_scores = []
    recall_scores = []
    
    # 모달리티별 어텐션 가중치 수집 (quadbranch_4modal_attention_unet_s만)
    collect_attention = (model_name == 'quadbranch_4modal_attention_unet_s')
    all_attention_weights = []  # 각 샘플별 어텐션 가중치 저장
    
    with torch.no_grad():
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        cm_accum = np.zeros((4, 4), dtype=np.int64)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 2D/3D 분기: 2D 모델은 그대로, 3D 모델은 depth 차원 추가
            if model_name != 'mobile_unetr' and len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(2)
                labels = labels.unsqueeze(2)
            
            # 3D 테스트: 슬라이딩 윈도우 추론
            # 모든 3D 모델은 전체 볼륨을 처리하기 위해 슬라이딩 윈도우 사용
            if inputs.dim() == 5 and inputs.size(0) == 1:  # 3D 볼륨
                if collect_attention:
                    # 어텐션 가중치 수집을 위해 슬라이딩 윈도우에서 직접 호출
                    # 슬라이딩 윈도우는 어텐션 가중치를 평균내야 하므로, 여기서는 전체 볼륨에 대해 직접 호출
                    real_model = model.module if hasattr(model, 'module') else model
                    if hasattr(real_model, 'forward') and 'return_attention' in real_model.forward.__code__.co_varnames:
                        # 전체 볼륨에 대해 직접 forward (슬라이딩 윈도우 없이, 메모리 허용 시)
                        # 실제로는 슬라이딩 윈도우를 사용하되, 각 패치의 어텐션을 평균내야 함
                        # 간단하게 전체 볼륨에 대해 직접 호출 (메모리 허용 시)
                        try:
                            logits, attention_dict = real_model(inputs, return_attention=True)
                            # 어텐션 가중치 저장 (평균 가중치 사용)
                            avg_weights = attention_dict['average'].cpu().numpy()  # [B, 4]
                            all_attention_weights.append(avg_weights[0])  # 첫 번째 샘플 (batch_size=1)
                        except RuntimeError as e:
                            # 메모리 부족 시 슬라이딩 윈도우 사용 (어텐션 수집 불가)
                            if "out of memory" in str(e).lower():
                                print(f"Warning: OOM during attention collection, using sliding window without attention")
                                logits = sliding_window_inference_3d(
                                    model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name
                                )
                            else:
                                raise
                    else:
                        logits = sliding_window_inference_3d(
                            model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name
                        )
                else:
                    logits = sliding_window_inference_3d(
                        model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name
                    )
            else:
                if collect_attention:
                    real_model = model.module if hasattr(model, 'module') else model
                    if hasattr(real_model, 'forward') and 'return_attention' in real_model.forward.__code__.co_varnames:
                        logits, attention_dict = real_model(inputs, return_attention=True)
                        avg_weights = attention_dict['average'].cpu().numpy()  # [B, 4]
                        for i in range(avg_weights.shape[0]):
                            all_attention_weights.append(avg_weights[i])
                    else:
                        logits = model(inputs)
                else:
                    logits = model(inputs)
            
            # Dice score 계산 (WT/TC/ET)
            dice_scores = calculate_wt_tc_et_dice(logits, labels)
            mean_dice = dice_scores.mean()
            bsz = inputs.size(0)
            test_dice += mean_dice.item() * bsz
            test_wt_sum += float(dice_scores[0].item()) * bsz
            test_tc_sum += float(dice_scores[1].item()) * bsz
            test_et_sum += float(dice_scores[2].item()) * bsz
            n_te += bsz
            
            # Precision, Recall 계산 (클래스별)
            pred = torch.argmax(logits, dim=1)
            pred_np = pred.cpu().numpy()
            labels_np = labels.cpu().numpy()
            # Accumulate 4-class confusion matrix (0..3)
            cm_batch = confusion_matrix(labels_np.flatten(), pred_np.flatten(), labels=list(range(4)))
            cm_accum += cm_batch
            
            for class_id in range(4):
                pred_class = (pred_np == class_id)
                true_class = (labels_np == class_id)
                
                if true_class.sum() > 0:
                    precision = (pred_class & true_class).sum() / pred_class.sum() if pred_class.sum() > 0 else 0
                    recall = (pred_class & true_class).sum() / true_class.sum()
                else:
                    precision = recall = 0
                    
                precision_scores.append(precision)
                recall_scores.append(recall)
    
    test_dice /= max(1, n_te)
    test_wt = test_wt_sum / max(1, n_te)
    test_tc = test_tc_sum / max(1, n_te)
    test_et = test_et_sum / max(1, n_te)
    # Reduce across processes if distributed
    if distributed and world_size > 1:
        import torch.distributed as dist
        td = torch.tensor([test_dice, test_wt, test_tc, test_et], device=device)
        dist.all_reduce(td, op=dist.ReduceOp.SUM)
        td = td / world_size
        test_dice, test_wt, test_tc, test_et = td.tolist()
        # Confusion matrix reduction is non-trivial without custom gather; skip CM plot on non-main ranks
    
    # Background 제외한 평균 (클래스 1, 2, 3만)
    avg_precision = np.mean(precision_scores[1::4])  # 클래스별로 평균
    avg_recall = np.mean(recall_scores[1::4])
    
    # 모달리티별 어텐션 가중치 분석 및 저장
    if collect_attention and len(all_attention_weights) > 0:
        try:
            attention_array = np.array(all_attention_weights)  # [n_samples, 4]
            modality_names = ['T1', 'T1CE', 'T2', 'FLAIR']
            
            # 평균 기여도 계산
            mean_contributions = attention_array.mean(axis=0)  # [4]
            std_contributions = attention_array.std(axis=0)  # [4]
            
            # 결과 출력
            print(f"\n{'='*60}")
            print(f"Modality Attention Analysis - {model_name}")
            print(f"{'='*60}")
            print(f"Total samples analyzed: {len(all_attention_weights)}")
            print(f"\nAverage Modality Contributions:")
            for i, mod_name in enumerate(modality_names):
                print(f"  {mod_name:6s}: {mean_contributions[i]:.4f} ± {std_contributions[i]:.4f}")
            
            # CSV 저장
            if results_dir:
                attention_df = pd.DataFrame(attention_array, columns=modality_names)
                attention_df['sample_id'] = range(len(attention_array))
                attention_df = attention_df[['sample_id'] + modality_names]
                
                csv_path = os.path.join(results_dir, f'modality_attention_{model_name}.csv')
                attention_df.to_csv(csv_path, index=False)
                print(f"\nAttention weights saved to: {csv_path}")
                
                # 요약 통계 저장
                summary_df = pd.DataFrame({
                    'modality': modality_names,
                    'mean_contribution': mean_contributions,
                    'std_contribution': std_contributions
                })
                summary_path = os.path.join(results_dir, f'modality_attention_summary_{model_name}.csv')
                summary_df.to_csv(summary_path, index=False)
                print(f"Attention summary saved to: {summary_path}")
            
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Failed to analyze/save attention weights: {e}")
    
    # Save confusion matrix heatmap on main (or non-distributed)
    try:
        if (not distributed) or (distributed and world_size > 0):
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_accum, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['BG', 'NCR/NET', 'ED', 'ET'],
                        yticklabels=['BG', 'NCR/NET', 'ED', 'ET'], ax=ax)
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            cm_path = os.path.join(results_dir or '.', f'confusion_matrix_{model_name}.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            if results_dir:
                print(f"Confusion matrix saved to: {cm_path}")
    except Exception as _e:
        print(f"Warning: failed to save confusion matrix: {_e}")

    return {
        'dice': test_dice,
        'wt': test_wt,
        'tc': test_tc,
        'et': test_et,
        'precision': avg_precision,
        'recall': avg_recall
    }


def run_integrated_experiment(data_path, epochs=10, batch_size=1, seeds=[24], models=None, datasets=None, dim='2d', use_pretrained=False, use_nnunet_loss=True, num_workers: int = 2, dataset_version='brats2018', use_5fold=False, use_mri_augmentation=False):
    """3D Segmentation 통합 실험 실행
    
    Args:
        data_path: 데이터셋 루트 디렉토리 경로 (기본: 'data')
        epochs: 훈련 에포크 수
        batch_size: 배치 크기
        use_nnunet_loss: If True, use nnU-Net style loss (Soft Dice with Squared Pred, Dice 70% + CE 30%)
        seeds: 실험 시드 리스트
        models: 사용할 모델 리스트 (기본: ['unet3d', 'unetr', 'swin_unetr', 'mobile_unetr'])
        datasets: 사용할 데이터셋 리스트 (기본: ['brats2021'])
                  지원: 'brats2021', 'auto' (자동 선택)
        dim: 데이터 차원 '2d' 또는 '3d' (기본: '2d')
        use_pretrained: pretrained 가중치 사용 여부 (기본: False, scratch 학습)
        dataset_version: 데이터셋 버전 'brats2021' 또는 'brats2018' (기본: 'brats2018')
        use_5fold: 5-fold cross-validation 사용 여부
    """
    
    # 실험 결과 저장 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"baseline_results/integrated_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Train data augmentation: {'MRI augmentations' if use_mri_augmentation else 'None'}")
    
    # 사용 가능한 모델들
    # Size suffix를 지원하는 모델 prefix들 (xs, s, m, l 모두 지원)
    size_suffix_models = {
        'unet3d_': ['xs', 's', 'm', 'l'],
        'unet3d_stride_': ['xs', 's', 'm', 'l'],
        'dualbranch_01_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_02_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_03_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_04_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_05_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_06_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_07_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_08_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_09_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_10_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_11_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_12_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_13_unet_': ['xs', 's', 'm', 'l'],
        'dualbranch_15_dilated125_both_': ['xs', 's', 'm', 'l'],
        'dualbranch_16_shufflenet_hybrid_': ['xs', 's', 'm', 'l'],
        'dualbranch_16_shufflenet_hybrid_ln_': ['xs', 's', 'm', 'l'],
        'dualbranch_17_shufflenet_pamlite_': ['xs', 's', 'm', 'l'],
    }
    
    # Size suffix를 지원하는 dualbranch_14 backbone들
    dualbranch_14_backbones = ['mobilenetv2_expand2', 'ghostnet', 'dilated', 'convnext', 
                                'shufflenetv2', 'shufflenetv2_crossattn', 'shufflenetv2_dilated', 'shufflenetv2_lk']
    
    # Size suffix를 지원하지 않는 모델들 (고정 이름)
    fixed_name_models = ['unetr', 'swin_unetr', 'mobile_unetr', 'mobile_unetr_3d',
                         'unet3d_2modal_s', 'unet3d_4modal_s', 'dualbranch_2modal_unet_s',
                         'quadbranch_4modal_unet_s', 'quadbranch_4modal_attention_unet_s']
    
    # 동적으로 available_models 생성
    if models is None:
        available_models = []
        # Size suffix를 지원하는 모델들 생성
        for prefix, sizes in size_suffix_models.items():
            for size in sizes:
                available_models.append(f"{prefix}{size}")
        # dualbranch_14 모델들 생성
        for backbone in dualbranch_14_backbones:
            for size in ['xs', 's', 'm', 'l']:
                available_models.append(f"dualbranch_14_{backbone}_{size}")
        # 고정 이름 모델들 추가
        available_models.extend(fixed_name_models)
    else:
        # 사용자가 지정한 모델들을 필터링 (prefix 기반으로 검증)
        available_models = []
        for model_name in models:
            # 고정 이름 모델인지 확인
            if model_name in fixed_name_models:
                available_models.append(model_name)
                continue
            # Size suffix를 지원하는 모델 prefix인지 확인
            is_valid = False
            for prefix, sizes in size_suffix_models.items():
                if model_name.startswith(prefix):
                    # Size suffix 추출
                    suffix = model_name[len(prefix):]
                    if suffix in sizes:
                        is_valid = True
                        break
            # dualbranch_14 모델인지 확인
            if not is_valid and model_name.startswith('dualbranch_14_'):
                parts = model_name.split('_', 2)
                if len(parts) >= 3:
                    backbone_and_size = parts[2]
                    for size in ['xs', 's', 'm', 'l']:
                        if backbone_and_size.endswith(f'_{size}'):
                            backbone = backbone_and_size[:-len(f'_{size}')]
                            if backbone in dualbranch_14_backbones:
                                is_valid = True
                                break
            if is_valid:
                available_models.append(model_name)
    
    # 결과 저장용
    all_results = []
    all_epochs_results = []
    all_stage_pam_results = []  # Stage별 PAM 결과 저장용
    
    # Distributed setup
    distributed, rank, local_rank, world_size = setup_distributed()
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process(rank):
            print(f"\nUsing DDP with world_size={world_size}, local_rank={local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
    
    # 데이터셋 경로 확인 (dataset_version에 따라)
    if dataset_version == 'brats2021':
        dataset_dir = os.path.join(data_path, 'BRATS2021', 'BraTS2021_Training_Data')
    elif dataset_version == 'brats2018':
        dataset_dir = os.path.join(data_path, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
    else:
        raise ValueError(f"Unknown dataset_version: {dataset_version}")
    
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset {dataset_version} not found at {dataset_dir}. Skipping...")
        return None, pd.DataFrame()
    
    print(f"\n{'#'*80}")
    print(f"Dataset Version: {dataset_version.upper()}")
    print(f"{'#'*80}")
        
    # 5-fold CV 또는 일반 실험
    if use_5fold:
        fold_list = list(range(5))
        print(f"\n{'='*60}")
        print(f"5-Fold Cross-Validation Mode")
        print(f"{'='*60}")
    else:
        fold_list = [None]  # 일반 모드에서는 fold 없음
    
    # 각 시드별로 실험
    for seed in seeds:
        # 각 fold별로 실험 (5-fold CV인 경우)
        for fold_idx in fold_list:
            if use_5fold:
                print(f"\n{'='*60}")
                print(f"Training 3D Segmentation Models - Dataset Version: {dataset_version.upper()}, Seed: {seed}, Fold: {fold_idx}")
                print(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print(f"Training 3D Segmentation Models - Dataset Version: {dataset_version.upper()}, Seed: {seed}")
                print(f"{'='*60}")
            
            # 전역 seed 설정 (데이터 분할, 학습 모두에 적용)
            set_seed(seed)
            
            # 각 모델별로 실험
            for model_name in available_models:
                try:
                    print(f"\nTraining {model_name.upper()}...")
                    
                    # 모델별 결정성 보장: 모델 초기화/샘플링 RNG 고정
                    set_seed(seed)
                    
                    # 모델에 따라 use_4modalities 및 n_channels 결정
                    use_4modalities = model_name in ['unet3d_4modal_s', 'quadbranch_4modal_unet_s', 'quadbranch_4modal_attention_unet_s']
                    if use_4modalities:
                        n_channels = 4
                    else:
                        n_channels = 2
                    
                    # 데이터 로더 생성 (모델별로 use_4modalities 설정)
                    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = get_data_loaders(
                        data_dir=data_path,
                        batch_size=batch_size,
                        num_workers=num_workers,  # /dev/shm 2GB 환경에서 기본 2 권장
                        max_samples=None,  # 전체 데이터 사용
                        dim=dim,  # 2D 또는 3D
                        dataset_version=dataset_version,  # 데이터셋 버전
                        seed=seed,  # 데이터 분할을 위한 seed
                        distributed=distributed,
                        world_size=world_size,
                        rank=rank,
                        use_4modalities=use_4modalities,  # 모델에 따라 설정
                        use_5fold=use_5fold,  # 5-fold CV 사용 여부
                        fold_idx=fold_idx,  # fold 인덱스 (None이면 일반 분할)
                        use_mri_augmentation=use_mri_augmentation
                    )

                    # 모델 생성
                    model = get_model(model_name, n_channels=n_channels, n_classes=4, dim=dim, use_pretrained=use_pretrained)
                    # DDP wrap
                    if distributed:
                        from torch.nn.parallel import DistributedDataParallel as DDP
                        model = model.to(device)
                        # GhostNet 모델은 일부 파라미터가 사용되지 않을 수 있으므로 find_unused_parameters=True 설정
                        use_find_unused = 'ghostnet' in model_name.lower()
                        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=use_find_unused)
                    
                    # 모델 정보 출력
                    print(f"\n=== {model_name.upper()} Model Information ===")
                    real_model = model.module if hasattr(model, 'module') else model
                    total_params = sum(p.numel() for p in real_model.parameters())
                    trainable_params = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
                    print(f"Total parameters: {total_params:,}")
                    print(f"Trainable parameters: {trainable_params:,}")
                    
                    # 모델 크기 계산 (real_model 사용)
                    param_size = 0
                    buffer_size = 0
                    for param in real_model.parameters():
                        param_size += param.nelement() * param.element_size()
                    for buffer in real_model.buffers():
                        buffer_size += buffer.nelement() * buffer.element_size()
                    model_size_mb = (param_size + buffer_size) / 1024 / 1024
                    if is_main_process(rank):
                        print(f"Model size: {model_size_mb:.2f} MB")
                        print("=" * 50)
                    
                    # 입력 크기 설정 (PAM, Latency 공용)
                    if dim == '2d':
                        input_size = (1, n_channels, *INPUT_SIZE_2D)
                    else:
                        input_size = (1, n_channels, *INPUT_SIZE_3D)

                    # PAM 계산 (모델 정보 출력 직후 바로 측정)
                    pam_train_list = []
                    pam_inference_list = []
                    pam_train_stages = {}
                    pam_inference_stages = {}
                    if is_main_process(rank):
                        try:
                            pam_train_list, pam_train_stages = calculate_pam(
                                model, input_size=input_size, mode='train', stage_wise=True, device=device
                            )
                            pam_inference_list, pam_inference_stages = calculate_pam(
                                model, input_size=input_size, mode='inference', stage_wise=True, device=device
                            )
                            if pam_train_list:
                                pam_train_mean = sum(pam_train_list) / len(pam_train_list)
                                print(f"PAM (Train, batch_size=1): {pam_train_mean / 1024**2:.2f} MB (mean of {len(pam_train_list)} runs)")
                            if pam_inference_list:
                                pam_inference_mean = sum(pam_inference_list) / len(pam_inference_list)
                                print(f"PAM (Inference, batch_size=1): {pam_inference_mean / 1024**2:.2f} MB (mean of {len(pam_inference_list)} runs)")
                            if pam_train_stages:
                                print("PAM (Train) by stage:")
                                for stage_name, mem_list in sorted(pam_train_stages.items()):
                                    if mem_list:
                                        mem_mean = sum(mem_list) / len(mem_list)
                                        print(f"  {stage_name}: {mem_mean / 1024**2:.2f} MB")
                            if pam_inference_stages:
                                print("PAM (Inference) by stage:")
                                for stage_name, mem_list in sorted(pam_inference_stages.items()):
                                    if mem_list:
                                        mem_mean = sum(mem_list) / len(mem_list)
                                        print(f"  {stage_name}: {mem_mean / 1024**2:.2f} MB")
                        except Exception as e:
                            print(f"Warning: Failed to calculate PAM: {e}")
                            pam_train_list = []
                            pam_inference_list = []
                            pam_train_stages = {}
                            pam_inference_stages = {}
                    
                    # 체크포인트 저장 경로 (실험 결과 폴더 내부)
                    if use_5fold:
                        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_fold_{fold_idx}_best.pth")
                    else:
                        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_best.pth")
                    
                    # 훈련
                    train_losses, val_dices, epoch_results, best_epoch, best_val_dice, best_val_wt, best_val_tc, best_val_et = train_model(
                        model, train_loader, val_loader, test_loader, epochs, device=device, model_name=model_name, seed=seed,
                        train_sampler=train_sampler, rank=rank,
                        sw_patch_size=(128, 128, 128), sw_overlap=0.10, dim=dim, use_nnunet_loss=use_nnunet_loss,
                        results_dir=results_dir, ckpt_path=ckpt_path
                    )
                    
                    # FLOPs 계산 (모델이 device에 있는 상태에서)
                    if dim == '2d':
                        flops = calculate_flops(model, input_size=(1, n_channels, *INPUT_SIZE_2D))
                    else:
                        flops = calculate_flops(model, input_size=(1, n_channels, *INPUT_SIZE_3D))
                    if is_main_process(rank):
                        print(f"FLOPs: {flops:,}")
                    
                    # Inference Latency 계산 (rank 0에서만, batch_size=1로 고정, 여러 번 측정)
                    inference_latency_list = []
                    inference_latency_stats = {}
                    if is_main_process(rank):
                        try:
                            inference_latency_list, inference_latency_stats = calculate_inference_latency(
                                model, input_size=input_size, device=device, num_warmup=10, num_runs=100
                            )
                            if inference_latency_list:
                                latency_mean = inference_latency_stats['mean']
                                latency_std = inference_latency_stats['std']
                                print(f"Inference Latency (batch_size=1): {latency_mean:.2f} ± {latency_std:.2f} ms (mean ± std of {len(inference_latency_list)} runs)")
                                print(f"  Min: {inference_latency_stats['min']:.2f} ms, Max: {inference_latency_stats['max']:.2f} ms")
                                print(f"  P50: {inference_latency_stats['p50']:.2f} ms, P95: {inference_latency_stats['p95']:.2f} ms, P99: {inference_latency_stats['p99']:.2f} ms")
                        except Exception as e:
                            print(f"Warning: Failed to calculate inference latency: {e}")
                            inference_latency_list = []
                            inference_latency_stats = {}
                    
                    # 최종 평가: Best 모델 로드 후 Test set 평가 (all ranks)
                    if is_main_process(rank):
                        print(f"\nLoading best model (epoch {best_epoch}, Val Dice: {best_val_dice:.4f}) for final test evaluation...")
                    # Best 체크포인트에서 모델 로드 (실험 결과 폴더 내부에서)
                    if use_5fold:
                        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_fold_{fold_idx}_best.pth")
                    else:
                        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_best.pth")
                    if os.path.exists(ckpt_path):
                        try:
                            real_model = model.module if hasattr(model, 'module') else model
                            state = torch.load(ckpt_path, map_location=device)
                            real_model.load_state_dict(state, strict=False)
                            if is_main_process(rank):
                                print(f"Loaded best checkpoint from {ckpt_path}")
                        except Exception as e:
                            if is_main_process(rank):
                                print(f"Warning: Failed to load checkpoint from {ckpt_path}: {e}")
                                print("Continuing with current model state...")
                    else:
                        if is_main_process(rank):
                            print(f"Warning: Checkpoint not found at {ckpt_path}, using current model state...")

                    # Switch RepLK blocks to deploy mode (before final test evaluation)
                    # RepLK blocks are fused into single 7x7x7 depthwise conv for efficient inference
                    # Check if model name starts with any RepLK model prefix (supports all sizes: xs, s, m, l)
                    replk_model_prefixes = ['dualbranch_04_unet_', 'dualbranch_05_unet_', 'dualbranch_06_unet_', 'dualbranch_07_unet_', 
                                            'dualbranch_08_unet_', 'dualbranch_09_unet_']
                    if any(model_name.startswith(prefix) for prefix in replk_model_prefixes):
                        real_model = model.module if hasattr(model, 'module') else model
                        if hasattr(real_model, 'switch_to_deploy'):
                            if is_main_process(rank):
                                print(f"Switching RepLK blocks to deploy mode (fusing branches)...")
                            real_model.switch_to_deploy()
                            # Recalculate parameters and FLOPs after deploy mode (fewer params/FLOPs due to fused branches)
                            total_params = sum(p.numel() for p in real_model.parameters())
                            if is_main_process(rank):
                                print(f"RepLK blocks switched to deploy mode.")
                                print(f"Parameters after deploy: {total_params:,} (branches fused)")
                            # Recalculate FLOPs for deploy mode
                            if dim == '2d':
                                flops = calculate_flops(model, input_size=(1, n_channels, *INPUT_SIZE_2D))
                            else:
                                flops = calculate_flops(model, input_size=(1, n_channels, *INPUT_SIZE_3D))
                            if is_main_process(rank):
                                print(f"FLOPs after deploy: {flops:,}")
                            # Recalculate PAM after deploy mode (may be different due to fused branches)
                            if is_main_process(rank):
                                try:
                                    if dim == '2d':
                                        input_size = (1, n_channels, *INPUT_SIZE_2D)
                                    else:
                                        input_size = (1, n_channels, *INPUT_SIZE_3D)
                                    pam_inference_list_after_deploy, _ = calculate_pam(
                                        model, input_size=input_size, mode='inference', stage_wise=True, device=device
                                    )
                                    if pam_inference_list_after_deploy:
                                        pam_inference_mean_after_deploy = sum(pam_inference_list_after_deploy) / len(pam_inference_list_after_deploy)
                                        print(f"PAM (Inference after deploy, batch_size=1): {pam_inference_mean_after_deploy / 1024**2:.2f} MB (mean of {len(pam_inference_list_after_deploy)} runs)")
                                except Exception as e:
                                    print(f"Warning: Failed to recalculate PAM after deploy: {e}")

                    # Test set 평가 (모든 랭크 동일 경로)
                    metrics = evaluate_model(model, test_loader, device, model_name, distributed=distributed, world_size=world_size,
                                              sw_patch_size=(128, 128, 128), sw_overlap=0.10, results_dir=results_dir)
                    
                    # Grad-CAM 생성 (rank 0에서만, 3D 모델만)
                    # TODO: Grad-CAM 기능은 아직 안정화되지 않아 주석 처리됨
                    # if is_main_process(rank) and dim == '3d':
                    #     try:
                    #         print(f"\nGenerating Grad-CAM visualizations for {model_name}...")
                    #         generate_gradcam_for_model(
                    #             model=model,
                    #             test_loader=test_loader,
                    #             device=device,
                    #             model_name=model_name,
                    #             results_dir=results_dir,
                    #             num_samples=3,  # 각 모델당 3개 샘플
                    #             target_layer=None  # 자동으로 찾음
                    #         )
                    #     except Exception as e:
                    #         print(f"Warning: Failed to generate Grad-CAM for {model_name}: {e}")
                    #         import traceback
                    #         traceback.print_exc()
                    
                    # 결과 저장 (각 run마다 하나의 행만 생성, PAM과 Latency는 평균값 사용)
                    # PAM과 Latency 평균값 계산
                    pam_train_mean = sum(pam_train_list) / len(pam_train_list) if pam_train_list else 0
                    pam_inference_mean = sum(pam_inference_list) / len(pam_inference_list) if pam_inference_list else 0
                    latency_mean = sum(inference_latency_list) / len(inference_latency_list) if inference_latency_list else 0
                    
                    result = {
                        'dataset': dataset_version,
                        'seed': seed,
                        'fold': fold_idx if use_5fold else None,
                        'model_name': model_name,
                        'total_params': total_params,
                        'flops': flops,
                        'pam_train': pam_train_mean,  # bytes, batch_size=1 기준 (평균값)
                        'pam_inference': pam_inference_mean,  # bytes, batch_size=1 기준 (평균값)
                        'inference_latency_ms': latency_mean,  # milliseconds, batch_size=1 기준 (평균값)
                        'test_dice': metrics['dice'],  # WT/TC/ET 평균
                        'test_wt': metrics.get('wt', None),
                        'test_tc': metrics.get('tc', None),
                        'test_et': metrics.get('et', None),
                        'val_dice': best_val_dice,  # WT/TC/ET 평균
                        'val_wt': best_val_wt,
                        'val_tc': best_val_tc,
                        'val_et': best_val_et,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'best_epoch': best_epoch
                    }
                    if is_main_process(rank):
                        all_results.append(result)
                        
                        # Stage별 PAM 결과 저장
                        if pam_train_stages:
                            for stage_name, mem_list in pam_train_stages.items():
                                if mem_list:
                                    mem_mean = sum(mem_list) / len(mem_list)
                                    mem_std = (sum((x - mem_mean) ** 2 for x in mem_list) / len(mem_list)) ** 0.5 if len(mem_list) > 1 else 0.0
                                    stage_pam_result = {
                                        'dataset': dataset_version,
                                        'seed': seed,
                                        'fold': fold_idx if use_5fold else None,
                                        'model_name': model_name,
                                        'mode': 'train',
                                        'stage_name': stage_name,
                                        'pam_mean': mem_mean,  # bytes
                                        'pam_std': mem_std,  # bytes
                                        'num_runs': len(mem_list)
                                    }
                                    all_stage_pam_results.append(stage_pam_result)
                        
                        if pam_inference_stages:
                            for stage_name, mem_list in pam_inference_stages.items():
                                if mem_list:
                                    mem_mean = sum(mem_list) / len(mem_list)
                                    mem_std = (sum((x - mem_mean) ** 2 for x in mem_list) / len(mem_list)) ** 0.5 if len(mem_list) > 1 else 0.0
                                    stage_pam_result = {
                                        'dataset': dataset_version,
                                        'seed': seed,
                                        'fold': fold_idx if use_5fold else None,
                                        'model_name': model_name,
                                        'mode': 'inference',
                                        'stage_name': stage_name,
                                        'pam_mean': mem_mean,  # bytes
                                        'pam_std': mem_std,  # bytes
                                        'num_runs': len(mem_list)
                                    }
                                    all_stage_pam_results.append(stage_pam_result)
                    
                    # 모든 epoch 결과 저장 (test_dice는 최종 평가 값으로 업데이트)
                    for epoch_result in epoch_results:
                        epoch_data = {
                            'dataset': dataset_version,
                            'seed': seed,
                            'fold': fold_idx if use_5fold else None,
                            'model_name': model_name,
                            'epoch': epoch_result['epoch'],
                            'train_loss': epoch_result['train_loss'],
                            'train_dice': epoch_result['train_dice'],
                            'val_loss': epoch_result['val_loss'],
                            'val_dice': epoch_result['val_dice'],
                            'val_wt': epoch_result.get('val_wt', None),
                            'val_tc': epoch_result.get('val_tc', None),
                            'val_et': epoch_result.get('val_et', None),
                            'test_dice': metrics['dice'] if epoch_result['epoch'] == best_epoch else None  # Best epoch에만 최종 test dice 기록
                        }
                        if is_main_process(rank):
                            all_epochs_results.append(epoch_data)
                    
                    if is_main_process(rank):
                        print(f"Final Val Dice: {best_val_dice:.4f} (WT {best_val_wt:.4f} | TC {best_val_tc:.4f} | ET {best_val_et:.4f}) (epoch {best_epoch})")
                        print(f"Final Test Dice: {metrics['dice']:.4f} (WT {metrics['wt']:.4f} | TC {metrics['tc']:.4f} | ET {metrics['et']:.4f}) | Prec {metrics['precision']:.4f} Rec {metrics['recall']:.4f}")
                
                except Exception as e:
                    if is_main_process(rank):
                        print(f"Error with {model_name}: {e}")
                        import traceback
                        print("Full traceback:")
                        traceback.print_exc()
                    continue
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(all_results)
    
    # 결과가 비어있는 경우 처리
    if results_df.empty:
        print("Warning: No results were collected. All experiments failed.")
        return results_dir, results_df
    
    # CSV로 저장
    if is_main_process(0) or not distributed:
        csv_path = os.path.join(results_dir, "integrated_experiment_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    # 모든 epoch 결과 저장
    epochs_df = None
    if all_epochs_results:
        epochs_df = pd.DataFrame(all_epochs_results)
        if is_main_process(0) or not distributed:
            epochs_csv_path = os.path.join(results_dir, "all_epochs_results.csv")
            epochs_df.to_csv(epochs_csv_path, index=False)
            print(f"All epochs results saved to: {epochs_csv_path}")
    
    # Stage별 PAM 결과 저장
    if all_stage_pam_results and (is_main_process(0) or not distributed):
        stage_pam_df = pd.DataFrame(all_stage_pam_results)
        stage_pam_csv_path = os.path.join(results_dir, "stage_wise_pam_results.csv")
        stage_pam_df.to_csv(stage_pam_csv_path, index=False)
        print(f"Stage-wise PAM results saved to: {stage_pam_csv_path}")
    
    # 모델별 성능 비교 분석
    if is_main_process(0) or not distributed:
        print("\nCreating model comparison analysis...")
    
    # 모델별 평균 성능
    try:
        agg_dict = {
            'test_dice': ['mean', 'std'],
            'val_dice': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }
        # PAM이 있는 경우에만 추가
        if 'pam_train' in results_df.columns:
            agg_dict['pam_train'] = ['mean', 'std']
        if 'pam_inference' in results_df.columns:
            agg_dict['pam_inference'] = ['mean', 'std']
        # Inference Latency 추가
        if 'inference_latency_ms' in results_df.columns:
            agg_dict['inference_latency_ms'] = ['mean', 'std']
        
        model_comparison = results_df.groupby('model_name').agg(agg_dict).round(4)
        if is_main_process(0) or not distributed:
            comparison_path = os.path.join(results_dir, "model_comparison.csv")
            model_comparison.to_csv(comparison_path)
            print(f"Model comparison saved to: {comparison_path}")
    except KeyError as e:
        print(f"Warning: Could not create model comparison: {e}")
    
    # 시각화 생성
    if is_main_process(0) or not distributed:
        print("\nCreating visualization charts...")
        try:
            create_comprehensive_analysis(results_df, epochs_df, results_dir)
            create_interactive_3d_plot(results_df, results_dir)
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    # 결과 출력
    if (is_main_process(0) or not distributed) and not results_df.empty:
        print("\n" + "="*80)
        print("3D SEGMENTATION INTEGRATED EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        print("\n--- Model Performance Summary ---")
        for model_name in available_models:
            model_results = results_df[results_df['model_name'] == model_name]
            if not model_results.empty:
                print(f"\n{model_name.upper()} Model:")
                for _, row in model_results.iterrows():
                    print(f"  Seed {row['seed']:3d} | Test Dice: {row['test_dice']:.4f} | Val Dice: {row['val_dice']:.4f} | Params: {row['total_params']:,}")
        
        # 모델별 평균 성능
        print("\n--- Model-wise Average Performance ---")
        for model_name in available_models:
            model_results = results_df[results_df['model_name'] == model_name]
            if not model_results.empty:
                avg_dice = model_results['test_dice'].mean()
                avg_val_dice = model_results['val_dice'].mean()
                avg_params = model_results['total_params'].mean()
                print(f"{model_name.upper():12}: Test Dice {avg_dice:.4f} | Val Dice {avg_val_dice:.4f} | Avg Params {avg_params:,.0f}")
    
    # cleanup
    if distributed:
        cleanup_distributed()
    return results_dir, results_df


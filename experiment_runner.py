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
    INPUT_SIZE_2D, INPUT_SIZE_3D, get_roi_model
)

# Import losses and metrics
from losses import combined_loss, combined_loss_nnunet_style
from metrics import calculate_wt_tc_et_dice, calculate_wt_tc_et_hd95, calculate_dice_score

# Import data loader and visualization
from dataloaders import get_data_loaders, get_brats_base_datasets
from visualization import create_comprehensive_analysis, create_interactive_3d_plot
from models.modules.se_modules import SEBlock3D
from models.modules.cbam_modules import CBAM3D, ChannelAttention3D

# Import Grad-CAM utilities
from utils.gradcam_utils import generate_gradcam_for_model

# Cascade utilities
from utils.cascade_utils import run_cascade_inference

# Import experiment configuration
from utils.experiment_config import (
    validate_and_filter_models,
    get_n_channels_for_model,
    get_model_config,
    get_roi_model_config,
    validate_roi_model,
)

# Import result utilities
from utils.result_utils import (
    create_result_dict,
    create_stage_pam_result,
    create_epoch_result_dict,
    save_results_to_csv
)


def train_roi_model(model, train_loader, val_loader, epochs, device, lr=1e-3,
                    criterion=None, ckpt_path=None, results_dir=None, model_name='roi_model',
                    train_sampler=None, rank: int = 0):
    """Train ROI detector on resized volumes (binary WT segmentation)."""
    if criterion is None:
        criterion = combined_loss_nnunet_style
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_dice = 0.0
    best_epoch = 0
    os.makedirs(results_dir or "baseline_results", exist_ok=True)
    disable_progress = not is_main_process(rank)

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        n_samples = 0
        for inputs, labels in tqdm(
            train_loader,
            desc=f"[ROI] Train {epoch+1}/{epochs}",
            leave=False,
            disable=disable_progress,
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            bsz = inputs.size(0)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bsz
            n_samples += bsz
        train_loss /= max(1, n_samples)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_dices = []
        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader,
                desc=f"[ROI] Val {epoch+1}/{epochs}",
                leave=False,
                disable=disable_progress,
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                bsz = inputs.size(0)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * bsz
                val_samples += bsz
                dice_scores = calculate_dice_score(logits.detach().cpu(), labels.detach().cpu(), num_classes=2)
                if dice_scores.numel() >= 2:
                    val_dices.append(dice_scores[1].item())
        val_loss /= max(1, val_samples)
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        scheduler.step(val_loss)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            if ckpt_path and is_main_process(rank):
                torch.save(model.state_dict(), ckpt_path)
        if is_main_process(rank):
            print(f"[ROI][Epoch {epoch+1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return {
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch,
    }


def evaluate_cascade_pipeline(roi_model, seg_model, base_dataset, device,
                              roi_resize=(64, 64, 64), crop_size=(96, 96, 96), include_coords=True,
                              crops_per_center=1, crop_overlap=0.5, use_blending=True):
    """
    Run cascade inference on base dataset and compute WT/TC/ET dice.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
    """
    roi_model.eval()
    seg_model.eval()
    dice_rows = []
    for idx in range(len(base_dataset)):
        image, target = base_dataset[idx]
        image = image.to(device)
        target = target.to(device)
        result = run_cascade_inference(
            roi_model=roi_model,
            seg_model=seg_model,
            image=image,
            device=device,
            roi_resize=roi_resize,
            crop_size=crop_size,
            include_coords=include_coords,
            crops_per_center=crops_per_center,
            crop_overlap=crop_overlap,
            use_blending=use_blending,
        )
        full_logits = result['full_logits'].unsqueeze(0).to(device)
        target_batch = target.unsqueeze(0)
        dice = calculate_wt_tc_et_dice(full_logits, target_batch).detach().cpu()
        dice_rows.append(dice)
    if not dice_rows:
        return {'wt': 0.0, 'tc': 0.0, 'et': 0.0, 'mean': 0.0}
    dice_tensor = torch.stack(dice_rows, dim=0)
    mean_scores = dice_tensor.mean(dim=0)
    return {
        'wt': float(mean_scores[0].item()),
        'tc': float(mean_scores[1].item()),
        'et': float(mean_scores[2].item()),
        'mean': float(mean_scores.mean().item())
    }


def load_roi_model_from_checkpoint(roi_model_name, weight_path, device, include_coords=True):
    """Load ROI model weights for inference.
    
    Automatically detects the number of input channels from checkpoint and adjusts include_coords accordingly.
    """
    cfg = get_roi_model_config(roi_model_name)
    
    # Load checkpoint first to detect input channels
    state = torch.load(weight_path, map_location=device)
    
    # Detect input channels from checkpoint
    # Try common first layer names for ROI models
    detected_channels = None
    for key in state.keys():
        if 'weight' in key and ('enc_blocks.0.net.0' in key or 'patch_embed' in key or 'conv' in key):
            if 'enc_blocks.0.net.0.weight' in key:
                # ROICascadeUNet3D: enc_blocks.0.net.0.weight shape is [out_channels, in_channels, ...]
                detected_channels = state[key].shape[1]
                break
            elif 'patch_embed' in key and 'weight' in key:
                # MobileUNETR_3D: patch_embed.proj.weight shape is [out_channels, in_channels, ...]
                detected_channels = state[key].shape[1]
                break
    
    # Auto-detect include_coords based on detected channels
    if detected_channels is not None:
        if detected_channels == 4:
            include_coords = False
            print(f"Detected 4-channel ROI model (no CoordConv). Using include_coords=False")
        elif detected_channels == 7:
            include_coords = True
            print(f"Detected 7-channel ROI model (with CoordConv). Using include_coords=True")
        else:
            print(f"Warning: Unexpected input channels {detected_channels} in checkpoint. Using provided include_coords={include_coords}")
    
    model = get_roi_model(
        roi_model_name,
        n_channels=7 if include_coords else 4,
        n_classes=2,
        roi_model_cfg=cfg,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def evaluate_segmentation_with_roi(
    seg_model,
    roi_model,
    data_dir,
    dataset_version,
    seed,
    roi_resize=(64, 64, 64),
    crop_size=(96, 96, 96),
    include_coords=True,
    use_5fold=False,
    fold_idx=None,
    max_samples=None,
    crops_per_center=1,
    crop_overlap=0.5,
    use_blending=True,
):
    """
    Evaluate trained segmentation model with pre-trained ROI detector.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
    """
    _, _, test_base = get_brats_base_datasets(
        data_dir=data_dir,
        dataset_version=dataset_version,
        max_samples=max_samples,
        seed=seed,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        use_4modalities=True,
    )
    seg_model.eval()
    roi_model.eval()
    return evaluate_cascade_pipeline(
        roi_model=roi_model,
        seg_model=seg_model,
        base_dataset=test_base,
        device=next(seg_model.parameters()).device,
        roi_resize=roi_resize,
        crop_size=crop_size,
        include_coords=include_coords,
        crops_per_center=crops_per_center,
        crop_overlap=crop_overlap,
        use_blending=use_blending,
    )


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


def evaluate_model(model, test_loader, device='cuda', model_name: str = 'model', distributed: bool = False, world_size: int = 1,
                   sw_patch_size=(128, 128, 128), sw_overlap=0.25, results_dir: str = None):
    """모델 평가 함수"""
    model.eval()
    real_model = model.module if hasattr(model, 'module') else model
    test_dice = 0.0
    test_wt_sum = test_tc_sum = test_et_sum = 0.0
    n_te = 0
    precision_scores = []
    recall_scores = []
    hd95_wt_sum = hd95_tc_sum = hd95_et_sum = 0.0
    hd95_wt_count = hd95_tc_count = hd95_et_count = 0
    save_examples = results_dir is not None
    rank0 = True
    if distributed and hasattr(torch, 'distributed') and torch.distributed.is_available():
        if torch.distributed.is_initialized():
            rank0 = torch.distributed.get_rank() == 0
    save_examples = save_examples and rank0

    collect_se = (results_dir is not None) and rank0
    se_blocks = []
    se_excitation_data = {}
    if collect_se:
        for name, module in real_model.named_modules():
            if isinstance(module, SEBlock3D):
                se_blocks.append((name, module))
                se_excitation_data[name] = []
    
    collect_cbam = (results_dir is not None) and rank0
    cbam_blocks = []
    channel_attention_blocks = []  # 블록 내부의 ChannelAttention3D도 수집
    cbam_channel_data = {}
    cbam_spatial_data = {}
    channel_attention_data = {}  # 블록 내부 Channel Attention 가중치
    if collect_cbam:
        for name, module in real_model.named_modules():
            if isinstance(module, CBAM3D):
                cbam_blocks.append((name, module))
                cbam_channel_data[name] = []
                cbam_spatial_data[name] = []
            elif isinstance(module, ChannelAttention3D):
                # 블록 내부의 ChannelAttention3D도 수집 (예: ShuffleNetV1Unit3D 내부)
                channel_attention_blocks.append((name, module))
                channel_attention_data[name] = []
        if rank0:
            print(f"[CBAM Debug] Found {len(cbam_blocks)} CBAM blocks: {[name for name, _ in cbam_blocks]}")
            if channel_attention_blocks:
                print(f"[CBAM Debug] Found {len(channel_attention_blocks)} ChannelAttention3D blocks (inside units): {[name for name, _ in channel_attention_blocks]}")
    example_dir = None
    example_limit = 10
    examples_saved = 0
    if save_examples:
        example_dir = os.path.join(results_dir, f'qualitative_examples_{model_name}')
        os.makedirs(example_dir, exist_ok=True)
    
    # 모달리티별 어텐션 가중치 수집 (quadbranch_4modal_attention_unet_s만)
    collect_attention = (model_name == 'quadbranch_4modal_attention_unet_s')
    all_attention_weights = []  # 각 샘플별 어텐션 가중치 저장
    
    with torch.no_grad():
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        from matplotlib.colors import ListedColormap
        seg_cmap = ListedColormap(['black', '#ff0000', '#00ff00', '#0000ff'])
        cm_accum = np.zeros((4, 4), dtype=np.int64)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 2D/3D 분기: 2D 모델은 그대로, 3D 모델은 depth 차원 추가
            if model_name != 'mobile_unetr' and len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(2)
                labels = labels.unsqueeze(2)
            
            # 3D 테스트: 슬라이딩 윈도우 추론
            # 모든 3D 모델은 전체 볼륨을 처리하기 위해 슬라이딩 윈도우 사용
            # 단, CBAM/SE 가중치 수집을 위해서는 전체 볼륨에 대해 직접 forward 호출 필요
            if inputs.dim() == 5 and inputs.size(0) == 1:  # 3D 볼륨
                # CBAM 또는 SE 가중치 수집이 필요한 경우, 전체 볼륨에 대해 직접 forward 호출
                if (collect_cbam and len(cbam_blocks) > 0) or (collect_se and len(se_blocks) > 0):
                    try:
                        # 전체 볼륨에 대해 직접 forward (가중치 수집을 위해)
                        logits = model(inputs)
                    except RuntimeError as e:
                        # 메모리 부족 시 슬라이딩 윈도우 사용 (가중치 수집 불가)
                        if "out of memory" in str(e).lower():
                            if rank0:
                                print(f"Warning: OOM during CBAM/SE weight collection, using sliding window (weights may not be collected)")
                            logits = sliding_window_inference_3d(
                                model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name
                            )
                        else:
                            raise
                elif collect_attention:
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
            hd95_batch = calculate_wt_tc_et_hd95(pred, labels)
            if hd95_batch.size > 0:
                for hd_wt, hd_tc, hd_et in hd95_batch:
                    if np.isfinite(hd_wt):
                        hd95_wt_sum += float(hd_wt)
                        hd95_wt_count += 1
                    if np.isfinite(hd_tc):
                        hd95_tc_sum += float(hd_tc)
                        hd95_tc_count += 1
                    if np.isfinite(hd_et):
                        hd95_et_sum += float(hd_et)
                        hd95_et_count += 1

            if collect_se and se_blocks:
                for block_name, block_module in se_blocks:
                    excitation = getattr(block_module, 'last_excitation', None)
                    if excitation is not None:
                        se_excitation_data[block_name].append(excitation.clone())
            
            # Collect CBAM weights (독립적으로 실행)
            if collect_cbam and cbam_blocks:
                for block_name, block_module in cbam_blocks:
                    channel_weights = getattr(block_module, 'last_channel_weights', None)
                    spatial_weights = getattr(block_module, 'last_spatial_weights', None)
                    if channel_weights is not None:
                        cbam_channel_data[block_name].append(channel_weights.clone())
                    else:
                        if rank0 and len(cbam_channel_data[block_name]) == 0:  # 첫 번째 배치에서만 경고
                            print(f"[CBAM Debug] Warning: No channel weights found for {block_name}")
                    if spatial_weights is not None:
                        cbam_spatial_data[block_name].append(spatial_weights.clone())
                    else:
                        if rank0 and len(cbam_spatial_data[block_name]) == 0:  # 첫 번째 배치에서만 경고
                            print(f"[CBAM Debug] Warning: No spatial weights found for {block_name}")
            
            # Collect ChannelAttention3D weights from inside blocks (e.g., ShuffleNetV1Unit3D)
            if collect_cbam and channel_attention_blocks:
                for block_name, block_module in channel_attention_blocks:
                    channel_weights = getattr(block_module, 'last_channel_weights', None)
                    if channel_weights is not None:
                        channel_attention_data[block_name].append(channel_weights.clone())
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

            # 예시 이미지 저장
            if save_examples and examples_saved < example_limit:
                inputs_np = inputs.detach().cpu().numpy()
                for bi in range(bsz):
                    if examples_saved >= example_limit:
                        break
                    input_sample = inputs_np[bi]  # (C, D, H, W) or (C, H, W)
                    label_sample = labels_np[bi]  # (D, H, W) or (H, W)
                    pred_sample = pred_np[bi]     # (D, H, W) or (H, W)

                    # 3D 볼륨인 경우: (C, H, W, D) 및 (H, W, D)
                    if input_sample.ndim == 4:  # (C, H, W, D)
                        C, H, W, D = input_sample.shape
                        channel_idx = 0 if C >= 2 else 0  # FLAIR 채널 (2채널이면 0번, 4채널이면 3번)
                        if C >= 4:
                            channel_idx = 3  # FLAIR는 4채널일 때 마지막
                        
                        # 마스크가 있는 슬라이스 찾기 (배경이 아닌 픽셀이 있는 슬라이스)
                        valid_slices = []
                        if label_sample.ndim == 3:  # (H, W, D)
                            for d_idx in range(D):
                                label_slice_2d = label_sample[:, :, d_idx]
                                if (label_slice_2d > 0).any():
                                    valid_slices.append(d_idx)
                        else:
                            # 2D인 경우
                            if (label_sample > 0).any():
                                valid_slices = [0]
                        
                        # 마스크가 있는 슬라이스가 없으면 스킵
                        if not valid_slices:
                            continue
                        
                        # 마스크가 있는 슬라이스 중에서 선택 (중간 부분 우선)
                        if len(valid_slices) > 1:
                            mid_idx = len(valid_slices) // 2
                            slice_idx = valid_slices[mid_idx]
                        else:
                            slice_idx = valid_slices[0]
                        
                        # 이미지 슬라이스 추출: (C, H, W, D) -> (H, W)
                        image_slice = input_sample[channel_idx, :, :, slice_idx]
                        
                        # 라벨/예측 슬라이스 추출: (H, W, D) -> (H, W)
                        if label_sample.ndim == 3:
                            label_slice = label_sample[:, :, slice_idx]
                            pred_slice = pred_sample[:, :, slice_idx]
                        else:
                            label_slice = label_sample
                            pred_slice = pred_sample
                    
                    else:  # 2D: (C, H, W) 및 (H, W)
                        C, H, W = input_sample.shape
                        channel_idx = min(C - 1, 0)  # FLAIR 채널
                        image_slice = input_sample[channel_idx, :, :]
                        
                        # 2D인 경우에도 마스크 확인
                        if label_sample.ndim == 2:
                            if not (label_sample > 0).any():
                                continue  # 마스크가 없으면 스킵
                            label_slice = label_sample
                            pred_slice = pred_sample
                        else:
                            # 3D 라벨이지만 2D 입력인 경우 (이상한 케이스)
                            continue

                    # 이미지 정규화
                    img_min, img_max = image_slice.min(), image_slice.max()
                    if img_max > img_min:
                        image_display = (image_slice - img_min) / (img_max - img_min)
                    else:
                        image_display = image_slice

                    # 시각화: 원본 이미지 위에 마스크 오버레이
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Input: 원본 이미지만
                    axes[0].imshow(image_display, cmap='gray', origin='lower')
                    axes[0].set_title(f'Input (FLAIR, slice {slice_idx if input_sample.ndim == 4 else "2D"})')
                    axes[0].axis('off')

                    # Ground Truth: 원본 이미지 + 마스크 오버레이
                    axes[1].imshow(image_display, cmap='gray', origin='lower')
                    # 마스크가 있는 부분만 오버레이 (alpha=0.5)
                    mask_gt = label_slice > 0
                    if mask_gt.any():
                        im_gt = axes[1].imshow(label_slice, cmap=seg_cmap, alpha=0.5, vmin=0, vmax=3, origin='lower')
                    axes[1].set_title('Ground Truth (overlay)')
                    axes[1].axis('off')

                    # Prediction: 원본 이미지 + 예측 마스크 오버레이
                    axes[2].imshow(image_display, cmap='gray', origin='lower')
                    # 예측 마스크가 있는 부분만 오버레이 (alpha=0.5)
                    mask_pred = pred_slice > 0
                    if mask_pred.any():
                        im_pred = axes[2].imshow(pred_slice, cmap=seg_cmap, alpha=0.5, vmin=0, vmax=3, origin='lower')
                    axes[2].set_title('Prediction (overlay)')
                    axes[2].axis('off')

                    plt.tight_layout()
                    example_path = os.path.join(
                        example_dir, f"{model_name}_example_{examples_saved + 1:02d}.png"
                    )
                    plt.savefig(example_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    examples_saved += 1

    
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
        hd_tensor = torch.tensor(
            [
                hd95_wt_sum,
                hd95_wt_count,
                hd95_tc_sum,
                hd95_tc_count,
                hd95_et_sum,
                hd95_et_count,
            ],
            device=device,
        )
        dist.all_reduce(hd_tensor, op=dist.ReduceOp.SUM)
        (
            hd95_wt_sum,
            hd95_wt_count,
            hd95_tc_sum,
            hd95_tc_count,
            hd95_et_sum,
            hd95_et_count,
        ) = hd_tensor.tolist()
        # Confusion matrix reduction is non-trivial without custom gather; skip CM plot on non-main ranks
    hd95_wt = (hd95_wt_sum / hd95_wt_count) if hd95_wt_count > 0 else None
    hd95_tc = (hd95_tc_sum / hd95_tc_count) if hd95_tc_count > 0 else None
    hd95_et = (hd95_et_sum / hd95_et_count) if hd95_et_count > 0 else None
    total_sum = hd95_wt_sum + hd95_tc_sum + hd95_et_sum
    total_count = hd95_wt_count + hd95_tc_count + hd95_et_count
    hd95_mean = (total_sum / total_count) if total_count > 0 else None
    
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

    # Save SE excitation statistics and histograms
    if collect_se and se_blocks:
        se_summary_rows = []
        for block_name, data_list in se_excitation_data.items():
            if not data_list:
                continue
            try:
                block_tensor = torch.cat(data_list, dim=0)
            except RuntimeError:
                block_tensor = torch.cat([d.cpu() for d in data_list], dim=0)
            block_np = block_tensor.numpy()
            channel_means = block_np.mean(axis=0)
            channel_stds = block_np.std(axis=0)
            for ch_idx, (m, s) in enumerate(zip(channel_means, channel_stds)):
                se_summary_rows.append({
                    'block_name': block_name,
                    'channel': ch_idx,
                    'mean_excitation': float(m),
                    'std_excitation': float(s)
                })
            if results_dir:
                safe_name = block_name.replace('.', '_')
                hist_path = os.path.join(
                    results_dir, f'se_excitation_hist_{model_name}_{safe_name}.png'
                )
                plt.figure(figsize=(6, 4))
                plt.hist(block_np.flatten(), bins=40, range=(0, 1), color='steelblue', alpha=0.8)
                plt.title(f'SE Excitation Histogram\n{model_name} - {block_name}')
                plt.xlabel('Excitation Weight')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
        if se_summary_rows and results_dir:
            se_df = pd.DataFrame(se_summary_rows)
            csv_path = os.path.join(results_dir, f'se_excitation_summary_{model_name}.csv')
            se_df.to_csv(csv_path, index=False)
            print(f"SE excitation summary saved to: {csv_path}")

    # Save CBAM attention statistics and histograms
    if collect_cbam and cbam_blocks:
        import matplotlib.pyplot as plt
        
        if rank0:
            print(f"[CBAM Debug] Saving CBAM statistics: {len(cbam_blocks)} blocks found")
            for block_name, data_list in cbam_channel_data.items():
                print(f"[CBAM Debug] Block {block_name}: {len(data_list)} channel weight samples collected")
            for block_name, data_list in cbam_spatial_data.items():
                print(f"[CBAM Debug] Block {block_name}: {len(data_list)} spatial weight samples collected")
        
        # Channel Attention weights
        cbam_channel_summary_rows = []
        for block_name, data_list in cbam_channel_data.items():
            if not data_list:
                if rank0:
                    print(f"[CBAM Debug] Warning: No channel data for {block_name}, skipping")
                continue
            try:
                block_tensor = torch.cat(data_list, dim=0)
            except RuntimeError:
                block_tensor = torch.cat([d.cpu() for d in data_list], dim=0)
            block_np = block_tensor.numpy()  # (N, C) where N is number of samples
            channel_means = block_np.mean(axis=0)
            channel_stds = block_np.std(axis=0)
            for ch_idx, (m, s) in enumerate(zip(channel_means, channel_stds)):
                cbam_channel_summary_rows.append({
                    'block_name': block_name,
                    'channel': ch_idx,
                    'mean_channel_weight': float(m),
                    'std_channel_weight': float(s)
                })
            if results_dir:
                safe_name = block_name.replace('.', '_')
                hist_path = os.path.join(
                    results_dir, f'cbam_channel_hist_{model_name}_{safe_name}.png'
                )
                plt.figure(figsize=(6, 4))
                plt.hist(block_np.flatten(), bins=40, range=(0, 1), color='coral', alpha=0.8)
                plt.title(f'CBAM Channel Attention Histogram\n{model_name} - {block_name}')
                plt.xlabel('Channel Attention Weight')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
        if cbam_channel_summary_rows and results_dir:
            cbam_channel_df = pd.DataFrame(cbam_channel_summary_rows)
            csv_path = os.path.join(results_dir, f'cbam_channel_summary_{model_name}.csv')
            cbam_channel_df.to_csv(csv_path, index=False)
            print(f"CBAM channel attention summary saved to: {csv_path}")
        
        # Spatial Attention weights
        cbam_spatial_summary_rows = []
        for block_name, data_list in cbam_spatial_data.items():
            if not data_list:
                if rank0:
                    print(f"[CBAM Debug] Warning: No spatial data for {block_name}, skipping")
                continue
            try:
                block_tensor = torch.cat(data_list, dim=0)
            except RuntimeError:
                block_tensor = torch.cat([d.cpu() for d in data_list], dim=0)
            block_np = block_tensor.numpy()  # (N, 1, D, H, W) where N is number of samples
            # Flatten spatial dimensions for statistics
            spatial_means = block_np.mean(axis=(0, 1))  # Average over batch and channel, shape: (D, H, W)
            spatial_stds = block_np.std(axis=(0, 1))
            spatial_flat_means = spatial_means.flatten()
            spatial_flat_stds = spatial_stds.flatten()
            for idx, (m, s) in enumerate(zip(spatial_flat_means, spatial_flat_stds)):
                cbam_spatial_summary_rows.append({
                    'block_name': block_name,
                    'spatial_idx': idx,
                    'mean_spatial_weight': float(m),
                    'std_spatial_weight': float(s)
                })
            if results_dir:
                safe_name = block_name.replace('.', '_')
                hist_path = os.path.join(
                    results_dir, f'cbam_spatial_hist_{model_name}_{safe_name}.png'
                )
                plt.figure(figsize=(6, 4))
                plt.hist(block_np.flatten(), bins=40, range=(0, 1), color='mediumseagreen', alpha=0.8)
                plt.title(f'CBAM Spatial Attention Histogram\n{model_name} - {block_name}')
                plt.xlabel('Spatial Attention Weight')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
        if cbam_spatial_summary_rows and results_dir:
            cbam_spatial_df = pd.DataFrame(cbam_spatial_summary_rows)
            csv_path = os.path.join(results_dir, f'cbam_spatial_summary_{model_name}.csv')
            cbam_spatial_df.to_csv(csv_path, index=False)
            print(f"CBAM spatial attention summary saved to: {csv_path}")
    
    # Save ChannelAttention3D weights from inside blocks (e.g., ShuffleNetV1Unit3D)
    if collect_cbam and channel_attention_blocks:
        import matplotlib.pyplot as plt
        
        if rank0:
            print(f"[CBAM Debug] Saving ChannelAttention3D statistics: {len(channel_attention_blocks)} blocks found")
            for block_name, data_list in channel_attention_data.items():
                print(f"[CBAM Debug] Block {block_name}: {len(data_list)} channel weight samples collected")
        
        # Channel Attention weights from inside blocks
        channel_attention_summary_rows = []
        for block_name, data_list in channel_attention_data.items():
            if not data_list:
                if rank0:
                    print(f"[CBAM Debug] Warning: No channel attention data for {block_name}, skipping")
                continue
            try:
                block_tensor = torch.cat(data_list, dim=0)
            except RuntimeError:
                block_tensor = torch.cat([d.cpu() for d in data_list], dim=0)
            block_np = block_tensor.numpy()  # (N, C) where N is number of samples
            channel_means = block_np.mean(axis=0)
            channel_stds = block_np.std(axis=0)
            for ch_idx, (m, s) in enumerate(zip(channel_means, channel_stds)):
                channel_attention_summary_rows.append({
                    'block_name': block_name,
                    'channel': ch_idx,
                    'mean_channel_weight': float(m),
                    'std_channel_weight': float(s)
                })
            if results_dir:
                safe_name = block_name.replace('.', '_')
                hist_path = os.path.join(
                    results_dir, f'channel_attention_hist_{model_name}_{safe_name}.png'
                )
                plt.figure(figsize=(6, 4))
                plt.hist(block_np.flatten(), bins=40, range=(0, 1), color='purple', alpha=0.8)
                plt.title(f'Channel Attention Histogram (Inside Blocks)\n{model_name} - {block_name}')
                plt.xlabel('Channel Attention Weight')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
        if channel_attention_summary_rows and results_dir:
            channel_attention_df = pd.DataFrame(channel_attention_summary_rows)
            csv_path = os.path.join(results_dir, f'channel_attention_summary_{model_name}.csv')
            channel_attention_df.to_csv(csv_path, index=False)
            print(f"Channel Attention (inside blocks) summary saved to: {csv_path}")

    return {
        'dice': test_dice,
        'wt': test_wt,
        'tc': test_tc,
        'et': test_et,
        'hd95_wt': hd95_wt,
        'hd95_tc': hd95_tc,
        'hd95_et': hd95_et,
        'hd95_mean': hd95_mean,
        'precision': avg_precision,
        'recall': avg_recall
    }


def run_integrated_experiment(data_path, epochs=10, batch_size=1, seeds=[24], models=None, datasets=None, dim='2d', use_pretrained=False, use_nnunet_loss=True, num_workers: int = 2, dataset_version='brats2018', use_5fold=False, use_mri_augmentation=False, cascade_infer_cfg=None, cascade_model_cfg=None):
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
    
    # 사용 가능한 모델들 검증 및 필터링
    available_models = validate_and_filter_models(models)
    
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
                    if is_main_process(rank):
                        print(f"\nTraining {model_name.upper()}...")
                    
                    # 모델별 결정성 보장: 모델 초기화/샘플링 RNG 고정
                    set_seed(seed)
                    
                    # Cascade 모델인 경우 ROI 모델 정보 확인
                    roi_model_name_for_result = None
                    if model_name.startswith('cascade_') and cascade_model_cfg:
                        roi_model_name_for_result = cascade_model_cfg.get('roi_model_name')
                        if is_main_process(rank):
                            print(f"Cascade model detected. ROI model: {roi_model_name_for_result}")
                    
                    # 모델에 따라 use_4modalities 및 n_channels 결정
                    model_config = get_model_config(model_name)
                    n_channels = model_config['n_channels']
                    use_4modalities = model_config['use_4modalities']
                    
                    # 데이터 로더 생성 (모델별로 use_4modalities 설정)
                    try:
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
                            use_mri_augmentation=use_mri_augmentation,
                            model_name=model_name,  # Cascade 모델 감지를 위해 전달
                        )
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"Error creating data loaders for {model_name}: {e}")
                            import traceback
                            traceback.print_exc()
                        continue

                    # 모델 생성
                    try:
                        if is_main_process(rank):
                            print(f"Creating model: {model_name}...")
                        model = get_model(model_name, n_channels=n_channels, n_classes=4, dim=dim, use_pretrained=use_pretrained)
                        if is_main_process(rank):
                            print(f"Model {model_name} created successfully.")
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"Error creating model {model_name}: {e}")
                            import traceback
                            print("Full traceback:")
                            traceback.print_exc()
                        continue
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
                    # Cascade 모델은 7채널 입력 (4 MRI + 3 CoordConv)
                    actual_n_channels = n_channels
                    if model_name.startswith('cascade_'):
                        actual_n_channels = 7  # 4 MRI + 3 CoordConv
                    
                    if dim == '2d':
                        input_size = (1, actual_n_channels, *INPUT_SIZE_2D)
                    else:
                        input_size = (1, actual_n_channels, *INPUT_SIZE_3D)

                    # PAM 계산 (모델 정보 출력 직후 바로 측정)
                    pam_train_list = []
                    pam_inference_list = []
                    pam_train_stages = {}
                    pam_inference_stages = {}
                    if is_main_process(rank):
                        if dim == '2d':
                            pam_input_size = (1, actual_n_channels, *INPUT_SIZE_2D)
                        else:
                            pam_input_size = (1, actual_n_channels, 128, 128, 128)
                        try:
                            pam_train_list, pam_train_stages = calculate_pam(
                                model, input_size=pam_input_size, mode='train', stage_wise=True, device=device
                            )
                            pam_inference_list, pam_inference_stages = calculate_pam(
                                model, input_size=pam_input_size, mode='inference', stage_wise=True, device=device
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
                        flops = calculate_flops(model, input_size=(1, actual_n_channels, *INPUT_SIZE_2D))
                    else:
                        flops = calculate_flops(model, input_size=(1, actual_n_channels, *INPUT_SIZE_3D))
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
                    replk_model_prefixes = [
                        'dualbranch_04_unet_',
                        'dualbranch_05_unet_',
                        'dualbranch_06_unet_',
                        'dualbranch_07_unet_',
                    ]
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
                                flops = calculate_flops(model, input_size=(1, actual_n_channels, *INPUT_SIZE_3D))
                            if is_main_process(rank):
                                print(f"FLOPs after deploy: {flops:,}")
                            # Recalculate PAM after deploy mode (may be different due to fused branches)
                            if is_main_process(rank):
                                try:
                                    if dim == '2d':
                                        input_size_after_deploy = (1, actual_n_channels, *INPUT_SIZE_2D)
                                    else:
                                        input_size_after_deploy = (1, actual_n_channels, *INPUT_SIZE_3D)
                                    pam_inference_list_after_deploy, _ = calculate_pam(
                                        model, input_size=input_size_after_deploy, mode='inference', stage_wise=True, device=device
                                    )
                                    if pam_inference_list_after_deploy:
                                        pam_inference_mean_after_deploy = sum(pam_inference_list_after_deploy) / len(pam_inference_list_after_deploy)
                                        print(f"PAM (Inference after deploy, batch_size=1): {pam_inference_mean_after_deploy / 1024**2:.2f} MB (mean of {len(pam_inference_list_after_deploy)} runs)")
                                except Exception as e:
                                    print(f"Warning: Failed to recalculate PAM after deploy: {e}")

                    # Test set 평가 (모든 랭크 동일 경로)
                    metrics = evaluate_model(
                        model,
                        test_loader,
                        device,
                        model_name,
                        distributed=distributed,
                        world_size=world_size,
                        sw_patch_size=(128, 128, 128),
                        sw_overlap=0.10,
                        results_dir=results_dir,
                    )

                    cascade_metrics = None
                    # Cascade 평가: --use_cascade_pipeline 또는 cascade 모델인 경우
                    should_run_cascade = (
                        dim == '3d'
                        and is_main_process(rank)
                        and (
                            (cascade_infer_cfg and cascade_infer_cfg.get('roi_weight_path'))
                            or (model_name.startswith('cascade_') and cascade_model_cfg)
                        )
                    )
                    
                    if should_run_cascade:
                        # ROI weight 경로 결정
                        roi_weight_path = None
                        roi_model_name_for_infer = None
                        roi_resize = (64, 64, 64)
                        crop_size = (96, 96, 96)
                        
                        if cascade_infer_cfg and cascade_infer_cfg.get('roi_weight_path'):
                            # 명시적으로 지정된 경우 (--use_cascade_pipeline)
                            roi_weight_path = cascade_infer_cfg['roi_weight_path']
                            roi_model_name_for_infer = cascade_infer_cfg['roi_model_name']
                            roi_resize = cascade_infer_cfg.get('roi_resize', roi_resize)
                            crop_size = cascade_infer_cfg.get('crop_size', crop_size)
                            crops_per_center = cascade_infer_cfg.get('crops_per_center', 1)
                            crop_overlap = cascade_infer_cfg.get('crop_overlap', 0.5)
                            use_blending = cascade_infer_cfg.get('use_blending', True)
                        elif model_name.startswith('cascade_') and cascade_model_cfg:
                            # Cascade 모델인 경우 자동으로 경로 생성
                            roi_model_name_for_infer = cascade_model_cfg['roi_model_name']
                            default_path = f"models/weights/cascade/roi_model/{roi_model_name_for_infer}/seed_{seed}/weights/best.pth"
                            if os.path.exists(default_path):
                                roi_weight_path = default_path
                                print(f"Using default ROI weight path for cascade model: {roi_weight_path}")
                            else:
                                print(f"Warning: Default ROI weight path not found: {default_path}. Skipping cascade evaluation.")
                                roi_weight_path = None
                            roi_resize = cascade_model_cfg.get('roi_resize', roi_resize)
                            crop_size = cascade_model_cfg.get('crop_size', crop_size)
                            # Cascade 모델인 경우 기본값 사용
                            crops_per_center = 1
                            crop_overlap = 0.5
                            use_blending = True
                        else:
                            # 기본값
                            crops_per_center = 1
                            crop_overlap = 0.5
                            use_blending = True
                        
                        if roi_weight_path and os.path.exists(roi_weight_path):
                            try:
                                roi_model = load_roi_model_from_checkpoint(
                                    roi_model_name_for_infer,
                                    roi_weight_path,
                                    device=device,
                                    include_coords=True,
                                )
                                real_model = model.module if hasattr(model, 'module') else model
                                cascade_metrics = evaluate_segmentation_with_roi(
                                    seg_model=real_model,
                                    roi_model=roi_model,
                                    data_dir=data_path,
                                    dataset_version=dataset_version,
                                    seed=seed,
                                    roi_resize=roi_resize,
                                    crop_size=crop_size,
                                    include_coords=True,
                                    use_5fold=use_5fold,
                                    fold_idx=fold_idx if use_5fold else None,
                                    crops_per_center=crops_per_center,
                                    crop_overlap=crop_overlap,
                                    use_blending=use_blending,
                                )
                                print(
                                    f"Cascade ROI→Seg Dice: {cascade_metrics['mean']:.4f} "
                                    f"(WT {cascade_metrics['wt']:.4f} | TC {cascade_metrics['tc']:.4f} | ET {cascade_metrics['et']:.4f})"
                                )
                            except Exception as e:
                                print(f"Warning: Cascade evaluation failed: {e}")
                                import traceback
                                traceback.print_exc()
                                cascade_metrics = None
                    
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
                    
                    if is_main_process(rank):
                        # 메인 결과 저장
                        result = create_result_dict(
                            dataset_version=dataset_version,
                            seed=seed,
                            fold_idx=fold_idx if use_5fold else None,
                            model_name=model_name,
                            total_params=total_params,
                            flops=flops,
                            pam_train_mean=pam_train_mean,
                            pam_inference_mean=pam_inference_mean,
                            latency_mean=latency_mean,
                            metrics=metrics,
                            best_val_dice=best_val_dice,
                            best_val_wt=best_val_wt,
                            best_val_tc=best_val_tc,
                            best_val_et=best_val_et,
                            best_epoch=best_epoch,
                            cascade_metrics=cascade_metrics,
                            roi_model_name=roi_model_name_for_result
                        )
                        all_results.append(result)
                        
                        # Stage별 PAM 결과 저장
                        if pam_train_stages:
                            for stage_name, mem_list in pam_train_stages.items():
                                stage_result = create_stage_pam_result(
                                    dataset_version=dataset_version,
                                    seed=seed,
                                    fold_idx=fold_idx if use_5fold else None,
                                    model_name=model_name,
                                    mode='train',
                                    stage_name=stage_name,
                                    mem_list=mem_list
                                )
                                if stage_result:
                                    all_stage_pam_results.append(stage_result)
                        
                        if pam_inference_stages:
                            for stage_name, mem_list in pam_inference_stages.items():
                                stage_result = create_stage_pam_result(
                                    dataset_version=dataset_version,
                                    seed=seed,
                                    fold_idx=fold_idx if use_5fold else None,
                                    model_name=model_name,
                                    mode='inference',
                                    stage_name=stage_name,
                                    mem_list=mem_list
                                )
                                if stage_result:
                                    all_stage_pam_results.append(stage_result)
                    
                    # 모든 epoch 결과 저장 (test_dice는 최종 평가 값으로 업데이트)
                    if is_main_process(rank):
                        for epoch_result in epoch_results:
                            epoch_data = create_epoch_result_dict(
                                dataset_version=dataset_version,
                                seed=seed,
                                fold_idx=fold_idx if use_5fold else None,
                                model_name=model_name,
                                epoch_result=epoch_result,
                                best_epoch=best_epoch,
                                test_dice=metrics['dice']
                            )
                            all_epochs_results.append(epoch_data)
                    
                    if is_main_process(rank):
                        print(f"Final Val Dice: {best_val_dice:.4f} (WT {best_val_wt:.4f} | TC {best_val_tc:.4f} | ET {best_val_et:.4f}) (epoch {best_epoch})")
                        print(f"Final Test Dice: {metrics['dice']:.4f} (WT {metrics['wt']:.4f} | TC {metrics['tc']:.4f} | ET {metrics['et']:.4f}) | Prec {metrics['precision']:.4f} Rec {metrics['recall']:.4f}")
                
                except Exception as e:
                    # 모든 프로세스에서 에러 로깅 (디버깅을 위해)
                    import traceback
                    error_msg = f"[Rank {rank}] Error with {model_name}: {e}"
                    print(error_msg)
                    print(f"[Rank {rank}] Full traceback:")
                    traceback.print_exc()
                    # Main process에서만 추가 정보 출력
                    if is_main_process(rank):
                        print(f"\n{'='*60}")
                        print(f"FAILED: {model_name.upper()}")
                        print(f"Error: {e}")
                        print(f"{'='*60}\n")
                    continue
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    # 결과가 비어있는 경우 처리
    if results_df.empty:
        if is_main_process(rank) or not distributed:
            print("\n" + "="*80)
            print("WARNING: No results were collected. All experiments failed.")
            print("="*80)
            print("Please check the error messages above for details.")
            print("Common issues:")
            print("  - Model import errors (check if model file exists)")
            print("  - Invalid model name or parameters")
            print("  - Data loading errors")
            print("  - CUDA/device errors")
            print("="*80 + "\n")
        return results_dir, results_df
    
    # CSV로 저장
    save_results_to_csv(
        results_dir=results_dir,
        all_results=all_results,
        all_epochs_results=all_epochs_results,
        all_stage_pam_results=all_stage_pam_results,
        is_main_process=(is_main_process(0) or not distributed)
    )
    
    # Epochs DataFrame 생성 (분석용)
    epochs_df = pd.DataFrame(all_epochs_results) if all_epochs_results else None
    
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


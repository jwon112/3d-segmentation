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
from utils.runner.cascade_evaluation import load_roi_model_from_checkpoint, evaluate_cascade_pipeline
from dataloaders import get_brats_base_datasets


def get_roi_center_prob_schedule(current_epoch: int, total_epochs: int, schedule_type: str = 'cosine', warmup_ratio: float = 0.0):
    """
    Epoch 진행률에 따라 ROI 중심 사용 확률을 점진적으로 증가시킴
    
    Args:
        current_epoch: 현재 epoch (0부터 시작)
        total_epochs: 전체 epoch 수
        schedule_type: 'cosine', 'linear', 'quadratic', 'sqrt' 중 선택
        warmup_ratio: 초반 이 비율만큼은 GT 중심만 사용 (0.0 ~ 1.0)
    
    Returns:
        ROI 중심 사용 확률 (0.0 ~ 1.0)
    """
    if total_epochs <= 1:
        return 0.0
    
    warmup_epochs = int(total_epochs * warmup_ratio)
    
    # Warmup 구간: GT 중심만 사용
    if current_epoch < warmup_epochs:
        return 0.0
    
    # Warmup 이후 진행률 계산
    remaining_epochs = total_epochs - warmup_epochs
    if remaining_epochs <= 1:
        return 1.0
    
    progress = (current_epoch - warmup_epochs) / (remaining_epochs - 1)
    
    if schedule_type == 'cosine':
        # Cosine annealing 스타일: 초반에는 천천히, 후반에 빠르게 증가
        # 1 - cos(π * progress / 2) 형태로, 초반에는 완만하게 증가
        roi_prob = 1.0 - np.cos(np.pi * progress / 2.0)
    elif schedule_type == 'linear':
        # 선형 증가: 일정하게 증가
        roi_prob = progress
    elif schedule_type == 'quadratic':
        # 2차 함수: 초반에는 매우 천천히, 후반에 빠르게
        roi_prob = progress ** 2
    elif schedule_type == 'sqrt':
        # 제곱근: 초반에 빠르게, 후반에 완만하게
        roi_prob = np.sqrt(progress)
    else:
        # 기본값: 선형
        roi_prob = progress
    
    return float(roi_prob)


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
                sw_patch_size=(128, 128, 128), sw_overlap=0.5, dim='3d', use_nnunet_loss=True, results_dir=None, ckpt_path=None, train_crops_per_center=1, dataset_version='brats2021',
                data_dir=None, cascade_infer_cfg=None, coord_type='none', preprocessed_dir=None, use_5fold=False, fold_idx=None, fold_split_dir=None,
                roi_center_schedule='cosine', roi_center_warmup_ratio=0.0):
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
    best_val_wt = best_val_tc = best_val_et = best_val_rc = 0.0
    epochs_without_improvement = 0  # Early stopping을 위한 카운터
    early_stopping_patience = 20  # 20 epoch 동안 개선 없으면 중단
    is_brats2024 = (dataset_version == 'brats2024')
    
    # Cascade 모델인지 확인
    is_cascade_model = model_name.startswith('cascade_')
    
    # Cascade 모델인 경우 ROI 모델 및 base dataset 준비
    roi_model = None
    val_base_dataset = None
    roi_use_4modalities = True
    if is_cascade_model:
        if data_dir is None:
            if is_main_process(rank):
                print(f"[Cascade Training] Warning: data_dir not provided. Validation will use crop-based evaluation.")
        else:
            # ROI 모델 경로 찾기
            possible_roi_model_names = ['roi_unet3d_small', 'roi_mobileunetr3d_tiny', 'roi_mobileunetr3d_small']
            roi_model_name = None
            roi_weight_path = None
            
            # results_dir에서 ROI 모델 체크포인트 찾기
            if results_dir:
                for candidate_roi_name in possible_roi_model_names:
                    roi_checkpoint_patterns = [
                        os.path.join(results_dir, f"{candidate_roi_name}_seed_{seed}_best.pth"),
                    ]
                    for pattern in roi_checkpoint_patterns:
                        if pattern and os.path.exists(pattern):
                            roi_weight_path = pattern
                            roi_model_name = candidate_roi_name
                            break
                    if roi_weight_path:
                        break
            
            # 기본 경로 시도
            if not roi_weight_path:
                for candidate_roi_name in possible_roi_model_names:
                    default_path = f"models/weights/cascade/roi_model/{candidate_roi_name}/seed_{seed}/weights/best.pth"
                    if os.path.exists(default_path):
                        roi_weight_path = default_path
                        roi_model_name = candidate_roi_name
                        break
            
            if roi_weight_path and os.path.exists(roi_weight_path):
                if is_main_process(rank):
                    print(f"[Cascade Training] Loading ROI model from {roi_weight_path}")
                try:
                    roi_model, roi_use_4modalities = load_roi_model_from_checkpoint(roi_model_name, roi_weight_path, device)
                    roi_model.eval()
                except Exception as e:
                    if is_main_process(rank):
                        print(f"[Cascade Training] Warning: Failed to load ROI model: {e}")
                        print(f"[Cascade Training] Validation will use crop-based evaluation.")
                    roi_model = None
            else:
                if is_main_process(rank):
                    print(f"[Cascade Training] Warning: ROI model not found. Validation will use crop-based evaluation.")
            
            # Validation용 base dataset 가져오기
            try:
                _, val_base_dataset, _ = get_brats_base_datasets(
                    data_dir=data_dir,
                    dataset_version=dataset_version,
                    seed=seed,
                    use_4modalities=True,
                    preprocessed_dir=preprocessed_dir,
                    use_5fold=use_5fold,
                    fold_idx=fold_idx,
                    fold_split_dir=fold_split_dir,
                )
            except Exception as e:
                if is_main_process(rank):
                    print(f"[Cascade Training] Warning: Failed to load base dataset for validation: {e}")
                    print(f"[Cascade Training] Validation will use crop-based evaluation.")
                val_base_dataset = None
    
    # Dataset에 ROI 모델 설정 (Training용)
    if is_cascade_model and roi_model is not None:
        if hasattr(train_loader.dataset, 'roi_model'):
            train_loader.dataset.roi_model = roi_model
        if hasattr(train_loader.dataset, 'roi_use_4modalities'):
            train_loader.dataset.roi_use_4modalities = roi_use_4modalities
        if hasattr(train_loader.dataset, 'set_roi_center_prob'):
            # 초기에는 GT 중심만 사용
            train_loader.dataset.set_roi_center_prob(0.0)
    
    # 체크포인트 저장 경로 (실험 결과 폴더 내부)
    if results_dir is None:
        results_dir = "experiment_result"
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
    
    for epoch in range(epochs):
        # 각 epoch 시작 시 seed 재설정하여 재현성 보장 (Stochastic depth 등 랜덤 연산 포함)
        # base_seed + epoch을 사용하여 각 epoch마다 다른 seed를 가지지만, 같은 seed로 시작하면 같은 순서로 재현 가능
        epoch_seed = seed + epoch
        set_seed(epoch_seed)
        
        # ROI 중심 비율 점진적 증가 (스케줄러 방식)
        if is_cascade_model and roi_model is not None:
            if hasattr(train_loader.dataset, 'set_roi_center_prob'):
                roi_prob = get_roi_center_prob_schedule(
                    current_epoch=epoch,
                    total_epochs=epochs,
                    schedule_type=roi_center_schedule,
                    warmup_ratio=roi_center_warmup_ratio
                )
                train_loader.dataset.set_roi_center_prob(roi_prob)
                # 10% 간격 또는 매 epoch 출력 (epochs가 10 이하인 경우)
                print_interval = max(1, epochs // 10) if epochs > 10 else 1
                if is_main_process(rank) and epoch % print_interval == 0:
                    print(f"[Curriculum Learning] Epoch {epoch+1}/{epochs}: ROI center probability = {roi_prob:.3f} (schedule: {roi_center_schedule})")
        
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
        train_iter = iter(train_loader)
        for step in tqdm(range(len(train_loader)), desc=f"Train {epoch+1}/{epochs}", leave=False):
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
            
            torch.cuda.synchronize()
            t_fwd = time.time()
            fwd_times.append(t_fwd - t_load)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            t_bwd = time.time()
            bwd_times.append(t_bwd - t_fwd)
            
            # BraTS composite Dice (WT, TC, ET, RC for BRATS2024)
            dice_scores = calculate_wt_tc_et_dice(logits.detach(), labels, dataset_version=dataset_version)
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
        
        # Cascade 모델인 경우 test와 동일한 방식으로 validation 수행
        if is_cascade_model and roi_model is not None and val_base_dataset is not None:
            if is_main_process(rank):
                print(f"[Cascade Training] Using cascade pipeline for validation (same as test)...")
            
            # Cascade inference 설정 가져오기
            eval_crops_per_center = cascade_infer_cfg.get('crops_per_center', 1) if cascade_infer_cfg else 1
            eval_crop_overlap = cascade_infer_cfg.get('crop_overlap', 0.5) if cascade_infer_cfg else 0.5
            eval_use_blending = cascade_infer_cfg.get('use_blending', True) if cascade_infer_cfg else True
            eval_batch_size = cascade_infer_cfg.get('batch_size', 1) if cascade_infer_cfg else 1
            eval_roi_batch_size = cascade_infer_cfg.get('roi_batch_size', None) if cascade_infer_cfg else None
            
            # coord_type에 따라 include_coords와 coord_encoding_type 결정
            if coord_type == 'none':
                include_coords = False
                coord_encoding_type = 'simple'
            elif coord_type == 'simple':
                include_coords = True
                coord_encoding_type = 'simple'
            elif coord_type == 'hybrid':
                include_coords = True
                coord_encoding_type = 'hybrid'
            else:
                include_coords = False
                coord_encoding_type = 'simple'
            
            # Cascade pipeline으로 validation 수행
            with torch.no_grad():
                # DDP 설정 확인
                distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
                world_size = torch.distributed.get_world_size() if distributed else 1
                
                cascade_result = evaluate_cascade_pipeline(
                    roi_model=roi_model,
                    seg_model=model,
                    base_dataset=val_base_dataset,
                    device=device,
                    roi_resize=(64, 64, 64),
                    crop_size=(96, 96, 96),
                    include_coords=include_coords,
                    coord_encoding_type=coord_encoding_type,
                    crops_per_center=eval_crops_per_center,
                    crop_overlap=eval_crop_overlap,
                    use_blending=eval_use_blending,
                    collect_attention=False,
                    results_dir=None,
                    model_name=model_name,
                    dataset_version=dataset_version,
                    roi_use_4modalities=True,  # val_base_dataset은 항상 4 modalities
                    batch_size=eval_batch_size,  # cascade_infer_cfg에서 가져온 값
                    roi_batch_size=eval_roi_batch_size,  # cascade_infer_cfg에서 가져온 값
                    distributed=distributed,
                    world_size=world_size,
                    rank=rank,
                    use_nnunet_loss=use_nnunet_loss,  # Training에서 사용하는 loss 타입 전달
                )
                
                # 결과 추출
                va_dice = cascade_result.get('mean', 0.0)  # 'mean' 키 사용 (evaluate_cascade_pipeline이 반환하는 키)
                va_wt = cascade_result.get('wt', 0.0)
                va_tc = cascade_result.get('tc', 0.0)
                va_et = cascade_result.get('et', 0.0)
                va_rc = cascade_result.get('rc', 0.0) if is_brats2024 else 0.0
                va_loss = cascade_result.get('loss', 0.0)  # Cascade pipeline에서 계산된 loss 사용
                
                # 디버깅: cascade_result 내용 확인 (모든 epoch에서 출력)
                if is_main_process(rank):
                    print(f"[Training Debug] Epoch {epoch+1} - cascade_result keys: {list(cascade_result.keys())}")
                    print(f"[Training Debug] Epoch {epoch+1} - cascade_result['loss']: {cascade_result.get('loss', 'NOT FOUND')}")
                    print(f"[Training Debug] Epoch {epoch+1} - va_loss immediately after get: {va_loss}")
                
                # 평균 계산 (샘플 수는 base_dataset 길이)
                n_va = len(val_base_dataset)
                va_dice_sum = va_dice * n_va
                va_wt_sum = va_wt * n_va
                va_tc_sum = va_tc * n_va
                va_et_sum = va_et * n_va
                if is_brats2024:
                    va_rc_sum = va_rc * n_va
        else:
            # 일반 모델 또는 cascade 모델이지만 ROI/base dataset이 없는 경우 기존 방식 사용
            with torch.no_grad():
                debug_printed = False
                all_sample_dices = []  # 디버깅: 모든 샘플의 Dice 수집
                for idx, batch_data in enumerate(tqdm(val_loader, desc=f"Val   {epoch+1}/{epochs}", leave=False)):
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
                    loss = criterion(logits, labels)
                    dice_scores = calculate_wt_tc_et_dice(logits, labels, dataset_version=dataset_version)
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
                
                # 디버깅: 모든 샘플의 Dice 통계 출력 (일반 모델만)
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
        
        # Cascade 모델의 경우 va_loss는 이미 평균값이므로 다시 나누지 않음
        # 일반 모델의 경우만 va_loss를 나눔
        if not is_cascade_model or val_base_dataset is None:
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
        
        # Learning rate scheduling (ReduceLROnPlateau는 validation metric 필요)
        # nnU-Net 스타일: validation loss를 모니터링
        if is_main_process(rank):
            print(f"[Training Debug] Epoch {epoch+1} - va_loss before scheduler.step: {va_loss}")
        scheduler.step(va_loss)
        
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
                torch.save(model_to_save.state_dict(), ckpt_path)
                print(f"[Epoch {epoch+1}] Saved best checkpoint (Val Dice: {va_dice:.4f}) to {ckpt_path}")
        else:
            epochs_without_improvement += 1  # 개선 없음
        
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


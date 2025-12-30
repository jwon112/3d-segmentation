"""
Cascade Evaluation
Cascade ROI→Seg 파이프라인 평가 관련 함수
"""

import os
import time
import json
import torch

from utils.experiment_utils import get_roi_model, is_main_process
from utils.experiment_config import get_roi_model_config
from utils.cascade_utils import run_cascade_inference
from dataloaders import get_brats_base_datasets
from metrics import calculate_wt_tc_et_dice


def evaluate_cascade_pipeline(roi_model, seg_model, base_dataset, device,
                              roi_resize=(64, 64, 64), crop_size=(96, 96, 96), include_coords=True,
                              coord_encoding_type='simple',
                              crops_per_center=1, crop_overlap=0.5, use_blending=True,
                              collect_attention=False, results_dir=None, model_name='model', dataset_version='brats2021',
                              roi_use_4modalities=True):
    """
    Run cascade inference on base dataset and compute WT/TC/ET dice.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
        collect_attention: True면 MobileViT attention weights 수집 및 분석
        results_dir: 결과 저장 디렉토리 (attention 분석용)
        model_name: 모델 이름 (attention 분석용)
    """
    roi_model.eval()
    seg_model.eval()
    dice_rows = []
    all_attention_weights = [] if collect_attention else None
    
    # 진행 상황 로그를 위한 rank 확인
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    
    total_samples = len(base_dataset)
    if is_main_process(rank):
        print(f"[Cascade Evaluation] Starting evaluation on {total_samples} samples...")
    
    # 시간 측정용 변수
    total_data_load_time = 0.0
    total_roi_time = 0.0
    total_seg_time = 0.0
    total_dice_time = 0.0
    
    for idx in range(len(base_dataset)):
        sample_start_time = time.time()
        
        # 진행 상황 로그 (10개마다 또는 첫 번째/마지막 샘플)
        if is_main_process(rank) and (idx == 0 or idx % 10 == 0 or idx == total_samples - 1):
            print(f"[Cascade Evaluation] Processing sample {idx+1}/{total_samples}...")
        
        try:
            # 데이터 로드 시간 측정
            data_load_start = time.time()
            # base_dataset이 (image, mask) 또는 (image, mask, fg_coords_dict) 반환 가능
            loaded_data = base_dataset[idx]
            if len(loaded_data) == 3:
                image, target, _ = loaded_data  # fg_coords_dict는 evaluation에서는 사용 안 함
            else:
                image, target = loaded_data
            image = image.to(device)
            target = target.to(device)
            data_load_time = time.time() - data_load_start
            total_data_load_time += data_load_time
            
            if is_main_process(rank) and idx == 0:
                print(f"[Cascade Evaluation] Sample {idx+1}: image.shape={image.shape}, target.shape={target.shape}")
                print(f"[Cascade Evaluation] Sample {idx+1}: Data load time: {data_load_time:.3f}s")
                # 볼륨 크기 확인 (디버깅용)
                print(f"[Cascade Evaluation] Sample {idx+1}: Volume size (H, W, D) = {image.shape[1:]}")
            
            # Cascade inference 시간 측정
            inference_start = time.time()
            result = run_cascade_inference(
                roi_model=roi_model,
                seg_model=seg_model,
                image=image,
                device=device,
                roi_resize=roi_resize,
                crop_size=crop_size,
                include_coords=include_coords,
                coord_encoding_type=coord_encoding_type,
                crops_per_center=crops_per_center,
                crop_overlap=crop_overlap,
                use_blending=use_blending,
                return_attention=collect_attention,
                roi_use_4modalities=roi_use_4modalities,
                return_timing=True,
                debug_sample_idx=idx,  # 첫 번째 샘플(idx==0)에 대해서만 로그 출력
            )
            inference_time = time.time() - inference_start
            
            # GPU 메모리 정리 (각 샘플 처리 후)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # ROI와 Segmentation 시간 분리
            timing_info = result.get('timing', {})
            roi_time = timing_info.get('roi_time', 0.0)
            seg_time = timing_info.get('seg_time', 0.0)
            num_centers = timing_info.get('num_centers', 1)
            num_crops = timing_info.get('num_crops', 1)
            
            total_roi_time += roi_time
            total_seg_time += seg_time
            
            # 첫 번째 샘플: 실제 종양 위치와 ROI 중심의 거리 계산
            if idx == 0 and is_main_process(rank):
                try:
                    import numpy as np
                    import os
                    log_dir = os.path.join(os.getcwd(), '.cursor')
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, 'debug.log')
                    
                    # 실제 종양 중심 계산
                    target_np = target.cpu().numpy()  # (H, W, D)
                    
                    # WT 마스크 (class > 0)
                    wt_mask = (target_np > 0).astype(np.float32)
                    
                    # 종양 중심 계산 (voxel 좌표)
                    tumor_center = None
                    tumor_volume = int(wt_mask.sum())
                    if tumor_volume > 0:
                        h_coords, w_coords, d_coords = np.where(wt_mask > 0)
                        tumor_center = (
                            float(h_coords.mean()),
                            float(w_coords.mean()),
                            float(d_coords.mean())
                        )
                        
                        # ROI 중심들과의 거리 계산
                        roi_info = result.get('roi', {})
                        roi_centers = roi_info.get('centers_full', [])
                        if not roi_centers:
                            # backward compatibility
                            center_full = roi_info.get('center_full')
                            if center_full:
                                roi_centers = [center_full]
                        
                        distances = []
                        for roi_center in roi_centers:
                            dist = np.sqrt(
                                (tumor_center[0] - roi_center[0])**2 +
                                (tumor_center[1] - roi_center[1])**2 +
                                (tumor_center[2] - roi_center[2])**2
                            )
                            distances.append(float(dist))
                        
                        with open(log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "tumor-center-check",
                                "hypothesisId": "H11",
                                "location": "cascade_evaluation.py:103",
                                "message": "Tumor center vs ROI centers distance",
                                "data": {
                                    "tumor_center": list(tumor_center),
                                    "tumor_volume": tumor_volume,
                                    "roi_centers": [[float(c) for c in center] for center in roi_centers],
                                    "distances": distances,
                                    "volume_shape": list(target_np.shape)
                                },
                                "timestamp": int(time.time() * 1000)
                            }, ensure_ascii=False) + "\n")
                            log_file.flush()
                except Exception as e:
                    pass
            
            if is_main_process(rank) and (idx == 0 or idx % 10 == 0):
                print(f"[Cascade Evaluation] Sample {idx+1}: ROI={roi_time:.3f}s, Seg={seg_time:.3f}s (centers={num_centers}, crops={num_crops}), Total={inference_time:.3f}s")
        except Exception as e:
            error_msg = f"[Cascade Evaluation] Error processing sample {idx+1}/{total_samples}: {type(e).__name__}: {e}"
            if is_main_process(rank):
                print(error_msg)
                import traceback
                traceback.print_exc()
            raise
        
        # Attention weights 수집 (dice 계산 전에 먼저 수집하여, 에러가 발생해도 보존)
        if collect_attention and all_attention_weights is not None:
            if 'attention_weights' in result and result['attention_weights']:
                all_attention_weights.extend(result['attention_weights'])
                if idx == 0 and is_main_process(rank):  # 첫 번째 샘플만 출력
                    print(f"[Cascade Evaluation] Collected attention weights from sample {idx+1}/{total_samples}")
        
        # full_logits shape 처리: run_cascade_inference는 (C, H, W, D) 형태를 반환
        full_logits_raw = result['full_logits'].to(device)
        if full_logits_raw.dim() == 4:  # (C, H, W, D) - 정상
            full_logits = full_logits_raw.unsqueeze(0)  # (1, C, H, W, D)
        elif full_logits_raw.dim() == 5:  # 이미 (1, C, H, W, D) - 이미 배치 차원 있음
            full_logits = full_logits_raw
        elif full_logits_raw.dim() == 6:  # (1, 1, C, H, W, D) - 배치 차원 중복
            # 첫 번째 배치 차원 제거
            full_logits = full_logits_raw.squeeze(0)  # (1, C, H, W, D)
            if full_logits.dim() == 6:
                # 여전히 6차원이면 다시 squeeze
                full_logits = full_logits.squeeze(0)  # (C, H, W, D)
                full_logits = full_logits.unsqueeze(0)  # (1, C, H, W, D)
        else:
            # 예상치 못한 shape인 경우 처리
            raise ValueError(f"Unexpected full_logits shape: {full_logits_raw.shape}, expected (C, H, W, D), (1, C, H, W, D), or (1, 1, C, H, W, D)")
        
        # target이 배치 차원이 없으면 추가, 있으면 그대로 사용
        if target.dim() == 3:  # (H, W, D)
            target_batch = target.unsqueeze(0)  # (1, H, W, D)
        elif target.dim() == 4:  # 이미 (1, H, W, D)
            target_batch = target
        else:
            target_batch = target.unsqueeze(0)  # 안전장치
        
        # Shape 확인 및 디버깅 정보 (첫 번째 샘플)
        if idx == 0:  # 첫 번째 샘플만 출력
            if is_main_process(rank):
                print(f"[Cascade Evaluation] full_logits.shape={full_logits.shape}, target_batch.shape={target_batch.shape}")
                # #region agent log
                import os
                log_dir = os.path.join(os.getcwd(), '.cursor')
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, 'debug.log')
                try:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "eval-check",
                            "hypothesisId": "H1",
                            "location": "cascade_evaluation.py:156",
                            "message": "Before Dice calculation - shapes and value ranges",
                            "data": {
                                "full_logits_shape": list(full_logits.shape),
                                "target_batch_shape": list(target_batch.shape),
                                "full_logits_min": float(full_logits.min().item()),
                                "full_logits_max": float(full_logits.max().item()),
                                "full_logits_mean": float(full_logits.mean().item()),
                                "target_batch_min": int(target_batch.min().item()),
                                "target_batch_max": int(target_batch.max().item()),
                                "target_batch_unique": [int(x) for x in torch.unique(target_batch).cpu().tolist()],
                                "dataset_version": dataset_version
                            },
                            "timestamp": int(time.time() * 1000)
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # Prediction 확인
                pred_argmax = torch.argmax(full_logits, dim=1)
                pred_unique = torch.unique(pred_argmax).cpu().tolist()
                target_unique = torch.unique(target_batch).cpu().tolist()
                print(f"[Cascade Evaluation] Sample {idx+1}: pred unique classes={pred_unique}, target unique classes={target_unique}")
                print(f"[Cascade Evaluation] Sample {idx+1}: pred class counts={dict(zip(*torch.unique(pred_argmax, return_counts=True)))}")
                print(f"[Cascade Evaluation] Sample {idx+1}: target class counts={dict(zip(*torch.unique(target_batch, return_counts=True)))}")
        
        try:
            dice_start = time.time()
            dice = calculate_wt_tc_et_dice(full_logits, target_batch, dataset_version=dataset_version, sample_idx=idx).detach().cpu()
            dice_time = time.time() - dice_start
            total_dice_time += dice_time
            
            # #region agent log
            if idx == 0:
                try:
                    import os
                    log_dir = os.path.join(os.getcwd(), '.cursor')
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, 'debug.log')
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "eval-check",
                            "hypothesisId": "H2",
                            "location": "cascade_evaluation.py:160",
                            "message": "After Dice calculation",
                            "data": {
                                "dice_values": [float(x) for x in dice.tolist()],
                                "dice_shape": list(dice.shape),
                                "dataset_version": dataset_version
                            },
                            "timestamp": int(time.time() * 1000)
                        }, ensure_ascii=False) + "\n")
                        log_file.flush()
                except Exception:
                    pass
            # #endregion
            
            dice_rows.append(dice)
            
            sample_total_time = time.time() - sample_start_time
            if is_main_process(rank) and (idx == 0 or idx % 10 == 0 or idx == total_samples - 1):
                if dataset_version == 'brats2024':
                    print(f"[Cascade Evaluation] Sample {idx+1}: Dice - WT={dice[0]:.4f}, TC={dice[1]:.4f}, ET={dice[2]:.4f}, RC={dice[3]:.4f} | Total: {sample_total_time:.3f}s")
                else:
                    print(f"[Cascade Evaluation] Sample {idx+1}: Dice - WT={dice[0]:.4f}, TC={dice[1]:.4f}, ET={dice[2]:.4f} | Total: {sample_total_time:.3f}s")
        except Exception as e:
            error_msg = f"[Cascade Evaluation] Error calculating dice for sample {idx+1}/{total_samples}: {type(e).__name__}: {e}"
            if is_main_process(rank):
                print(error_msg)
                import traceback
                traceback.print_exc()
            raise
    
    if is_main_process(rank):
        total_eval_time = total_data_load_time + total_seg_time + total_dice_time
        print(f"[Cascade Evaluation] Completed processing all {total_samples} samples.")
        print(f"[Cascade Evaluation] Timing summary:")
        print(f"  - Data loading: {total_data_load_time:.2f}s (avg: {total_data_load_time/total_samples:.3f}s/sample, {total_data_load_time/total_eval_time*100:.1f}%)")
        print(f"  - ROI localization: {total_roi_time:.2f}s (avg: {total_roi_time/total_samples:.3f}s/sample, {total_roi_time/total_eval_time*100:.1f}%)")
        print(f"  - Segmentation inference: {total_seg_time:.2f}s (avg: {total_seg_time/total_samples:.3f}s/sample, {total_seg_time/total_eval_time*100:.1f}%)")
        print(f"  - Dice calculation: {total_dice_time:.2f}s (avg: {total_dice_time/total_samples:.3f}s/sample, {total_dice_time/total_eval_time*100:.1f}%)")
        print(f"  - Total: {total_eval_time:.2f}s")
        print(f"[Cascade Evaluation] Calculating final metrics...")
    
    is_brats2024 = (dataset_version == 'brats2024')
    if not dice_rows:
        if is_main_process(rank):
            print(f"[Cascade Evaluation] Warning: No dice scores calculated. Returning zero metrics.")
        if is_brats2024:
            return {'wt': 0.0, 'tc': 0.0, 'et': 0.0, 'rc': 0.0, 'mean': 0.0}
        else:
            return {'wt': 0.0, 'tc': 0.0, 'et': 0.0, 'mean': 0.0}
    dice_tensor = torch.stack(dice_rows, dim=0)
    mean_scores = dice_tensor.mean(dim=0)
    
    if is_main_process(rank):
        if is_brats2024:
            print(f"[Cascade Evaluation] Final metrics - WT={mean_scores[0]:.4f}, TC={mean_scores[1]:.4f}, ET={mean_scores[2]:.4f}, RC={mean_scores[3]:.4f}, Mean={mean_scores.mean():.4f}")
        else:
            print(f"[Cascade Evaluation] Final metrics - WT={mean_scores[0]:.4f}, TC={mean_scores[1]:.4f}, ET={mean_scores[2]:.4f}, Mean={mean_scores.mean():.4f}")
    
    # MobileViT attention 분석
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    
    if collect_attention:
        if all_attention_weights and len(all_attention_weights) > 0:
            if is_main_process(rank):
                print(f"\n[Cascade Evaluation] Analyzing {len(all_attention_weights)} attention weight samples...")
            if results_dir:
                try:
                    from utils.mvit_attention_utils import analyze_mvit_attention_weights, check_mvit_attention_learning
                    
                    analysis_result = analyze_mvit_attention_weights(
                        all_attention_weights,
                        results_dir=results_dir,
                        model_name=model_name,
                    )
                    
                    is_learning, message = check_mvit_attention_learning(all_attention_weights)
                    if is_main_process(rank):
                        print(f"\nMobileViT Attention Learning Status (Cascade): {message}")
                        if not is_learning:
                            print(f"⚠️  Warning: MobileViT attention may not be learning properly!")
                except Exception as e:
                    if is_main_process(rank):
                        print(f"Warning: Failed to analyze/save MobileViT attention weights: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                if is_main_process(rank):
                    print(f"[Cascade Evaluation] Warning: results_dir is None, skipping attention analysis")
        else:
            if is_main_process(rank):
                print(f"[Cascade Evaluation] Warning: No attention weights collected (collect_attention={collect_attention}, len={len(all_attention_weights) if all_attention_weights else 0})")
    
    result = {
        'wt': float(mean_scores[0].item()),
        'tc': float(mean_scores[1].item()),
        'et': float(mean_scores[2].item()),
        'mean': float(mean_scores.mean().item())
    }
    if is_brats2024 and len(mean_scores) >= 4:
        result['rc'] = float(mean_scores[3].item())
    return result


def load_roi_model_from_checkpoint(roi_model_name, weight_path, device):
    """Load ROI model weights for inference.
    
    ROI 모델은 항상 4채널(4 modalities, no coords) 또는 2채널(2 modalities, no coords)만 사용합니다.
    Automatically detects use_4modalities from checkpoint if available, otherwise falls back to channel detection.
    
    Returns:
        model: Loaded ROI model
        use_4modalities: Detected or default use_4modalities value (True for 4 modalities, False for 2 modalities)
    """
    cfg = get_roi_model_config(roi_model_name)
    
    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location=device)
    
    # Check if checkpoint has metadata (new format) or is just state_dict (old format)
    # ROI 모델은 항상 coords를 사용하지 않으므로 use_4modalities만 감지
    if isinstance(checkpoint, dict) and 'metadata' in checkpoint:
        # New format with metadata
        metadata = checkpoint['metadata']
        state = checkpoint['state_dict']
        use_4modalities = metadata.get('use_4modalities', True)
        # rank 확인
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if is_main_process(rank):
            print(f"Loaded ROI model with metadata: use_4modalities={use_4modalities} (4 channels, no coords)")
    else:
        # Old format: just state_dict, detect from channels
        state = checkpoint
        
        # Detect input channels from checkpoint
        # ROI 모델은 항상 2채널 또는 4채널만 사용 (coords 없음)
        detected_channels = None
        for key in state.keys():
            if 'weight' in key and ('enc_blocks.0.net.0' in key or 'patch_embed' in key or 'conv' in key):
                if 'enc_blocks.0.net.0.weight' in key:
                    detected_channels = state[key].shape[1]
                    break
                elif 'patch_embed' in key and 'weight' in key:
                    detected_channels = state[key].shape[1]
                    break
        
        # Auto-detect use_4modalities based on detected channels
        # 2 channels = 2 modalities, no coords
        # 4 channels = 4 modalities, no coords
        # rank 확인
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        
        if detected_channels is not None:
            if detected_channels == 2:
                use_4modalities = False
                if is_main_process(rank):
                    print(f"Detected 2-channel ROI model (2 modalities, no coords). Using use_4modalities=False")
            elif detected_channels == 4:
                use_4modalities = True
                if is_main_process(rank):
                    print(f"Detected 4-channel ROI model (4 modalities, no coords). Using use_4modalities=True")
            else:
                # 기본값 사용 (4 modalities)
                use_4modalities = True
                if is_main_process(rank):
                    print(f"Warning: Unexpected input channels {detected_channels} in ROI checkpoint. Expected 2 or 4. Using default: use_4modalities=True")
        else:
            # 채널을 감지하지 못한 경우 기본값 사용
            use_4modalities = True
            if is_main_process(rank):
                print(f"Warning: Could not detect input channels from ROI checkpoint. Using default: use_4modalities=True")
    
    # ROI 모델 입력 채널 수 계산 (modalities만, coords 없음)
    n_channels = 4 if use_4modalities else 2
    
    model = get_roi_model(
        roi_model_name,
        n_channels=n_channels,
        n_classes=2,
        roi_model_cfg=cfg,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, use_4modalities


def evaluate_segmentation_with_roi(
    seg_model,
    roi_model,
    data_dir,
    dataset_version,
    seed,
    roi_resize=(64, 64, 64),
    crop_size=(96, 96, 96),
    include_coords=True,
    coord_encoding_type='simple',
    use_5fold=False,
    fold_idx=None,
    fold_split_dir=None,
    max_samples=None,
    crops_per_center=1,
    crop_overlap=0.5,
    use_blending=True,
    results_dir=None,
    model_name='model',
    preprocessed_dir=None,
    roi_use_4modalities=True,
):
    """
    Evaluate trained segmentation model with pre-trained ROI detector.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
        results_dir: 결과 저장 디렉토리 (MobileViT attention 분석용)
        model_name: 모델 이름 (MobileViT attention 분석용)
        fold_split_dir: fold split 디렉토리 경로 (지정 시 해당 fold의 train/val/test 사용)
    """
    _, _, test_base = get_brats_base_datasets(
        data_dir=data_dir,
        dataset_version=dataset_version,
        max_samples=max_samples,
        seed=seed,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        fold_split_dir=fold_split_dir,
        use_4modalities=True,
        preprocessed_dir=preprocessed_dir,
    )
    seg_model.eval()
    roi_model.eval()
    
    # MobileViT attention 수집 여부 확인
    collect_attention = False
    if results_dir is not None:
        try:
            from models.modules.mvit_modules import MobileViT3DBlock, MobileViT3DBlockV3
            real_model = seg_model.module if hasattr(seg_model, 'module') else seg_model
            mvit_blocks_found = []
            for name, module in real_model.named_modules():
                if isinstance(module, (MobileViT3DBlock, MobileViT3DBlockV3)):
                    mvit_blocks_found.append((name, module))
            
            import inspect
            sig = inspect.signature(real_model.forward)
            has_return_attention = 'return_attention' in sig.parameters
            
            if len(mvit_blocks_found) > 0 and has_return_attention:
                collect_attention = True
                print(f"[Cascade MobileViT] Found {len(mvit_blocks_found)} MobileViT blocks, will collect attention weights")
        except Exception as e:
            print(f"[Cascade MobileViT] Error checking for MobileViT blocks: {e}")
    
    return evaluate_cascade_pipeline(
        roi_model=roi_model,
        seg_model=seg_model,
        base_dataset=test_base,
        device=next(seg_model.parameters()).device,
        roi_resize=roi_resize,
        crop_size=crop_size,
        include_coords=include_coords,
        coord_encoding_type=coord_encoding_type,
        crops_per_center=crops_per_center,
        crop_overlap=crop_overlap,
        use_blending=use_blending,
        collect_attention=collect_attention,
        results_dir=results_dir,
        model_name=model_name,
        dataset_version=dataset_version,
        roi_use_4modalities=roi_use_4modalities,
    )


"""
Cascade Evaluation
Cascade ROI→Seg 파이프라인 평가 관련 함수
"""

import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from utils.experiment_utils import get_roi_model, is_main_process
from utils.experiment_config import get_roi_model_config
from utils.cascade_utils import run_cascade_inference, run_cascade_inference_batch, _generate_multi_crop_centers
from dataloaders import get_brats_base_datasets
from metrics import calculate_wt_tc_et_dice


def evaluate_cascade_pipeline(roi_model, seg_model, base_dataset, device,
                              roi_resize=(64, 64, 64), crop_size=(96, 96, 96), include_coords=True,
                              coord_encoding_type='simple',
                              crops_per_center=1, crop_overlap=0.5, use_blending=True,
                              collect_attention=False, results_dir=None, model_name='model', dataset_version='brats2021',
                              roi_use_4modalities=True, batch_size=1, roi_batch_size=None,
                              distributed=False, world_size=1, rank=0):
    """
    Run cascade inference on base dataset and compute WT/TC/ET dice.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
        collect_attention: True면 MobileViT attention weights 수집 및 분석
        results_dir: 결과 저장 디렉토리 (attention 분석용)
        model_name: 모델 이름 (attention 분석용)
        batch_size: 배치로 처리할 볼륨 수 (기본값 1, 순차 처리)
                    batch_size > 1이면 여러 볼륨을 배치로 처리:
                    - ROI 단계: 배치 내 모든 볼륨의 ROI를 동시에 처리 (roi_batch_size로 제한 가능)
                    - Segmentation 단계: 모든 볼륨의 모든 crop을 모아서 배치로 처리
                    - 각 crop이 어느 볼륨에 속하는지 추적하여 볼륨별로 blending 수행
        roi_batch_size: ROI 단계에서 한 번에 처리할 최대 볼륨 수 (None이면 batch_size와 동일하게 처리)
                       메모리가 부족한 경우 ROI 단계만 더 작은 배치로 나눌 수 있음
        distributed: DDP 사용 여부
        world_size: 전체 프로세스 수
        rank: 현재 프로세스 rank
    """
    roi_model.eval()
    seg_model.eval()
    dice_rows = []
    all_attention_weights = [] if collect_attention else None
    
    # DDP 설정 확인 및 업데이트
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        distributed = True
    
    # DDP 환경에서 데이터셋 분할
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            base_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False  # 평가는 shuffle 불필요
        )
        # 샘플러를 사용하여 각 프로세스가 다른 데이터를 처리하도록 함
        indices = list(sampler)
        total_samples_per_process = len(indices)
    else:
        indices = list(range(len(base_dataset)))
        total_samples_per_process = len(indices)
    
    total_samples = len(base_dataset)
    if is_main_process(rank):
        print(f"[Cascade Evaluation] Starting evaluation on {total_samples_per_process} samples per process (total: {total_samples} samples)...")
        print(f"[Cascade Evaluation] Batch size: {batch_size}, Distributed: {distributed}, World size: {world_size}, Rank: {rank}")
    
    # 시간 측정용 변수
    total_data_load_time = 0.0
    total_roi_time = 0.0
    total_seg_time = 0.0
    total_dice_time = 0.0
    
    # 배치 처리 로직
    if batch_size > 1:
        # 배치 단위로 처리 (DDP 분할된 indices 사용)
        for batch_start in range(0, len(indices), batch_size):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_images = []
            batch_targets = []
            batch_indices = []
            
            # 배치 데이터 로드
            batch_data_load_start = time.time()
            for local_idx in range(batch_start, batch_end):
                idx = indices[local_idx]  # DDP 분할된 인덱스 사용
                loaded_data = base_dataset[idx]
                if len(loaded_data) == 3:
                    image, target, _ = loaded_data
                else:
                    image, target = loaded_data
                image = image.to(device)
                target = target.to(device)
                batch_images.append(image)
                batch_targets.append(target)
                batch_indices.append(idx)
            
            batch_data_load_time = time.time() - batch_data_load_start
            total_data_load_time += batch_data_load_time
            
            if is_main_process(rank) and batch_start == 0:
                print(f"[Cascade Evaluation] Batch {batch_start//batch_size + 1}: Loaded {len(batch_images)} samples")
                print(f"[Cascade Evaluation] Sample 1: image.shape={batch_images[0].shape}, target.shape={batch_targets[0].shape}")
            
            # 배치 cascade inference
            batch_inference_start = time.time()
            batch_results = run_cascade_inference_batch(
                roi_model=roi_model,
                seg_model=seg_model,
                images=batch_images,
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
                roi_batch_size=roi_batch_size,
            )
            batch_inference_time = time.time() - batch_inference_start
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 각 샘플별로 결과 처리
            for batch_idx, (idx, result, target) in enumerate(zip(batch_indices, batch_results, batch_targets)):
                local_idx = batch_start + batch_idx  # 현재 프로세스의 indices 내 인덱스
                timing_info = result.get('timing', {})
                roi_time = timing_info.get('roi_time', 0.0)
                seg_time = timing_info.get('seg_time', 0.0)
                num_centers = timing_info.get('num_centers', 1)
                num_crops = timing_info.get('num_crops', 1)
                
                # 배치 처리에서는 ROI와 seg 시간이 배치 전체에 대한 것이므로, 샘플당 평균으로 계산
                total_roi_time += roi_time / len(batch_images)
                total_seg_time += seg_time / len(batch_images)
                
                if is_main_process(rank) and (local_idx == 0 or local_idx % 10 == 0 or local_idx == len(indices) - 1):
                    print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): ROI={roi_time/len(batch_images):.3f}s, Seg={seg_time/len(batch_images):.3f}s (centers={num_centers}, crops={num_crops})")
                
                # full_logits 처리
                if 'full_logits' not in result or result['full_logits'] is None:
                    if is_main_process(rank):
                        print(f"[Cascade Evaluation] Warning: Batch sample {batch_idx+1} (global {idx+1}): full_logits is None or missing in result. Skipping dice calculation.")
                    continue  # 이 샘플은 건너뛰고 다음 샘플로
                
                full_logits_raw = result['full_logits'].to(device)
                if full_logits_raw.dim() == 4:
                    full_logits = full_logits_raw.unsqueeze(0)
                elif full_logits_raw.dim() == 5:
                    full_logits = full_logits_raw
                else:
                    full_logits = full_logits_raw.unsqueeze(0)
                
                # target 처리
                if target.dim() == 3:
                    target_batch = target.unsqueeze(0)
                elif target.dim() == 4:
                    target_batch = target
                else:
                    target_batch = target.unsqueeze(0)
                
                # Dice 계산
                try:
                    dice_start = time.time()
                    dice = calculate_wt_tc_et_dice(full_logits, target_batch, dataset_version=dataset_version, sample_idx=idx).detach().cpu()
                    dice_time = time.time() - dice_start
                    total_dice_time += dice_time
                    
                    dice_rows.append(dice)
                except Exception as e:
                    if is_main_process(rank):
                        print(f"[Cascade Evaluation] Error calculating dice for batch sample {batch_idx+1} (global {idx+1}): {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                    # 예외가 발생해도 다음 샘플로 계속 진행
                
                # 예측 마스크 시각화 및 저장 (results_dir가 제공된 경우)
                if results_dir and is_main_process(rank):
                    try:
                        # patient 이름 추출
                        patient_path = base_dataset.samples[idx] if hasattr(base_dataset, 'samples') else None
                        if patient_path:
                            if str(patient_path).endswith('.h5'):
                                patient_name = os.path.basename(patient_path).replace('.h5', '')
                            else:
                                patient_name = os.path.basename(patient_path)
                        else:
                            patient_name = f"sample_{idx+1:04d}"
                        
                        # 예측 마스크 추출 (배치 처리에서는 result에서 직접 가져옴)
                        pred_mask = result.get('full_mask')
                        if pred_mask is None:
                            # full_mask가 없으면 full_logits에서 생성
                            full_logits_for_mask = result['full_logits']
                            if full_logits_for_mask.dim() == 4:  # (C, H, W, D)
                                pred_mask = torch.argmax(full_logits_for_mask, dim=0)
                            else:
                                pred_mask = torch.argmax(full_logits_for_mask.squeeze(0), dim=0)
                        
                        # CPU로 이동 및 numpy 변환
                        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
                        
                        # 시각화 디렉토리 생성
                        predictions_dir = os.path.join(results_dir, 'predictions')
                        os.makedirs(predictions_dir, exist_ok=True)
                        
                        # 이미지와 마스크를 사용하여 시각화 생성
                        # 배치 처리에서는 batch_images와 batch_targets에서 가져옴
                        batch_image = batch_images[batch_idx]  # (C, H, W, D)
                        batch_target = batch_targets[batch_idx]  # (H, W, D)
                        image_np = batch_image.cpu().numpy()  # (C, H, W, D)
                        target_np = batch_target.cpu().numpy()  # (H, W, D)
                        
                        # 주요 슬라이스 선택 (Depth 방향)
                        H, W, D = pred_mask_np.shape
                        slice_indices = [D//4, D//2, 3*D//4] if D > 3 else list(range(D))
                        
                        # FLAIR 채널 사용 (일반적으로 마지막 채널 또는 인덱스 3)
                        flair_idx = min(3, image_np.shape[0] - 1) if image_np.shape[0] >= 4 else image_np.shape[0] - 1
                        
                        # 시각화 생성
                        n_slices = len(slice_indices)
                        fig, axes = plt.subplots(n_slices, 3, figsize=(15, 5*n_slices))
                        if n_slices == 1:
                            axes = axes.reshape(1, -1)
                        
                        # Segmentation colormap (evaluation.py와 동일)
                        seg_cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow'])
                        
                        # ROI 중심점과 crop 영역 정보 추출
                        roi_info = result.get('roi', {})
                        roi_centers = roi_info.get('centers_full', []) or []
                        if not roi_centers and roi_info.get('center_full'):
                            roi_centers = [roi_info.get('center_full')]
                        
                        # Crop 크기 정보
                        crop_size_tuple = crop_size if isinstance(crop_size, tuple) else tuple(crop_size)
                        crop_half_h, crop_half_w, crop_half_d = crop_size_tuple[0] // 2, crop_size_tuple[1] // 2, crop_size_tuple[2] // 2
                        
                        for i, slice_idx in enumerate(slice_indices):
                            # 원본 이미지 (FLAIR)
                            img_slice = image_np[flair_idx, :, :, slice_idx]
                            img_min, img_max = img_slice.min(), img_slice.max()
                            if img_max > img_min:
                                img_display = (img_slice - img_min) / (img_max - img_min)
                            else:
                                img_display = img_slice
                            
                            axes[i, 0].imshow(img_display, cmap='gray', origin='lower')
                            # ROI 중심점 표시 (해당 슬라이스 근처에 있는 경우)
                            for center_idx, center in enumerate(roi_centers):
                                if center and len(center) >= 3:
                                    center_h, center_w, center_d = float(center[0]), float(center[1]), float(center[2])
                                    # 슬라이스 범위 내에 있는지 확인 (crop_size의 절반 범위)
                                    if abs(center_d - slice_idx) <= crop_half_d:
                                        axes[i, 0].plot(center_w, center_h, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='yellow', label='ROI Center' if center_idx == 0 else '')
                                        # Crop 영역 표시 (ROI 중심점 기준)
                                        rect = Rectangle(
                                            (center_w - crop_half_w, center_h - crop_half_h),
                                            crop_size_tuple[1], crop_size_tuple[0],
                                            linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--', alpha=0.5
                                        )
                                        axes[i, 0].add_patch(rect)
                            axes[i, 0].set_title(f'FLAIR - Slice {slice_idx}')
                            axes[i, 0].axis('off')
                            
                            # Ground Truth 오버레이
                            axes[i, 1].imshow(img_display, cmap='gray', origin='lower')
                            target_slice = target_np[:, :, slice_idx]
                            mask_gt = target_slice > 0
                            if mask_gt.any():
                                axes[i, 1].imshow(target_slice, cmap=seg_cmap, alpha=0.5, vmin=0, vmax=4, origin='lower')
                            # ROI 중심점 표시
                            for center_idx, center in enumerate(roi_centers):
                                if center and len(center) >= 3:
                                    center_h, center_w, center_d = float(center[0]), float(center[1]), float(center[2])
                                    if abs(center_d - slice_idx) <= crop_half_d:
                                        axes[i, 1].plot(center_w, center_h, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='yellow')
                                        rect = Rectangle(
                                            (center_w - crop_half_w, center_h - crop_half_h),
                                            crop_size_tuple[1], crop_size_tuple[0],
                                            linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--', alpha=0.5
                                        )
                                        axes[i, 1].add_patch(rect)
                            axes[i, 1].set_title(f'Ground Truth - Slice {slice_idx}')
                            axes[i, 1].axis('off')
                            
                            # Prediction 오버레이
                            axes[i, 2].imshow(img_display, cmap='gray', origin='lower')
                            pred_slice = pred_mask_np[:, :, slice_idx]
                            mask_pred = pred_slice > 0
                            if mask_pred.any():
                                axes[i, 2].imshow(pred_slice, cmap=seg_cmap, alpha=0.5, vmin=0, vmax=4, origin='lower')
                            
                            # ROI 중심점과 crop 영역 표시
                            for center_idx, center in enumerate(roi_centers):
                                if center and len(center) >= 3:
                                    center_h, center_w, center_d = float(center[0]), float(center[1]), float(center[2])
                                    # 슬라이스 범위 내에 있는지 확인
                                    if abs(center_d - slice_idx) <= crop_half_d:
                                        # ROI 중심점 표시
                                        axes[i, 2].plot(center_w, center_h, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='yellow', label='ROI Center' if center_idx == 0 else '')
                                        
                                        # Crop 영역 표시 (ROI 중심점 기준)
                                        rect = Rectangle(
                                            (center_w - crop_half_w, center_h - crop_half_h),
                                            crop_size_tuple[1], crop_size_tuple[0],
                                            linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--', alpha=0.7
                                        )
                                        axes[i, 2].add_patch(rect)
                            
                            axes[i, 2].set_title(f'Prediction - Slice {slice_idx}')
                            axes[i, 2].axis('off')
                        
                        plt.tight_layout()
                        jpg_path = os.path.join(predictions_dir, f"{patient_name}_pred.jpg")
                        plt.savefig(jpg_path, dpi=200, bbox_inches='tight', format='jpg')
                        plt.close()
                        
                        if local_idx == 0 or local_idx % 50 == 0 or local_idx == len(indices) - 1:
                            print(f"[Cascade Evaluation] Saved prediction visualization: {jpg_path}")
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"[Cascade Evaluation] Warning: Failed to save prediction visualization for sample {local_idx+1} (global {idx+1}): {e}")
                            import traceback
                            traceback.print_exc()
                
                if is_main_process(rank) and (local_idx == 0 or local_idx % 10 == 0 or local_idx == len(indices) - 1):
                    if dataset_version == 'brats2024':
                        print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): Dice - WT={dice[0]:.4f}, TC={dice[1]:.4f}, ET={dice[2]:.4f}, RC={dice[3]:.4f}")
                    else:
                        print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): Dice - WT={dice[0]:.4f}, TC={dice[1]:.4f}, ET={dice[2]:.4f}")
            
            # 배치 처리 완료 후 메모리 정리
            del batch_images, batch_targets, batch_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 기존 순차 처리 (batch_size == 1 또는 호환성)
    else:
        for local_idx, idx in enumerate(indices):  # DDP 분할된 인덱스 사용
            sample_start_time = time.time()
            
            # 진행 상황 로그 (10개마다 또는 첫 번째/마지막 샘플)
            if is_main_process(rank) and (local_idx == 0 or local_idx % 10 == 0 or local_idx == len(indices) - 1):
                print(f"[Cascade Evaluation] Processing sample {local_idx+1}/{len(indices)} (global idx: {idx+1}/{total_samples})...")
            
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
            
            if is_main_process(rank) and local_idx == 0:
                print(f"[Cascade Evaluation] Sample {local_idx+1} (global {idx+1}): image.shape={image.shape}, target.shape={target.shape}")
                print(f"[Cascade Evaluation] Sample {local_idx+1} (global {idx+1}): Data load time: {data_load_time:.3f}s")
                # 볼륨 크기 확인 (디버깅용)
                print(f"[Cascade Evaluation] Sample {local_idx+1} (global {idx+1}): Volume size (H, W, D) = {image.shape[1:]}")
            
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
                debug_sample_idx=idx if local_idx == 0 else -1,  # 첫 번째 샘플(local_idx==0)에 대해서만 로그 출력
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
            
            if is_main_process(rank) and (local_idx == 0 or local_idx % 10 == 0):
                print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): ROI={roi_time:.3f}s, Seg={seg_time:.3f}s (centers={num_centers}, crops={num_crops}), Total={inference_time:.3f}s")
            
            # Attention weights 수집 (dice 계산 전에 먼저 수집하여, 에러가 발생해도 보존)
            if collect_attention and all_attention_weights is not None:
                if 'attention_weights' in result and result['attention_weights']:
                    all_attention_weights.extend(result['attention_weights'])
                    if local_idx == 0 and is_main_process(rank):  # 첫 번째 샘플만 출력
                        print(f"[Cascade Evaluation] Collected attention weights from sample {local_idx+1}/{len(indices)} (global {idx+1}/{total_samples})")
            
            # full_logits shape 처리: run_cascade_inference는 (C, H, W, D) 형태를 반환
            if 'full_logits' not in result or result['full_logits'] is None:
                if is_main_process(rank):
                    print(f"[Cascade Evaluation] Warning: Sample {local_idx+1} (global {idx+1}): full_logits is None or missing in result. Skipping dice calculation.")
                continue  # 이 샘플은 건너뛰고 다음 샘플로
            
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
                if is_main_process(rank):
                    print(f"[Cascade Evaluation] Warning: Sample {local_idx+1} (global {idx+1}): Unexpected full_logits shape: {full_logits_raw.shape}, expected (C, H, W, D), (1, C, H, W, D), or (1, 1, C, H, W, D). Skipping dice calculation.")
                continue  # 이 샘플은 건너뛰고 다음 샘플로
            
            # target이 배치 차원이 없으면 추가, 있으면 그대로 사용
            if target.dim() == 3:  # (H, W, D)
                target_batch = target.unsqueeze(0)  # (1, H, W, D)
            elif target.dim() == 4:  # 이미 (1, H, W, D)
                target_batch = target
            else:
                target_batch = target.unsqueeze(0)  # 안전장치
            
            # Shape 확인 및 디버깅 정보 (첫 번째 샘플)
            if local_idx == 0 and is_main_process(rank):  # 첫 번째 샘플만 출력
                print(f"[Cascade Evaluation] full_logits.shape={full_logits.shape}, target_batch.shape={target_batch.shape}")
                # Prediction 확인
                pred_argmax = torch.argmax(full_logits, dim=1)
                pred_unique = torch.unique(pred_argmax).cpu().tolist()
                target_unique = torch.unique(target_batch).cpu().tolist()
                print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): pred unique classes={pred_unique}, target unique classes={target_unique}")
                print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): pred class counts={dict(zip(*torch.unique(pred_argmax, return_counts=True)))}")
                print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): target class counts={dict(zip(*torch.unique(target_batch, return_counts=True)))}")
            
            try:
                    dice_start = time.time()
                    dice = calculate_wt_tc_et_dice(full_logits, target_batch, dataset_version=dataset_version, sample_idx=idx).detach().cpu()
                    dice_time = time.time() - dice_start
                    total_dice_time += dice_time
                    
                    # dice_rows에 추가
                    dice_rows.append(dice)
                    
                    # 예측 마스크 시각화 및 저장 (results_dir가 제공된 경우)
                    if results_dir and is_main_process(rank):
                        try:
                            # patient 이름 추출
                            patient_path = base_dataset.samples[idx] if hasattr(base_dataset, 'samples') else None
                            if patient_path:
                                if str(patient_path).endswith('.h5'):
                                    patient_name = os.path.basename(patient_path).replace('.h5', '')
                                else:
                                    patient_name = os.path.basename(patient_path)
                            else:
                                patient_name = f"sample_{idx+1:04d}"
                            
                            # 예측 마스크 추출
                            pred_mask = result.get('full_mask')
                            if pred_mask is None:
                                # full_mask가 없으면 full_logits에서 생성
                                full_logits_for_mask = result['full_logits']
                                if full_logits_for_mask.dim() == 4:  # (C, H, W, D)
                                    pred_mask = torch.argmax(full_logits_for_mask, dim=0)
                                else:
                                    pred_mask = torch.argmax(full_logits_for_mask.squeeze(0), dim=0)
                            
                            # CPU로 이동 및 numpy 변환
                            pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
                            
                            # 시각화 디렉토리 생성
                            predictions_dir = os.path.join(results_dir, 'predictions')
                            os.makedirs(predictions_dir, exist_ok=True)
                            
                            # 이미지와 마스크를 사용하여 시각화 생성
                            # image는 (C, H, W, D), target은 (H, W, D), pred_mask_np는 (H, W, D)
                            image_np = image.cpu().numpy()  # (C, H, W, D)
                            target_np = target.cpu().numpy()  # (H, W, D)
                            
                            # 주요 슬라이스 선택 (Depth 방향)
                            H, W, D = pred_mask_np.shape
                            slice_indices = [D//4, D//2, 3*D//4] if D > 3 else list(range(D))
                            
                            # FLAIR 채널 사용 (일반적으로 마지막 채널 또는 인덱스 3)
                            flair_idx = min(3, image_np.shape[0] - 1) if image_np.shape[0] >= 4 else image_np.shape[0] - 1
                            
                            # 시각화 생성
                            n_slices = len(slice_indices)
                            fig, axes = plt.subplots(n_slices, 3, figsize=(15, 5*n_slices))
                            if n_slices == 1:
                                axes = axes.reshape(1, -1)
                            
                            # Segmentation colormap (evaluation.py와 동일)
                            seg_cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow'])
                            
                            # ROI 중심점과 crop 영역 정보 추출
                            roi_info = result.get('roi', {})
                            roi_centers = roi_info.get('centers_full', []) or []
                            if not roi_centers and roi_info.get('center_full'):
                                roi_centers = [roi_info.get('center_full')]
                            
                            # Crop 크기 정보
                            crop_size_tuple = crop_size if isinstance(crop_size, tuple) else tuple(crop_size)
                            crop_half_h, crop_half_w, crop_half_d = crop_size_tuple[0] // 2, crop_size_tuple[1] // 2, crop_size_tuple[2] // 2
                            
                            # 모든 crop 중심점 생성 (crops_per_center > 1인 경우)
                            all_crop_centers = []
                            for roi_center in roi_centers:
                                if roi_center and len(roi_center) >= 3:
                                    crop_centers = _generate_multi_crop_centers(
                                        center=roi_center,
                                        crop_size=crop_size_tuple,
                                        crops_per_center=crops_per_center,
                                        crop_overlap=crop_overlap,
                                    )
                                    all_crop_centers.extend(crop_centers)
                            
                            for i, slice_idx in enumerate(slice_indices):
                                # 원본 이미지 (FLAIR)
                                img_slice = image_np[flair_idx, :, :, slice_idx]
                                img_min, img_max = img_slice.min(), img_slice.max()
                                if img_max > img_min:
                                    img_display = (img_slice - img_min) / (img_max - img_min)
                                else:
                                    img_display = img_slice
                                
                                axes[i, 0].imshow(img_display, cmap='gray', origin='lower')
                                # ROI 중심점 표시 (해당 슬라이스 근처에 있는 경우)
                                for center_idx, center in enumerate(roi_centers):
                                    if center and len(center) >= 3:
                                        center_h, center_w, center_d = float(center[0]), float(center[1]), float(center[2])
                                        # 슬라이스 범위 내에 있는지 확인 (crop_size의 절반 범위)
                                        if abs(center_d - slice_idx) <= crop_half_d:
                                            axes[i, 0].plot(center_w, center_h, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='yellow', label='ROI Center' if center_idx == 0 else '')
                                
                                # 모든 crop 영역 표시
                                for crop_idx, crop_center in enumerate(all_crop_centers):
                                    if crop_center and len(crop_center) >= 3:
                                        crop_h, crop_w, crop_d = float(crop_center[0]), float(crop_center[1]), float(crop_center[2])
                                        # 슬라이스 범위 내에 있는지 확인
                                        if abs(crop_d - slice_idx) <= crop_half_d:
                                            # 각 crop 영역을 표시 (약간 투명하게, 여러 개가 겹치므로)
                                            rect = Rectangle(
                                                (crop_w - crop_half_w, crop_h - crop_half_h),
                                                crop_size_tuple[1], crop_size_tuple[0],
                                                linewidth=1.0, edgecolor='cyan', facecolor='none', 
                                                linestyle='--', alpha=0.3
                                            )
                                            axes[i, 0].add_patch(rect)
                                
                                axes[i, 0].set_title(f'FLAIR - Slice {slice_idx}')
                                axes[i, 0].axis('off')
                                
                                # Ground Truth 오버레이
                                axes[i, 1].imshow(img_display, cmap='gray', origin='lower')
                                target_slice = target_np[:, :, slice_idx]
                                mask_gt = target_slice > 0
                                if mask_gt.any():
                                    axes[i, 1].imshow(target_slice, cmap=seg_cmap, alpha=0.5, vmin=0, vmax=4, origin='lower')
                                # ROI 중심점 표시
                                for center_idx, center in enumerate(roi_centers):
                                    if center and len(center) >= 3:
                                        center_h, center_w, center_d = float(center[0]), float(center[1]), float(center[2])
                                        if abs(center_d - slice_idx) <= crop_half_d:
                                            axes[i, 1].plot(center_w, center_h, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='yellow')
                                
                                # 모든 crop 영역 표시
                                for crop_idx, crop_center in enumerate(all_crop_centers):
                                    if crop_center and len(crop_center) >= 3:
                                        crop_h, crop_w, crop_d = float(crop_center[0]), float(crop_center[1]), float(crop_center[2])
                                        if abs(crop_d - slice_idx) <= crop_half_d:
                                            rect = Rectangle(
                                                (crop_w - crop_half_w, crop_h - crop_half_h),
                                                crop_size_tuple[1], crop_size_tuple[0],
                                                linewidth=1.0, edgecolor='cyan', facecolor='none', 
                                                linestyle='--', alpha=0.3
                                            )
                                            axes[i, 1].add_patch(rect)
                                
                                axes[i, 1].set_title(f'Ground Truth - Slice {slice_idx}')
                                axes[i, 1].axis('off')
                                
                                # Prediction 오버레이
                                axes[i, 2].imshow(img_display, cmap='gray', origin='lower')
                                pred_slice = pred_mask_np[:, :, slice_idx]
                                mask_pred = pred_slice > 0
                                if mask_pred.any():
                                    axes[i, 2].imshow(pred_slice, cmap=seg_cmap, alpha=0.5, vmin=0, vmax=4, origin='lower')
                                
                                # ROI 중심점 표시
                                for center_idx, center in enumerate(roi_centers):
                                    if center and len(center) >= 3:
                                        center_h, center_w, center_d = float(center[0]), float(center[1]), float(center[2])
                                        if abs(center_d - slice_idx) <= crop_half_d:
                                            axes[i, 2].plot(center_w, center_h, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='yellow', label='ROI Center' if center_idx == 0 else '')
                                
                                # 모든 crop 영역 표시
                                for crop_idx, crop_center in enumerate(all_crop_centers):
                                    if crop_center and len(crop_center) >= 3:
                                        crop_h, crop_w, crop_d = float(crop_center[0]), float(crop_center[1]), float(crop_center[2])
                                        if abs(crop_d - slice_idx) <= crop_half_d:
                                            rect = Rectangle(
                                                (crop_w - crop_half_w, crop_h - crop_half_h),
                                                crop_size_tuple[1], crop_size_tuple[0],
                                                linewidth=1.0, edgecolor='cyan', facecolor='none', 
                                                linestyle='--', alpha=0.3
                                            )
                                            axes[i, 2].add_patch(rect)
                                
                                axes[i, 2].set_title(f'Prediction - Slice {slice_idx}')
                                axes[i, 2].axis('off')
                            
                            plt.tight_layout()
                            jpg_path = os.path.join(predictions_dir, f"{patient_name}_pred.jpg")
                            plt.savefig(jpg_path, dpi=200, bbox_inches='tight', format='jpg')
                            plt.close()
                            
                            if local_idx == 0 or local_idx % 50 == 0 or local_idx == len(indices) - 1:
                                print(f"[Cascade Evaluation] Saved prediction visualization: {jpg_path}")
                        except Exception as e:
                            if is_main_process(rank):
                                print(f"[Cascade Evaluation] Warning: Failed to save prediction visualization for sample {local_idx+1} (global {idx+1}): {e}")
                                import traceback
                                traceback.print_exc()
                    
                    sample_total_time = time.time() - sample_start_time
                    if is_main_process(rank) and (local_idx == 0 or local_idx % 10 == 0 or local_idx == len(indices) - 1):
                        if dataset_version == 'brats2024':
                            print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): Dice - WT={dice[0]:.4f}, TC={dice[1]:.4f}, ET={dice[2]:.4f}, RC={dice[3]:.4f} | Total: {sample_total_time:.3f}s")
                        else:
                            print(f"[Cascade Evaluation] Sample {local_idx+1}/{len(indices)} (global {idx+1}): Dice - WT={dice[0]:.4f}, TC={dice[1]:.4f}, ET={dice[2]:.4f} | Total: {sample_total_time:.3f}s")
            except Exception as e:
                error_msg = f"[Cascade Evaluation] Error processing sample {local_idx+1}/{len(indices)} (global {idx+1}/{total_samples}): {type(e).__name__}: {e}"
                if is_main_process(rank):
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                # 예외가 발생해도 다음 샘플로 계속 진행
                        # 예외가 발생해도 다음 샘플로 계속 진행 (dice 계산 실패는 기록만 하고 계속)
                        # raise  # 주석 처리: 예외를 발생시키지 않고 다음 샘플로 계속
    
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
    batch_size=1,
    roi_batch_size=None,
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
    
    # DDP 설정 확인
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    
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
        batch_size=batch_size,
        roi_batch_size=roi_batch_size,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
    )


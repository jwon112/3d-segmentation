#!/usr/bin/env python3
"""
Cascade segmentation helpers (ROI + crop + CoordConv pipeline).

현재 구현:
- ROI detector로 WT binary를 예측하고, connected components 기반으로 여러 중심을 추출
- 각 중심 주변에서 multi-crop을 수행하여 segmentation 모델을 여러 번 실행
- crop별 logits를 원본 공간으로 복원한 뒤 cosine blending 또는 voxel-wise max로 병합
- 각 중심마다 여러 crop을 수행하여 큰 종양도 완전히 커버 (crops_per_center > 1)
"""

from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from dataloaders import (
    get_normalized_coord_map,
    get_coord_map,
    resize_volume,
    crop_volume_with_center,
    paste_patch_to_volume,
    compute_tumor_center,
)
from utils.experiment_utils import is_main_process


def _prepare_roi_input(image: torch.Tensor, roi_resize: Sequence[int], use_4modalities: bool = True) -> torch.Tensor:
    """
    Prepare ROI input by extracting modalities only (no coords).
    
    ROI 모델은 항상 4채널(4 modalities, no coords) 또는 2채널(2 modalities, no coords)만 사용합니다.
    Segmentation 모델의 coord_type과 무관하게 ROI 모델은 coords를 사용하지 않습니다.
    
    Args:
        image: Input image tensor (C, H, W, D) where C can be:
               - 2: 2 modalities only (T1CE, FLAIR)
               - 4: 4 modalities only (T1, T1CE, T2, FLAIR)
               - 5: 2 modalities + 3 simple coords
               - 7: 4 modalities + 3 simple coords
               - 11: 2 modalities + 9 hybrid coords
               - 13: 4 modalities + 9 hybrid coords
        use_4modalities: Whether ROI model uses 4 modalities (True) or 2 modalities (False)
    
    입력 이미지가 이미 좌표를 포함하고 있을 수 있으므로, modalities만 추출합니다.
    """
    
    # 입력 이미지의 채널 수 확인
    n_channels = image.shape[0]
    
    # ROI 모델이 사용하는 modalities 수에 따라 추출
    if use_4modalities:
        # 4 modalities 사용: T1, T1CE, T2, FLAIR
        if n_channels >= 4:
            image_modalities = image[:4]
        else:
            raise ValueError(
                f"ROI model expects 4 modalities, but input image has only {n_channels} channels. "
                f"Please ensure the input image contains all 4 modalities (T1, T1CE, T2, FLAIR)."
            )
    else:
        # 2 modalities 사용: T1CE, FLAIR
        # 입력 이미지가 4 modalities인 경우 T1CE(1)와 FLAIR(3)만 선택
        if n_channels >= 4:
            # 4 modalities가 있는 경우: T1CE (index 1)와 FLAIR (index 3)만 선택
            image_modalities = image[[1, 3], :, :, :]
        elif n_channels >= 2:
            # 이미 2 modalities만 있는 경우
            image_modalities = image[:2]
        else:
            raise ValueError(
                f"ROI model expects 2 modalities (T1CE, FLAIR), but input image has only {n_channels} channels."
            )
    
    # ROI 모델은 항상 modalities만 사용 (coords 추가 안 함)
    roi_input = image_modalities
    
    # 최종 ROI 입력 채널 수 확인 및 검증
    # ROI 모델은 항상 modalities만 사용 (coords 없음)
    expected_channels = 4 if use_4modalities else 2
    if roi_input.shape[0] != expected_channels:
        raise ValueError(
            f"[_prepare_roi_input] Unexpected ROI input channels: {roi_input.shape[0]}. "
            f"Expected: {expected_channels} (modalities only, no coords). "
            f"Input image had {n_channels} channels. "
            f"image_modalities shape: {image_modalities.shape}, "
            f"use_4modalities: {use_4modalities}"
        )
    
    roi_input = resize_volume(roi_input, roi_resize, mode='trilinear')
    return roi_input


def _scale_center(center_roi: Tuple[float, float, float], original_shape: Sequence[int], roi_shape: Sequence[int], debug_sample_idx: int = -1) -> Tuple[float, float, float]:
    """Map ROI-space center coordinates back to original volume space."""
    scales = [original_shape[i] / float(max(1, roi_shape[i])) for i in range(3)]
    scaled = (
        center_roi[0] * scales[0],
        center_roi[1] * scales[1],
        center_roi[2] * scales[2],
    )
    
    # 첫 번째 샘플에 대해서만 디버그 로그 출력
    if debug_sample_idx == 0:
        import json
        import time
        import os
        # 현재 작업 디렉토리 기준으로 로그 파일 경로 설정
        log_dir = os.path.join(os.getcwd(), '.cursor')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "coord-check",
                    "hypothesisId": "H6",
                    "location": "cascade_utils.py:_scale_center",
                    "message": "Scale ROI center to full volume space",
                    "data": {
                        "center_roi": [float(center_roi[0]), float(center_roi[1]), float(center_roi[2])],
                        "original_shape": list(original_shape),
                        "roi_shape": list(roi_shape),
                        "scales": [float(s) for s in scales],
                        "scaled_center": [float(scaled[0]), float(scaled[1]), float(scaled[2])],
                        "log_path": log_path,
                        "cwd": os.getcwd()
                    },
                    "timestamp": int(time.time() * 1000)
                }, ensure_ascii=False) + "\n")
                log_file.flush()  # 즉시 디스크에 쓰기
        except Exception as e:
            # 예외 발생 시에도 로그 기록 시도 (디버깅용)
            try:
                error_log_path = os.path.join(os.getcwd(), 'debug_error.log')
                with open(error_log_path, 'a', encoding='utf-8') as error_file:
                    error_file.write(f"[H6 Error] {str(e)}\n")
            except:
                pass
    
    return scaled
3

def _extract_roi_centers(
    roi_mask: torch.Tensor,
    max_instances: int = 3,
    min_component_size: int = 50,
) -> List[Tuple[float, float, float]]:
    """
    WT binary mask에서 connected components를 찾아
    각 컴포넌트의 중심을 구하고, 크기 순으로 상위 max_instances개를 반환.
    """
    if roi_mask.dtype != torch.bool:
        roi_mask_bin = roi_mask > 0
    else:
        roi_mask_bin = roi_mask

    np_mask = roi_mask_bin.cpu().numpy().astype(np.bool_)
    if not np_mask.any():
        # 포그라운드가 전혀 없으면 전체 중심 하나만 반환
        h, w, d = np_mask.shape
        return [(h / 2.0, w / 2.0, d / 2.0)]

    labeled, num = ndimage.label(np_mask)
    if num == 0:
        h, w, d = np_mask.shape
        return [(h / 2.0, w / 2.0, d / 2.0)]

    centers: List[Tuple[float, float, float]] = []
    sizes: List[int] = []
    for comp_id in range(1, num + 1):
        comp_mask = (labeled == comp_id)
        size = int(comp_mask.sum())
        if size < max(1, min_component_size):
            continue
        coords = np.argwhere(comp_mask)  # (N, 3) / (y, x, z)
        center = coords.mean(axis=0)
        cy, cx, cz = center.tolist()
        centers.append((float(cy), float(cx), float(cz)))
        sizes.append(size)

    if not centers:
        # 모든 컴포넌트가 너무 작으면 전체 WT 중심으로 fallback
        cy, cx, cz = compute_tumor_center(torch.from_numpy(np_mask.astype(np.int64)))
        return [(cy, cx, cz)]

    # 크기 기준 내림차순 정렬 후 상위 max_instances 선택
    order = np.argsort(-np.array(sizes))
    centers_sorted = [centers[i] for i in order[:max_instances]]
    return centers_sorted


def run_roi_localization_batch(
    roi_model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    roi_resize: Sequence[int] = (64, 64, 64),
    max_instances: int = 3,
    min_component_size: int = 50,
    roi_use_4modalities: bool = True,
) -> List[Dict]:
    """
    Run ROI detector on multiple volumes in batch.
    
    Args:
        images: List of image tensors, each with shape (C, H, W, D)
    
    Returns:
        List of ROI info dicts, each with the same structure as run_roi_localization
    """
    roi_model.eval()
    
    # Prepare ROI inputs for all volumes
    roi_inputs = []
    original_shapes = []
    for image in images:
        roi_input_prepared = _prepare_roi_input(image, roi_resize, use_4modalities=roi_use_4modalities)
        roi_inputs.append(roi_input_prepared)
        original_shapes.append(image.shape[1:])
    
    # Stack into batch: (B, C, H, W, D)
    roi_batch = torch.stack(roi_inputs, dim=0).to(device)
    
    # Validate batch input channels
    expected_roi_channels = 4 if roi_use_4modalities else 2
    if roi_batch.shape[1] != expected_roi_channels:
        raise ValueError(
            f"[run_roi_localization_batch] ROI batch channel mismatch: "
            f"Expected {expected_roi_channels} channels, but got {roi_batch.shape[1]} channels."
        )
    
    # Batch ROI inference
    with torch.no_grad():
        roi_logits_batch = roi_model(roi_batch)
    roi_probs_batch = torch.softmax(roi_logits_batch, dim=1)
    roi_masks_batch = torch.argmax(roi_probs_batch, dim=1).cpu()  # (B, H, W, D)
    
    # GPU memory cleanup
    roi_logits_cpu_batch = roi_logits_batch.detach().cpu()
    del roi_batch, roi_logits_batch, roi_probs_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process each volume's ROI results
    results = []
    for i, (roi_mask, original_shape) in enumerate(zip(roi_masks_batch, original_shapes)):
        centers_roi = _extract_roi_centers(
            roi_mask=roi_mask,
            max_instances=max_instances,
            min_component_size=min_component_size,
        )
        centers_full = [
            _scale_center(c, original_shape, roi_resize, debug_sample_idx=-1) for c in centers_roi
        ]
        
        results.append({
            'centers_full': centers_full,
            'centers_roi': centers_roi,
            'roi_mask': roi_mask,
            'roi_logits': roi_logits_cpu_batch[i],
            # backward compatibility
            'center_full': centers_full[0] if centers_full else None,
            'center_roi': centers_roi[0] if centers_roi else None,
        })
    
    return results


def run_roi_localization(
    roi_model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_resize: Sequence[int] = (64, 64, 64),
    max_instances: int = 1,
    min_component_size: int = 50,
    roi_use_4modalities: bool = True,
    debug_sample_idx: int = -1,
) -> Dict:
    """
    Run ROI detector to predict coarse WT center(s).

    Returns:
        {
            'centers_full': [ (cy, cx, cz), ... ]  # original volume 좌표계
            'centers_roi':  [ (cy, cx, cz), ... ]  # ROI 해상도 좌표계
            'roi_mask':     Tensor(H, W, D)        # WT binary (argmax)
            'roi_logits':   Tensor(1, 2, H, W, D)
            # backward compatibility
            'center_full':  첫 번째 중심
            'center_roi':   첫 번째 중심
        }
    """
    roi_model.eval()
    # ROI 모델은 항상 coords를 사용하지 않음 (modalities만 사용)
    roi_input_prepared = _prepare_roi_input(image, roi_resize, use_4modalities=roi_use_4modalities)
    roi_input = roi_input_prepared.unsqueeze(0).to(device)
    
    # 최종 검증: ROI 입력 채널 수가 ROI 모델이 기대하는 채널 수와 일치하는지 확인
    # ROI 모델은 항상 modalities만 사용 (coords 없음)
    expected_roi_channels = 4 if roi_use_4modalities else 2
    if roi_input.shape[1] != expected_roi_channels:
        raise ValueError(
            f"[run_roi_localization] ROI input channel mismatch: "
            f"Expected {expected_roi_channels} channels (modalities only, no coords), "
            f"but got {roi_input.shape[1]} channels. "
            f"Input image had {image.shape[0]} channels. "
            f"roi_use_4modalities={roi_use_4modalities}"
        )
    
    with torch.no_grad():
        roi_logits = roi_model(roi_input)
    roi_probs = torch.softmax(roi_logits, dim=1)
    roi_mask = torch.argmax(roi_probs, dim=1).squeeze(0).cpu()  # (H, W, D) with 0/1
    
    # GPU 메모리 정리 (ROI inference 후)
    roi_logits_cpu = roi_logits.detach().cpu()
    del roi_input, roi_logits, roi_probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    centers_roi = _extract_roi_centers(
        roi_mask=roi_mask,
        max_instances=max_instances,
        min_component_size=min_component_size,
    )
    centers_full = [
        _scale_center(c, image.shape[1:], roi_resize, debug_sample_idx=debug_sample_idx) for c in centers_roi
    ]

    return {
        'centers_full': centers_full,
        'centers_roi': centers_roi,
        'roi_mask': roi_mask,
        'roi_logits': roi_logits_cpu,
        # backward compatibility (단일 중심만 사용하던 코드 대비)
        'center_full': centers_full[0],
        'center_roi': centers_roi[0],
    }


def build_segmentation_input(
    image: torch.Tensor,
    center: Sequence[float],
    crop_size: Sequence[int],
    include_coords: bool = True,
    coord_encoding_type: str = 'simple',
    coord_map: Optional[torch.Tensor] = None,
    debug_sample_idx: int = -1,
) -> Dict:
    """Create crop with coord channels around predicted center.
    
    Args:
        coord_encoding_type: 'simple' (3 channels) or 'hybrid' (9 channels)
        coord_map: Pre-computed coordinate map (optional). If None, will be created on-the-fly.
        debug_sample_idx: 디버그 로그를 출력할 샘플 인덱스 (-1이면 출력 안 함)
    """
    # 첫 번째 샘플의 첫 번째 crop에 대해서만 로그 출력
    if debug_sample_idx == 0:
        import json
        import time
        import os
        # 현재 작업 디렉토리 기준으로 로그 파일 경로 설정
        log_dir = os.path.join(os.getcwd(), '.cursor')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "coord-check",
                    "hypothesisId": "H7",
                    "location": "cascade_utils.py:build_segmentation_input",
                    "message": "Build segmentation input - center received",
                    "data": {
                        "center": [float(c) for c in center],
                        "image_shape": list(image.shape),
                        "crop_size": list(crop_size),
                    },
                    "timestamp": int(time.time() * 1000)
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    seg_patch, origin = crop_volume_with_center(image, center, crop_size, return_origin=True, debug_sample_idx=debug_sample_idx)
    inputs = seg_patch
    if include_coords:
        if coord_map is None:
            # Fallback: create coord_map if not provided (for backward compatibility)
            # 이 경우는 발생하지 않아야 함 (run_cascade_inference에서 coord_map을 전달해야 함)
            import warnings
            warnings.warn(
                f"build_segmentation_input: coord_map is None, creating on-the-fly. "
                f"This should not happen in cascade inference. "
                f"image.shape={image.shape}, center={center}, crop_size={crop_size}"
            )
            coord_map = get_coord_map(image.shape[1:], device=image.device, encoding_type=coord_encoding_type)
        coord_patch = crop_volume_with_center(coord_map, center, crop_size, return_origin=False, debug_sample_idx=-1)  # coord는 로그 불필요
        inputs = torch.cat([seg_patch, coord_patch], dim=0)
    return {
        'inputs': inputs,
        'origin': origin,
    }


def _generate_multi_crop_centers(
    center: Tuple[float, float, float],
    crop_size: Sequence[int],
    crops_per_center: int = 1,
    crop_overlap: float = 0.5,
    debug_sample_idx: int = -1,
    center_idx: int = -1,
) -> List[Tuple[float, float, float]]:
    """
    중심 주변에서 여러 crop 위치를 생성합니다.
    
    Args:
        center: 중심 좌표 (cy, cx, cz)
        crop_size: crop 크기 (h, w, d)
        crops_per_center: 중심당 crop 개수 (1, 2, 3 등)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        debug_sample_idx: 디버그 로그를 출력할 샘플 인덱스 (-1이면 출력 안 함)
        center_idx: 중심 인덱스 (디버그용)
    
    Returns:
        crop 중심 좌표 리스트
    """
    if crops_per_center <= 1:
        return [center]
    
    # 겹침에 따른 offset 계산
    # 예: crop_size=96, overlap=0.5이면 stride=48
    stride = [int(s * (1.0 - crop_overlap)) for s in crop_size]
    
    # crops_per_center에 따라 grid 크기 결정
    # crops_per_center=2 -> 2x2x2=8개
    # crops_per_center=3 -> 3x3x3=27개
    grid_size = crops_per_center
    
    # 첫 번째 샘플의 첫 번째 중심에 대해서만 로그 출력
    if debug_sample_idx == 0 and center_idx == 0:
        import json
        import time
        import os
        # 현재 작업 디렉토리 기준으로 로그 파일 경로 설정
        log_dir = os.path.join(os.getcwd(), '.cursor')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "coord-check",
                    "hypothesisId": "H8",
                    "location": "cascade_utils.py:_generate_multi_crop_centers",
                    "message": "Generate multi crop centers - input center",
                    "data": {
                        "input_center": [float(center[0]), float(center[1]), float(center[2])],
                        "crop_size": list(crop_size),
                        "crops_per_center": crops_per_center,
                        "crop_overlap": crop_overlap,
                        "stride": stride,
                        "grid_size": grid_size,
                        "log_path": log_path,
                        "cwd": os.getcwd()
                    },
                    "timestamp": int(time.time() * 1000)
                }, ensure_ascii=False) + "\n")
                log_file.flush()  # 즉시 디스크에 쓰기
        except Exception as e:
            # 예외 발생 시에도 로그 기록 시도 (디버깅용)
            try:
                error_log_path = os.path.join(os.getcwd(), 'debug_error.log')
                with open(error_log_path, 'a', encoding='utf-8') as error_file:
                    error_file.write(f"[H8 Error] {str(e)}\n")
            except:
                pass
    
    centers = []
    cy, cx, cz = center
    
    # 중심을 기준으로 grid 생성
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # grid 위치를 offset으로 변환
                # 중심에서 offset만큼 이동
                offset_y = (i - (grid_size - 1) / 2.0) * stride[0]
                offset_x = (j - (grid_size - 1) / 2.0) * stride[1]
                offset_z = (k - (grid_size - 1) / 2.0) * stride[2]
                
                new_center = (
                    cy + offset_y,
                    cx + offset_x,
                    cz + offset_z,
                )
                centers.append(new_center)
                
                # 첫 번째 샘플의 첫 번째 중심의 첫 번째 crop에 대해서만 로그 출력
                if debug_sample_idx == 0 and center_idx == 0 and i == 0 and j == 0 and k == 0:
                    import json
                    import time
                    import os
                    # 현재 작업 디렉토리 기준으로 로그 파일 경로 설정
                    log_dir = os.path.join(os.getcwd(), '.cursor')
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, 'debug.log')
                    try:
                        with open(log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "coord-check",
                                "hypothesisId": "H8",
                                "location": "cascade_utils.py:_generate_multi_crop_centers",
                                "message": "Generate multi crop centers - first crop center",
                                "data": {
                                    "input_center": [float(cy), float(cx), float(cz)],
                                    "offset": [float(offset_y), float(offset_x), float(offset_z)],
                                    "new_center": [float(new_center[0]), float(new_center[1]), float(new_center[2])],
                                },
                                "timestamp": int(time.time() * 1000)
                            }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
    
    return centers


def run_cascade_inference_batch(
    roi_model: torch.nn.Module,
    seg_model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    roi_resize: Sequence[int] = (64, 64, 64),
    crop_size: Sequence[int] = (96, 96, 96),
    include_coords: bool = True,
    coord_encoding_type: str = 'simple',
    max_instances: int = 3,
    min_component_size: int = 50,
    crops_per_center: int = 1,
    crop_overlap: float = 0.5,
    use_blending: bool = True,
    return_attention: bool = False,
    roi_use_4modalities: bool = True,
    return_timing: bool = False,
    roi_batch_size: Optional[int] = None,
) -> List[Dict]:
    """
    Batch cascade inference: ROI -> multi-crop -> segmentation -> merge & uncrop.
    
    Processes multiple volumes in batch:
    1. Batch ROI localization for all volumes (with optional batch size limit)
    2. Collect all crops from all volumes with metadata
    3. Batch segmentation for all crops
    4. Per-volume blending
    
    Args:
        images: List of image tensors, each with shape (C, H, W, D)
        roi_batch_size: ROI 단계에서 한 번에 처리할 최대 볼륨 수 (None이면 모든 볼륨을 한 번에 처리)
    
    Returns:
        List of result dicts, each with the same structure as run_cascade_inference
    """
    import time
    total_start = time.time()
    
    # 1. Batch ROI localization (with optional batch size limit)
    roi_start = time.time()
    if roi_batch_size is None or roi_batch_size >= len(images):
        # 모든 볼륨을 한 번에 처리
        roi_infos = run_roi_localization_batch(
            roi_model=roi_model,
            images=images,
            device=device,
            roi_resize=roi_resize,
            max_instances=max_instances,
            min_component_size=min_component_size,
            roi_use_4modalities=roi_use_4modalities,
        )
    else:
        # ROI 단계에서 배치 크기 제한
        roi_infos = []
        for roi_batch_start in range(0, len(images), roi_batch_size):
            roi_batch_end = min(roi_batch_start + roi_batch_size, len(images))
            roi_batch_images = images[roi_batch_start:roi_batch_end]
            roi_batch_infos = run_roi_localization_batch(
                roi_model=roi_model,
                images=roi_batch_images,
                device=device,
                roi_resize=roi_resize,
                max_instances=max_instances,
                min_component_size=min_component_size,
                roi_use_4modalities=roi_use_4modalities,
            )
            roi_infos.extend(roi_batch_infos)
    roi_time = time.time() - roi_start
    
    seg_model.eval()
    
    # 2. Collect all crops from all volumes with metadata
    crop_metadata = []  # List of (volume_idx, center, origin, image_shape)
    volume_coord_maps = {}  # volume_idx -> coord_map
    
    for volume_idx, (image, roi_info) in enumerate(zip(images, roi_infos)):
        centers_full = roi_info.get('centers_full') or [roi_info.get('center_full')]
        if not centers_full or centers_full[0] is None:
            # No centers found, skip this volume
            continue
        
        # Generate coord_map for this volume (reused for all crops)
        if include_coords:
            if volume_idx not in volume_coord_maps:
                coord_map_full = get_coord_map(image.shape[1:], device=image.device, encoding_type=coord_encoding_type)
                volume_coord_maps[volume_idx] = coord_map_full
        
        # Generate crop centers for all ROI centers
        for center_idx, center in enumerate(centers_full):
            crop_centers = _generate_multi_crop_centers(
                center=center,
                crop_size=crop_size,
                crops_per_center=crops_per_center,
                crop_overlap=crop_overlap,
                debug_sample_idx=-1,  # No debug logs in batch mode
                center_idx=center_idx,
            )
            
            # Prepare crop inputs and collect metadata
            for crop_center in crop_centers:
                seg_inputs = build_segmentation_input(
                    image=image,
                    center=crop_center,
                    crop_size=crop_size,
                    include_coords=include_coords,
                    coord_encoding_type=coord_encoding_type,
                    coord_map=volume_coord_maps.get(volume_idx),
                    debug_sample_idx=-1,
                )
                crop_metadata.append({
                    'volume_idx': volume_idx,
                    'center': crop_center,
                    'origin': seg_inputs['origin'],
                    'image_shape': image.shape[1:],
                    'seg_input': seg_inputs['inputs'],  # (C, h, w, d)
                })
    
    if not crop_metadata:
        # No crops found, return empty results
        return [{'full_logits': None, 'roi': roi_info, 'timing': {'roi_time': roi_time, 'seg_time': 0.0}} for roi_info in roi_infos]
    
    # 3. Batch segmentation for all crops
    seg_start = time.time()
    seg_inputs_batch = torch.stack([meta['seg_input'] for meta in crop_metadata], dim=0).to(device)  # (B, C, h, w, d)
    
    # Check if model supports return_attention
    import inspect
    real_seg_model = seg_model.module if hasattr(seg_model, 'module') else seg_model
    sig = inspect.signature(real_seg_model.forward)
    supports_return_attention = 'return_attention' in sig.parameters
    
    with torch.no_grad():
        if return_attention and supports_return_attention:
            # Only collect attention from first crop
            seg_result = seg_model(seg_inputs_batch[:1], return_attention=True)
            if isinstance(seg_result, tuple):
                seg_logits_batch_first, attn_dict = seg_result
                # Process remaining crops without attention
                if seg_inputs_batch.shape[0] > 1:
                    seg_logits_batch_rest = seg_model(seg_inputs_batch[1:], return_attention=False)
                    seg_logits_batch = torch.cat([seg_logits_batch_first, seg_logits_batch_rest], dim=0)
                else:
                    seg_logits_batch = seg_logits_batch_first
            else:
                seg_logits_batch = seg_result
        else:
            seg_logits_batch = seg_model(seg_inputs_batch, return_attention=False)
    
    seg_time = time.time() - seg_start
    
    # GPU memory cleanup
    del seg_inputs_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 4. Per-volume blending
    seg_logits_list = seg_logits_batch.cpu().unbind(0)  # List of (C, h, w, d)
    
    # Group crops by volume
    volume_crops = {}  # volume_idx -> List of (logits, origin, image_shape)
    for meta, logits in zip(crop_metadata, seg_logits_list):
        vol_idx = meta['volume_idx']
        if vol_idx not in volume_crops:
            volume_crops[vol_idx] = []
        volume_crops[vol_idx].append({
            'logits': logits,
            'origin': meta['origin'],
            'image_shape': meta['image_shape'],
        })
    
    # Determine number of classes from first logits
    n_classes = seg_logits_list[0].shape[0] if seg_logits_list else 5
    
    # Blend per volume
    results = []
    blend_weights = _make_blend_weights_3d(crop_size) if use_blending else None
    
    for volume_idx, image in enumerate(images):
        roi_info = roi_infos[volume_idx]
        
        if volume_idx not in volume_crops:
            # No crops for this volume
            results.append({
                'full_logits': torch.zeros((n_classes,) + image.shape[1:], dtype=torch.float32),
                'roi': roi_info,
                'timing': {'roi_time': roi_time, 'seg_time': 0.0},
            })
            continue
        
        crops = volume_crops[volume_idx]
        image_shape = image.shape[1:]
        
        # Initialize accumulators for this volume
        if use_blending and blend_weights is not None:
            full_logits = torch.zeros((n_classes,) + image_shape, dtype=torch.float32)
            weight_sum = torch.zeros(image_shape, dtype=torch.float32)
        else:
            full_logits = None
            weight_sum = None
        
        # Blend all crops for this volume
        for crop_info in crops:
            patch_logits = crop_info['logits']  # (C, h, w, d)
            origin = crop_info['origin']
            
            if use_blending and blend_weights is not None:
                # Cosine blending
                patch_full_logits = paste_patch_to_volume(
                    patch_logits,
                    origin,
                    image_shape,
                    debug_sample_idx=-1,
                )  # (C, H, W, D)
                patch_weights = paste_patch_to_volume(
                    blend_weights.squeeze(0).squeeze(0),  # (h, w, d)
                    origin,
                    image_shape,
                    debug_sample_idx=-1,
                )  # (H, W, D)
                
                full_logits += patch_full_logits * patch_weights.unsqueeze(0)
                weight_sum += patch_weights
                
                del patch_full_logits, patch_weights
            else:
                # Voxel-wise max
                patch_full_logits = paste_patch_to_volume(
                    patch_logits,
                    origin,
                    image_shape,
                    debug_sample_idx=-1,
                )
                
                if full_logits is None:
                    full_logits = patch_full_logits
                else:
                    full_logits = torch.maximum(full_logits, patch_full_logits)
                
                del patch_full_logits
            
            del patch_logits
        
        # Normalize blending weights
        if use_blending and weight_sum is not None:
            weight_sum = weight_sum.clamp_min(1e-8)
            full_logits = full_logits / weight_sum.unsqueeze(0)
        
        if full_logits is None:
            full_logits = torch.zeros((n_classes,) + image_shape, dtype=torch.float32)
        
        num_centers = len(roi_info.get('centers_full', [roi_info.get('center_full')]))
        num_crops = len(crops)
        
        results.append({
            'full_logits': full_logits,
            'roi': roi_info,
            'timing': {
                'roi_time': roi_time,
                'seg_time': seg_time,
                'num_centers': num_centers,
                'num_crops': num_crops,
            },
        })
    
    return results


def _make_blend_weights_3d(patch_size: Sequence[int]) -> torch.Tensor:
    """
    Create separable cosine blending weights for 3D patches.
    Returns tensor of shape (1, 1, H, W, D).
    """
    import math
    ph, pw, pd = patch_size
    
    def one_dim(n):
        t = torch.arange(n, dtype=torch.float32)
        # raised cosine from 0..pi
        w = 0.5 * (1 - torch.cos(math.pi * (t + 0.5) / n))
        # avoid zeros on borders completely
        return w.clamp_min(1e-4)
    
    wh = one_dim(ph)
    ww = one_dim(pw)
    wd = one_dim(pd)
    w3 = wh.view(ph, 1, 1) * ww.view(1, pw, 1) * wd.view(1, 1, pd)
    w3 = w3 / w3.max()
    return w3.view(1, 1, ph, pw, pd)


def run_cascade_inference(
    roi_model: torch.nn.Module,
    seg_model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_resize: Sequence[int] = (64, 64, 64),
    crop_size: Sequence[int] = (96, 96, 96),
    include_coords: bool = True,
    coord_encoding_type: str = 'simple',
    max_instances: int = 3,
    min_component_size: int = 50,
    crops_per_center: int = 1,
    crop_overlap: float = 0.5,
    use_blending: bool = True,
    return_attention: bool = False,
    roi_use_4modalities: bool = True,
    return_timing: bool = False,
    debug_sample_idx: int = -1,
) -> Dict:
    """
    Full cascade inference: ROI -> multi-crop -> segmentation -> merge & uncrop.

    - ROI detector에서 WT binary를 예측하고 여러 컴포넌트 중심을 얻음
    - 각 중심마다 여러 crop을 수행 (crops_per_center > 1인 경우)
    - crop별 logits를 원본 공간으로 복원 후 blending 또는 max로 병합
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
        return_attention: True면 MobileViT attention weights도 반환
    """
    import time
    roi_start = time.time()
    roi_info = run_roi_localization(
        roi_model=roi_model,
        image=image,
        device=device,
        roi_resize=roi_resize,
        max_instances=max_instances,
        min_component_size=min_component_size,
        roi_use_4modalities=roi_use_4modalities,
        debug_sample_idx=debug_sample_idx,
    )
    roi_time = time.time() - roi_start

    centers_full = roi_info.get('centers_full') or [roi_info['center_full']]
    seg_time_total = 0.0
    num_seg_calls = 0

    seg_model.eval()
    
    # coord_map을 한 번만 생성하여 재사용 (성능 최적화)
    # 이전에는 각 crop마다 get_coord_map을 호출했지만, 이제는 샘플당 1번만 생성
    # get_hybrid_coord_map도 이제 캐시를 사용하므로 같은 shape에 대해서는 매우 빠름
    coord_map_full = None
    coord_map_time = 0.0
    if include_coords:
        coord_map_start = time.time()
        coord_map_full = get_coord_map(image.shape[1:], device=image.device, encoding_type=coord_encoding_type)
        coord_map_time = time.time() - coord_map_start
    
    # 모델이 return_attention 파라미터를 지원하는지 확인
    import inspect
    real_seg_model = seg_model.module if hasattr(seg_model, 'module') else seg_model
    sig = inspect.signature(real_seg_model.forward)
    supports_return_attention = 'return_attention' in sig.parameters
    
    # MobileViT attention 수집 (첫 번째 crop만)
    all_attention_weights = [] if return_attention else None
    
    # 모든 crop 중심 수집 (첫 번째 샘플만, 디버깅용)
    all_crop_centers = [] if debug_sample_idx == 0 else None
    
    # Blending을 사용하는 경우 accumulator와 weight sum 초기화
    if use_blending and crops_per_center > 1:
        # 출력 채널 수 확인
        # Cascade segmentation 모델은 항상 CoordConv 포함 입력을 기대
        dummy_input = build_segmentation_input(
            image=image,
            center=centers_full[0],
            crop_size=crop_size,
            include_coords=include_coords,
            coord_encoding_type=coord_encoding_type,
            coord_map=coord_map_full,  # 재사용
        )['inputs'].unsqueeze(0).to(device)
        with torch.no_grad():
            dummy_logits = seg_model(dummy_input)
        n_classes = dummy_logits.shape[1]
        
        full_logits = torch.zeros((n_classes,) + image.shape[1:], dtype=torch.float32)  # (C, H, W, D)
        weight_sum = torch.zeros(image.shape[1:], dtype=torch.float32)  # (H, W, D) - 배치 차원 없음
        blend_weights = _make_blend_weights_3d(crop_size)
    else:
        full_logits: Optional[torch.Tensor] = None
        weight_sum = None
        blend_weights = None

    for center_idx, center in enumerate(centers_full):
        # 각 중심마다 여러 crop 위치 생성
        crop_centers = _generate_multi_crop_centers(
            center=center,
            crop_size=crop_size,
            crops_per_center=crops_per_center,
            crop_overlap=crop_overlap,
            debug_sample_idx=debug_sample_idx,
            center_idx=center_idx,
        )
        
        for crop_idx, crop_center in enumerate(crop_centers):
            # 모든 crop 중심 수집 (첫 번째 샘플만)
            if all_crop_centers is not None:
                all_crop_centers.append((float(crop_center[0]), float(crop_center[1]), float(crop_center[2])))
            
            # Cascade segmentation 모델은 항상 CoordConv 포함 입력을 기대
            # 첫 번째 샘플의 첫 번째 중심의 첫 번째 crop만 로그 출력
            is_first_crop = (debug_sample_idx == 0 and center_idx == 0 and crop_idx == 0)
            seg_inputs = build_segmentation_input(
                image=image,
                center=crop_center,
                crop_size=crop_size,
                include_coords=include_coords,
                coord_encoding_type=coord_encoding_type,
                coord_map=coord_map_full,  # 재사용: 한 번 생성한 coord_map을 모든 crop에서 사용
                debug_sample_idx=0 if is_first_crop else -1,
            )
            seg_tensor = seg_inputs['inputs'].unsqueeze(0).to(device)
            with torch.no_grad():
                seg_start = time.time()
                if return_attention and all_attention_weights is not None and len(all_attention_weights) == 0:
                    # MobileViT attention 수집 (첫 번째 crop만)
                    try:
                        seg_result = seg_model(seg_tensor, return_attention=True)
                        if isinstance(seg_result, tuple):
                            seg_logits, attn_dict = seg_result
                            if attn_dict is not None and len(attn_dict) > 0:
                                all_attention_weights.append(attn_dict)
                                rank = 0
                                if torch.distributed.is_available() and torch.distributed.is_initialized():
                                    rank = torch.distributed.get_rank()
                                if is_main_process(rank):
                                    print(f"[Cascade] Collected MobileViT attention weights: {len(attn_dict)} layers")
                            else:
                                rank = 0
                                if torch.distributed.is_available() and torch.distributed.is_initialized():
                                    rank = torch.distributed.get_rank()
                                if is_main_process(rank):
                                    print(f"[Cascade] Warning: seg_model returned tuple but attn_dict is None or empty")
                                seg_logits = seg_result[0] if len(seg_result) > 0 else seg_result
                        else:
                            rank = 0
                            if torch.distributed.is_available() and torch.distributed.is_initialized():
                                rank = torch.distributed.get_rank()
                            if is_main_process(rank):
                                print(f"[Cascade] Warning: seg_model did not return tuple for attention. Got type: {type(seg_result)}")
                            seg_logits = seg_result
                    except Exception as e:
                        rank = 0
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            rank = torch.distributed.get_rank()
                        if is_main_process(rank):
                            print(f"[Cascade] Warning: Failed to collect attention weights: {e}")
                        # return_attention을 지원하는 경우에만 파라미터 전달
                        if supports_return_attention:
                            seg_logits = seg_model(seg_tensor, return_attention=False)
                        else:
                            seg_logits = seg_model(seg_tensor)
                else:
                    # return_attention을 지원하는 경우에만 파라미터 전달
                    if supports_return_attention:
                        seg_logits = seg_model(seg_tensor, return_attention=False)
                    else:
                        seg_logits = seg_model(seg_tensor)
                seg_time_total += time.time() - seg_start
                num_seg_calls += 1
            patch_logits = seg_logits.squeeze(0).cpu()  # (C, h, w, d)
            
            # GPU 메모리 정리: 중간 텐서 즉시 삭제
            del seg_tensor, seg_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if use_blending and crops_per_center > 1 and blend_weights is not None:
                # Cosine blending 사용
                patch_full_logits = paste_patch_to_volume(
                    patch_logits,
                    seg_inputs['origin'],
                    image.shape[1:],
                    debug_sample_idx=0 if is_first_crop else -1,
                )  # (C, H, W, D)
                patch_weights = paste_patch_to_volume(
                    blend_weights.squeeze(0).squeeze(0),  # (h, w, d)
                    seg_inputs['origin'],
                    image.shape[1:],
                    debug_sample_idx=-1,  # weights는 로그 불필요
                )  # (H, W, D)
                
                # patch_weights를 (1, H, W, D)로 확장하여 broadcasting
                full_logits += patch_full_logits * patch_weights.unsqueeze(0)  # (C, H, W, D) * (1, H, W, D) -> (C, H, W, D)
                weight_sum += patch_weights  # (H, W, D)
                
                # 중간 텐서 삭제 (메모리 절약)
                del patch_full_logits, patch_weights
            else:
                # Voxel-wise max 사용 (기존 방식)
                patch_full_logits = paste_patch_to_volume(
                    patch_logits,
                    seg_inputs['origin'],
                    image.shape[1:],
                    debug_sample_idx=0 if is_first_crop else -1,
                )
                
                if full_logits is None:
                    full_logits = patch_full_logits
                else:
                    full_logits = torch.maximum(full_logits, patch_full_logits)
                
                # 중간 텐서 삭제 (메모리 절약)
                del patch_full_logits
            
            # patch_logits도 더 이상 필요 없으므로 삭제
            del patch_logits

    if full_logits is None:
        # 안전장치: ROI가 완전히 비었을 경우, 전부 background로 설정
        c = seg_model.out_channels if hasattr(seg_model, "out_channels") else 4
        full_logits = torch.zeros((c,) + image.shape[1:], dtype=torch.float32)  # (C, H, W, D)
    elif use_blending and weight_sum is not None:
        # Blending 가중치로 정규화
        # full_logits: (C, H, W, D), weight_sum: (H, W, D)
        
        # 디버깅: weight_sum 분포 확인 (첫 번째 샘플만)
        if debug_sample_idx == 0:
            import json
            import time
            import os
            log_dir = os.path.join(os.getcwd(), '.cursor')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            try:
                weight_sum_min = float(weight_sum.min().item())
                weight_sum_max = float(weight_sum.max().item())
                weight_sum_mean = float(weight_sum.mean().item())
                weight_sum_zero_count = int((weight_sum == 0).sum().item())
                weight_sum_total = int(weight_sum.numel())
                weight_sum_zero_ratio = weight_sum_zero_count / weight_sum_total if weight_sum_total > 0 else 0.0
                
                # weight_sum이 0인 영역의 공간적 분포 분석
                weight_sum_zero_mask = (weight_sum == 0)
                volume_shape = weight_sum.shape  # (H, W, D)
                
                # 경계 영역 정의 (각 차원의 10% 영역)
                boundary_threshold = 0.1
                h_bound = int(volume_shape[0] * boundary_threshold)
                w_bound = int(volume_shape[1] * boundary_threshold)
                d_bound = int(volume_shape[2] * boundary_threshold)
                
                # 경계 영역 마스크
                boundary_mask = torch.zeros_like(weight_sum, dtype=torch.bool)
                boundary_mask[:h_bound, :, :] = True
                boundary_mask[-h_bound:, :, :] = True
                boundary_mask[:, :w_bound, :] = True
                boundary_mask[:, -w_bound:, :] = True
                boundary_mask[:, :, :d_bound] = True
                boundary_mask[:, :, -d_bound:] = True
                
                # 경계 영역에서 zero 비율
                zero_in_boundary = (weight_sum_zero_mask & boundary_mask).sum().item()
                zero_in_center = (weight_sum_zero_mask & ~boundary_mask).sum().item()
                boundary_total = boundary_mask.sum().item()
                center_total = (~boundary_mask).sum().item()
                
                full_logits_before_norm_min = float(full_logits.min().item())
                full_logits_before_norm_max = float(full_logits.max().item())
                full_logits_before_norm_mean = float(full_logits.mean().item())
                
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "blending-check",
                        "hypothesisId": "H9",
                        "location": "cascade_utils.py:run_cascade_inference",
                        "message": "Blending normalization - weight_sum and full_logits stats",
                        "data": {
                            "weight_sum_stats": {
                                "min": weight_sum_min,
                                "max": weight_sum_max,
                                "mean": weight_sum_mean,
                                "zero_count": weight_sum_zero_count,
                                "total_voxels": weight_sum_total,
                                "zero_ratio": weight_sum_zero_ratio
                            },
                            "weight_sum_spatial_distribution": {
                                "zero_in_boundary": int(zero_in_boundary),
                                "zero_in_center": int(zero_in_center),
                                "boundary_total": int(boundary_total),
                                "center_total": int(center_total),
                                "zero_ratio_in_boundary": zero_in_boundary / boundary_total if boundary_total > 0 else 0.0,
                                "zero_ratio_in_center": zero_in_center / center_total if center_total > 0 else 0.0,
                                "boundary_threshold": boundary_threshold
                            },
                            "full_logits_before_norm": {
                                "min": full_logits_before_norm_min,
                                "max": full_logits_before_norm_max,
                                "mean": full_logits_before_norm_mean,
                                "shape": list(full_logits.shape)
                            },
                            "num_centers": len(centers_full),
                            "num_crops": num_seg_calls,
                            "use_blending": use_blending,
                            "roi_centers": [[float(c) for c in center] for center in centers_full],
                            "crop_centers": all_crop_centers if all_crop_centers is not None else None,
                            "crop_centers_bounds": {
                                "h_min": float(min(c[0] for c in all_crop_centers)) if all_crop_centers and len(all_crop_centers) > 0 else None,
                                "h_max": float(max(c[0] for c in all_crop_centers)) if all_crop_centers and len(all_crop_centers) > 0 else None,
                                "w_min": float(min(c[1] for c in all_crop_centers)) if all_crop_centers and len(all_crop_centers) > 0 else None,
                                "w_max": float(max(c[1] for c in all_crop_centers)) if all_crop_centers and len(all_crop_centers) > 0 else None,
                                "d_min": float(min(c[2] for c in all_crop_centers)) if all_crop_centers and len(all_crop_centers) > 0 else None,
                                "d_max": float(max(c[2] for c in all_crop_centers)) if all_crop_centers and len(all_crop_centers) > 0 else None,
                                "volume_shape": list(image.shape[1:])
                            } if all_crop_centers is not None else None
                        },
                        "timestamp": int(time.time() * 1000)
                    }, ensure_ascii=False) + "\n")
                    log_file.flush()
            except Exception as e:
                pass
        
        full_logits = full_logits / weight_sum.unsqueeze(0).clamp_min(1e-6)  # (C, H, W, D) / (1, H, W, D) -> (C, H, W, D)
        
        # 디버깅: 정규화 후 full_logits 분포 확인 (첫 번째 샘플만)
        if debug_sample_idx == 0:
            try:
                full_logits_after_norm_min = float(full_logits.min().item())
                full_logits_after_norm_max = float(full_logits.max().item())
                full_logits_after_norm_mean = float(full_logits.mean().item())
                
                pred_argmax = torch.argmax(full_logits, dim=0)
                pred_unique = torch.unique(pred_argmax).cpu().tolist()
                pred_class_counts = {int(k): int(v) for k, v in zip(*torch.unique(pred_argmax, return_counts=True))}
                
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "blending-check",
                        "hypothesisId": "H10",
                        "location": "cascade_utils.py:run_cascade_inference",
                        "message": "After blending normalization - full_logits stats and prediction",
                        "data": {
                            "full_logits_after_norm": {
                                "min": full_logits_after_norm_min,
                                "max": full_logits_after_norm_max,
                                "mean": full_logits_after_norm_mean,
                                "shape": list(full_logits.shape)
                            },
                            "prediction_stats": {
                                "unique_classes": pred_unique,
                                "class_counts": pred_class_counts,
                                "total_voxels": int(pred_argmax.numel())
                            }
                        },
                        "timestamp": int(time.time() * 1000)
                    }, ensure_ascii=False) + "\n")
                    log_file.flush()
            except Exception as e:
                pass
    
    # full_logits는 항상 (C, H, W, D) 형태여야 함 (배치 차원 없음)
    assert full_logits.dim() == 4, f"full_logits should be 4D (C, H, W, D), got {full_logits.shape}"

    full_mask = torch.argmax(full_logits, dim=0)

    result = {
        'roi': roi_info,
        'full_logits': full_logits,
        'full_mask': full_mask,
        # 참고용: 첫 번째 crop 정보만 유지 (필요시 확장 가능)
        'crop_origin': None,
    }
    if return_attention and all_attention_weights is not None:
        result['attention_weights'] = all_attention_weights
    if return_timing:
        result['timing'] = {
            'roi_time': roi_time,
            'seg_time': seg_time_total,
            'coord_map_time': coord_map_time,
            'num_seg_calls': num_seg_calls,
            'num_centers': len(centers_full),
            'num_crops': num_seg_calls,
        }
    return result


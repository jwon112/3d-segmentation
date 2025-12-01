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
    resize_volume,
    crop_volume_with_center,
    paste_patch_to_volume,
    compute_tumor_center,
)


def _prepare_roi_input(image: torch.Tensor, roi_resize: Sequence[int], include_coords: bool = True) -> torch.Tensor:
    """Concatenate coord map and resize to ROI resolution."""
    coord_map = get_normalized_coord_map(image.shape[1:], device=image.device)
    roi_input = torch.cat([image, coord_map], dim=0) if include_coords else image
    roi_input = resize_volume(roi_input, roi_resize, mode='trilinear')
    return roi_input


def _scale_center(center_roi: Tuple[float, float, float], original_shape: Sequence[int], roi_shape: Sequence[int]) -> Tuple[float, float, float]:
    """Map ROI-space center coordinates back to original volume space."""
    scales = [original_shape[i] / float(max(1, roi_shape[i])) for i in range(3)]
    return (
        center_roi[0] * scales[0],
        center_roi[1] * scales[1],
        center_roi[2] * scales[2],
    )


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


def run_roi_localization(
    roi_model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_resize: Sequence[int] = (64, 64, 64),
    include_coords: bool = True,
    max_instances: int = 1,
    min_component_size: int = 50,
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
    roi_input = _prepare_roi_input(image, roi_resize, include_coords=include_coords).unsqueeze(0).to(device)
    with torch.no_grad():
        roi_logits = roi_model(roi_input)
    roi_probs = torch.softmax(roi_logits, dim=1)
    roi_mask = torch.argmax(roi_probs, dim=1).squeeze(0).cpu()  # (H, W, D) with 0/1

    centers_roi = _extract_roi_centers(
        roi_mask=roi_mask,
        max_instances=max_instances,
        min_component_size=min_component_size,
    )
    centers_full = [
        _scale_center(c, image.shape[1:], roi_resize) for c in centers_roi
    ]

    return {
        'centers_full': centers_full,
        'centers_roi': centers_roi,
        'roi_mask': roi_mask,
        'roi_logits': roi_logits.detach().cpu(),
        # backward compatibility (단일 중심만 사용하던 코드 대비)
        'center_full': centers_full[0],
        'center_roi': centers_roi[0],
    }


def build_segmentation_input(
    image: torch.Tensor,
    center: Sequence[float],
    crop_size: Sequence[int],
    include_coords: bool = True,
) -> Dict:
    """Create 7-channel crop (4 MRI + 3 coord) around predicted center."""
    seg_patch, origin = crop_volume_with_center(image, center, crop_size, return_origin=True)
    inputs = seg_patch
    if include_coords:
        coord_map = get_normalized_coord_map(image.shape[1:], device=image.device)
        coord_patch = crop_volume_with_center(coord_map, center, crop_size, return_origin=False)
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
) -> List[Tuple[float, float, float]]:
    """
    중심 주변에서 여러 crop 위치를 생성합니다.
    
    Args:
        center: 중심 좌표 (cy, cx, cz)
        crop_size: crop 크기 (h, w, d)
        crops_per_center: 중심당 crop 개수 (1, 2, 3 등)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
    
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
    
    return centers


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
    max_instances: int = 3,
    min_component_size: int = 50,
    crops_per_center: int = 1,
    crop_overlap: float = 0.5,
    use_blending: bool = True,
    return_attention: bool = False,
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
    roi_info = run_roi_localization(
        roi_model=roi_model,
        image=image,
        device=device,
        roi_resize=roi_resize,
        include_coords=include_coords,
        max_instances=max_instances,
        min_component_size=min_component_size,
    )

    centers_full = roi_info.get('centers_full') or [roi_info['center_full']]

    seg_model.eval()
    
    # MobileViT attention 수집 (첫 번째 crop만)
    all_attention_weights = [] if return_attention else None
    
    # Blending을 사용하는 경우 accumulator와 weight sum 초기화
    if use_blending and crops_per_center > 1:
        # 출력 채널 수 확인
        # Cascade segmentation 모델은 항상 7채널 입력(4 MRI + 3 CoordConv)을 기대
        dummy_input = build_segmentation_input(
            image=image,
            center=centers_full[0],
            crop_size=crop_size,
            include_coords=True,  # Cascade segmentation 모델은 항상 CoordConv 포함
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

    for center in centers_full:
        # 각 중심마다 여러 crop 위치 생성
        crop_centers = _generate_multi_crop_centers(
            center=center,
            crop_size=crop_size,
            crops_per_center=crops_per_center,
            crop_overlap=crop_overlap,
        )
        
        for crop_center in crop_centers:
            # Cascade segmentation 모델은 항상 7채널 입력(4 MRI + 3 CoordConv)을 기대
            seg_inputs = build_segmentation_input(
                image=image,
                center=crop_center,
                crop_size=crop_size,
                include_coords=True,  # Cascade segmentation 모델은 항상 CoordConv 포함
            )
            seg_tensor = seg_inputs['inputs'].unsqueeze(0).to(device)
            with torch.no_grad():
                if return_attention and all_attention_weights is not None and len(all_attention_weights) == 0:
                    # MobileViT attention 수집 (첫 번째 crop만)
                    try:
                        seg_result = seg_model(seg_tensor, return_attention=True)
                        if isinstance(seg_result, tuple):
                            seg_logits, attn_dict = seg_result
                            if attn_dict is not None and len(attn_dict) > 0:
                                all_attention_weights.append(attn_dict)
                                print(f"[Cascade] Collected MobileViT attention weights: {len(attn_dict)} layers")
                            else:
                                print(f"[Cascade] Warning: seg_model returned tuple but attn_dict is None or empty")
                                seg_logits = seg_result[0] if len(seg_result) > 0 else seg_result
                        else:
                            print(f"[Cascade] Warning: seg_model did not return tuple for attention. Got type: {type(seg_result)}")
                            seg_logits = seg_result
                    except Exception as e:
                        print(f"[Cascade] Warning: Failed to collect attention weights: {e}")
                        import traceback
                        traceback.print_exc()
                        seg_logits = seg_model(seg_tensor, return_attention=False)
                else:
                    seg_logits = seg_model(seg_tensor, return_attention=False)
            patch_logits = seg_logits.squeeze(0).cpu()  # (C, h, w, d)
            
            if use_blending and crops_per_center > 1 and blend_weights is not None:
                # Cosine blending 사용
                patch_full_logits = paste_patch_to_volume(
                    patch_logits,
                    seg_inputs['origin'],
                    image.shape[1:],
                )  # (C, H, W, D)
                patch_weights = paste_patch_to_volume(
                    blend_weights.squeeze(0).squeeze(0),  # (h, w, d)
                    seg_inputs['origin'],
                    image.shape[1:],
                )  # (H, W, D)
                
                # patch_weights를 (1, H, W, D)로 확장하여 broadcasting
                full_logits += patch_full_logits * patch_weights.unsqueeze(0)  # (C, H, W, D) * (1, H, W, D) -> (C, H, W, D)
                weight_sum += patch_weights  # (H, W, D)
            else:
                # Voxel-wise max 사용 (기존 방식)
                patch_full_logits = paste_patch_to_volume(
                    patch_logits,
                    seg_inputs['origin'],
                    image.shape[1:],
                )
                
                if full_logits is None:
                    full_logits = patch_full_logits
                else:
                    full_logits = torch.maximum(full_logits, patch_full_logits)

    if full_logits is None:
        # 안전장치: ROI가 완전히 비었을 경우, 전부 background로 설정
        c = seg_model.out_channels if hasattr(seg_model, "out_channels") else 4
        full_logits = torch.zeros((c,) + image.shape[1:], dtype=torch.float32)  # (C, H, W, D)
    elif use_blending and weight_sum is not None:
        # Blending 가중치로 정규화
        # full_logits: (C, H, W, D), weight_sum: (H, W, D)
        full_logits = full_logits / weight_sum.unsqueeze(0).clamp_min(1e-6)  # (C, H, W, D) / (1, H, W, D) -> (C, H, W, D)
    
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
    return result


#!/usr/bin/env python3
"""
Cascade segmentation helpers (ROI + crop + CoordConv pipeline).

현재 구현:
- ROI detector로 WT binary를 예측하고, connected components 기반으로 여러 중심을 추출
- 각 중심 주변에서 multi-crop을 수행하여 segmentation 모델을 여러 번 실행
- crop별 logits를 원본 공간으로 복원한 뒤 voxel-wise max로 병합
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
) -> Dict:
    """
    Full cascade inference: ROI -> multi-crop -> segmentation -> merge & uncrop.

    - ROI detector에서 WT binary를 예측하고 여러 컴포넌트 중심을 얻음
    - 각 중심마다 crop → segmentation 수행
    - crop별 logits를 원본 공간으로 복원 후 voxel-wise max로 병합
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
    full_logits: Optional[torch.Tensor] = None

    for center in centers_full:
        seg_inputs = build_segmentation_input(
            image=image,
            center=center,
            crop_size=crop_size,
            include_coords=include_coords,
        )
        seg_tensor = seg_inputs['inputs'].unsqueeze(0).to(device)
        with torch.no_grad():
            seg_logits = seg_model(seg_tensor)
        patch_logits = seg_logits.squeeze(0).cpu()  # (C, h, w, d)
        patch_full_logits = paste_patch_to_volume(
            patch_logits,
            seg_inputs['origin'],
            image.shape[1:],
        )

        if full_logits is None:
            full_logits = patch_full_logits
        else:
            # voxel-wise max merge (logits 기준)
            full_logits = torch.maximum(full_logits, patch_full_logits)

    if full_logits is None:
        # 안전장치: ROI가 완전히 비었을 경우, 전부 background로 설정
        c = seg_model.out_channels if hasattr(seg_model, "out_channels") else 4
        full_logits = torch.zeros((c,) + image.shape[1:], dtype=torch.float32)

    full_mask = torch.argmax(full_logits, dim=0)

    return {
        'roi': roi_info,
        'full_logits': full_logits,
        'full_logits': full_logits,
        'full_mask': full_mask,
        # 참고용: 첫 번째 crop 정보만 유지 (필요시 확장 가능)
        'crop_origin': None,
    }


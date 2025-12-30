"""
Cascade / CoordConv 관련 Dataset 및 Loader.
"""

import math
import random
from typing import Sequence, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .brats_base import BratsDataset3D, get_brats_base_datasets

_COORD_MAP_CACHE: Dict[Tuple[int, int, int], torch.Tensor] = {}


def _to_3tuple(size: Sequence[int]) -> Tuple[int, int, int]:
    if len(size) != 3:
        raise ValueError(f"Expected 3 spatial dims, got {size}")
    return int(size[0]), int(size[1]), int(size[2])


def get_normalized_coord_map(spatial_shape: Sequence[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """Simple 3-channel normalized coordinate map (0-1 range)."""
    shape = _to_3tuple(spatial_shape)
    if shape not in _COORD_MAP_CACHE:
        h, w, d = shape
        ys = torch.linspace(0.0, 1.0, steps=h, dtype=torch.float32)
        xs = torch.linspace(0.0, 1.0, steps=w, dtype=torch.float32)
        zs = torch.linspace(0.0, 1.0, steps=d, dtype=torch.float32)
        grid = torch.meshgrid(ys, xs, zs, indexing='ij')
        coord = torch.stack(grid, dim=0)
        _COORD_MAP_CACHE[shape] = coord.contiguous()
    coord_map = _COORD_MAP_CACHE[shape]
    if device is not None:
        coord_map = coord_map.to(device)
    return coord_map


def get_hybrid_coord_map(spatial_shape: Sequence[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """Hybrid coordinate map: 3 linear channels + 6 Fourier feature channels (9 channels total).
    
    Returns:
        Tensor of shape (9, H, W, D): 3 linear + 6 Fourier features
    """
    shape = _to_3tuple(spatial_shape)
    
    # 캐시 확인 (같은 shape에 대해서는 재사용)
    cache_key = ('hybrid', shape)
    if cache_key not in _COORD_MAP_CACHE:
        h, w, d = shape
        
        # Linear coordinates (normalized 0-1)
        ys = torch.linspace(0.0, 1.0, steps=h, dtype=torch.float32)
        xs = torch.linspace(0.0, 1.0, steps=w, dtype=torch.float32)
        zs = torch.linspace(0.0, 1.0, steps=d, dtype=torch.float32)
        grid_y, grid_x, grid_z = torch.meshgrid(ys, xs, zs, indexing='ij')
        
        # Linear coordinates (3 channels)
        linear_coord = torch.stack([grid_y, grid_x, grid_z], dim=0)
        
        # Fourier features: 6 channels (sin/cos for 3 axes at 1 frequency)
        # Use frequency 1.0 for all axes
        freq = 1.0
        fourier_list = [
            torch.sin(2 * torch.pi * freq * grid_y),
            torch.cos(2 * torch.pi * freq * grid_y),
            torch.sin(2 * torch.pi * freq * grid_x),
            torch.cos(2 * torch.pi * freq * grid_x),
            torch.sin(2 * torch.pi * freq * grid_z),
            torch.cos(2 * torch.pi * freq * grid_z),
        ]
        
        # Stack to get 6 channels (3 axes × 2 sin/cos)
        # Total: 3 linear + 6 Fourier = 9 channels
        fourier_coord = torch.stack(fourier_list, dim=0)  # (6, H, W, D)
        
        # Combine: 3 linear + 6 Fourier = 9 channels
        hybrid_coord = torch.cat([linear_coord, fourier_coord], dim=0)  # (9, H, W, D)
        _COORD_MAP_CACHE[cache_key] = hybrid_coord.contiguous()
    
    hybrid_coord = _COORD_MAP_CACHE[cache_key]
    if device is not None:
        hybrid_coord = hybrid_coord.to(device)
    return hybrid_coord


def get_coord_map(spatial_shape: Sequence[int], device: Optional[torch.device] = None, encoding_type: str = 'simple') -> torch.Tensor:
    """Get coordinate map based on encoding type.
    
    Args:
        spatial_shape: Spatial dimensions (H, W, D)
        device: Target device
        encoding_type: 'simple' (3 channels) or 'hybrid' (9 channels)
    
    Returns:
        Coordinate map tensor: (3, H, W, D) for 'simple', (9, H, W, D) for 'hybrid'
    """
    if encoding_type == 'simple':
        return get_normalized_coord_map(spatial_shape, device)
    elif encoding_type == 'hybrid':
        return get_hybrid_coord_map(spatial_shape, device)
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}. Must be 'simple' or 'hybrid'")


def _apply_anisotropy_resize(
    img: torch.Tensor,
    mask: torch.Tensor,
    target_size: Sequence[int],
    prob: float = 0.4,
    scale_range: Tuple[float, float] = (0.7, 1.3),
    sample_idx: Optional[int] = None,
    deterministic: bool = False,
):
    """
    Depth 축만 이방성 스케일 후 다시 target_size로 리샘플해 through-plane 변화를 모사.
    - img: (C, D, H, W), mask: (D, H, W)
    - deterministic=True일 때는 sample_idx 기반으로 재현 가능한 augmentation 적용
    """
    # Deterministic 모드: sample_idx 기반으로 고정된 랜덤 값 생성
    if deterministic and sample_idx is not None:
        # 샘플 인덱스를 seed로 사용하여 재현 가능한 랜덤 값 생성
        rng = random.Random(sample_idx)
        should_apply = rng.random() < prob
        if not should_apply:
            return img, mask
        
        # 고정된 scale factor 생성
        log_r = rng.uniform(math.log(scale_range[0]), math.log(scale_range[1]))
        scale_factor = math.exp(log_r)
    else:
        # Training 모드: 랜덤하게 적용
        if torch.rand(1).item() >= prob:
            return img, mask
        log_r = torch.empty(1, device=img.device).uniform_(math.log(scale_range[0]), math.log(scale_range[1])).exp().item()
        scale_factor = log_r

    C, D, H, W = img.shape
    tD, tH, tW = _to_3tuple(target_size)
    new_D = max(2, int(round(D * scale_factor)))

    def _interp(tensor, size, mode):
        vol = tensor.unsqueeze(0)  # (1, C, D, H, W) 또는 (1, 1, D, H, W)
        kwargs = {"size": size, "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        out = F.interpolate(vol, **kwargs)
        return out.squeeze(0)

    img_scaled = _interp(img, (new_D, H, W), "trilinear")
    mask_scaled = _interp(mask.unsqueeze(0).float(), (new_D, H, W), "nearest").squeeze(0).long()

    img_resized = _interp(img_scaled, (tD, tH, tW), "trilinear")
    mask_resized = _interp(mask_scaled.unsqueeze(0).float(), (tD, tH, tW), "nearest").squeeze(0).long()
    return img_resized, mask_resized


def resize_volume(volume: torch.Tensor, target_size: Sequence[int], mode: str = 'trilinear') -> torch.Tensor:
    if volume.ndim != 4:
        raise ValueError(f"Expected tensor with shape (C, H, W, D), got {volume.shape}")
    target = _to_3tuple(target_size)
    if list(volume.shape[1:]) == list(target):
        return volume
    vol = volume.unsqueeze(0)
    vol = vol.permute(0, 1, 4, 2, 3)
    interp_kwargs = {}
    if mode in {'linear', 'bilinear', 'bicubic', 'trilinear'}:
        interp_kwargs['align_corners'] = False
    target_dhw = (target[2], target[0], target[1])
    vol = F.interpolate(vol, size=target_dhw, mode=mode, **interp_kwargs)
    vol = vol.permute(0, 1, 3, 4, 2)
    return vol.squeeze(0)


def crop_volume_with_center(tensor: torch.Tensor, center: Sequence[float], crop_size: Sequence[int], return_origin: bool = False, debug_sample_idx: int = -1):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dims, got {tensor.shape}")

    c, h, w, d = tensor.shape
    size = _to_3tuple(crop_size)
    # center는 (cy, cx, cz) 순서로 들어옴
    # tensor.shape는 (C, H, W, D) = (C, h, w, d) 순서
    # 따라서 center[0]=cy는 h 차원, center[1]=cx는 w 차원, center[2]=cz는 d 차원에 매칭
    cy, cx, cz = [float(c_val) for c_val in center]
    half_h, half_w, half_d = [s / 2.0 for s in size]
    
    # 각 차원별로 시작 위치 계산
    start_h = int(round(cy - half_h))
    start_w = int(round(cx - half_w))
    start_d = int(round(cz - half_d))
    starts = [start_h, start_w, start_d]

    src_ranges = []
    dst_ranges = []
    origins = []
    for dim, start, sz in zip((h, w, d), starts, size):
        end = start + sz
        src_start = max(0, start)
        src_end = min(end, dim)
        dst_start = max(0, -start)
        copy_len = max(0, src_end - src_start)
        dst_end = dst_start + copy_len
        origins.append(src_start)
        src_ranges.append((src_start, src_end))
        dst_ranges.append((dst_start, dst_end))
    
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
                    "hypothesisId": "H5",
                    "location": "cascade.py:crop_volume_with_center",
                    "message": "Crop volume with center - coordinate mapping",
                    "data": {
                        "input_center": [float(cy), float(cx), float(cz)],
                        "tensor_shape": [int(c), int(h), int(w), int(d)],
                        "crop_size": list(size),
                        "half_sizes": [float(half_h), float(half_w), float(half_d)],
                        "starts": [int(start_h), int(start_w), int(start_d)],
                        "origins": list(origins),
                        "src_ranges": [[int(r[0]), int(r[1])] for r in src_ranges],
                        "dst_ranges": [[int(r[0]), int(r[1])] for r in dst_ranges]
                    },
                    "timestamp": int(time.time() * 1000)
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    patch = tensor.new_zeros((c, size[0], size[1], size[2]))
    if all(r[1] - r[0] > 0 for r in src_ranges):
        patch[
            :,
            dst_ranges[0][0]:dst_ranges[0][1],
            dst_ranges[1][0]:dst_ranges[1][1],
            dst_ranges[2][0]:dst_ranges[2][1],
        ] = tensor[
            :,
            src_ranges[0][0]:src_ranges[0][1],
            src_ranges[1][0]:src_ranges[1][1],
            src_ranges[2][0]:src_ranges[2][1],
        ]

    if squeeze:
        patch = patch.squeeze(0)
    if return_origin:
        return patch, tuple(origins)
    return patch


def paste_patch_to_volume(tensor: torch.Tensor, origin: Sequence[int], full_shape: Sequence[int], debug_sample_idx: int = -1):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dims, got {tensor.shape}")

    full_shape = _to_3tuple(full_shape)
    full = tensor.new_zeros((tensor.shape[0],) + full_shape)
    y0, x0, z0 = [int(max(0, o)) for o in origin]
    y1 = min(y0 + tensor.shape[1], full_shape[0])
    x1 = min(x0 + tensor.shape[2], full_shape[1])
    z1 = min(z0 + tensor.shape[3], full_shape[2])
    
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
                    "hypothesisId": "H5",
                    "location": "cascade.py:paste_patch_to_volume",
                    "message": "Paste patch to volume - coordinate mapping",
                    "data": {
                        "input_origin": list(origin),
                        "tensor_shape": list(tensor.shape),
                        "full_shape": list(full_shape),
                        "paste_coords": {
                            "y0": int(y0), "y1": int(y1),
                            "x0": int(x0), "x1": int(x1),
                            "z0": int(z0), "z1": int(z1)
                        },
                        "paste_size": [int(y1-y0), int(x1-x0), int(z1-z0)]
                    },
                    "timestamp": int(time.time() * 1000)
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    if y1 <= y0 or x1 <= x0 or z1 <= z0:
        return full.squeeze(0) if squeeze else full
    fy = y1 - y0
    fx = x1 - x0
    fz = z1 - z0
    full[:, y0:y1, x0:x1, z0:z1] = tensor[:, :fy, :fx, :fz]
    return full.squeeze(0) if squeeze else full


def compute_tumor_center(mask: torch.Tensor) -> Tuple[float, float, float]:
    fg = (mask > 0).nonzero(as_tuple=False)
    if fg.numel() == 0:
        h, w, d = mask.shape
        return (h / 2.0, w / 2.0, d / 2.0)
    center = fg.float().mean(dim=0)
    cy, cx, cz = center.tolist()
    return float(cy), float(cx), float(cz)


def _generate_multi_crop_centers(
    center: Tuple[float, float, float],
    crop_size: Sequence[int],
    crops_per_center: int = 1,
    crop_overlap: float = 0.5,
) -> list:
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


class BratsCascadeROIDataset(Dataset):
    """Resize full volume + append coord maps (for ROI detection)."""

    def __init__(
        self,
        base_dataset: Dataset,
        target_size: Sequence[int] = (64, 64, 64),
        include_coords: bool = True,
        coord_encoding_type: str = 'simple',
        augment: bool = False,
        anisotropy_augment: bool = False,
        deterministic: bool = False,  # Validation/Test에서 재현 가능한 augmentation을 위한 플래그
    ):
        self.base_dataset = base_dataset
        self.target_size = _to_3tuple(target_size)
        self.include_coords = include_coords
        self.coord_encoding_type = coord_encoding_type
        self.augment = augment
        self.anisotropy_augment = anisotropy_augment
        self.deterministic = deterministic

    def __len__(self):
        return len(self.base_dataset)

    def _maybe_augment(self, img_vol: torch.Tensor, mask_vol: torch.Tensor, sample_idx: Optional[int] = None):
        if not self.augment and not self.anisotropy_augment:
            return img_vol, mask_vol
        
        if self.anisotropy_augment:
            img_vol, mask_vol = _apply_anisotropy_resize(
                img_vol, mask_vol, target_size=self.target_size, prob=0.4, scale_range=(0.7, 1.3),
                sample_idx=sample_idx, deterministic=self.deterministic
            )
            # anisotropy만 적용하고 intensity aug는 건너뛸 수 있도록 분리
            if not self.augment:
                return img_vol, mask_vol
        
        # 1. Multi-axis flipping (각 축에 대해 독립적으로)
        # img_vol: (C, D, H, W), mask_vol: (D, H, W)
        if torch.rand(1).item() < 0.5:
            img_vol = torch.flip(img_vol, dims=(1,))  # Depth flip
            mask_vol = torch.flip(mask_vol, dims=(0,))  # Depth flip
        if torch.rand(1).item() < 0.5:
            img_vol = torch.flip(img_vol, dims=(2,))  # Height flip
            mask_vol = torch.flip(mask_vol, dims=(1,))  # Height flip
        if torch.rand(1).item() < 0.5:
            img_vol = torch.flip(img_vol, dims=(3,))  # Width flip
            mask_vol = torch.flip(mask_vol, dims=(2,))  # Width flip
        
        # 2. 90-degree rotations (의료 영상에 적합)
        if torch.rand(1).item() < 0.5:
            # XY plane rotation (90, 180, 270도)
            k = torch.randint(1, 4, (1,)).item()
            img_vol = torch.rot90(img_vol, k=k, dims=(2, 3))  # Height, Width
            mask_vol = torch.rot90(mask_vol, k=k, dims=(1, 2))  # Height, Width
        
        # 3. Intensity augmentation: Scale + Shift (강화)
        if torch.rand(1).item() < 0.5:
            scale = 1.0 + 0.15 * torch.randn(1).item()  # 0.1 -> 0.15
            shift = 0.08 * torch.randn(1).item()  # 0.05 -> 0.08
            img_vol = img_vol * scale + shift
        
        # 4. Gamma correction (강화)
        if torch.rand(1).item() < 0.4:  # 0.3 -> 0.4
            gamma = 0.6 + 0.8 * torch.rand(1).item()  # 0.7~1.3 -> 0.6~1.4
            sign = torch.sign(img_vol)
            img_vol = sign * (torch.abs(img_vol) ** gamma)
        
        # 5. Contrast adjustment
        if torch.rand(1).item() < 0.3:
            # Contrast: (x - mean) * factor + mean
            mean = img_vol.mean()
            factor = 0.7 + 0.6 * torch.rand(1).item()  # 0.7~1.3
            img_vol = (img_vol - mean) * factor + mean
        
        # 6. Brightness adjustment
        if torch.rand(1).item() < 0.3:
            brightness = 0.1 * torch.randn(1).item()
            img_vol = img_vol + brightness
        
        # 7. Gaussian noise (강화)
        if torch.rand(1).item() < 0.4:  # 0.3 -> 0.4
            noise_std = 0.02 + 0.03 * torch.rand(1).item()  # 0.02~0.05 (기존 0.03 고정)
            noise = torch.randn_like(img_vol) * noise_std
            img_vol = img_vol + noise
        
        return img_vol, mask_vol

    def __getitem__(self, idx):
        # base_dataset이 (image, mask) 또는 (image, mask, fg_coords_dict) 반환 가능
        loaded_data = self.base_dataset[idx]
        if len(loaded_data) == 3:
            image, mask, _ = loaded_data  # fg_coords_dict는 cascade에서는 사용 안 함
        else:
            image, mask = loaded_data
        coord_map = get_coord_map(mask.shape, device=image.device, encoding_type=self.coord_encoding_type)
        if self.include_coords:
            roi_input = torch.cat([image, coord_map], dim=0)
        else:
            roi_input = image
        resized_image = resize_volume(roi_input, self.target_size, mode='trilinear')
        wt_mask = (mask > 0).float().unsqueeze(0)
        resized_mask = resize_volume(wt_mask, self.target_size, mode='nearest').squeeze(0).long()
        resized_image, resized_mask = self._maybe_augment(resized_image, resized_mask, sample_idx=idx)
        return resized_image, resized_mask


class BratsCascadeSegmentationDataset(Dataset):
    """GT 중심 기반 crop + CoordConv를 위한 Dataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        crop_size: Sequence[int] = (96, 96, 96),
        include_coords: bool = True,
        coord_encoding_type: str = 'simple',
        center_jitter: int = 0,
        augment: bool = False,
        anisotropy_augment: bool = False,
        return_metadata: bool = False,
        crops_per_center: int = 1,
        crop_overlap: float = 0.5,
        deterministic: bool = False,  # Validation/Test에서 재현 가능한 augmentation을 위한 플래그
    ):
        self.base_dataset = base_dataset
        self.crop_size = _to_3tuple(crop_size)
        self.include_coords = include_coords
        self.coord_encoding_type = coord_encoding_type
        self.center_jitter = max(0, int(center_jitter))
        self.augment = augment
        self.anisotropy_augment = anisotropy_augment
        self.return_metadata = return_metadata
        self.crops_per_center = max(1, int(crops_per_center))
        self.crop_overlap = max(0.0, min(1.0, float(crop_overlap)))
        self.deterministic = deterministic

    def __len__(self):
        # Multi-crop 모드: 각 crop을 별도의 샘플로 취급
        # crops_per_center=2 -> 2^3=8개, crops_per_center=3 -> 3^3=27개
        if self.crops_per_center > 1:
            num_crops_per_sample = self.crops_per_center ** 3
            return len(self.base_dataset) * num_crops_per_sample
        return len(self.base_dataset)

    def _maybe_augment(self, img_patch: torch.Tensor, msk_patch: torch.Tensor, sample_idx: Optional[int] = None):
        if not self.augment and not self.anisotropy_augment:
            return img_patch, msk_patch
        
        if self.anisotropy_augment:
            img_patch, msk_patch = _apply_anisotropy_resize(
                img_patch, msk_patch, target_size=self.crop_size, prob=0.4, scale_range=(0.7, 1.3),
                sample_idx=sample_idx, deterministic=self.deterministic
            )
            if not self.augment:
                return img_patch, msk_patch
        
        # 1. Multi-axis flipping (각 축에 대해 독립적으로)
        # img_patch: (C, D, H, W), msk_patch: (D, H, W)
        if torch.rand(1).item() < 0.5:
            img_patch = torch.flip(img_patch, dims=(1,))  # Depth flip
            msk_patch = torch.flip(msk_patch, dims=(0,))  # Depth flip
        if torch.rand(1).item() < 0.5:
            img_patch = torch.flip(img_patch, dims=(2,))  # Height flip
            msk_patch = torch.flip(msk_patch, dims=(1,))  # Height flip
        if torch.rand(1).item() < 0.5:
            img_patch = torch.flip(img_patch, dims=(3,))  # Width flip
            msk_patch = torch.flip(msk_patch, dims=(2,))  # Width flip
        
        # 2. 90-degree rotations (의료 영상에 적합)
        if torch.rand(1).item() < 0.5:
            # XY plane rotation (90, 180, 270도)
            k = torch.randint(1, 4, (1,)).item()
            img_patch = torch.rot90(img_patch, k=k, dims=(2, 3))  # Height, Width
            msk_patch = torch.rot90(msk_patch, k=k, dims=(1, 2))  # Height, Width
        
        # 3. Intensity augmentation: Scale + Shift (강화)
        if torch.rand(1).item() < 0.5:
            scale = 1.0 + 0.15 * torch.randn(1).item()  # 0.1 -> 0.15
            shift = 0.08 * torch.randn(1).item()  # 0.05 -> 0.08
            img_patch = img_patch * scale + shift
        
        # 4. Gamma correction (강화)
        if torch.rand(1).item() < 0.4:  # 0.3 -> 0.4
            gamma = 0.6 + 0.8 * torch.rand(1).item()  # 0.7~1.3 -> 0.6~1.4
            sign = torch.sign(img_patch)
            img_patch = sign * (torch.abs(img_patch) ** gamma)
        
        # 5. Contrast adjustment
        if torch.rand(1).item() < 0.3:
            # Contrast: (x - mean) * factor + mean
            mean = img_patch.mean()
            factor = 0.7 + 0.6 * torch.rand(1).item()  # 0.7~1.3
            img_patch = (img_patch - mean) * factor + mean
        
        # 6. Brightness adjustment
        if torch.rand(1).item() < 0.3:
            brightness = 0.1 * torch.randn(1).item()
            img_patch = img_patch + brightness
        
        # 7. Gaussian noise (강화)
        if torch.rand(1).item() < 0.4:  # 0.3 -> 0.4
            noise_std = 0.02 + 0.03 * torch.rand(1).item()  # 0.02~0.05 (기존 0.03 고정)
            noise = torch.randn_like(img_patch) * noise_std
            img_patch = img_patch + noise
        
        return img_patch, msk_patch

    def __getitem__(self, idx):
        # Multi-crop 모드: 각 crop을 별도의 샘플로 취급
        if self.crops_per_center > 1:
            # base dataset 인덱스와 crop 인덱스 계산
            # crops_per_center=2 -> 2^3=8개 crop per sample
            num_crops_per_sample = self.crops_per_center ** 3
            base_idx = idx // num_crops_per_sample
            crop_idx = idx % num_crops_per_sample
            
            # base_dataset이 (image, mask) 또는 (image, mask, fg_coords_dict) 반환 가능
            loaded_data = self.base_dataset[base_idx]
            if len(loaded_data) == 3:
                image, mask, _ = loaded_data  # fg_coords_dict는 cascade에서는 사용 안 함
            else:
                image, mask = loaded_data
            center = compute_tumor_center(mask)
            if self.center_jitter > 0:
                jitter = torch.randint(-self.center_jitter, self.center_jitter + 1, (3,))
                center = tuple(float(c + j) for c, j in zip(center, jitter.tolist()))
            
            # 모든 crop center 생성
            crop_centers = _generate_multi_crop_centers(
                center=center,
                crop_size=self.crop_size,
                crops_per_center=self.crops_per_center,
                crop_overlap=self.crop_overlap,
            )
            
            # 특정 crop만 선택
            center = crop_centers[crop_idx]
        else:
            # 단일 crop 모드 (기존 방식)
            # base_dataset이 (image, mask) 또는 (image, mask, fg_coords_dict) 반환 가능
            loaded_data = self.base_dataset[idx]
            if len(loaded_data) == 3:
                image, mask, _ = loaded_data  # fg_coords_dict는 cascade에서는 사용 안 함
            else:
                image, mask = loaded_data
            center = compute_tumor_center(mask)
            if self.center_jitter > 0:
                jitter = torch.randint(-self.center_jitter, self.center_jitter + 1, (3,))
                center = tuple(float(c + j) for c, j in zip(center, jitter.tolist()))
        
        img_crop, origin = crop_volume_with_center(image, center, self.crop_size, return_origin=True)
        mask_crop = crop_volume_with_center(mask, center, self.crop_size, return_origin=False).long()
        if self.include_coords:
            coord_map = get_coord_map(mask.shape, device=image.device, encoding_type=self.coord_encoding_type)
            coord_crop = crop_volume_with_center(coord_map, center, self.crop_size, return_origin=False)
            img_crop = torch.cat([img_crop, coord_crop], dim=0)
        img_crop, mask_crop = self._maybe_augment(img_crop, mask_crop, sample_idx=idx)
        if self.return_metadata:
            metadata = {
                'crop_origin': torch.tensor(origin, dtype=torch.long),
                'original_shape': torch.tensor(mask.shape, dtype=torch.long),
                'center': torch.tensor(center, dtype=torch.float32),
                'index': torch.tensor(idx, dtype=torch.long),
            }
            return img_crop, mask_crop, metadata
        return img_crop, mask_crop


def get_cascade_data_loaders(
    data_dir,
    roi_batch_size: int = 2,
    seg_batch_size: int = 1,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    dataset_version: str = 'brats2021',
    seed: Optional[int] = None,
    use_5fold: bool = False,
    fold_idx: Optional[int] = None,
    fold_split_dir: Optional[str] = None,
    roi_resize: Sequence[int] = (64, 64, 64),
    seg_crop_size: Sequence[int] = (96, 96, 96),
    include_coords: bool = True,
    coord_encoding_type: str = 'simple',
    center_jitter: int = 0,
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    use_mri_augmentation: bool = False,
    anisotropy_augment: bool = False,
    train_crops_per_center: int = 1,
    train_crop_overlap: float = 0.5,
    use_4modalities: bool = True,
    preprocessed_dir: Optional[str] = None,
):
    """ROI detection + cascade segmentation loaders with CoordConv support."""
    train_base, val_base, test_base = get_brats_base_datasets(
        data_dir=data_dir,
        dataset_version=dataset_version,
        max_samples=max_samples,
        seed=seed,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        fold_split_dir=fold_split_dir,
        use_4modalities=use_4modalities,
        max_cache_size=0,  # 캐싱 비활성화: 순수 I/O 성능 측정
        preprocessed_dir=preprocessed_dir,
    )

    train_base_dataset = train_base.dataset if hasattr(train_base, 'dataset') else train_base
    
    # Validation/Test dataset은 전체 볼륨을 로드하므로 캐시를 비활성화하여 메모리 사용량 최소화
    if hasattr(val_base, 'dataset'):
        val_base.dataset.max_cache_size = 0
    if hasattr(test_base, 'dataset'):
        test_base.dataset.max_cache_size = 0
    
    # Train dataset 캐싱 비활성화: 순수 I/O 성능 측정
    train_base_dataset.max_cache_size = 0

    roi_train_ds = BratsCascadeROIDataset(
        train_base,
        target_size=roi_resize,
        include_coords=include_coords,
        coord_encoding_type=coord_encoding_type,
        augment=use_mri_augmentation,
        anisotropy_augment=anisotropy_augment,
    )
    roi_val_ds = BratsCascadeROIDataset(
        val_base, 
        target_size=roi_resize, 
        include_coords=include_coords,
        coord_encoding_type=coord_encoding_type,
        augment=False,  # 일반 augmentation은 validation에서 사용 안 함
        anisotropy_augment=anisotropy_augment,  # Validation에도 이방성 augmentation 적용
        deterministic=True,  # Validation에서는 재현 가능한 augmentation 적용
    )
    roi_test_ds = BratsCascadeROIDataset(
        test_base, 
        target_size=roi_resize, 
        include_coords=include_coords,
        coord_encoding_type=coord_encoding_type,
        augment=False,  # 일반 augmentation은 test에서 사용 안 함
        anisotropy_augment=anisotropy_augment,  # Test에도 이방성 augmentation 적용
        deterministic=True,  # Test에서는 재현 가능한 augmentation 적용
    )

    seg_train_ds = BratsCascadeSegmentationDataset(
        train_base,
        crop_size=seg_crop_size,
        include_coords=include_coords,
        coord_encoding_type=coord_encoding_type,
        center_jitter=center_jitter,
        augment=use_mri_augmentation,
        anisotropy_augment=anisotropy_augment,
        return_metadata=False,
        crops_per_center=train_crops_per_center,
        crop_overlap=train_crop_overlap,
    )
    seg_val_ds = BratsCascadeSegmentationDataset(
        val_base,
        crop_size=seg_crop_size,
        include_coords=include_coords,
        coord_encoding_type=coord_encoding_type,
        center_jitter=0,
        augment=False,
        anisotropy_augment=anisotropy_augment,  # Validation에도 이방성 augmentation 적용
        return_metadata=False,
    )
    seg_test_ds = BratsCascadeSegmentationDataset(
        test_base,
        crop_size=seg_crop_size,
        include_coords=include_coords,
        coord_encoding_type=coord_encoding_type,
        center_jitter=0,
        augment=False,
        anisotropy_augment=anisotropy_augment,  # Test에도 이방성 augmentation 적용
        return_metadata=False,  # 일반 평가에서는 metadata 불필요
    )

    roi_train_sampler = roi_val_sampler = roi_test_sampler = None
    seg_train_sampler = seg_val_sampler = seg_test_sampler = None
    if distributed and world_size is not None and rank is not None:
        roi_train_sampler = DistributedSampler(roi_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        roi_val_sampler = DistributedSampler(roi_val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        roi_test_sampler = DistributedSampler(roi_test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        seg_train_sampler = DistributedSampler(seg_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        seg_val_sampler = DistributedSampler(seg_val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        seg_test_sampler = DistributedSampler(seg_test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    def _worker_init_fn(worker_id):
        # 각 worker마다 고유한 seed 설정 (재현성 보장)
        # base_seed + worker_id를 사용하여 각 worker가 다른 seed를 가지도록 함
        base_seed = (seed if seed is not None else 0)
        worker_seed = base_seed + worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    _generator = None
    if seed is not None:
        _generator = torch.Generator()
        _generator.manual_seed(seed)

    def _build_loader(dataset, batch_size, shuffle, sampler=None, pin_memory=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(8 if num_workers > 0 else None),
            worker_init_fn=_worker_init_fn,
            generator=_generator,
        )
    
    def _build_val_test_loader(dataset, batch_size, sampler=None):
        """Validation/Test loader: num_workers=3 사용"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=3,  # val/test 데이터로더에 num_workers 사용
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=_worker_init_fn,
            generator=_generator,
        )

    roi_train_loader = _build_loader(roi_train_ds, roi_batch_size, shuffle=True, sampler=roi_train_sampler)
    roi_val_loader = _build_val_test_loader(roi_val_ds, roi_batch_size, sampler=roi_val_sampler)
    roi_test_loader = _build_val_test_loader(roi_test_ds, roi_batch_size, sampler=roi_test_sampler)

    # Multi-crop 사용 시: 배치 크기는 그대로 유지 (VRAM 여유에 맞게 조정 가능)
    # 각 배치 내에서 모든 샘플의 crop들을 순차 처리 (Gradient Accumulation)
    seg_train_loader = _build_loader(seg_train_ds, seg_batch_size, shuffle=True, sampler=seg_train_sampler)
    seg_val_loader = _build_val_test_loader(seg_val_ds, seg_batch_size, sampler=seg_val_sampler)
    seg_test_loader = _build_val_test_loader(seg_test_ds, seg_batch_size, sampler=seg_test_sampler)

    return {
        'roi': {
            'train': roi_train_loader,
            'val': roi_val_loader,
            'test': roi_test_loader,
            'train_sampler': roi_train_sampler,
            'val_sampler': roi_val_sampler,
            'test_sampler': roi_test_sampler,
        },
        'seg': {
            'train': seg_train_loader,
            'val': seg_val_loader,
            'test': seg_test_loader,
            'train_sampler': seg_train_sampler,
            'val_sampler': seg_val_sampler,
            'test_sampler': seg_test_sampler,
        },
        'base_datasets': {
            'train': train_base,
            'val': val_base,
            'test': test_base,
        },
    }


def get_roi_data_loaders(
    data_dir,
    batch_size: int = 2,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    dataset_version: str = 'brats2021',
    seed: Optional[int] = None,
    use_5fold: bool = False,
    fold_idx: Optional[int] = None,
    roi_resize: Sequence[int] = (64, 64, 64),
    include_coords: bool = True,
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    use_mri_augmentation: bool = False,
    anisotropy_augment: bool = False,
    use_4modalities: bool = True,
):
    """ROI-only dataloaders for training/evaluation."""
    train_base, val_base, test_base = get_brats_base_datasets(
        data_dir=data_dir,
        dataset_version=dataset_version,
        max_samples=max_samples,
        seed=seed,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        use_4modalities=use_4modalities,
    )

    # train_base의 base dataset은 캐시를 유지
    train_base_dataset = train_base.dataset if hasattr(train_base, 'dataset') else train_base
    
    # Validation/Test dataset은 전체 볼륨을 로드하므로 캐시를 비활성화하여 메모리 사용량 최소화
    if hasattr(val_base, 'dataset'):
        val_base.dataset.max_cache_size = 0
    if hasattr(test_base, 'dataset'):
        test_base.dataset.max_cache_size = 0
    
    # Train dataset 캐싱 비활성화: 순수 I/O 성능 측정
    train_base_dataset.max_cache_size = 0

    train_ds = BratsCascadeROIDataset(
        train_base,
        target_size=roi_resize,
        include_coords=include_coords,
        augment=use_mri_augmentation,
        anisotropy_augment=anisotropy_augment,
    )
    val_ds = BratsCascadeROIDataset(val_base, target_size=roi_resize, include_coords=include_coords)
    test_ds = BratsCascadeROIDataset(test_base, target_size=roi_resize, include_coords=include_coords)

    train_sampler = val_sampler = test_sampler = None
    if distributed and world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    def _worker_init_fn(worker_id):
        # 각 worker마다 고유한 seed 설정 (재현성 보장)
        # base_seed + worker_id를 사용하여 각 worker가 다른 seed를 가지도록 함
        base_seed = (seed if seed is not None else 0)
        worker_seed = base_seed + worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    _generator = None
    if seed is not None:
        _generator = torch.Generator()
        _generator.manual_seed(seed)

    def _build_loader(dataset, sampler=None, shuffle=False, pin_memory=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(8 if num_workers > 0 else None),  # 캐싱 대신 prefetch_factor 증가로 I/O 병목 해결
            worker_init_fn=_worker_init_fn,
            generator=_generator,
        )
    
    def _build_val_test_loader(dataset, sampler=None):
        """Validation/Test loader: num_workers=3 사용"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=3,  # val/test 데이터로더에 num_workers 사용
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=_worker_init_fn,
            generator=_generator,
        )

    return {
        'train': _build_loader(train_ds, sampler=train_sampler, shuffle=True),
        'val': _build_val_test_loader(val_ds, sampler=val_sampler),
        'test': _build_val_test_loader(test_ds, sampler=test_sampler),
        'train_sampler': train_sampler,
        'val_sampler': val_sampler,
        'test_sampler': test_sampler,
        'base_datasets': {
            'train': train_base,
            'val': val_base,
            'test': test_base,
        },
    }


__all__ = [
    "get_normalized_coord_map",
    "resize_volume",
    "crop_volume_with_center",
    "paste_patch_to_volume",
    "compute_tumor_center",
    "BratsCascadeROIDataset",
    "BratsCascadeSegmentationDataset",
    "get_cascade_data_loaders",
    "get_roi_data_loaders",
]




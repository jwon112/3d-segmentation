"""
Cascade / CoordConv 관련 Dataset 및 Loader.
"""

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


def crop_volume_with_center(tensor: torch.Tensor, center: Sequence[float], crop_size: Sequence[int], return_origin: bool = False):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dims, got {tensor.shape}")

    c, h, w, d = tensor.shape
    size = _to_3tuple(crop_size)
    center = [float(c_val) for c_val in center]
    half = [s / 2.0 for s in size]
    starts = [int(round(c_val - h_val)) for c_val, h_val in zip(center, half)]

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


def paste_patch_to_volume(tensor: torch.Tensor, origin: Sequence[int], full_shape: Sequence[int]):
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
        augment: bool = False,
    ):
        self.base_dataset = base_dataset
        self.target_size = _to_3tuple(target_size)
        self.include_coords = include_coords
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def _maybe_augment(self, img_vol: torch.Tensor, mask_vol: torch.Tensor):
        if not self.augment:
            return img_vol, mask_vol
        if torch.rand(1).item() < 0.5:
            img_vol = torch.flip(img_vol, dims=(2,))
            mask_vol = torch.flip(mask_vol, dims=(1,))
        if torch.rand(1).item() < 0.5:
            scale = 1.0 + 0.1 * torch.randn(1).item()
            shift = 0.05 * torch.randn(1).item()
            img_vol = img_vol * scale + shift
        if torch.rand(1).item() < 0.3:
            gamma = 0.7 + 0.6 * torch.rand(1).item()
            sign = torch.sign(img_vol)
            img_vol = sign * (torch.abs(img_vol) ** gamma)
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(img_vol) * 0.03
            img_vol = img_vol + noise
        return img_vol, mask_vol

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        coord_map = get_normalized_coord_map(mask.shape, device=image.device)
        if self.include_coords:
            roi_input = torch.cat([image, coord_map], dim=0)
        else:
            roi_input = image
        resized_image = resize_volume(roi_input, self.target_size, mode='trilinear')
        wt_mask = (mask > 0).float().unsqueeze(0)
        resized_mask = resize_volume(wt_mask, self.target_size, mode='nearest').squeeze(0).long()
        resized_image, resized_mask = self._maybe_augment(resized_image, resized_mask)
        return resized_image, resized_mask


class BratsCascadeSegmentationDataset(Dataset):
    """GT 중심 기반 crop + CoordConv를 위한 Dataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        crop_size: Sequence[int] = (96, 96, 96),
        include_coords: bool = True,
        center_jitter: int = 0,
        augment: bool = False,
        return_metadata: bool = False,
        crops_per_center: int = 1,
        crop_overlap: float = 0.5,
    ):
        self.base_dataset = base_dataset
        self.crop_size = _to_3tuple(crop_size)
        self.include_coords = include_coords
        self.center_jitter = max(0, int(center_jitter))
        self.augment = augment
        self.return_metadata = return_metadata
        self.crops_per_center = max(1, int(crops_per_center))
        self.crop_overlap = max(0.0, min(1.0, float(crop_overlap)))

    def __len__(self):
        return len(self.base_dataset)

    def _maybe_augment(self, img_patch: torch.Tensor, msk_patch: torch.Tensor):
        if not self.augment:
            return img_patch, msk_patch
        if torch.rand(1).item() < 0.5:
            img_patch = torch.flip(img_patch, dims=(2,))
            msk_patch = torch.flip(msk_patch, dims=(1,))
        if torch.rand(1).item() < 0.5:
            scale = 1.0 + 0.1 * torch.randn(1).item()
            shift = 0.05 * torch.randn(1).item()
            img_patch = img_patch * scale + shift
        if torch.rand(1).item() < 0.3:
            gamma = 0.7 + 0.6 * torch.rand(1).item()
            sign = torch.sign(img_patch)
            img_patch = sign * (torch.abs(img_patch) ** gamma)
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(img_patch) * 0.03
            img_patch = img_patch + noise
        return img_patch, msk_patch

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        center = compute_tumor_center(mask)
        if self.center_jitter > 0:
            jitter = torch.randint(-self.center_jitter, self.center_jitter + 1, (3,))
            center = tuple(float(c + j) for c, j in zip(center, jitter.tolist()))
        
        # Multi-crop 샘플링: 여러 crop 위치 중 하나를 랜덤하게 선택
        if self.crops_per_center > 1:
            crop_centers = _generate_multi_crop_centers(
                center=center,
                crop_size=self.crop_size,
                crops_per_center=self.crops_per_center,
                crop_overlap=self.crop_overlap,
            )
            # 랜덤하게 하나 선택
            center = crop_centers[random.randint(0, len(crop_centers) - 1)]
        
        img_crop, origin = crop_volume_with_center(image, center, self.crop_size, return_origin=True)
        mask_crop = crop_volume_with_center(mask, center, self.crop_size, return_origin=False).long()
        if self.include_coords:
            coord_map = get_normalized_coord_map(mask.shape, device=image.device)
            coord_crop = crop_volume_with_center(coord_map, center, self.crop_size, return_origin=False)
            img_crop = torch.cat([img_crop, coord_crop], dim=0)
        img_crop, mask_crop = self._maybe_augment(img_crop, mask_crop)
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
    roi_resize: Sequence[int] = (64, 64, 64),
    seg_crop_size: Sequence[int] = (96, 96, 96),
    include_coords: bool = True,
    center_jitter: int = 0,
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    use_mri_augmentation: bool = False,
    train_crops_per_center: int = 1,
    train_crop_overlap: float = 0.5,
):
    """ROI detection + cascade segmentation loaders with CoordConv support."""
    train_base, val_base, test_base = get_brats_base_datasets(
        data_dir=data_dir,
        dataset_version=dataset_version,
        max_samples=max_samples,
        seed=seed,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        use_4modalities=True,
    )

    roi_train_ds = BratsCascadeROIDataset(
        train_base,
        target_size=roi_resize,
        include_coords=include_coords,
        augment=use_mri_augmentation,
    )
    roi_val_ds = BratsCascadeROIDataset(val_base, target_size=roi_resize, include_coords=include_coords)
    roi_test_ds = BratsCascadeROIDataset(test_base, target_size=roi_resize, include_coords=include_coords)

    seg_train_ds = BratsCascadeSegmentationDataset(
        train_base,
        crop_size=seg_crop_size,
        include_coords=include_coords,
        center_jitter=center_jitter,
        augment=use_mri_augmentation,
        return_metadata=False,
        crops_per_center=train_crops_per_center,
        crop_overlap=train_crop_overlap,
    )
    seg_val_ds = BratsCascadeSegmentationDataset(
        val_base,
        crop_size=seg_crop_size,
        include_coords=include_coords,
        center_jitter=0,
        augment=False,
        return_metadata=False,
    )
    seg_test_ds = BratsCascadeSegmentationDataset(
        test_base,
        crop_size=seg_crop_size,
        include_coords=include_coords,
        center_jitter=0,
        augment=False,
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
        base_seed = (seed if seed is not None else 0)
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        random.seed(base_seed)

    def _build_loader(dataset, batch_size, shuffle, sampler=None, pin_memory=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(4 if num_workers > 0 else None),
            worker_init_fn=_worker_init_fn,
        )

    roi_train_loader = _build_loader(roi_train_ds, roi_batch_size, shuffle=True, sampler=roi_train_sampler)
    roi_val_loader = _build_loader(roi_val_ds, roi_batch_size, shuffle=False, sampler=roi_val_sampler, pin_memory=False)
    roi_test_loader = _build_loader(roi_test_ds, roi_batch_size, shuffle=False, sampler=roi_test_sampler, pin_memory=False)

    seg_train_loader = _build_loader(seg_train_ds, seg_batch_size, shuffle=True, sampler=seg_train_sampler)
    seg_val_loader = _build_loader(seg_val_ds, seg_batch_size, shuffle=False, sampler=seg_val_sampler, pin_memory=False)
    seg_test_loader = _build_loader(seg_test_ds, seg_batch_size, shuffle=False, sampler=seg_test_sampler, pin_memory=False)

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
):
    """ROI-only dataloaders for training/evaluation."""
    train_base, val_base, test_base = get_brats_base_datasets(
        data_dir=data_dir,
        dataset_version=dataset_version,
        max_samples=max_samples,
        seed=seed,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        use_4modalities=True,
    )

    train_ds = BratsCascadeROIDataset(
        train_base,
        target_size=roi_resize,
        include_coords=include_coords,
        augment=use_mri_augmentation,
    )
    val_ds = BratsCascadeROIDataset(val_base, target_size=roi_resize, include_coords=include_coords)
    test_ds = BratsCascadeROIDataset(test_base, target_size=roi_resize, include_coords=include_coords)

    train_sampler = val_sampler = test_sampler = None
    if distributed and world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    def _worker_init_fn(worker_id):
        base_seed = (seed if seed is not None else 0)
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        random.seed(base_seed)

    def _build_loader(dataset, sampler=None, shuffle=False, pin_memory=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(4 if num_workers > 0 else None),
            worker_init_fn=_worker_init_fn,
        )

    return {
        'train': _build_loader(train_ds, sampler=train_sampler, shuffle=True),
        'val': _build_loader(val_ds, sampler=val_sampler, shuffle=False, pin_memory=False),
        'test': _build_loader(test_ds, sampler=test_sampler, shuffle=False, pin_memory=False),
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




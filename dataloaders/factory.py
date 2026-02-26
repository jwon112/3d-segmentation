"""
Top-level dataloader factory.

통합 진입점: get_data_loaders
"""

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .brats_base import BratsDataset3D, split_brats_dataset
from .patch_3d import BratsPatchDataset3D


def get_data_loaders(
    data_dir,
    batch_size: int = 1,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    dim: str = '3d',
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    dataset_version: str = 'brats2021',
    seed: Optional[int] = None,
    use_4modalities: bool = False,
    use_5fold: bool = False,
    fold_idx: Optional[int] = None,
    fold_split_dir: Optional[str] = None,
    use_mri_augmentation: bool = False,
    model_name: Optional[str] = None,
    train_crops_per_center: int = 1,
    train_crop_overlap: float = 0.5,
    anisotropy_augment: bool = False,
    use_nnunet_augmentation: bool = False,
    coord_type: str = 'none',
    preprocessed_dir: Optional[str] = None,
):
    """공통 get_data_loaders 진입점 (3D 전용).
    
    Args:
        model_name: 모델 이름 (선택, 로깅/확장용).
        coord_type: 좌표 인코딩 타입 ('none', 'simple', 'hybrid')
    """
    # coord_type에 따라 include_coords와 coord_encoding_type 결정
    if coord_type == 'none':
        include_coords = False
        coord_encoding_type = 'simple'  # 사용 안 하지만 기본값
    elif coord_type == 'simple':
        include_coords = True
        coord_encoding_type = 'simple'
    elif coord_type == 'hybrid':
        include_coords = True
        coord_encoding_type = 'hybrid'
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}. Must be 'none', 'simple', or 'hybrid'")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # fold_split_dir이 지정된 경우: fold별 디렉토리에서 직접 로드 (use_5fold와 무관)
    # use_5fold=False여도 fold_split_dir을 지정하면 해당 fold의 train/val/test를 사용
    if fold_split_dir:
        if dim != '3d':
            raise ValueError("Only 3D is supported (dim='3d')")
        # fold_idx가 지정되지 않았으면 기본값 0 사용
        if fold_idx is None:
            fold_idx = 0
        # Fold별 디렉토리에서 직접 로드
        train_dataset = BratsDataset3D(
            data_dir,
            split='train',
            max_samples=max_samples,
            dataset_version=dataset_version,
            use_4modalities=use_4modalities,
            max_cache_size=0,
            fold_split_dir=fold_split_dir,
            fold_idx=fold_idx,
        )
        val_dataset = BratsDataset3D(
            data_dir,
            split='val',
            max_samples=max_samples,
            dataset_version=dataset_version,
            use_4modalities=use_4modalities,
            max_cache_size=0,
            fold_split_dir=fold_split_dir,
            fold_idx=fold_idx,
        )
        test_dataset = BratsDataset3D(
            data_dir,
            split='test',
            max_samples=max_samples,
            dataset_version=dataset_version,
            use_4modalities=use_4modalities,
            max_cache_size=0,
            fold_split_dir=fold_split_dir,
            fold_idx=fold_idx,
        )
    else:
        # 일반 모드: 3D 전용
        if dim != '3d':
            raise ValueError("Only 3D is supported (dim='3d')")
        dataset_class = BratsDataset3D
        # Training용: 캐싱 비활성화 (순수 I/O 성능 측정)
        full_dataset = dataset_class(
                data_dir,
                split='train',
                max_samples=max_samples,
                dataset_version=dataset_version,
                use_4modalities=use_4modalities,
                max_cache_size=0,  # 캐싱 비활성화: 순수 I/O 성능 측정
                preprocessed_dir=preprocessed_dir,
            )

        train_dataset, val_dataset, test_dataset = split_brats_dataset(
            full_dataset=full_dataset,
            data_dir=data_dir,
            dataset_version=dataset_version,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            use_5fold=use_5fold,
            fold_idx=fold_idx,
            seed=seed,
        )

    train_base_dataset = train_dataset

    # Val/Test는 전체 볼륨 로드 → 캐시 비활성화로 메모리 절약
    if hasattr(val_dataset, 'dataset'):
        if hasattr(val_dataset.dataset, 'max_cache_size'):
            val_dataset.dataset.max_cache_size = 0
    if hasattr(test_dataset, 'dataset'):
        if hasattr(test_dataset.dataset, 'max_cache_size'):
            test_dataset.dataset.max_cache_size = 0

    # Train dataset 캐싱: BratsPatchDataset3D 사용
    if hasattr(train_base_dataset, 'dataset') and hasattr(train_base_dataset.dataset, 'max_cache_size'):
        train_base_dataset.dataset.max_cache_size = 0
    elif hasattr(train_base_dataset, 'max_cache_size'):
        train_base_dataset.max_cache_size = 0

    patch_size = (128, 128, 128)
    if patch_size[0] == 96:
        samples_per_volume = 8
    elif patch_size[0] == 128:
        samples_per_volume = 3
    else:
        samples_per_volume = 16

    train_dataset = BratsPatchDataset3D(
        base_dataset=train_base_dataset,
        patch_size=patch_size,
        samples_per_volume=samples_per_volume,
        augment=use_mri_augmentation if not use_nnunet_augmentation else False,
        anisotropy_augment=anisotropy_augment,
        nnunet_augmentation=use_nnunet_augmentation,
        max_cache_size=50,
    )

    train_sampler = val_sampler = test_sampler = None
    if distributed and world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    def _worker_init_fn(worker_id):
        base_seed = (seed if seed is not None else 0) + worker_id
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        random.seed(base_seed)

    _generator = torch.Generator().manual_seed(seed) if seed is not None else None
    nw = num_workers if num_workers is not None else 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=nw,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=(nw > 0),
        prefetch_factor=(8 if nw > 0 else None),
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        sampler=test_sampler,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


__all__ = ["get_data_loaders"]




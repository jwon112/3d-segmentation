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

from .brats_base import BratsDataset3D, BratsDataset2D, split_brats_dataset
from .patch_3d import BratsPatchDataset3D


def get_data_loaders(
    data_dir,
    batch_size: int = 1,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    dim: str = '3d',
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    dataset_version: str = 'brats2021',
    seed: Optional[int] = None,
    use_4modalities: bool = False,
    use_5fold: bool = False,
    fold_idx: Optional[int] = None,
    use_mri_augmentation: bool = False,
):
    """공통 get_data_loaders 진입점 (기존 data_loader.get_data_loaders와 동일 인터페이스)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    if dim == '2d':
        dataset_class = BratsDataset2D
        full_dataset = dataset_class(data_dir, split='train', max_samples=max_samples, dataset_version=dataset_version)
    else:
        dataset_class = BratsDataset3D
        full_dataset = dataset_class(
            data_dir,
            split='train',
            max_samples=max_samples,
            dataset_version=dataset_version,
            use_4modalities=use_4modalities,
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

    if dim == '3d':
        train_dataset = BratsPatchDataset3D(
            base_dataset=train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset,
            patch_size=(128, 128, 128),
            samples_per_volume=16,
            augment=use_mri_augmentation,
        )

    train_sampler = val_sampler = test_sampler = None
    if distributed and world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    def _worker_init_fn(worker_id):
        base_seed = (seed if seed is not None else 0)
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        random.seed(base_seed)

    _generator = torch.Generator()
    if seed is not None:
        _generator = _generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=(num_workers if num_workers is not None else 8),
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=((num_workers if num_workers is not None else 8) > 0),
        prefetch_factor=(4 if (num_workers if num_workers is not None else 8) > 0 else None),
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )

    v_workers = 4
    t_workers = 4
    val_bs = 1 if dim == '3d' else batch_size
    test_bs = 1 if dim == '3d' else batch_size

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=v_workers,
        pin_memory=False,
        sampler=val_sampler,
        persistent_workers=False,
        prefetch_factor=(4 if v_workers > 0 else None),
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=t_workers,
        pin_memory=False,
        sampler=test_sampler,
        persistent_workers=False,
        prefetch_factor=(4 if t_workers > 0 else None),
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


__all__ = ["get_data_loaders"]




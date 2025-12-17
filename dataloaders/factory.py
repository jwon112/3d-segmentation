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
from .cascade import get_cascade_data_loaders


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
    model_name: Optional[str] = None,
    train_crops_per_center: int = 1,
    train_crop_overlap: float = 0.5,
    anisotropy_augment: bool = False,
):
    """공통 get_data_loaders 진입점 (기존 data_loader.get_data_loaders와 동일 인터페이스).
    
    Args:
        model_name: 모델 이름. cascade 모델인 경우 자동으로 cascade 데이터로더 사용.
    """
    # Cascade 모델인 경우 cascade 데이터로더 사용 (segmentation 단계만)
    if model_name and model_name.startswith('cascade_'):
        cascade_loaders = get_cascade_data_loaders(
            data_dir=data_dir,
            roi_batch_size=batch_size,
            seg_batch_size=batch_size,
            num_workers=num_workers,
            max_samples=max_samples,
            dataset_version=dataset_version,
            seed=seed,
            use_5fold=use_5fold,
            fold_idx=fold_idx,
            roi_resize=(64, 64, 64),  # ROI 모델 기본 크기
            seg_crop_size=(96, 96, 96),  # Segmentation 모델 기본 크기
            include_coords=True,
            center_jitter=0,  # 학습 시에는 jitter 없음 (또는 옵션으로 추가 가능)
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            use_mri_augmentation=use_mri_augmentation,
            anisotropy_augment=anisotropy_augment,
            train_crops_per_center=train_crops_per_center,
            train_crop_overlap=train_crop_overlap,
        )
        # Segmentation 데이터로더만 반환 (일반 get_data_loaders와 동일한 형식)
        seg_loaders = cascade_loaders['seg']
        return (
            seg_loaders['train'],
            seg_loaders['val'],
            seg_loaders['test'],
            seg_loaders['train_sampler'],
            seg_loaders['val_sampler'],
            seg_loaders['test_sampler'],
        )
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
        # Training용: GPU당 num_workers=6 기준 최적화
        full_dataset = dataset_class(
            data_dir,
            split='train',
            max_samples=max_samples,
            dataset_version=dataset_version,
            use_4modalities=use_4modalities,
            max_cache_size=80,  # Application RAM 여유 있음: worker당 80개 볼륨 캐시
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
        # train_dataset의 base dataset은 캐시를 유지 (max_cache_size=80)
        train_base_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
        # train_base_dataset의 max_cache_size는 이미 80으로 설정되어 있음
        
        # Validation/Test dataset은 전체 볼륨을 로드하므로 캐시를 비활성화하여 메모리 사용량 최소화
        # 단, train_base_dataset과는 다른 인스턴스이므로 train에는 영향 없음
        if hasattr(val_dataset, 'dataset'):
            # val_dataset과 train_dataset이 같은 base dataset을 공유하는지 확인
            if val_dataset.dataset is train_base_dataset:
                # 같은 인스턴스를 공유하는 경우, train용으로 별도 복사본 생성
                # 하지만 Subset은 원본을 참조하므로, train_base_dataset의 max_cache_size는 유지
                # 대신 validation/test에서만 캐시를 사용하지 않도록 처리
                # (실제로는 _load_nifti_data에서 max_cache_size > 0 체크를 하므로 문제 없음)
                pass
            val_dataset.dataset.max_cache_size = 0
        if hasattr(test_dataset, 'dataset'):
            if test_dataset.dataset is train_base_dataset:
                pass
            test_dataset.dataset.max_cache_size = 0
        
        # BratsPatchDataset3D가 자체 _volume_cache를 가지고 있으므로,
        # base_dataset의 _nifti_cache는 비활성화하여 중복 캐싱 방지
        # (BratsPatchDataset3D의 캐시만으로도 충분함)
        train_base_dataset.max_cache_size = 0
        
        train_dataset = BratsPatchDataset3D(
            base_dataset=train_base_dataset,
            patch_size=(128, 128, 128),
            samples_per_volume=16,
            augment=use_mri_augmentation,
            anisotropy_augment=anisotropy_augment,
            max_cache_size=80,  # Application RAM 여유 있음: worker당 80개 볼륨 캐시
        )

    train_sampler = val_sampler = test_sampler = None
    if distributed and world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=(num_workers if num_workers is not None else 8),
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=((num_workers if num_workers is not None else 8) > 0),
        prefetch_factor=(8 if (num_workers if num_workers is not None else 8) > 0 else None),  # GPU당 num_workers=6 기준 최적화: wait time 감소를 위해 증가
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )

    # Validation/Test는 전체 볼륨을 로드하므로 메모리 사용량이 큼
    # num_workers와 prefetch_factor를 줄여서 OOM 방지
    v_workers = 2
    t_workers = 2
    val_bs = 1 if dim == '3d' else batch_size
    test_bs = 1 if dim == '3d' else batch_size

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=0,  # 메인 프로세스에서 직접 로드하여 메모리 사용량 최소화
        pin_memory=False,
        sampler=val_sampler,
        persistent_workers=False,
        prefetch_factor=None,
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=0,  # 메인 프로세스에서 직접 로드하여 메모리 사용량 최소화
        pin_memory=False,
        sampler=test_sampler,
        persistent_workers=False,
        prefetch_factor=None,
        worker_init_fn=_worker_init_fn,
        generator=_generator,
    )

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


__all__ = ["get_data_loaders"]




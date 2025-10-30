#!/usr/bin/env python3
"""
BraTS 데이터 로더
표준 NIfTI 데이터를 로드하는 데이터 로더
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import nibabel as nib
from scipy import ndimage


def _normalize_volume_np(vol):
    """퍼센타일 클리핑 + Z-score 정규화 (비영점 마스크 기준)

    Args:
        vol (np.ndarray): (H, W, D) 또는 (H, W) 강도 볼륨
    Returns:
        np.ndarray: 정규화된 볼륨
    """
    nz = vol > 0
    if nz.sum() == 0:
        return vol
    v = vol[nz]
    lo, hi = np.percentile(v, [0.5, 99.5])
    vol = np.clip(vol, lo, hi)
    m = v.mean()
    s = v.std() + 1e-8
    vol[nz] = (vol[nz] - m) / s
    return vol

class BratsDataset3D(Dataset):
    """3D BraTS 데이터셋 (Depth, Height, Width)"""
    
    def __init__(self, data_dir, split='train', max_samples=None):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        
        # 데이터 파일 목록 수집
        self.samples = self._load_samples()
        
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
    
    def _load_samples(self):
        """데이터 샘플 목록 로드"""
        samples = []
        
        # BraTS2021 데이터셋 확인
        brats2021_dir = os.path.join(self.data_dir, 'BraTS2021_Training_Data')
        if not os.path.exists(brats2021_dir):
            raise FileNotFoundError(
                f"BraTS2021 dataset not found at {brats2021_dir}\n"
                f"Please ensure the dataset is extracted in the correct directory."
            )
        
        print(f"Loading BraTS2021 dataset from {brats2021_dir}")
        for patient_dir in sorted(os.listdir(brats2021_dir)):
            patient_path = os.path.join(brats2021_dir, patient_dir)
            if os.path.isdir(patient_path):
                samples.append(patient_path)
        
        if not samples:
            raise ValueError(f"No patient data found in {brats2021_dir}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        return self._load_nifti_data(sample_path)
    
    
    def _load_nifti_data(self, patient_dir):
        """NIfTI 파일 로드 (T1CE, FLAIR만 사용)"""
        files = os.listdir(patient_dir)
        
        # modality 파일 찾기 (T1CE, FLAIR만)
        t1ce_file = [f for f in files if 't1ce' in f.lower()][0]
        flair_file = [f for f in files if 'flair.nii' in f.lower()][0]
        seg_file = [f for f in files if 'seg' in f.lower()][0]
        
        # 데이터 로드
        t1ce = nib.load(os.path.join(patient_dir, t1ce_file)).get_fdata()
        flair = nib.load(os.path.join(patient_dir, flair_file)).get_fdata()
        seg = nib.load(os.path.join(patient_dir, seg_file)).get_fdata()
        
        # 정규화 (채널별, 비영점 기준 퍼센타일 클리핑 + Z-score)
        t1ce = _normalize_volume_np(t1ce)
        flair = _normalize_volume_np(flair)

        # (H, W, D) 형태로 결합 (T1CE, FLAIR만)
        image = np.stack([t1ce, flair], axis=-1)  # (H, W, D, 2)
        
        # 텐서로 변환
        image = torch.from_numpy(np.transpose(image, (3, 0, 1, 2))).float()  # (2, H, W, D)
        mask = torch.from_numpy(seg).long()
        
        # BraTS label 매핑: 4 -> 3 (4는 전체 tumor이고, ET(3)로 매핑)
        mask = torch.where(mask == 4, torch.tensor(3).long(), mask)
        
        return image, mask


class BratsPatchDataset3D(Dataset):
    """3D BraTS 패치 데이터셋 (학습용)

    - 원본 볼륨에서 128x128x128 패치를 랜덤 샘플링
    - 포그라운드(>0) 비율 임계값을 만족하도록 중심 후보 우선 샘플링
    - 패딩으로 경계 안전 처리
    """
    def __init__(self, base_dataset: BratsDataset3D, patch_size=(128, 128, 128),
                 samples_per_volume: int = 8, min_fg_ratio: float = 0.05, max_tries: int = 20):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.min_fg_ratio = min_fg_ratio
        self.max_tries = max_tries

        # 인덱스 매핑: (volume_index, sample_idx)
        self.index_map = []
        for vidx in range(len(self.base_dataset)):
            for s in range(self.samples_per_volume):
                self.index_map.append((vidx, s))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vidx, _ = self.index_map[idx]
        image, mask = self.base_dataset[vidx]  # image: (2, H, W, D), mask: (H, W, D)

        C, H, W, D = image.shape
        ph, pw, pd = self.patch_size

        # 필요 시 패딩 (경계 안전)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        pad_d = max(0, pd - D)
        if pad_h or pad_w or pad_d:
            # pad: (left, right, top, bottom, front, back)
            image = F.pad(image, (0, pad_d, 0, pad_w, 0, pad_h))  # (2, H, W, D)
            mask = F.pad(mask, (0, pad_d, 0, pad_w, 0, pad_h))
            C, H, W, D = image.shape

        # 포그라운드 비율 조건을 만족하는 패치 시도
        tries = 0
        best_patch = None
        best_ratio = -1.0
        while tries < self.max_tries:
            # 라벨에서 포그라운드 좌표 찾기
            fg = (mask > 0).nonzero(as_tuple=False)
            if fg.numel() > 0:
                # 포그라운드에서 랜덤 중심 선택
                ci = fg[torch.randint(0, fg.shape[0], (1,)).item()]
                cy, cx, cz = ci.tolist()
            else:
                # 포그라운드가 없다면 랜덤 중심
                cy = torch.randint(0, H, (1,)).item()
                cx = torch.randint(0, W, (1,)).item()
                cz = torch.randint(0, D, (1,)).item()

            sy = max(0, min(cy - ph // 2, H - ph))
            sx = max(0, min(cx - pw // 2, W - pw))
            sz = max(0, min(cz - pd // 2, D - pd))

            img_patch = image[:, sy:sy+ph, sx:sx+pw, sz:sz+pd]
            msk_patch = mask[sy:sy+ph, sx:sx+pw, sz:sz+pd]

            fg_ratio = (msk_patch > 0).float().mean().item()
            if fg_ratio >= self.min_fg_ratio:
                return img_patch, msk_patch

            if fg_ratio > best_ratio:
                best_ratio = fg_ratio
                best_patch = (img_patch, msk_patch)

            tries += 1

        # 조건 만족 실패 시 가장 좋은 패치 반환
        return best_patch


class BratsDataset2D(Dataset):
    """2D BraTS 데이터셋 (Height, Width) - 슬라이스 단위"""
    
    def __init__(self, data_dir, split='train', max_samples=None):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        
        # 3D 볼륨 샘플 로드
        self.volumes = self._load_samples()
        
        # 각 볼륨의 실제 depth 확인 및 슬라이스 생성
        self.samples = []
        for volume in self.volumes:
            # 각 환자의 depth 확인
            files = os.listdir(volume)
            t1_file = [f for f in files if 't1.nii' in f.lower() and 'seg' not in f and 't1ce' not in f][0]
            t1 = nib.load(os.path.join(volume, t1_file)).get_fdata()
            depth = t1.shape[2]
            
            # 각 슬라이스 생성
            for slice_idx in range(depth):
                self.samples.append((volume, slice_idx))
        
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
    
    def _load_samples(self):
        """데이터 샘플 목록 로드 (3D 볼륨)"""
        samples = []
        
        # BraTS2021 데이터셋 확인
        brats2021_dir = os.path.join(self.data_dir, 'BraTS2021_Training_Data')
        if not os.path.exists(brats2021_dir):
            raise FileNotFoundError(
                f"BraTS2021 dataset not found at {brats2021_dir}\n"
                f"Please ensure the dataset is extracted in the correct directory."
            )
        
        print(f"Loading BraTS2021 dataset for 2D slices from {brats2021_dir}")
        for patient_dir in sorted(os.listdir(brats2021_dir)):
            patient_path = os.path.join(brats2021_dir, patient_dir)
            if os.path.isdir(patient_path):
                samples.append(patient_path)
        
        if not samples:
            raise ValueError(f"No patient data found in {brats2021_dir}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        patient_dir, slice_idx = self.samples[idx]
        return self._load_nifti_slice(patient_dir, slice_idx)
    
    def _load_nifti_slice(self, patient_dir, slice_idx):
        """NIfTI에서 특정 슬라이스 추출 (T1CE, FLAIR만 사용)"""
        files = os.listdir(patient_dir)
        
        # modality 파일 찾기 (T1CE, FLAIR만)
        t1ce_file = [f for f in files if 't1ce' in f.lower()][0]
        flair_file = [f for f in files if 'flair.nii' in f.lower()][0]
        seg_file = [f for f in files if 'seg' in f.lower()][0]
        
        # 데이터 로드
        t1ce = nib.load(os.path.join(patient_dir, t1ce_file)).get_fdata()
        flair = nib.load(os.path.join(patient_dir, flair_file)).get_fdata()
        seg = nib.load(os.path.join(patient_dir, seg_file)).get_fdata()
        
        # 정규화 (채널별, 퍼센타일 클리핑 + Z-score)
        t1ce = _normalize_volume_np(t1ce)
        flair = _normalize_volume_np(flair)

        # 슬라이스 추출 (depth 차원)
        image = np.stack([t1ce, flair], axis=0)  # (2, H, W, D)
        image_slice = image[:, :, :, slice_idx]  # (2, H, W)
        mask_slice = seg[:, :, slice_idx]  # (H, W)
        
        # 텐서로 변환
        image = torch.from_numpy(image_slice).float()  # (2, H, W)
        mask = torch.from_numpy(mask_slice).long()  # (H, W)
        
        # BraTS label 매핑: 4 -> 3 (4는 전체 tumor이고, ET(3)로 매핑)
        mask = torch.where(mask == 4, torch.tensor(3).long(), mask)
        
        # MobileUNETR을 위해 256x256으로 resize (16의 배수)
        if image.shape[1] == 240 or image.shape[2] == 240:
            image = image.unsqueeze(0)  # (1, 2, H, W)
            image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
            image = image.squeeze(0)  # (2, 256, 256)
            
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            mask = F.interpolate(mask.float(), size=(256, 256), mode='nearest')
            mask = mask.squeeze(0).squeeze(0).long()  # (256, 256)
        
        return image, mask


def get_data_loaders(data_dir, batch_size=1, num_workers=0, max_samples=None, 
                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, dim='3d',
                     distributed=False, world_size=None, rank=None):
    """
    데이터 로더 생성
    
    Args:
        data_dir: 데이터 디렉토리 경로
        batch_size: 배치 크기
        num_workers: 워커 수
        max_samples: 최대 샘플 수 (None이면 전체)
        train_ratio: 훈련 세트 비율
        val_ratio: 검증 세트 비율
        test_ratio: 테스트 세트 비율
        dim: '2d' 또는 '3d'
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 데이터셋 선택
    if dim == '2d':
        dataset_class = BratsDataset2D
    else:
        dataset_class = BratsDataset3D
    
    # 전체 데이터셋 로드
    full_dataset = dataset_class(data_dir, split='train', max_samples=max_samples)
    
    # 데이터셋 분할
    total_len = len(full_dataset)
    
    # 샘플이 너무 적으면 모든 데이터를 train으로 사용
    if total_len <= 10:
        train_dataset, val_dataset, test_dataset = full_dataset, full_dataset, full_dataset
    else:
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        test_len = total_len - train_len - val_len
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_len, val_len, test_len]
        )

    # 3D 학습은 패치 데이터셋으로 래핑 (128^3, FG-aware)
    if dim == '3d':
        train_dataset = BratsPatchDataset3D(
            base_dataset=train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset,
            patch_size=(128, 128, 128),
            samples_per_volume=8,
            min_fg_ratio=0.05,
            max_tries=20,
        )
    
    # Distributed samplers
    train_sampler = val_sampler = test_sampler = None
    if distributed and world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    # DataLoader 생성 (분산 시 shuffle=False, sampler 사용)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else 2,
    )
    # Validation/Test: conservative loader settings to avoid /dev/shm issues
    v_workers = 0
    t_workers = 0
    # 3D 검증/테스트는 슬라이딩 윈도우 특성상 batch_size=1 권장
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
        prefetch_factor=(2 if v_workers > 0 else None),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=t_workers,
        pin_memory=False,
        sampler=test_sampler,
        persistent_workers=False,
        prefetch_factor=(2 if t_workers > 0 else None),
    )
    
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler




if __name__ == "__main__":
    # 테스트
    train_loader, val_loader, test_loader = get_data_loaders('data', batch_size=1, max_samples=None)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 샘플 확인
    for image, mask in train_loader:
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        break

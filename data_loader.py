#!/usr/bin/env python3
"""
BraTS 데이터 로더
표준 NIfTI 데이터를 로드하는 데이터 로더
"""

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import nibabel as nib
from scipy import ndimage


def _normalize_volume_np(vol):
    """퍼센타일 클리핑 + Z-score 정규화 (비영점 마스크 기준, 배경은 0 유지)

    - 퍼센타일 추정과 클리핑, 평균/표준편차 계산은 비영점(nz) 영역에서만 수행
    - 정규화 후 배경(~nz)은 항상 0으로 유지
    """
    nz = vol > 0
    if nz.sum() == 0:
        return vol.astype(np.float32)

    v = vol[nz]
    lo, hi = np.percentile(v, [0.5, 99.5])
    v_clipped = np.clip(v, lo, hi)
    m = v_clipped.mean()
    s = v_clipped.std()
    if s < 1e-8:
        s = 1e-8

    out = np.zeros_like(vol, dtype=np.float32)
    out[nz] = ((v_clipped - m) / s).astype(np.float32)
    # 배경(~nz)은 0 유지
    return out

class BratsDataset3D(Dataset):
    """3D BraTS 데이터셋 (Depth, Height, Width)"""
    
    def __init__(self, data_dir, split='train', max_samples=None, dataset_version='brats2021'):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.dataset_version = dataset_version
        
        # 데이터 파일 목록 수집
        self.samples = self._load_samples()
        
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
    
    def _load_samples(self):
        """데이터 샘플 목록 로드"""
        samples = []
        
        if self.dataset_version == 'brats2021':
            # BraTS2021 데이터셋 확인
            # 공통 경로/data_dir에서 BRATS2021/BraTS2021_Training_Data 경로 구성
            brats2021_dir = os.path.join(self.data_dir, 'BRATS2021', 'BraTS2021_Training_Data')
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
        
        elif self.dataset_version == 'brats2018':
            # BraTS2018 데이터셋 확인 (HGG와 LGG 폴더 모두 확인)
            # 공통 경로/data_dir에서 BRATS2018/MICCAI_BraTS_2018_Data_Training 경로 구성
            brats2018_dir = os.path.join(self.data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
            if not os.path.exists(brats2018_dir):
                raise FileNotFoundError(
                    f"BraTS2018 dataset not found at {brats2018_dir}\n"
                    f"Please ensure the dataset is extracted in the correct directory."
                )
            
            hgg_dir = os.path.join(brats2018_dir, 'HGG')
            lgg_dir = os.path.join(brats2018_dir, 'LGG')
            
            print(f"Loading BraTS2018 dataset from {brats2018_dir}")
            
            # HGG 샘플 수집
            if os.path.exists(hgg_dir):
                for patient_dir in sorted(os.listdir(hgg_dir)):
                    patient_path = os.path.join(hgg_dir, patient_dir)
                    if os.path.isdir(patient_path):
                        samples.append(patient_path)
                print(f"  Found {len([s for s in samples if hgg_dir in s])} HGG samples")
            
            # LGG 샘플 수집
            lgg_count = 0
            if os.path.exists(lgg_dir):
                for patient_dir in sorted(os.listdir(lgg_dir)):
                    patient_path = os.path.join(lgg_dir, patient_dir)
                    if os.path.isdir(patient_path):
                        samples.append(patient_path)
                        lgg_count += 1
                print(f"  Found {lgg_count} LGG samples")
            
            if not samples:
                raise ValueError(f"No patient data found in {brats2018_dir} (checked HGG and LGG folders)")
        else:
            raise ValueError(f"Unknown dataset_version: {self.dataset_version}. Must be 'brats2021' or 'brats2018'")
        
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
    """3D BraTS 패치 데이터셋 (학습용) - nnU-Net 스타일 샘플링

    nnU-Net의 표준 패치 샘플링 전략을 따름:
    1. 1/3 (33.3%): 포그라운드 오버샘플링 (Foreground Oversampling)
       - 포그라운드 클래스 중 하나를 무작위 선택
       - 해당 클래스의 복셀 중 하나를 무작위 선택하여 중심으로 패치 추출
       - 포그라운드가 없으면 무작위 샘플링으로 대체
    2. 2/3 (66.7%): 완전 무작위 샘플링 (Random Sampling)
       - 케이스 내에서 위치에 상관없이 완전히 무작위로 패치 중심 선택
    
    이 전략은 포그라운드 학습을 보장하면서도 전체 데이터 분포를 유지합니다.
    """
    def __init__(self, base_dataset: BratsDataset3D, patch_size=(128, 128, 128),
                 samples_per_volume: int = 16):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume

        # 인덱스 매핑: (volume_index, sample_idx, sampling_type)
        # sampling_type: 0=포그라운드 오버샘플링, 1=완전 무작위
        self.index_map = []
        for vidx in range(len(self.base_dataset)):
            for s in range(self.samples_per_volume):
                # nnU-Net 스타일: 1/3 포그라운드 오버샘플링, 2/3 완전 무작위
                # s % 3 == 0이면 포그라운드 오버샘플링 (1/3)
                sampling_type = 0 if (s % 3 == 0) else 1
                self.index_map.append((vidx, s, sampling_type))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vidx, _, sampling_type = self.index_map[idx]
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

        # nnU-Net 스타일 샘플링
        if sampling_type == 0:
            # 1. 포그라운드 오버샘플링 (Foreground Oversampling) - 33.3%
            # 포그라운드 클래스(1, 2, 3) 중 하나를 무작위 선택
            fg_classes = []
            for cls in [1, 2, 3]:
                if (mask == cls).any():
                    fg_classes.append(cls)
            
            if len(fg_classes) > 0:
                # 포그라운드 클래스 중 하나를 무작위 선택
                selected_class = fg_classes[torch.randint(0, len(fg_classes), (1,)).item()]
                # 해당 클래스의 복셀 중 하나를 무작위 선택
                fg_coords = (mask == selected_class).nonzero(as_tuple=False)
                if fg_coords.numel() > 0:
                    ci = fg_coords[torch.randint(0, fg_coords.shape[0], (1,)).item()]
                    cy, cx, cz = ci.tolist()
                else:
                    # 포그라운드가 없으면 무작위 샘플링으로 대체
                    cy = torch.randint(0, H, (1,)).item()
                    cx = torch.randint(0, W, (1,)).item()
                    cz = torch.randint(0, D, (1,)).item()
            else:
                # 포그라운드가 전혀 없으면 무작위 샘플링으로 대체
                cy = torch.randint(0, H, (1,)).item()
                cx = torch.randint(0, W, (1,)).item()
                cz = torch.randint(0, D, (1,)).item()
        else:
            # 2. 완전 무작위 샘플링 (Random Sampling) - 66.7%
            # 케이스 내에서 위치에 상관없이 완전히 무작위로 패치 중심 선택
            cy = torch.randint(0, H, (1,)).item()
            cx = torch.randint(0, W, (1,)).item()
            cz = torch.randint(0, D, (1,)).item()

        # 패치 좌표 계산
        sy = max(0, min(cy - ph // 2, H - ph))
        sx = max(0, min(cx - pw // 2, W - pw))
        sz = max(0, min(cz - pd // 2, D - pd))

        img_patch = image[:, sy:sy+ph, sx:sx+pw, sz:sz+pd]
        msk_patch = mask[sy:sy+ph, sx:sx+pw, sz:sz+pd]

        return img_patch, msk_patch


class BratsDataset2D(Dataset):
    """2D BraTS 데이터셋 (Height, Width) - 슬라이스 단위"""
    
    def __init__(self, data_dir, split='train', max_samples=None, dataset_version='brats2021'):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.dataset_version = dataset_version
        
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
        
        if self.dataset_version == 'brats2021':
            # BraTS2021 데이터셋 확인
            # 공통 경로/data_dir에서 BRATS2021/BraTS2021_Training_Data 경로 구성
            brats2021_dir = os.path.join(self.data_dir, 'BRATS2021', 'BraTS2021_Training_Data')
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
        
        elif self.dataset_version == 'brats2018':
            # BraTS2018 데이터셋 확인 (HGG와 LGG 폴더 모두 확인)
            # 공통 경로/data_dir에서 BRATS2018/MICCAI_BraTS_2018_Data_Training 경로 구성
            brats2018_dir = os.path.join(self.data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
            if not os.path.exists(brats2018_dir):
                raise FileNotFoundError(
                    f"BraTS2018 dataset not found at {brats2018_dir}\n"
                    f"Please ensure the dataset is extracted in the correct directory."
                )
            
            hgg_dir = os.path.join(brats2018_dir, 'HGG')
            lgg_dir = os.path.join(brats2018_dir, 'LGG')
            
            print(f"Loading BraTS2018 dataset for 2D slices from {brats2018_dir}")
            
            # HGG 샘플 수집
            if os.path.exists(hgg_dir):
                for patient_dir in sorted(os.listdir(hgg_dir)):
                    patient_path = os.path.join(hgg_dir, patient_dir)
                    if os.path.isdir(patient_path):
                        samples.append(patient_path)
                print(f"  Found {len([s for s in samples if hgg_dir in s])} HGG samples")
            
            # LGG 샘플 수집
            lgg_count = 0
            if os.path.exists(lgg_dir):
                for patient_dir in sorted(os.listdir(lgg_dir)):
                    patient_path = os.path.join(lgg_dir, patient_dir)
                    if os.path.isdir(patient_path):
                        samples.append(patient_path)
                        lgg_count += 1
                print(f"  Found {lgg_count} LGG samples")
            
            if not samples:
                raise ValueError(f"No patient data found in {brats2018_dir} (checked HGG and LGG folders)")
        else:
            raise ValueError(f"Unknown dataset_version: {self.dataset_version}. Must be 'brats2021' or 'brats2018'")
        
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
                     distributed=False, world_size=None, rank=None, dataset_version='brats2021'):
    """
    데이터 로더 생성
    
    Args:
        data_dir: 데이터 디렉토리 경로
        batch_size: 배치 크기
        num_workers: 워커 수
        max_samples: 최대 샘플 수 (None이면 전체)
        train_ratio: 훈련 세트 비율 (BRATS2021에서 사용, BRATS2018은 7:1.5:1.5 고정)
        val_ratio: 검증 세트 비율 (BRATS2021에서 사용, BRATS2018은 7:1.5:1.5 고정)
        test_ratio: 테스트 세트 비율 (BRATS2021에서 사용, BRATS2018은 7:1.5:1.5 고정)
        dim: '2d' 또는 '3d'
        dataset_version: 'brats2021' 또는 'brats2018' (기본값: 'brats2021')
    
    Returns:
        train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler
    """
    # 데이터셋 선택
    if dim == '2d':
        dataset_class = BratsDataset2D
    else:
        dataset_class = BratsDataset3D
    
    # 전체 데이터셋 로드
    full_dataset = dataset_class(data_dir, split='train', max_samples=max_samples, dataset_version=dataset_version)
    
    # 데이터셋 분할
    total_len = len(full_dataset)
    
    # 샘플이 너무 적으면 모든 데이터를 train으로 사용
    if total_len <= 10:
        train_dataset, val_dataset, test_dataset = full_dataset, full_dataset, full_dataset
    else:
        if dataset_version == 'brats2018':
            # BRATS2018: HGG와 LGG를 각각 7:1.5:1.5 비율로 분할 후 섞기
            # 샘플 경로에서 HGG/LGG 구분
            hgg_indices = []
            lgg_indices = []
            
            brats2018_dir = os.path.join(data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
            hgg_dir = os.path.join(brats2018_dir, 'HGG')
            lgg_dir = os.path.join(brats2018_dir, 'LGG')
            
            for idx in range(total_len):
                sample_path = full_dataset.samples[idx]
                if hgg_dir in sample_path:
                    hgg_indices.append(idx)
                elif lgg_dir in sample_path:
                    lgg_indices.append(idx)
            
            # 각 그룹을 7:1.5:1.5 비율로 분할
            def split_indices(indices, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
                """인덱스 리스트를 비율에 맞게 분할"""
                random.shuffle(indices)
                n = len(indices)
                train_len = int(n * train_ratio)
                val_len = int(n * val_ratio)
                test_len = n - train_len - val_len
                
                train_indices = indices[:train_len]
                val_indices = indices[train_len:train_len + val_len]
                test_indices = indices[train_len + val_len:]
                
                return train_indices, val_indices, test_indices
            
            # HGG와 LGG 각각 분할
            hgg_train, hgg_val, hgg_test = split_indices(hgg_indices, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
            lgg_train, lgg_val, lgg_test = split_indices(lgg_indices, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
            
            # 분할된 것을 섞어서 결합
            train_indices = hgg_train + lgg_train
            val_indices = hgg_val + lgg_val
            test_indices = hgg_test + lgg_test
            
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Subset으로 데이터셋 생성
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)
            
            print(f"BRATS2018 split: Train={len(train_indices)} (HGG={len(hgg_train)}, LGG={len(lgg_train)}), "
                  f"Val={len(val_indices)} (HGG={len(hgg_val)}, LGG={len(lgg_val)}), "
                  f"Test={len(test_indices)} (HGG={len(hgg_test)}, LGG={len(lgg_test)})")
        else:
            # BRATS2021: 기존 로직 유지
            train_len = int(total_len * train_ratio)
            val_len = int(total_len * val_ratio)
            test_len = total_len - train_len - val_len
            
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset, [train_len, val_len, test_len]
            )

    # 3D 학습은 패치 데이터셋으로 래핑 (128^3, nnU-Net 스타일 샘플링)
    if dim == '3d':
        train_dataset = BratsPatchDataset3D(
            base_dataset=train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset,
            patch_size=(128, 128, 128),
            samples_per_volume=16,
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
        num_workers=(num_workers if num_workers is not None else 8),
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=((num_workers if num_workers is not None else 8) > 0),
        prefetch_factor=(4 if (num_workers if num_workers is not None else 8) > 0 else None),
    )
    # Validation/Test workers
    v_workers = 4
    t_workers = 4
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
        prefetch_factor=(4 if v_workers > 0 else None),
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

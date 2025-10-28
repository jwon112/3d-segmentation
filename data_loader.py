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
import nibabel as nib
from scipy import ndimage


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
        
        # (H, W, D) 형태로 결합 (T1CE, FLAIR만)
        image = np.stack([t1ce, flair], axis=-1)  # (H, W, D, 2)
        
        # 텐서로 변환
        image = torch.from_numpy(np.transpose(image, (3, 0, 1, 2))).float()  # (2, H, W, D)
        mask = torch.from_numpy(seg).long()
        
        return image, mask


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
                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, dim='3d'):
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
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader




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

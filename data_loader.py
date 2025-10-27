#!/usr/bin/env python3
"""
BraTS 데이터 로더
표준 NIfTI 데이터를 로드하는 데이터 로더
"""

import os
import torch
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
        if os.path.exists(brats2021_dir):
            print(f"Loading BraTS2021 dataset from {brats2021_dir}")
            for patient_dir in sorted(os.listdir(brats2021_dir)):
                patient_path = os.path.join(brats2021_dir, patient_dir)
                if os.path.isdir(patient_path):
                    samples.append(patient_path)
            if samples:
                return samples
        
        # 데이터셋을 찾을 수 없으면 더미 데이터 생성
        print(f"Warning: No BraTS dataset found in {self.data_dir}. Creating dummy data.")
        return self._create_dummy_samples()
    
    def _create_dummy_samples(self):
        """더미 데이터 생성"""
        print("Creating dummy data for demonstration...")
        return [None] * 5  # 5개의 더미 샘플
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        
        # 더미 데이터인 경우
        if sample_path is None or not os.path.exists(sample_path):
            return self._get_dummy_data()
        
        # NIfTI 포맷인 경우 (디렉토리 경로)
        return self._load_nifti_data(sample_path)
    
    
    def _load_nifti_data(self, patient_dir):
        """NIfTI 파일 로드"""
        files = os.listdir(patient_dir)
        
        # modality 파일 찾기 (BraTS2021 지원)
        t1_file = [f for f in files if 't1.nii' in f.lower() and 'seg' not in f and 't1ce' not in f][0]
        t1ce_file = [f for f in files if 't1ce' in f.lower()][0]
        t2_file = [f for f in files if 't2.nii' in f.lower() and 'seg' not in f][0]
        flair_file = [f for f in files if 'flair.nii' in f.lower()][0]
        seg_file = [f for f in files if 'seg' in f.lower()][0]
        
        # 데이터 로드
        t1 = nib.load(os.path.join(patient_dir, t1_file)).get_fdata()
        t1ce = nib.load(os.path.join(patient_dir, t1ce_file)).get_fdata()
        t2 = nib.load(os.path.join(patient_dir, t2_file)).get_fdata()
        flair = nib.load(os.path.join(patient_dir, flair_file)).get_fdata()
        seg = nib.load(os.path.join(patient_dir, seg_file)).get_fdata()
        
        # (H, W, D) 형태로 결합
        image = np.stack([t1, t1ce, t2, flair], axis=-1)  # (H, W, D, 4)
        
        # 리사이즈
        if image.shape[:3] != (64, 64, 64):
            from scipy.ndimage import zoom
            factors = (64/image.shape[0], 64/image.shape[1], 64/image.shape[2], 1)
            image = zoom(image, factors)
            seg = zoom(seg, (64/seg.shape[0], 64/seg.shape[1], 64/seg.shape[2]))
        
        # 텐서로 변환
        image = torch.from_numpy(np.transpose(image, (3, 0, 1, 2))).float()  # (4, H, W, D)
        mask = torch.from_numpy(seg).long()
        
        return image, mask
    
    def _get_dummy_data(self):
        """더미 데이터 생성"""
        # 랜덤 이미지 (4 channels, 3D)
        image = torch.randn(4, 64, 64, 64)
        mask = torch.randint(0, 4, (64, 64, 64))
        
        return image, mask


class BratsDataset2D(Dataset):
    """2D BraTS 데이터셋 (Height, Width) - 슬라이스 단위"""
    
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
        
        training_dir = os.path.join(self.data_dir, 'MICCAI_BraTS2020_TrainingData')
        
        if not os.path.exists(training_dir):
            print(f"Warning: {training_dir} not found. Creating dummy data.")
            return self._create_dummy_samples()
        
        data_dir = os.path.join(training_dir, 'BraTS2020_training_data', 'content', 'data')
        
        if os.path.exists(data_dir):
            return self._load_h5_samples(data_dir)
        else:
            return []
    
    def _load_h5_samples(self, data_dir):
        """H5 파일 기반 샘플 로드"""
        import h5py
        samples = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.h5'):
                samples.append(os.path.join(data_dir, filename))
        
        return samples
    
    def _create_dummy_samples(self):
        """더미 데이터 생성"""
        print("Creating dummy data for demonstration...")
        return [None] * 5
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        
        if sample_path is None:
            return self._get_dummy_data()
        
        if sample_path.endswith('.h5'):
            return self._load_h5_data(sample_path)
    
    def _load_h5_data(self, file_path):
        """H5 파일 로드 (2D 슬라이스)"""
        import h5py
        with h5py.File(file_path, 'r') as f:
            image = f['image'][:]  # (240, 240, 4)
            mask = f['mask'][:]    # (240, 240)
        
        # (4, H, W) 형태로 변환
        image = np.transpose(image, (2, 0, 1))
        
        # 리사이즈
        from scipy.ndimage import zoom
        image = zoom(image, (1, 64/240, 64/240))  # (4, 64, 64)
        mask = zoom(mask, (64/240, 64/240))  # (64, 64)
        
        # 텐서로 변환
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def _get_dummy_data(self):
        """더미 데이터 생성"""
        image = torch.randn(4, 64, 64)
        mask = torch.randint(0, 4, (64, 64))
        return image, mask


def get_data_loaders(data_dir, batch_size=1, num_workers=0, max_samples=None, 
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, dim='3d'):
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
    train_loader, val_loader, test_loader = get_data_loaders('.', batch_size=1, max_samples=5)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 샘플 확인
    for image, mask in train_loader:
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        break

import os
import numpy as np
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import glob
from collections import defaultdict

class BratsKaggleDataset(Dataset):
    """
    Kaggle BraTS 2020 데이터셋을 위한 PyTorch Dataset 클래스
    H5 파일 형식으로 저장된 슬라이스 데이터 처리
    """
    def __init__(self, data_dir, split='train', transform=None, max_samples=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """데이터셋 샘플 목록을 로드"""
        samples = []
        
        # CSV 파일에서 메타데이터 로드
        csv_path = os.path.join(self.data_dir, 'MICCAI_BraTS2020_TrainingData', 'BraTS20 Training Metadata.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Creating dummy data for demonstration.")
            return self._create_dummy_samples()
        
        print("CSV 파일 로딩 중...")
        df = pd.read_csv(csv_path)
        print(f"총 {len(df)} 개의 슬라이스 발견")
        
        # 볼륨별로 그룹화
        volume_groups = defaultdict(list)
        for _, row in df.iterrows():
            volume_id = row['volume']
            volume_groups[volume_id].append({
                'slice_path': row['slice_path'],
                'slice_idx': row['slice'],
                'target': row['target'],
                'volume': volume_id
            })
        
        print(f"총 {len(volume_groups)} 개의 볼륨 발견")
        
        # 각 볼륨에 대해 샘플 생성
        for volume_id, slices in volume_groups.items():
            if self.max_samples and len(samples) >= self.max_samples:
                break
                
            # 슬라이스들을 정렬
            slices.sort(key=lambda x: x['slice_idx'])
            
            # 4가지 모달리티 데이터 수집
            volume_data = {
                'volume_id': volume_id,
                'flair_slices': [],
                't1_slices': [],
                't1ce_slices': [],
                't2_slices': [],
                'seg_slices': [],
                'slice_indices': []
            }
            
            for slice_info in slices:
                slice_path = slice_info['slice_path']
                if os.path.exists(slice_path):
                    try:
                        with h5py.File(slice_path, 'r') as f:
                            # H5 파일에서 데이터 추출 (shape: 240, 240, 4)
                            if 'image' in f.keys() and 'mask' in f.keys():
                                image_data = f['image'][:]  # (240, 240, 4)
                                mask_data = f['mask'][:]    # (240, 240, 1)
                                
                                # 4채널을 분리 (H, W, C) -> (C, H, W)
                                volume_data['flair_slices'].append(image_data[:, :, 0])
                                volume_data['t1_slices'].append(image_data[:, :, 1])
                                volume_data['t1ce_slices'].append(image_data[:, :, 2])
                                volume_data['t2_slices'].append(image_data[:, :, 3])
                                volume_data['seg_slices'].append(mask_data[:, :, 0])
                                
                                volume_data['slice_indices'].append(slice_info['slice_idx'])
                    except Exception as e:
                        print(f"Error loading {slice_path}: {e}")
                        continue
            
            # 모든 모달리티가 있는 경우만 샘플로 추가
            if (len(volume_data['flair_slices']) > 0 and 
                len(volume_data['t1_slices']) > 0 and 
                len(volume_data['t1ce_slices']) > 0 and 
                len(volume_data['t2_slices']) > 0 and 
                len(volume_data['seg_slices']) > 0):
                
                samples.append(volume_data)
        
        print(f"로드된 샘플 수: {len(samples)}")
        return samples
    
    def _create_dummy_samples(self):
        """더미 데이터 생성 (테스트용)"""
        print("더미 데이터 생성 중...")
        dummy_samples = []
        for i in range(5):  # 5개의 더미 샘플
            dummy_samples.append({
                'volume_id': f'dummy_{i:03d}',
                'flair_slices': [np.random.randn(240, 240) for _ in range(10)],
                't1_slices': [np.random.randn(240, 240) for _ in range(10)],
                't1ce_slices': [np.random.randn(240, 240) for _ in range(10)],
                't2_slices': [np.random.randn(240, 240) for _ in range(10)],
                'seg_slices': [np.random.randint(0, 4, (240, 240)) for _ in range(10)],
                'slice_indices': list(range(10))
            })
        return dummy_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 슬라이스들을 3D 배열로 스택
        try:
            flair_volume = np.stack(sample['flair_slices'], axis=0)
            t1_volume = np.stack(sample['t1_slices'], axis=0)
            t1ce_volume = np.stack(sample['t1ce_slices'], axis=0)
            t2_volume = np.stack(sample['t2_slices'], axis=0)
            seg_volume = np.stack(sample['seg_slices'], axis=0)
            
            # 4채널 입력으로 스택 (C, D, H, W)
            image = np.stack([flair_volume, t1_volume, t1ce_volume, t2_volume], axis=0)
            
            # 정규화
            image = self._normalize(image)
            
            # Segmentation 마스크 전처리
            seg_volume = self._process_segmentation(seg_volume)
            
            # 원본 크기 유지 (리사이징 제거)
            
            # 텐서로 변환
            image = torch.FloatTensor(image)
            seg_volume = torch.LongTensor(seg_volume)
            
            if self.transform:
                image = self.transform(image)
                
            return {
                'image': image,
                'segmentation': seg_volume,
                'patient_id': sample.get('volume_id', f'volume_{idx}')  # volume_id를 patient_id로 통일
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._get_dummy_data(sample)
    
    def _get_dummy_data(self, sample):
        """더미 데이터 생성"""
        # 랜덤한 3D 이미지 생성 (4채널, 64x64x64)
        image = np.random.randn(4, 64, 64, 64).astype(np.float32)
        image = self._normalize(image)
        
        # 랜덤한 segmentation 마스크 생성
        seg = np.random.randint(0, 4, (64, 64, 64)).astype(np.int64)
        
        return {
            'image': torch.FloatTensor(image),
            'segmentation': torch.LongTensor(seg),
            'patient_id': sample.get('volume_id', f'dummy_{idx}')  # volume_id를 patient_id로 통일
        }
    
    def _normalize(self, image):
        """이미지 정규화"""
        for i in range(image.shape[0]):
            channel = image[i]
            # 0.5%와 99.5% 퍼센타일로 클리핑
            p1, p99 = np.percentile(channel, [0.5, 99.5])
            channel = np.clip(channel, p1, p99)
            # Z-score 정규화
            mean = np.mean(channel)
            std = np.std(channel)
            if std > 0:
                channel = (channel - mean) / std
            image[i] = channel
        return image
    
    def _process_segmentation(self, seg):
        """Segmentation 마스크 전처리"""
        # BraTS 라벨: 0: 배경, 1: NCR/NET, 2: ED, 4: ET
        # 4클래스로 변환: 0: 배경, 1: NCR/NET, 2: ED, 3: ET
        seg_processed = np.zeros_like(seg, dtype=np.int64)
        seg_processed[seg == 1] = 1  # NCR/NET
        seg_processed[seg == 2] = 2  # ED
        seg_processed[seg == 4] = 3  # ET
        return seg_processed
    
    def _resize_volume(self, volume, target_size):
        """3D 볼륨 리사이징 (이미지용)"""
        from scipy.ndimage import zoom
        # volume shape: (C, D, H, W)
        current_size = volume.shape[1:]  # (D, H, W)
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        zoom_factors = [1.0] + zoom_factors  # 채널 차원은 그대로
        
        resized = np.zeros((volume.shape[0],) + target_size, dtype=volume.dtype)
        for c in range(volume.shape[0]):
            resized[c] = zoom(volume[c], zoom_factors[1:], order=1, mode='nearest')
        
        return resized
    
    def _resize_segmentation(self, seg_volume, target_size):
        """3D 세그멘테이션 리사이징 (라벨용)"""
        from scipy.ndimage import zoom
        current_size = seg_volume.shape
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        
        resized = zoom(seg_volume, zoom_factors, order=0, mode='nearest')
        return resized.astype(np.int64)

def get_data_loaders_kaggle(data_dir, batch_size=1, num_workers=0, max_samples=None):
    """Kaggle 데이터 로더 생성"""
    train_dataset = BratsKaggleDataset(data_dir, split='train', max_samples=max_samples)
    val_dataset = BratsKaggleDataset(data_dir, split='val', max_samples=max_samples//2 if max_samples else None)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # 테스트
    dataset = BratsKaggleDataset(".", max_samples=3)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Segmentation shape: {sample['segmentation'].shape}")
        print(f"Volume ID: {sample['volume_id']}")

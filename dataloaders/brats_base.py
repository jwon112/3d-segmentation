"""
BraTS base datasets and split helpers.

3D/2D BraTS 볼륨 로드, 정규화, train/val/test 분할 로직을 포함합니다.
"""

import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Tuple, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def get_project_root():
    """프로젝트 루트 디렉토리 찾기 (dataloaders 패키지의 부모 디렉토리)"""
    return Path(__file__).parent.parent.absolute()


def get_default_preprocessed_dir(dataset_version='brats2021'):
    """기본 전처리 디렉토리 경로 반환"""
    project_root = get_project_root()
    data_dir = project_root / 'data'
    if dataset_version == 'brats2021':
        return str(data_dir / 'BRATS2021_preprocessed')
    elif dataset_version == 'brats2018':
        return str(data_dir / 'BRATS2018_preprocessed')
    return None


def _normalize_volume_np(vol):
    """퍼센타일 클리핑 + Z-score 정규화 (비영점 마스크 기준, 배경은 0 유지)"""
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
    return out


class BratsDataset3D(Dataset):
    """3D BraTS 데이터셋 (Depth, Height, Width)"""

    def __init__(self, data_dir, split='train', max_samples=None, dataset_version='brats2021', use_4modalities=False, max_cache_size: int = 50, preprocessed_dir=None):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.dataset_version = dataset_version
        self.use_4modalities = use_4modalities
        # 전처리된 데이터 디렉토리
        # 1. 파라미터로 지정된 경우
        # 2. 환경변수로 지정된 경우
        # 3. 기본값: 프로젝트 루트의 data/ 디렉토리
        if preprocessed_dir:
            self.preprocessed_dir = preprocessed_dir
        elif os.environ.get('BRATS_PREPROCESSED_DIR'):
            self.preprocessed_dir = os.environ.get('BRATS_PREPROCESSED_DIR')
        else:
            # 기본값: 프로젝트 루트의 data/ 디렉토리
            default_dir = get_default_preprocessed_dir(dataset_version)
            self.preprocessed_dir = default_dir if default_dir and os.path.exists(default_dir) else None
        self.max_cache_size = max_cache_size

        self.samples = self._load_samples()
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        # LRU 캐시 (worker별로 독립적으로 유지됨)
        self._nifti_cache = OrderedDict()
        # 파일 이름 캐시 (lazy caching, worker별로 독립적으로 유지됨)
        # os.listdir() 호출을 줄이기 위해 파일 이름을 캐싱
        self._file_names_cache = {}

    def _load_samples(self):
        samples = []
        if self.dataset_version == 'brats2021':
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
            brats2018_dir = os.path.join(self.data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
            if not os.path.exists(brats2018_dir):
                raise FileNotFoundError(
                    f"BraTS2018 dataset not found at {brats2018_dir}\n"
                    f"Please ensure the dataset is extracted in the correct directory."
                )
            hgg_dir = os.path.join(brats2018_dir, 'HGG')
            lgg_dir = os.path.join(brats2018_dir, 'LGG')
            print(f"Loading BraTS2018 dataset from {brats2018_dir}")
            if os.path.exists(hgg_dir):
                for patient_dir in sorted(os.listdir(hgg_dir)):
                    patient_path = os.path.join(hgg_dir, patient_dir)
                    if os.path.isdir(patient_path):
                        samples.append(patient_path)
                print(f"  Found {len([s for s in samples if hgg_dir in s])} HGG samples")
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
        # LRU 캐시 확인 및 업데이트 (max_cache_size > 0일 때만)
        if self.max_cache_size > 0 and patient_dir in self._nifti_cache:
            # 캐시 히트: 해당 항목을 맨 뒤로 이동 (가장 최근 사용)
            result = self._nifti_cache.pop(patient_dir)
            self._nifti_cache[patient_dir] = result
            return result
        
        # 전처리된 데이터 우선 로드 (HDF5)
        # 1. 별도 디렉토리 확인 (환경변수 또는 파라미터로 지정)
        if self.preprocessed_dir:
            patient_name = os.path.basename(patient_dir)
            preprocessed_path = os.path.join(self.preprocessed_dir, f"{patient_name}.h5")
        else:
            # 2. 각 샘플 폴더 내부 확인 (하위 호환성)
            preprocessed_path = os.path.join(patient_dir, 'preprocessed.h5')
        
        if HAS_H5PY and os.path.exists(preprocessed_path):
            try:
                with h5py.File(preprocessed_path, 'r') as f:
                    image = torch.from_numpy(f['image'][:]).float()
                    mask = torch.from_numpy(f['mask'][:]).long()
                    
                    # 포그라운드 좌표 로드 (있으면 사용, 없으면 None)
                    fg_coords_dict = {}
                    if f.attrs.get('has_fg_coords', False):
                        for cls in [1, 2, 3]:
                            coords_key = f'fg_coords_{cls}'
                            if coords_key in f:
                                coords_array = f[coords_key][:]
                                # 포그라운드 좌표 크기 확인 (디버깅용)
                                if coords_array.size > 0:
                                    coords_shape = coords_array.shape
                                    # 매우 큰 좌표 배열은 int32 범위를 넘을 수 있음
                                    if coords_shape[0] > 2**31 - 1:
                                        import warnings
                                        warnings.warn(f"Foreground coordinates for class {cls} exceed int32 range: {coords_shape[0]}")
                                fg_coords_dict[cls] = torch.from_numpy(coords_array).long()
                
                # 모달리티 수 확인
                if self.use_4modalities and image.shape[0] != 4:
                    # 전처리된 데이터가 2모달리티인데 4모달리티가 필요한 경우
                    raise ValueError("Preprocessed data has wrong number of modalities")
                elif not self.use_4modalities and image.shape[0] != 2:
                    # 전처리된 데이터가 4모달리티인데 2모달리티가 필요한 경우
                    raise ValueError("Preprocessed data has wrong number of modalities")
                
                # 포그라운드 좌표가 있으면 함께 반환
                if fg_coords_dict:
                    result = (image, mask, fg_coords_dict)
                else:
                    result = (image, mask)
                
                # 캐시 크기 제한 확인 (max_cache_size > 0일 때만 캐시 사용)
                if self.max_cache_size > 0:
                    if len(self._nifti_cache) >= self.max_cache_size:
                        self._nifti_cache.popitem(last=False)
                    self._nifti_cache[patient_dir] = result
                
                return result
            except Exception as e:
                # 전처리된 데이터 로드 실패 시 원본 로드로 fallback
                pass
        
        # 원본 NIfTI 파일 로드 (fallback 또는 전처리된 데이터가 없는 경우)
        # 캐시 미스: 파일 이름 캐싱 (lazy caching)
        # 각 worker가 독립적으로 파일 이름을 캐싱하여 os.listdir() 호출 최소화
        if patient_dir not in self._file_names_cache:
            files = os.listdir(patient_dir)
            t1ce_file = [f for f in files if 't1ce' in f.lower()][0]
            flair_file = [f for f in files if 'flair.nii' in f.lower()][0]
            seg_file = [f for f in files if 'seg' in f.lower()][0]
            if self.use_4modalities:
                t1_file = [f for f in files if 't1.nii' in f.lower() and 't1ce' not in f.lower()][0]
                t2_file = [f for f in files if 't2.nii' in f.lower()][0]
                self._file_names_cache[patient_dir] = (t1ce_file, flair_file, seg_file, t1_file, t2_file)
            else:
                self._file_names_cache[patient_dir] = (t1ce_file, flair_file, seg_file)
        
        # 캐시된 파일 이름 사용
        if self.use_4modalities:
            t1ce_file, flair_file, seg_file, t1_file, t2_file = self._file_names_cache[patient_dir]
        else:
            t1ce_file, flair_file, seg_file = self._file_names_cache[patient_dir]

        t1ce = nib.load(os.path.join(patient_dir, t1ce_file)).get_fdata()
        flair = nib.load(os.path.join(patient_dir, flair_file)).get_fdata()
        seg = nib.load(os.path.join(patient_dir, seg_file)).get_fdata()

        t1ce = _normalize_volume_np(t1ce)
        flair = _normalize_volume_np(flair)

        if self.use_4modalities:
            t1 = nib.load(os.path.join(patient_dir, t1_file)).get_fdata()
            t2 = nib.load(os.path.join(patient_dir, t2_file)).get_fdata()
            t1 = _normalize_volume_np(t1)
            t2 = _normalize_volume_np(t2)
            image = np.stack([t1, t1ce, t2, flair], axis=-1)
            image = torch.from_numpy(np.transpose(image, (3, 0, 1, 2))).float()
        else:
            image = np.stack([t1ce, flair], axis=-1)
            image = torch.from_numpy(np.transpose(image, (3, 0, 1, 2))).float()

        mask = torch.from_numpy(seg).long()
        mask = torch.where(mask == 4, torch.tensor(3).long(), mask)
        
        result = (image, mask)
        
        # 캐시 크기 제한 확인 (max_cache_size > 0일 때만 캐시 사용)
        if self.max_cache_size > 0:
            if len(self._nifti_cache) >= self.max_cache_size:
                # 가장 오래된 항목 제거 (LRU)
                self._nifti_cache.popitem(last=False)
            # 새 항목 추가
            self._nifti_cache[patient_dir] = result
        
        return result


class BratsDataset2D(Dataset):
    """2D BraTS 데이터셋 (Height, Width) - 슬라이스 단위"""

    def __init__(self, data_dir, split='train', max_samples=None, dataset_version='brats2021', preprocessed_dir=None):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.dataset_version = dataset_version
        # 전처리된 데이터 디렉토리
        # 1. 파라미터로 지정된 경우
        # 2. 환경변수로 지정된 경우
        # 3. 기본값: 프로젝트 루트의 data/ 디렉토리
        if preprocessed_dir:
            self.preprocessed_dir = preprocessed_dir
        elif os.environ.get('BRATS_PREPROCESSED_DIR'):
            self.preprocessed_dir = os.environ.get('BRATS_PREPROCESSED_DIR')
        else:
            # 기본값: 프로젝트 루트의 data/ 디렉토리
            default_dir = get_default_preprocessed_dir(dataset_version)
            self.preprocessed_dir = default_dir if default_dir and os.path.exists(default_dir) else None
        self.volumes = self._load_samples()
        self.samples = []
        # 파일 이름 캐시 (lazy caching을 위해 초기화)
        self._file_names_cache = {}
        for volume in self.volumes:
            files = os.listdir(volume)
            t1_file = [f for f in files if 't1.nii' in f.lower() and 'seg' not in f and 't1ce' not in f][0]
            t1ce_file = [f for f in files if 't1ce' in f.lower()][0]
            flair_file = [f for f in files if 'flair.nii' in f.lower()][0]
            seg_file = [f for f in files if 'seg' in f.lower()][0]
            # 파일 이름 캐싱
            self._file_names_cache[volume] = (t1ce_file, flair_file, seg_file)
            t1 = nib.load(os.path.join(volume, t1_file)).get_fdata()
            depth = t1.shape[2]
            for slice_idx in range(depth):
                self.samples.append((volume, slice_idx))
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _load_samples(self):
        samples = []
        if self.dataset_version == 'brats2021':
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
            brats2018_dir = os.path.join(self.data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
            if not os.path.exists(brats2018_dir):
                raise FileNotFoundError(
                    f"BraTS2018 dataset not found at {brats2018_dir}\n"
                    f"Please ensure the dataset is extracted in the correct directory."
                )
            hgg_dir = os.path.join(brats2018_dir, 'HGG')
            lgg_dir = os.path.join(brats2018_dir, 'LGG')
            print(f"Loading BraTS2018 dataset for 2D slices from {brats2018_dir}")
            if os.path.exists(hgg_dir):
                for patient_dir in sorted(os.listdir(hgg_dir)):
                    patient_path = os.path.join(hgg_dir, patient_dir)
                    if os.path.isdir(patient_path):
                        samples.append(patient_path)
                print(f"  Found {len([s for s in samples if hgg_dir in s])} HGG samples")
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
        # 전처리된 데이터 우선 로드 (HDF5)
        # 1. 별도 디렉토리 확인 (환경변수 또는 파라미터로 지정)
        if self.preprocessed_dir:
            patient_name = os.path.basename(patient_dir)
            preprocessed_path = os.path.join(self.preprocessed_dir, f"{patient_name}.h5")
        else:
            # 2. 각 샘플 폴더 내부 확인 (하위 호환성)
            preprocessed_path = os.path.join(patient_dir, 'preprocessed.h5')
        
        if HAS_H5PY and os.path.exists(preprocessed_path):
            try:
                with h5py.File(preprocessed_path, 'r') as f:
                    image_vol = torch.from_numpy(f['image'][:]).float()  # (C, H, W, D)
                    mask_vol = torch.from_numpy(f['mask'][:]).long()      # (H, W, D)
                
                # 슬라이스 추출
                image_slice = image_vol[:, :, :, slice_idx]  # (C, H, W)
                mask_slice = mask_vol[:, :, slice_idx]        # (H, W)
                
                # 라벨 매핑 (이미 전처리에서 처리되었지만 안전을 위해)
                mask_slice = torch.where(mask_slice == 4, torch.tensor(3).long(), mask_slice)
                
                # 리사이즈 (240 -> 256)
                if image_slice.shape[1] == 240 or image_slice.shape[2] == 240:
                    image_slice = image_slice.unsqueeze(0)
                    image_slice = torch.nn.functional.interpolate(image_slice, size=(256, 256), mode='bilinear', align_corners=False)
                    image_slice = image_slice.squeeze(0)
                    mask_slice = mask_slice.unsqueeze(0).unsqueeze(0)
                    mask_slice = torch.nn.functional.interpolate(mask_slice.float(), size=(256, 256), mode='nearest')
                    mask_slice = mask_slice.squeeze(0).squeeze(0).long()
                
                return image_slice, mask_slice
            except Exception as e:
                # 전처리된 데이터 로드 실패 시 원본 로드로 fallback
                pass
        
        # 원본 NIfTI 파일 로드 (fallback 또는 전처리된 데이터가 없는 경우)
        # 파일 이름 캐싱 (lazy caching)
        if patient_dir not in self._file_names_cache:
            files = os.listdir(patient_dir)
            t1ce_file = [f for f in files if 't1ce' in f.lower()][0]
            flair_file = [f for f in files if 'flair.nii' in f.lower()][0]
            seg_file = [f for f in files if 'seg' in f.lower()][0]
            self._file_names_cache[patient_dir] = (t1ce_file, flair_file, seg_file)
        else:
            t1ce_file, flair_file, seg_file = self._file_names_cache[patient_dir]
        
        t1ce = nib.load(os.path.join(patient_dir, t1ce_file)).get_fdata()
        flair = nib.load(os.path.join(patient_dir, flair_file)).get_fdata()
        seg = nib.load(os.path.join(patient_dir, seg_file)).get_fdata()
        t1ce = _normalize_volume_np(t1ce)
        flair = _normalize_volume_np(flair)
        image = np.stack([t1ce, flair], axis=0)
        image_slice = image[:, :, :, slice_idx]
        mask_slice = seg[:, :, slice_idx]
        image_t = torch.from_numpy(image_slice).float()
        mask_t = torch.from_numpy(mask_slice).long()
        mask_t = torch.where(mask_t == 4, torch.tensor(3).long(), mask_t)
        if image_t.shape[1] == 240 or image_t.shape[2] == 240:
            image_t = image_t.unsqueeze(0)
            image_t = torch.nn.functional.interpolate(image_t, size=(256, 256), mode='bilinear', align_corners=False)
            image_t = image_t.squeeze(0)
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            mask_t = torch.nn.functional.interpolate(mask_t.float(), size=(256, 256), mode='nearest')
            mask_t = mask_t.squeeze(0).squeeze(0).long()
        return image_t, mask_t


def split_brats_dataset(
    full_dataset: Dataset,
    data_dir: str,
    dataset_version: str = 'brats2021',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    use_5fold: bool = False,
    fold_idx: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Shared dataset splitting logic (train/val/test or 5-fold)."""
    total_len = len(full_dataset)
    if total_len <= 10:
        return full_dataset, full_dataset, full_dataset

    if seed is not None:
        random.seed(seed)

    if use_5fold:
        if fold_idx is None or fold_idx < 0 or fold_idx >= 5:
            raise ValueError("fold_idx must be in [0, 4] when use_5fold=True.")
        indices = list(range(total_len))
        random.shuffle(indices)
        fold_size = total_len // 5
        fold_sizes = [fold_size] * 5
        remainder = total_len % 5
        for i in range(remainder):
            fold_sizes[i] += 1
        fold_starts = [0]
        for size in fold_sizes:
            fold_starts.append(fold_starts[-1] + size)
        test_start = fold_starts[fold_idx]
        test_end = fold_starts[fold_idx + 1]
        test_indices = indices[test_start:test_end]
        val_fold_idx = (fold_idx + 1) % 5
        val_start = fold_starts[val_fold_idx]
        val_end = fold_starts[val_fold_idx + 1]
        val_indices = indices[val_start:val_end]
        train_indices = []
        for i in range(5):
            if i in (fold_idx, val_fold_idx):
                continue
            train_start = fold_starts[i]
            train_end = fold_starts[i + 1]
            train_indices.extend(indices[train_start:train_end])
        return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices), Subset(full_dataset, test_indices)

    if dataset_version == 'brats2018':
        hgg_indices, lgg_indices = [], []
        brats2018_dir = os.path.join(data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
        hgg_dir = os.path.join(brats2018_dir, 'HGG')
        lgg_dir = os.path.join(brats2018_dir, 'LGG')
        for idx in range(total_len):
            sample_path = full_dataset.samples[idx]
            if hgg_dir in sample_path:
                hgg_indices.append(idx)
            elif lgg_dir in sample_path:
                lgg_indices.append(idx)

        def split_indices(indices, tr=0.7, vr=0.15):
            random.shuffle(indices)
            n = len(indices)
            train_len = int(n * tr)
            val_len = int(n * vr)
            train_idx = indices[:train_len]
            val_idx = indices[train_len:train_len + val_len]
            test_idx = indices[train_len + val_len:]
            return train_idx, val_idx, test_idx

        hgg_train, hgg_val, hgg_test = split_indices(hgg_indices)
        lgg_train, lgg_val, lgg_test = split_indices(lgg_indices)
        train_indices = hgg_train + lgg_train
        val_indices = hgg_val + lgg_val
        test_indices = hgg_test + lgg_test
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)
        return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices), Subset(full_dataset, test_indices)

    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)
    return random_split(full_dataset, [train_len, val_len, test_len], generator=generator)


def get_brats_base_datasets(
    data_dir,
    dataset_version: str = 'brats2021',
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    use_5fold: bool = False,
    fold_idx: Optional[int] = None,
    use_4modalities: bool = True,
    max_cache_size: Optional[int] = None,
):
    """Create base BratsDataset3D and return train/val/test splits."""
    base_dataset = BratsDataset3D(
        data_dir,
        split='train',
        max_samples=max_samples,
        dataset_version=dataset_version,
        use_4modalities=use_4modalities,
        max_cache_size=max_cache_size if max_cache_size is not None else 50,
    )
    return split_brats_dataset(
        full_dataset=base_dataset,
        data_dir=data_dir,
        dataset_version=dataset_version,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        seed=seed,
    )


__all__ = [
    "_normalize_volume_np",
    "BratsDataset3D",
    "BratsDataset2D",
    "split_brats_dataset",
    "get_brats_base_datasets",
]




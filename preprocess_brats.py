#!/usr/bin/env python3
"""
BraTS 데이터 전처리 스크립트

모든 NIfTI 파일을 전처리하여 HDF5 형식으로 저장합니다.
이렇게 하면 학습 시 로딩 속도가 7-10배 빨라집니다.

Usage:
    python preprocess_brats.py --data_dir /path/to/data --dataset_version brats2021
    python preprocess_brats.py --data_dir /path/to/data --dataset_version brats2018 --use_4modalities
"""

import os
import argparse
from tqdm import tqdm
import h5py
import nibabel as nib
import numpy as np


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


def preprocess_volume(patient_dir, use_4modalities=False, output_path=None):
    """
    단일 볼륨을 전처리하여 저장
    
    Args:
        patient_dir: 환자 디렉토리 경로
        use_4modalities: 4개 모달리티 사용 여부
        output_path: 저장할 파일 경로 (None이면 patient_dir/preprocessed.h5)
    
    Returns:
        bool: 성공 여부
    """
    if output_path is None:
        output_path = os.path.join(patient_dir, 'preprocessed.h5')
    
    # 이미 전처리되어 있으면 스킵
    if os.path.exists(output_path):
        return True
    
    try:
        # 파일 이름 찾기
        files = os.listdir(patient_dir)
        t1ce_file = [f for f in files if 't1ce' in f.lower()][0]
        flair_file = [f for f in files if 'flair.nii' in f.lower()][0]
        seg_file = [f for f in files if 'seg' in f.lower()][0]
        
        # NIfTI 파일 로드
        t1ce = nib.load(os.path.join(patient_dir, t1ce_file)).get_fdata()
        flair = nib.load(os.path.join(patient_dir, flair_file)).get_fdata()
        seg = nib.load(os.path.join(patient_dir, seg_file)).get_fdata()
        
        # 정규화
        t1ce = _normalize_volume_np(t1ce)
        flair = _normalize_volume_np(flair)
        
        if use_4modalities:
            t1_file = [f for f in files if 't1.nii' in f.lower() and 't1ce' not in f.lower()][0]
            t2_file = [f for f in files if 't2.nii' in f.lower()][0]
            t1 = nib.load(os.path.join(patient_dir, t1_file)).get_fdata()
            t2 = nib.load(os.path.join(patient_dir, t2_file)).get_fdata()
            t1 = _normalize_volume_np(t1)
            t2 = _normalize_volume_np(t2)
            image = np.stack([t1, t1ce, t2, flair], axis=-1)
        else:
            image = np.stack([t1ce, flair], axis=-1)
        
        # (H, W, D, C) -> (C, H, W, D)로 변환
        image = np.transpose(image, (3, 0, 1, 2)).astype(np.float32)
        
        # Mask 처리
        mask = seg.astype(np.int64)
        mask = np.where(mask == 4, 3, mask)
        
        # HDF5로 저장
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('image', data=image, compression='gzip', compression_opts=4)
            f.create_dataset('mask', data=mask, compression='gzip', compression_opts=4)
        
        return True
    except Exception as e:
        print(f"Error processing {patient_dir}: {e}")
        return False


def preprocess_all_volumes(data_dir, dataset_version='brats2021', use_4modalities=False, 
                          output_dir=None, skip_existing=True):
    """
    모든 볼륨을 전처리하여 저장
    
    Args:
        data_dir: 데이터 루트 디렉토리
        dataset_version: 'brats2021' 또는 'brats2018'
        use_4modalities: 4개 모달리티 사용 여부
        output_dir: 저장할 디렉토리 (None이면 원본 디렉토리 내부)
        skip_existing: 이미 전처리된 파일 스킵 여부
    """
    # 데이터셋 경로 설정
    if dataset_version == 'brats2021':
        brats_dir = os.path.join(data_dir, 'BRATS2021', 'BraTS2021_Training_Data')
    elif dataset_version == 'brats2018':
        brats_dir = os.path.join(data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
        hgg_dir = os.path.join(brats_dir, 'HGG')
        lgg_dir = os.path.join(brats_dir, 'LGG')
    else:
        raise ValueError(f"Unknown dataset_version: {dataset_version}")
    
    if not os.path.exists(brats_dir):
        raise FileNotFoundError(f"Dataset not found at {brats_dir}")
    
    # 환자 디렉토리 수집
    patient_dirs = []
    if dataset_version == 'brats2021':
        for patient_dir in sorted(os.listdir(brats_dir)):
            patient_path = os.path.join(brats_dir, patient_dir)
            if os.path.isdir(patient_path):
                patient_dirs.append(patient_path)
    else:  # brats2018
        if os.path.exists(hgg_dir):
            for patient_dir in sorted(os.listdir(hgg_dir)):
                patient_path = os.path.join(hgg_dir, patient_dir)
                if os.path.isdir(patient_path):
                    patient_dirs.append(patient_path)
        if os.path.exists(lgg_dir):
            for patient_dir in sorted(os.listdir(lgg_dir)):
                patient_path = os.path.join(lgg_dir, patient_dir)
                if os.path.isdir(patient_path):
                    patient_dirs.append(patient_path)
    
    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Dataset version: {dataset_version}")
    print(f"Use 4 modalities: {use_4modalities}")
    print(f"Output directory: {output_dir if output_dir else 'Original directory (preprocessed.h5)'}")
    print()
    
    # 전처리 실행
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for patient_dir in tqdm(patient_dirs, desc="Preprocessing"):
        if output_dir:
            patient_name = os.path.basename(patient_dir)
            output_path = os.path.join(output_dir, f"{patient_name}.h5")
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_path = os.path.join(patient_dir, 'preprocessed.h5')
        
        # 이미 전처리되어 있으면 스킵
        if skip_existing and os.path.exists(output_path):
            skip_count += 1
            continue
        
        if preprocess_volume(patient_dir, use_4modalities, output_path):
            success_count += 1
        else:
            error_count += 1
    
    print()
    print(f"Preprocessing completed:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess BraTS dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data root directory')
    parser.add_argument('--dataset_version', type=str, default='brats2021',
                        choices=['brats2021', 'brats2018'],
                        help='Dataset version')
    parser.add_argument('--use_4modalities', action='store_true',
                        help='Use 4 modalities (T1, T1ce, T2, FLAIR)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for preprocessed files (None: save in original directory)')
    parser.add_argument('--no_skip_existing', action='store_true',
                        help='Do not skip existing preprocessed files')
    
    args = parser.parse_args()
    
    preprocess_all_volumes(
        data_dir=args.data_dir,
        dataset_version=args.dataset_version,
        use_4modalities=args.use_4modalities,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip_existing
    )


if __name__ == '__main__':
    main()


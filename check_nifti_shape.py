#!/usr/bin/env python3
"""
원본 NIfTI 파일의 shape을 확인하는 스크립트

Usage:
    python check_nifti_shape.py --patient_dir /path/to/patient/dir
    python check_nifti_shape.py --data_dir /path/to/brats/data --dataset_version brats2024 --sample_count 5
"""

import os
import argparse
import nibabel as nib
from pathlib import Path


def check_single_patient(patient_dir):
    """단일 환자 디렉토리의 NIfTI 파일 shape 확인"""
    files = os.listdir(patient_dir)
    files_lower = [f.lower() for f in files]
    
    def _find_exact_suffix(suffix: str):
        for f in files:
            if f.endswith(suffix):
                return f
        return None
    
    def _find_first_file(contains_list, exclude_list=None):
        if exclude_list is None:
            exclude_list = []
        candidates = []
        for orig, low in zip(files, files_lower):
            if not (low.endswith(".nii") or low.endswith(".nii.gz")):
                continue
            if all(c in low for c in contains_list) and not any(ex in low for ex in exclude_list):
                candidates.append(orig)
        return candidates[0] if candidates else None
    
    # BRATS2023/2024 패턴 먼저 시도
    t1_file = _find_exact_suffix("-t1n.nii.gz")
    t1ce_file = _find_exact_suffix("-t1c.nii.gz")
    flair_file = _find_exact_suffix("-t2f.nii.gz")
    t2_file = _find_exact_suffix("-t2w.nii.gz")
    seg_file = _find_exact_suffix("-seg.nii.gz")
    
    if not all(f is not None for f in [t1_file, t1ce_file, flair_file, t2_file, seg_file]):
        # 일반 패턴 시도
        t1ce_file = _find_first_file(["t1ce"], []) or _find_first_file(["t1c"], [])
        flair_file = _find_first_file(["flair"], [])
        seg_file = _find_first_file(["seg"], []) or _find_first_file(["segmentation"], [])
        t1_file = _find_first_file(["t1"], ["t1ce", "t1c"])
        t2_file = _find_first_file(["t2"], [])
    
    if not all(f is not None for f in [t1_file, t1ce_file, flair_file, t2_file, seg_file]):
        print(f"[ERROR] Missing files in {patient_dir}")
        return None
    
    print(f"\n[Patient] {os.path.basename(patient_dir)}")
    print(f"  Directory: {patient_dir}")
    
    shapes = {}
    for name, file in [("T1", t1_file), ("T1CE", t1ce_file), ("T2", t2_file), ("FLAIR", flair_file), ("SEG", seg_file)]:
        file_path = os.path.join(patient_dir, file)
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            shapes[name] = data.shape
            print(f"  {name:6s}: {file:30s} -> shape={data.shape}")
        except Exception as e:
            print(f"  {name:6s}: {file:30s} -> ERROR: {e}")
            return None
    
    # 모든 모달리티의 shape이 같은지 확인
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        print(f"  [WARNING] Different shapes detected: {shapes}")
    else:
        print(f"  [OK] All modalities have the same shape: {list(unique_shapes)[0]}")
    
    return shapes


def check_multiple_patients(data_dir, dataset_version='brats2024', sample_count=5):
    """여러 환자 디렉토리의 NIfTI 파일 shape 확인"""
    if dataset_version == 'brats2024':
        brats2024_root = os.path.join(data_dir, 'BRATS2024')
        train_dir1 = os.path.join(brats2024_root, 'training_data1_v2')
        train_dir2 = os.path.join(brats2024_root, 'training_data_additional')
        
        patient_dirs = []
        for root_dir in [train_dir1, train_dir2]:
            if os.path.exists(root_dir):
                for patient_dir in sorted(os.listdir(root_dir)):
                    patient_path = os.path.join(root_dir, patient_dir)
                    if os.path.isdir(patient_path):
                        patient_dirs.append(patient_path)
    else:
        print(f"[ERROR] dataset_version '{dataset_version}' not supported yet")
        return
    
    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Checking first {min(sample_count, len(patient_dirs))} samples...\n")
    
    all_shapes = {}
    for i, patient_dir in enumerate(patient_dirs[:sample_count]):
        shapes = check_single_patient(patient_dir)
        if shapes:
            for name, shape in shapes.items():
                if name not in all_shapes:
                    all_shapes[name] = []
                all_shapes[name].append(shape)
    
    # 통계 출력
    print(f"\n[Summary] Shape statistics across {min(sample_count, len(patient_dirs))} samples:")
    for name, shape_list in all_shapes.items():
        unique_shapes = set(tuple(s) for s in shape_list)
        print(f"  {name:6s}: {len(unique_shapes)} unique shape(s)")
        for shape in unique_shapes:
            count = sum(1 for s in shape_list if tuple(s) == shape)
            print(f"           - {shape}: {count} samples")


def main():
    parser = argparse.ArgumentParser(description='Check NIfTI file shapes')
    parser.add_argument('--patient_dir', type=str, default=None,
                        help='Single patient directory to check')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data root directory (for checking multiple patients)')
    parser.add_argument('--dataset_version', type=str, default='brats2024',
                        choices=['brats2024'],
                        help='Dataset version (only brats2024 supported for now)')
    parser.add_argument('--sample_count', type=int, default=5,
                        help='Number of samples to check when using --data_dir')
    
    args = parser.parse_args()
    
    if args.patient_dir:
        check_single_patient(args.patient_dir)
    elif args.data_dir:
        check_multiple_patients(args.data_dir, args.dataset_version, args.sample_count)
    else:
        parser.print_help()
        print("\n[ERROR] Either --patient_dir or --data_dir must be provided")


if __name__ == '__main__':
    main()


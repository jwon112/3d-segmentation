#!/usr/bin/env python3
"""
5-Fold Cross-Validation 데이터셋 분할 준비 스크립트

전처리된 H5 파일들을 5-fold로 분할하여 fold별 디렉토리 구조를 생성합니다.
심볼릭 링크를 사용하여 저장 공간을 절약합니다.
여러 BRATS 버전을 합쳐서 하나의 큰 데이터셋으로 만들 수 있습니다.

Usage:
    # 단일 버전
    python prepare_5fold_splits.py --data_dir /path/to/data --dataset_version brats2021
    # 여러 버전 합치기
    python prepare_5fold_splits.py --preprocessed_base_dir /home/work/3D_/processed_data --dataset_versions brats2017 brats2018 brats2019 brats2020 brats2021 brats2023
"""

import os
import json
import random
import argparse
from pathlib import Path


def get_project_root():
    """프로젝트 루트 디렉토리 찾기"""
    return Path(__file__).parent.absolute()


def normalize_patient_name(patient_name, dataset_version):
    """
    환자 이름을 통일된 형식(Brats{YY}_...)으로 변환
    
    Args:
        patient_name: 원본 환자 이름
        dataset_version: 데이터셋 버전 (brats2021, brats2023 등)
    
    Returns:
        변환된 환자 이름
    """
    if dataset_version == 'brats2021':
        # BRATS2021: BraTS2021_00000 -> Brats21_00000
        if patient_name.startswith('BraTS2021_'):
            # BraTS2021_00000 -> Brats21_00000
            number_part = patient_name.replace('BraTS2021_', '')
            return f"Brats21_{number_part}"
        elif patient_name.startswith('BraTS2021'):
            # BraTS202100000 -> Brats21_00000 (하이픈 없는 경우)
            number_part = patient_name.replace('BraTS2021', '')
            return f"Brats21_{number_part}"
    elif dataset_version == 'brats2023':
        # BRATS2023: BraTS-GLI-00000-000 -> Brats23_00000_000
        if patient_name.startswith('BraTS-GLI-'):
            # BraTS-GLI-00000-000 -> Brats23_00000_000
            parts = patient_name.replace('BraTS-GLI-', '').split('-')
            if len(parts) == 2:
                return f"Brats23_{parts[0]}_{parts[1]}"
            else:
                # 하이픈이 하나만 있는 경우
                return f"Brats23_{parts[0]}"
        elif patient_name.startswith('BraTS-GLI'):
            # BraTS-GLI00000-000 형식
            parts = patient_name.replace('BraTS-GLI', '').split('-')
            if len(parts) >= 2:
                return f"Brats23_{parts[0]}_{parts[1]}"
            else:
                return f"Brats23_{parts[0]}"
    
    # 이미 올바른 형식이거나 다른 버전은 그대로 반환
    return patient_name


def prepare_5fold_splits(
    data_dir=None,
    dataset_version=None,
    dataset_versions=None,
    seed=24,
    preprocessed_dir=None,
    preprocessed_base_dir=None,
    output_base_dir=None
):
    """
    5-fold 분할을 미리 준비하여 fold별 디렉토리 구조 생성
    여러 BRATS 버전을 합쳐서 하나의 큰 데이터셋으로 만들 수 있습니다.
    
    Args:
        data_dir: 원본 데이터 디렉토리 (단일 버전 모드에서만 사용)
        dataset_version: 단일 버전 사용 시 'brats2021' 등 (단일 버전 모드)
        dataset_versions: 여러 버전 리스트 ['brats2017', 'brats2018', ...] (다중 버전 모드)
        seed: 분할 시드
        preprocessed_dir: 전처리된 H5 파일이 있는 디렉토리 (단일 버전 모드)
        preprocessed_base_dir: 전처리된 파일들의 기본 디렉토리 (다중 버전 모드, 예: /home/work/3D_/processed_data)
        output_base_dir: fold별 디렉토리를 생성할 기본 경로
    
    Returns:
        output_base_dir: 생성된 fold 디렉토리의 기본 경로
    """
    # 다중 버전 모드인지 확인
    if dataset_versions is not None:
        # 다중 버전 모드: 여러 버전의 전처리된 파일들을 합치기
        if preprocessed_base_dir is None:
            preprocessed_base_dir = Path("/home/work/3D_/processed_data")
        else:
            preprocessed_base_dir = Path(preprocessed_base_dir)
        
        if not preprocessed_base_dir.exists():
            raise FileNotFoundError(f"Preprocessed base directory not found at {preprocessed_base_dir}")
        
        # 각 버전의 전처리된 파일 수집
        all_h5_files = []  # (patient_name, source_path, dataset_version) 튜플 리스트
        version_counts = {}
        
        for version in dataset_versions:
            # 버전명을 대문자로 변환 (brats2017 -> BRATS2017)
            version_upper = version.upper()
            version_dir = preprocessed_base_dir / version_upper
            if not version_dir.exists():
                print(f"Warning: Preprocessed directory not found for {version}: {version_dir}")
                continue
            
            # H5 파일 찾기
            h5_files = list(version_dir.glob("*.h5"))
            version_counts[version] = len(h5_files)
            
            for h5_file in h5_files:
                original_name = h5_file.stem  # .h5 확장자 제거
                # 파일명을 통일된 형식으로 변환 (2021, 2023만 변환)
                normalized_name = normalize_patient_name(original_name, version)
                all_h5_files.append((normalized_name, h5_file, version, original_name))
        
        if not all_h5_files:
            raise ValueError("No preprocessed H5 files found in any of the specified dataset versions")
        
        # 환자 이름으로 정렬 (버전 정보는 메타데이터로 유지)
        all_h5_files.sort(key=lambda x: x[0])  # normalized_name으로 정렬
        
        sorted_patient_names = [x[0] for x in all_h5_files]  # 정규화된 이름
        sorted_h5_files = [x[1] for x in all_h5_files]  # 원본 파일 경로
        sorted_versions = [x[2] for x in all_h5_files]  # 버전
        sorted_original_names = [x[3] for x in all_h5_files]  # 원본 파일명 (디버깅용)
        
        # 중복 체크: 정규화된 이름이 중복되는지 확인
        from collections import Counter
        name_counts = Counter(sorted_patient_names)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        if duplicates:
            print(f"Warning: Found {len(duplicates)} duplicate normalized names:")
            for name, count in list(duplicates.items())[:10]:  # 최대 10개만 출력
                indices = [i for i, n in enumerate(sorted_patient_names) if n == name]
                print(f"  {name}: {count} files")
                for idx in indices:
                    print(f"    - {sorted_original_names[idx]} ({sorted_versions[idx]})")
            if len(duplicates) > 10:
                print(f"  ... and {len(duplicates) - 10} more duplicates")
            print()
        
        total_samples = len(all_h5_files)
        print(f"Found {total_samples} preprocessed H5 files from {len(dataset_versions)} dataset versions:")
        for version, count in version_counts.items():
            print(f"  {version}: {count} files")
        print()
        
        # 출력 디렉토리 설정
        if output_base_dir is None:
            versions_str = "_".join([v.upper().replace('BRATS', '') for v in sorted(dataset_versions)])
            output_base_dir = preprocessed_base_dir / f"COMBINED_{versions_str}_5fold_splits"
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {output_base_dir}")
        print(f"Seed: {seed}")
        print()
        
    else:
        # 단일 버전 모드 (기존 로직)
        if dataset_version is None:
            dataset_version = 'brats2021'
        
        # 샘플 수집 (이름으로 정렬하여 순서 고정)
        if dataset_version == 'brats2021':
            brats_dir = os.path.join(data_dir, 'BRATS2021', 'BraTS2021_Training_Data')
            if not os.path.exists(brats_dir):
                raise FileNotFoundError(f"BraTS2021 dataset not found at {brats_dir}")
            
            patient_dirs = []
            for patient_dir in sorted(os.listdir(brats_dir)):
                patient_path = os.path.join(brats_dir, patient_dir)
                if os.path.isdir(patient_path):
                    patient_dirs.append(patient_path)
        elif dataset_version == 'brats2018':
            brats_dir = os.path.join(data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
            if not os.path.exists(brats_dir):
                raise FileNotFoundError(f"BraTS2018 dataset not found at {brats_dir}")
            
            patient_dirs = []
            hgg_dir = os.path.join(brats_dir, 'HGG')
            lgg_dir = os.path.join(brats_dir, 'LGG')
            
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
        else:
            raise ValueError(f"Unknown dataset_version: {dataset_version}")
        
        if not patient_dirs:
            raise ValueError(f"No patient data found in {brats_dir}")
        
        # 샘플 이름으로 정렬 (파일 시스템 순서와 무관)
        patient_names = [os.path.basename(p) for p in patient_dirs]
        sorted_indices = sorted(range(len(patient_names)), key=lambda i: patient_names[i])
        sorted_patient_dirs = [patient_dirs[i] for i in sorted_indices]
        sorted_patient_names = [patient_names[i] for i in sorted_indices]
        
        print(f"Found {len(sorted_patient_dirs)} patient directories")
        
        # 출력 디렉토리 설정
        if output_base_dir is None:
            project_root = get_project_root()
            output_base_dir = project_root / 'data' / f'{dataset_version.upper()}_5fold_splits'
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 전처리된 파일 디렉토리
        if preprocessed_dir is None:
            project_root = get_project_root()
            preprocessed_dir = project_root / 'data' / f'{dataset_version.upper()}_preprocessed'
        preprocessed_dir = Path(preprocessed_dir)
        
        if not preprocessed_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed directory not found at {preprocessed_dir}\n"
                f"Please run preprocess_brats.py first to create preprocessed H5 files."
            )
        
        sorted_h5_files = None  # 단일 버전 모드에서는 사용하지 않음
        sorted_versions = None
        total_samples = len(sorted_patient_names)
        
        print(f"Preprocessed directory: {preprocessed_dir}")
        print(f"Output directory: {output_base_dir}")
        print(f"Seed: {seed}")
        print()
    
    # 5-fold 분할
    random.seed(seed)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    fold_size = len(indices) // 5
    fold_sizes = [fold_size] * 5
    remainder = len(indices) % 5
    for i in range(remainder):
        fold_sizes[i] += 1
    
    fold_starts = [0]
    for size in fold_sizes:
        fold_starts.append(fold_starts[-1] + size)
    
    # 각 fold별로 디렉토리 생성 및 심볼릭 링크
    split_info = {}
    for fold_idx in range(5):
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
        
        # Fold 디렉토리 생성
        fold_dir = output_base_dir / f'fold_{fold_idx}'
        train_dir = fold_dir / 'train'
        val_dir = fold_dir / 'val'
        test_dir = fold_dir / 'test'
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 심볼릭 링크 생성
        def create_links(indices_list, target_dir, split_name):
            created = 0
            missing = 0
            for idx in indices_list:
                patient_name = sorted_patient_names[idx]
                
                # 다중 버전 모드인지 확인
                if sorted_h5_files is not None:
                    # 다중 버전 모드: 이미 수집된 H5 파일 경로 사용
                    source = sorted_h5_files[idx]
                else:
                    # 단일 버전 모드: 기존 로직
                    source = preprocessed_dir / f"{patient_name}.h5"
                
                # 정규화된 이름 사용 (이미 통일된 형식이므로 버전 정보 불필요)
                # 하지만 중복 방지를 위해 버전 정보를 포함할 수도 있음
                unique_name = f"{patient_name}.h5"
                
                # 중복 체크: 같은 이름이 이미 존재하는지 확인
                # (다른 버전에서 같은 정규화된 이름이 나올 수 있음)
                if sorted_versions is not None:
                    version = sorted_versions[idx]
                    # 중복을 방지하기 위해 버전 정보를 포함 (선택적)
                    # 주석 처리: 정규화된 이름이 고유하다고 가정
                    # unique_name = f"{patient_name}_{version}.h5"
                
                target = target_dir / unique_name
                
                if source.exists():
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    try:
                        target.symlink_to(source.absolute())
                        created += 1
                    except OSError as e:
                        # Windows에서는 관리자 권한이 필요할 수 있음
                        print(f"Warning: Failed to create symlink for {patient_name}: {e}")
                        # Windows에서는 복사로 대체
                        import shutil
                        shutil.copy2(source, target)
                        created += 1
                else:
                    missing += 1
                    print(f"Warning: Preprocessed file not found: {source}")
            return created, missing
        
        train_created, train_missing = create_links(train_indices, train_dir, 'train')
        val_created, val_missing = create_links(val_indices, val_dir, 'val')
        test_created, test_missing = create_links(test_indices, test_dir, 'test')
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_indices)} samples ({train_created} created, {train_missing} missing)")
        print(f"  Val:   {len(val_indices)} samples ({val_created} created, {val_missing} missing)")
        print(f"  Test:  {len(test_indices)} samples ({test_created} created, {test_missing} missing)")
        
        # 분할 정보 저장
        if sorted_versions is not None:
            # 다중 버전 모드: 버전 정보 포함
            split_info[f'fold_{fold_idx}'] = {
                'train': [{'name': sorted_patient_names[i], 'version': sorted_versions[i]} for i in train_indices],
                'val': [{'name': sorted_patient_names[i], 'version': sorted_versions[i]} for i in val_indices],
                'test': [{'name': sorted_patient_names[i], 'version': sorted_versions[i]} for i in test_indices],
            }
        else:
            # 단일 버전 모드: 기존 형식 유지
            split_info[f'fold_{fold_idx}'] = {
                'train': [sorted_patient_names[i] for i in train_indices],
                'val': [sorted_patient_names[i] for i in val_indices],
                'test': [sorted_patient_names[i] for i in test_indices],
            }
    
    # 분할 정보를 JSON으로 저장
    split_info_path = output_base_dir / 'split_info.json'
    info_dict = {
        'seed': seed,
        'total_samples': total_samples,
        'splits': split_info
    }
    
    if sorted_versions is not None:
        info_dict['dataset_versions'] = dataset_versions
        info_dict['version_counts'] = version_counts
    else:
        info_dict['dataset_version'] = dataset_version
    
    with open(split_info_path, 'w') as f:
        json.dump(info_dict, f, indent=2)
    
    print()
    print(f"5-fold splits prepared in {output_base_dir}")
    print(f"Split info saved to {split_info_path}")
    
    return str(output_base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare 5-fold cross-validation splits')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Root directory containing BRATS dataset (single version mode)')
    parser.add_argument('--dataset_version', type=str, default=None,
                       choices=['brats2017', 'brats2018', 'brats2019', 'brats2020', 'brats2021', 'brats2023', 'brats2024'],
                       help='Single dataset version (single version mode)')
    parser.add_argument('--dataset_versions', type=str, nargs='+', default=None,
                       choices=['brats2017', 'brats2018', 'brats2019', 'brats2020', 'brats2021', 'brats2023', 'brats2024'],
                       help='Multiple dataset versions to combine (multi-version mode)')
    parser.add_argument('--seed', type=int, default=24,
                       help='Random seed for splitting (default: 24)')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory containing preprocessed H5 files (single version mode, default: data/{DATASET}_preprocessed)')
    parser.add_argument('--preprocessed_base_dir', type=str, default=None,
                       help='Base directory containing preprocessed H5 files (multi-version mode, default: /home/work/3D_/processed_data)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for fold splits')
    
    args = parser.parse_args()
    
    # 다중 버전 모드인지 확인
    if args.dataset_versions is not None:
        # 다중 버전 모드
        prepare_5fold_splits(
            dataset_versions=args.dataset_versions,
            seed=args.seed,
            preprocessed_base_dir=args.preprocessed_base_dir,
            output_base_dir=args.output_dir,
        )
    else:
        # 단일 버전 모드
        if args.dataset_version is None:
            args.dataset_version = 'brats2021'
        if args.data_dir is None:
            parser.error("--data_dir is required in single version mode")
        
        prepare_5fold_splits(
            data_dir=args.data_dir,
            dataset_version=args.dataset_version,
            seed=args.seed,
            preprocessed_dir=args.preprocessed_dir,
            output_base_dir=args.output_dir,
        )


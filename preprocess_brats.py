#!/usr/bin/env python3
"""
BraTS 데이터 전처리 스크립트

모든 NIfTI 파일을 전처리하여 HDF5 형식으로 저장합니다.
이렇게 하면 학습 시 로딩 속도가 7-10배 빨라집니다.

Usage:
    python preprocess_brats.py --data_dir /home/work/3D_/BT --dataset_version brats2017
    python preprocess_brats.py --data_dir /home/work/3D_/BT --dataset_version brats2018
    python preprocess_brats.py --data_dir /home/work/3D_/BT --dataset_version brats2019
    python preprocess_brats.py --data_dir /home/work/3D_/BT --dataset_version brats2020
    python preprocess_brats.py --data_dir /home/work/3D_/BT --dataset_version brats2021
    python preprocess_brats.py --data_dir /home/work/3D_/BT --dataset_version brats2023
    python preprocess_brats.py --data_dir /home/work/3D_/BT --dataset_version brats2024
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import nibabel as nib
import numpy as np


def get_project_root():
    """프로젝트 루트 디렉토리 찾기 (preprocess_brats.py가 있는 디렉토리)"""
    return Path(__file__).parent.absolute()


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


def preprocess_volume(patient_dir, use_4modalities=False, output_path=None, force_overwrite=False):
    """
    단일 볼륨을 전처리하여 저장
    
    Note:
        use_4modalities 파라미터는 무시됩니다 (하위 호환성을 위해 유지).
        H5 파일은 항상 4개 모달리티 (T1, T1CE, T2, FLAIR)로 저장됩니다.
        실험 시 필요한 모달리티만 선택적으로 로드할 수 있습니다.
    
    Args:
        patient_dir: 환자 디렉토리 경로
        use_4modalities: 무시됨 (하위 호환성을 위해 유지). H5 파일은 항상 4개 모달리티로 저장됩니다.
        output_path: 저장할 파일 경로 (None이면 patient_dir/preprocessed.h5)
        force_overwrite: 강제 덮어쓰기 여부
    
    Returns:
        bool: 성공 여부
    """
    if output_path is None:
        output_path = os.path.join(patient_dir, 'preprocessed.h5')
    
    # 파일이 존재하는 경우 확인
    if os.path.exists(output_path) and not force_overwrite:
        # 모달리티 구성 확인 (항상 4개 모달리티로 저장되므로 4개인지 확인)
        try:
            with h5py.File(output_path, 'r') as f:
                if 'image' in f:
                    existing_channels = f['image'].shape[0]
                    # H5 파일은 항상 4개 모달리티로 저장됨
                    if existing_channels == 4:
                        return True
                    # 모달리티 구성이 다르면 재처리 필요 (파일 삭제)
                    os.remove(output_path)
        except Exception:
            # 파일 읽기 실패 시 재처리
            os.remove(output_path)
    
    try:
        # 파일 이름 찾기
        files = os.listdir(patient_dir)
        files_lower = [f.lower() for f in files]

        # 1) BRATS2023/2024 전용 패턴 먼저 시도
        # 예) BraTS-GLI-00005-100-t1n.nii.gz, -t1c.nii.gz, -t2f.nii.gz, -t2w.nii.gz, -seg.nii.gz
        def _find_exact_suffix(suffix: str):
            for f in files:
                if f.endswith(suffix):
                    return f
            return None

        t1_file = _find_exact_suffix("-t1n.nii.gz")
        t1ce_file = _find_exact_suffix("-t1c.nii.gz")
        flair_file = _find_exact_suffix("-t2f.nii.gz")  # FLAIR
        t2_file = _find_exact_suffix("-t2w.nii.gz")     # T2-weighted
        seg_file = _find_exact_suffix("-seg.nii.gz")

        if any(f is not None for f in [t1_file, t1ce_file, flair_file, t2_file, seg_file]):
            # BRATS2023/2024 패턴으로 인식된 경우: 이 매핑을 우선 사용
            missing = []
            if t1_file is None:
                missing.append("T1(-t1n.nii.gz)")
            if t1ce_file is None:
                missing.append("T1CE(-t1c.nii.gz)")
            if t2_file is None:
                missing.append("T2(-t2w.nii.gz)")
            if flair_file is None:
                missing.append("FLAIR(-t2f.nii.gz)")
            if seg_file is None:
                missing.append("SEG(-seg.nii.gz)")

            if missing:
                print(
                    f"[WARN] Skipping {patient_dir}: missing BRATS2023/2024-style modalities {missing}. "
                    f"Files in dir: {files}"
                )
                return False
        else:
            # 2) BRATS2021/2018 및 기타 일반 패턴 (기존 로직을 보다 안전하게)
            def _find_first_file(contains_list, exclude_list=None):
                """
                주어진 키워드를 포함하고(복수 가능), 제외 키워드는 포함하지 않는
                첫 번째 NIfTI 파일(.nii 또는 .nii.gz)을 찾는다.
                """
                if exclude_list is None:
                    exclude_list = []
                candidates = []
                for orig, low in zip(files, files_lower):
                    if not (low.endswith(".nii") or low.endswith(".nii.gz")):
                        continue
                    if all(c in low for c in contains_list) and not any(ex in low for ex in exclude_list):
                        candidates.append(orig)
                return candidates[0] if candidates else None

            # - T1CE: "t1ce" 또는 "t1c"
            # - FLAIR: "flair"
            # - T1: "t1" 이면서 "t1ce"/"t1c" 는 제외
            # - T2: "t2"
            # - seg: "seg" 또는 "segmentation"
            t1ce_file = _find_first_file(["t1ce"], []) or _find_first_file(["t1c"], [])
            flair_file = _find_first_file(["flair"], [])
            seg_file = _find_first_file(["seg"], []) or _find_first_file(["segmentation"], [])
            t1_file = _find_first_file(["t1"], ["t1ce", "t1c"])
            t2_file = _find_first_file(["t2"], [])

            missing = []
            if t1_file is None:
                missing.append("T1")
            if t1ce_file is None:
                missing.append("T1CE")
            if t2_file is None:
                missing.append("T2")
            if flair_file is None:
                missing.append("FLAIR")
            if seg_file is None:
                missing.append("SEG")

            if missing:
                print(
                    f"[WARN] Skipping {patient_dir}: missing modalities {missing}. "
                    f"Files in dir: {files}"
                )
                return False

        # NIfTI 파일 로드
        t1ce = nib.load(os.path.join(patient_dir, t1ce_file)).get_fdata()
        flair = nib.load(os.path.join(patient_dir, flair_file)).get_fdata()
        seg = nib.load(os.path.join(patient_dir, seg_file)).get_fdata()

        # 정규화
        t1ce = _normalize_volume_np(t1ce)
        flair = _normalize_volume_np(flair)

        # H5 파일은 항상 4개 모달리티로 저장 (T1, T1CE, T2, FLAIR)
        # 실험 시 필요한 모달리티만 선택적으로 로드 가능
        t1 = nib.load(os.path.join(patient_dir, t1_file)).get_fdata()
        t2 = nib.load(os.path.join(patient_dir, t2_file)).get_fdata()
        t1 = _normalize_volume_np(t1)
        t2 = _normalize_volume_np(t2)
        image = np.stack([t1, t1ce, t2, flair], axis=-1)  # (H, W, D, 4)
        
        # (H, W, D, C) -> (C, H, W, D)로 변환
        image = np.transpose(image, (3, 0, 1, 2)).astype(np.float32)
        
        # Mask 처리
        mask = seg.astype(np.int64)
        mask = np.where(mask == 4, 3, mask)
        
        # 포그라운드 좌표 사전 계산 (패치 샘플링 최적화)
        # 클래스별 포그라운드 좌표를 미리 계산하여 저장
        fg_coords_dict = {}
        for cls in [1, 2, 3]:
            coords = np.argwhere(mask == cls)
            if len(coords) > 0:
                fg_coords_dict[f'fg_coords_{cls}'] = coords
        
        # HDF5로 저장 (메타데이터 포함, gzip level 4 - 벤치마크 결과 최적값)
        # 벤치마크 결과: level 4가 Cold cache 로딩(270ms)에서 가장 빠름, 파일 크기(6.89MB)도 최소
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('image', data=image, compression='gzip', compression_opts=4)
            f.create_dataset('mask', data=mask, compression='gzip', compression_opts=4)
            # 포그라운드 좌표 저장 (패치 샘플링 최적화)
            for cls_key, coords in fg_coords_dict.items():
                f.create_dataset(cls_key, data=coords, compression='gzip', compression_opts=1)
            # 메타데이터 저장 (모달리티 구성 확인용)
            # H5 파일은 항상 4개 모달리티로 저장되므로 use_4modalities는 항상 True
            f.attrs['use_4modalities'] = True
            f.attrs['num_channels'] = image.shape[0]  # 항상 4
            f.attrs['has_fg_coords'] = len(fg_coords_dict) > 0
        
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
        dataset_version: 'brats2017', 'brats2018', 'brats2019', 'brats2020', 'brats2021', 'brats2023', 'brats2024'
        use_4modalities: 무시됨 (하위 호환성을 위해 유지). H5 파일은 항상 4개 모달리티로 저장됩니다.
        output_dir: 저장할 디렉토리 (None이면 기본 전처리 디렉토리 사용)
        skip_existing: 이미 전처리된 파일 스킵 여부
    
    Note:
        - H5 파일은 항상 4개 모달리티 (T1, T1CE, T2, FLAIR)로 저장됩니다.
        - 실험 시 필요한 모달리티만 선택적으로 로드할 수 있습니다 (use_4modalities 파라미터 사용).
        - dataset_version에 따라 기본 output_dir이 달라집니다.
          * brats2017-2021: 프로젝트 루트의 data/ 디렉토리 하위에 저장
          * brats2023, brats2024: /home/work/3D_/processed_data/ 하위에 저장
    """
    # 기본 전처리 디렉토리 설정 (output_dir이 None인 경우)
    if output_dir is None:
        if dataset_version in ('brats2017', 'brats2018', 'brats2019', 'brats2020', 'brats2021'):
            # 기존 브라츠 버전: 프로젝트 루트의 data/ 디렉토리 하위에 저장
            project_root = get_project_root()
            data_dir_path = project_root / 'data'
            version_map = {
                'brats2017': 'BRATS2017_preprocessed',
                'brats2018': 'BRATS2018_preprocessed',
                'brats2019': 'BRATS2019_preprocessed',
                'brats2020': 'BRATS2020_preprocessed',
                'brats2021': 'BRATS2021_preprocessed',
            }
            output_dir = data_dir_path / version_map[dataset_version]
            output_dir = str(output_dir)
        elif dataset_version in ('brats2023', 'brats2024'):
            # BRATS2023, BRATS2024: /home/work/3D_/processed_data/ 하위에 저장
            processed_root = Path("/home/work/3D_/processed_data")
            if dataset_version == 'brats2023':
                output_dir = processed_root / "BRATS2023"
            elif dataset_version == 'brats2024':
                output_dir = processed_root / "BRATS2024"
            output_dir = str(output_dir)
        else:
            raise ValueError(f"Unknown dataset_version: {dataset_version}")
        os.makedirs(output_dir, exist_ok=True)
    # 데이터셋 경로 설정
    if dataset_version == 'brats2017':
        # BRATS2017/Brats17TrainingData → HGG/LGG → Brats17_2013_2_1, Brats17_CBICA_AAB_1
        brats_dir = os.path.join(data_dir, 'BRATS2017', 'Brats17TrainingData')
        hgg_dir = os.path.join(brats_dir, 'HGG')
        lgg_dir = os.path.join(brats_dir, 'LGG')
    elif dataset_version == 'brats2018':
        # BRATS2018/MICCAI_BraTS_2018_Data_Training → HGG/LGG → Brats18_2013_2_1, Brats18_CBICA_AAB_1
        brats_dir = os.path.join(data_dir, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
        hgg_dir = os.path.join(brats_dir, 'HGG')
        lgg_dir = os.path.join(brats_dir, 'LGG')
    elif dataset_version == 'brats2019':
        # BRATS2019/MICCAI_BraTS_2019_Data_Training → HGG/LGG → Brats19_2013_2_1, Brats19_CBICA_AAB_1
        # (루트 경로명이 명시되지 않았지만 일반적인 패턴으로 추정)
        brats_dir = os.path.join(data_dir, 'BRATS2019', 'MICCAI_BraTS_2019_Data_Training')
        hgg_dir = os.path.join(brats_dir, 'HGG')
        lgg_dir = os.path.join(brats_dir, 'LGG')
    elif dataset_version == 'brats2020':
        # BRATS2020/MICCAI_BraTS2020_TrainingData → BraTS20_Training_001 (HGG/LGG 없음)
        brats_dir = os.path.join(data_dir, 'BRATS2020', 'MICCAI_BraTS2020_TrainingData')
    elif dataset_version == 'brats2021':
        # BRATS2021/BraTS2021_Training_Data → BraTS2021_00000 (HGG/LGG 없음)
        brats_dir = os.path.join(data_dir, 'BRATS2021', 'BraTS2021_Training_Data')
    elif dataset_version == 'brats2023':
        # BRATS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData → BraTS-GLI-00000-000 (HGG/LGG 없음)
        brats_dir = os.path.join(data_dir, 'BRATS2023', 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
    elif dataset_version == 'brats2024':
        # BRATS2024:
        #   /home/work/3D_/BT/BRATS2024/training_data1_v2
        #   /home/work/3D_/BT/BRATS2024/training_data_additional
        brats2024_root = os.path.join(data_dir, 'BRATS2024')
        train_dir1 = os.path.join(brats2024_root, 'training_data1_v2')
        train_dir2 = os.path.join(brats2024_root, 'training_data_additional')
        # existence check는 아래에서 patient_dirs 수집 시 함께 수행
    else:
        raise ValueError(f"Unknown dataset_version: {dataset_version}")
    
    # 데이터셋 존재 여부 확인
    if dataset_version in ('brats2017', 'brats2018', 'brats2019'):
        # HGG/LGG 구조를 가진 버전들
        if not os.path.exists(brats_dir):
            raise FileNotFoundError(f"Dataset not found at {brats_dir}")
        if not os.path.exists(hgg_dir) and not os.path.exists(lgg_dir):
            raise FileNotFoundError(
                f"Neither HGG nor LGG directory found in {brats_dir}\n"
                f"Expected: {hgg_dir} or {lgg_dir}"
            )
    elif dataset_version in ('brats2020', 'brats2021', 'brats2023'):
        # 단일 디렉토리 구조를 가진 버전들
        if not os.path.exists(brats_dir):
            raise FileNotFoundError(f"Dataset not found at {brats_dir}")
    elif dataset_version == 'brats2024':
        # BRATS2024는 두 개의 학습 디렉토리를 사용하므로 각각 존재 여부 확인
        missing_dirs = []
        for d in [train_dir1, train_dir2]:
            if not os.path.exists(d):
                missing_dirs.append(d)
        if missing_dirs:
            raise FileNotFoundError(
                "BRATS2024 dataset directories not found:\n" +
                "\n".join(f"  - {p}" for p in missing_dirs)
            )
    
    # 환자 디렉토리 수집
    patient_dirs = []
    if dataset_version in ('brats2017', 'brats2018', 'brats2019'):
        # HGG/LGG 구조를 가진 버전들
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
    elif dataset_version in ('brats2020', 'brats2021', 'brats2023'):
        # 단일 디렉토리 구조를 가진 버전들
        for patient_dir in sorted(os.listdir(brats_dir)):
            patient_path = os.path.join(brats_dir, patient_dir)
            if os.path.isdir(patient_path):
                patient_dirs.append(patient_path)
    elif dataset_version == 'brats2024':
        # 두 개의 학습 디렉토리에서 환자 디렉토리를 모두 수집하여 단일 리스트로 통합
        for root_dir in [train_dir1, train_dir2]:
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"BRATS2024 train directory not found: {root_dir}")
            for patient_dir in sorted(os.listdir(root_dir)):
                patient_path = os.path.join(root_dir, patient_dir)
                if os.path.isdir(patient_path):
                    patient_dirs.append(patient_path)
    
    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Dataset version: {dataset_version}")
    print(f"Modalities: 4 (T1, T1CE, T2, FLAIR) - always saved as 4 modalities")
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
        
        force_overwrite = not skip_existing
        if preprocess_volume(patient_dir, use_4modalities, output_path, force_overwrite):
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
                        help='Data root directory (e.g., /home/work/3D_/BT). Should contain BRATS2017, BRATS2018, etc. subdirectories')
    parser.add_argument('--dataset_version', type=str, default='brats2021',
                        choices=['brats2017', 'brats2018', 'brats2019', 'brats2020', 'brats2021', 'brats2023', 'brats2024'],
                        help='Dataset version. '
                             'brats2017-2019: HGG/LGG structure. '
                             'brats2020-2021: single directory. '
                             'brats2023: single directory. '
                             'brats2024: uses BRATS2024/training_data1_v2 and training_data_additional, '
                             'outputs to /home/work/3D_/processed_data/BRATS2024 by default')
    parser.add_argument('--use_4modalities', action='store_true',
                        help='(Deprecated) H5 files are always saved with 4 modalities (T1, T1CE, T2, FLAIR). This flag is ignored but kept for backward compatibility.')
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


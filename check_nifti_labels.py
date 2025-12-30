#!/usr/bin/env python3
"""BRATS2024 원본 NIfTI 파일에서 라벨 분포 확인"""

import os
import numpy as np
import nibabel as nib

def check_brats2024_nifti_labels(data_dir=None, max_samples=50):
    """BRATS2024 원본 NIfTI 파일에서 라벨 분포 확인"""
    
    if data_dir is None:
        # 기본 경로: Windows 로컬 경로
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data', 'BRATS2024', 'brats2024-brats-gli-trainingdata')
    
    # BRATS2024 디렉토리 구조 확인
    patient_dirs = []
    
    # 경로가 직접 환자 디렉토리인지 확인
    if os.path.basename(data_dir) == 'brats2024-brats-gli-trainingdata' or os.path.exists(data_dir):
        # 환자 디렉토리들이 바로 이 디렉토리에 있는지 확인
        try:
            items = os.listdir(data_dir)
            # 첫 번째 항목이 디렉토리이고 seg 파일이 있는지 확인
            for item in items[:5]:  # 처음 5개만 확인
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    # seg 파일이 있는지 확인
                    files = os.listdir(item_path)
                    if any('seg' in f.lower() for f in files):
                        # 환자 디렉토리들이 바로 여기에 있음
                        for patient_item in os.listdir(data_dir):
                            patient_path = os.path.join(data_dir, patient_item)
                            if os.path.isdir(patient_path):
                                patient_dirs.append(patient_path)
                        break
        except:
            pass
    
    # 위에서 찾지 못했으면 기존 구조 시도
    if not patient_dirs:
        train_dir1 = os.path.join(data_dir, 'training_data1_v2')
        train_dir2 = os.path.join(data_dir, 'training_data_additional')
        
        for root_dir in [train_dir1, train_dir2]:
            if os.path.exists(root_dir):
                for patient_dir in sorted(os.listdir(root_dir)):
                    patient_path = os.path.join(root_dir, patient_dir)
                    if os.path.isdir(patient_path):
                        patient_dirs.append(patient_path)
    
    patient_dirs = []
    for root_dir in [train_dir1, train_dir2]:
        if os.path.exists(root_dir):
            for patient_dir in sorted(os.listdir(root_dir)):
                patient_path = os.path.join(root_dir, patient_dir)
                if os.path.isdir(patient_path):
                    patient_dirs.append(patient_path)
    
    if not patient_dirs:
        print(f"No patient directories found in: {data_dir}")
        return
    
    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Checking first {max_samples} samples...\n")
    
    label_3_count = 0  # 라벨 3을 가진 샘플 수
    label_4_count = 0  # 라벨 4를 가진 샘플 수
    both_labels_count = 0  # 라벨 3과 4를 모두 가진 샘플 수
    total_label_3_voxels = 0
    total_label_4_voxels = 0
    samples_checked = 0
    
    for idx, patient_dir in enumerate(patient_dirs[:max_samples]):
        try:
            files = os.listdir(patient_dir)
            seg_file = None
            for f in files:
                if 'seg' in f.lower() and f.endswith('.nii.gz'):
                    seg_file = f
                    break
            
            if seg_file is None:
                continue
            
            seg_path = os.path.join(patient_dir, seg_file)
            
            # 라벨 로드
            seg = nib.load(seg_path).get_fdata()
            unique_labels = np.unique(seg).astype(int).tolist()
            
            has_label_3 = 3 in unique_labels
            has_label_4 = 4 in unique_labels
            
            if has_label_3:
                label_3_count += 1
                count_3 = np.sum(seg == 3)
                total_label_3_voxels += count_3
            if has_label_4:
                label_4_count += 1
                count_4 = np.sum(seg == 4)
                total_label_4_voxels += count_4
            if has_label_3 and has_label_4:
                both_labels_count += 1
            
            samples_checked += 1
            
            if idx < 10:  # 처음 10개만 상세 출력
                print(f"[{idx+1}] {os.path.basename(patient_dir)}")
                print(f"  Unique labels: {unique_labels}")
                print(f"  Has label 3: {has_label_3}, Has label 4: {has_label_4}")
                
                if has_label_3:
                    count_3 = np.sum(seg == 3)
                    print(f"    Label 3 voxels: {count_3:,} ({count_3/seg.size*100:.3f}%)")
                if has_label_4:
                    count_4 = np.sum(seg == 4)
                    print(f"    Label 4 voxels: {count_4:,} ({count_4/seg.size*100:.3f}%)")
                
                # 라벨 1, 2도 확인
                if 1 in unique_labels:
                    count_1 = np.sum(seg == 1)
                    print(f"    Label 1 (NCR/NET) voxels: {count_1:,} ({count_1/seg.size*100:.3f}%)")
                if 2 in unique_labels:
                    count_2 = np.sum(seg == 2)
                    print(f"    Label 2 (ED) voxels: {count_2:,} ({count_2/seg.size*100:.3f}%)")
                print()
        except Exception as e:
            print(f"Error processing {patient_dir}: {e}")
            continue
    
    print("="*60)
    print(f"Summary over {samples_checked} samples:")
    print(f"  Samples with label 3: {label_3_count}/{samples_checked}")
    print(f"  Samples with label 4: {label_4_count}/{samples_checked}")
    print(f"  Samples with both 3 and 4: {both_labels_count}/{samples_checked}")
    if label_3_count > 0:
        print(f"  Average label 3 voxels per sample: {total_label_3_voxels/label_3_count:,.0f}")
    if label_4_count > 0:
        print(f"  Average label 4 voxels per sample: {total_label_4_voxels/label_4_count:,.0f}")
    print("="*60)
    print("\nCurrent code assumption:")
    print("  - Label 3 = RC (Resection Cavity)")
    print("  - Label 4 = ET (Enhancing Tumor)")
    print("\nInterpretation:")
    if label_3_count > label_4_count:
        print("  ✓ Label 3 is more common -> Current mapping (RC=3, ET=4) is likely CORRECT")
        print("    (RC is typically more common than ET in post-surgical cases)")
    elif label_4_count > label_3_count:
        print("  ⚠ Label 4 is more common -> May need to SWAP (RC=4, ET=3)")
        print("    (But ET is usually less common, so this might indicate a mapping issue)")
    else:
        print("  ? Both labels are equally common -> Need to check BRATS2024 documentation")
    
    # 추가 분석: 라벨 3과 4의 공간적 관계
    if both_labels_count > 0:
        print("\n" + "="*60)
        print("Additional analysis: Samples with both label 3 and 4")
        print("="*60)
        overlap_count = 0
        for idx, patient_dir in enumerate(patient_dirs[:max_samples]):
            try:
                files = os.listdir(patient_dir)
                seg_file = None
                for f in files:
                    if 'seg' in f.lower() and f.endswith('.nii.gz'):
                        seg_file = f
                        break
                if seg_file is None:
                    continue
                
                seg_path = os.path.join(patient_dir, seg_file)
                seg = nib.load(seg_path).get_fdata()
                
                has_label_3 = 3 in np.unique(seg)
                has_label_4 = 4 in np.unique(seg)
                
                if has_label_3 and has_label_4:
                    # 공간적 겹침 확인
                    overlap = np.sum((seg == 3) & (seg == 4))
                    if overlap > 0:
                        overlap_count += 1
                        if overlap_count <= 3:  # 처음 3개만 출력
                            print(f"\n{os.path.basename(patient_dir)}:")
                            print(f"  Overlapping voxels (3 & 4): {overlap:,}")
                            print(f"  Label 3 only: {np.sum((seg == 3) & (seg != 4)):,}")
                            print(f"  Label 4 only: {np.sum((seg == 4) & (seg != 3)):,}")
            except:
                continue
        
        if overlap_count == 0:
            print("  No spatial overlap between label 3 and 4 (they are mutually exclusive)")
        else:
            print(f"\n  Found {overlap_count} samples with overlapping labels 3 and 4")

if __name__ == '__main__':
    check_brats2024_nifti_labels()


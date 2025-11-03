#!/usr/bin/env python3
"""
BraTS 라벨 통계 직접 확인 스크립트

NIfTI 파일을 직접 읽어서 라벨 분포를 확인합니다.
"""

import os
import numpy as np
import nibabel as nib
import argparse

def check_label_statistics(data_dir, max_samples=10):
    """실제 NIfTI 파일에서 라벨 통계 확인"""
    
    brats2021_dir = os.path.join(data_dir, 'BraTS2021_Training_Data')
    if not os.path.exists(brats2021_dir):
        print(f"[ERROR] 데이터셋을 찾을 수 없습니다: {brats2021_dir}")
        return
    
    patient_dirs = sorted([d for d in os.listdir(brats2021_dir) 
                          if os.path.isdir(os.path.join(brats2021_dir, d))])
    
    if max_samples:
        patient_dirs = patient_dirs[:max_samples]
    
    print("="*80)
    print("BraTS 라벨 통계 직접 확인 (NIfTI 파일에서)")
    print("="*80)
    print(f"확인할 환자 수: {len(patient_dirs)}")
    print()
    
    all_stats = []
    
    for idx, patient_dir in enumerate(patient_dirs):
        patient_path = os.path.join(brats2021_dir, patient_dir)
        files = os.listdir(patient_path)
        
        # seg 파일 찾기
        seg_file = [f for f in files if 'seg' in f.lower()][0]
        seg_path = os.path.join(patient_path, seg_file)
        
        # 라벨 로드
        seg = nib.load(seg_path).get_fdata()
        
        # 원본 라벨 확인 (매핑 전)
        unique_labels_orig = np.unique(seg).astype(int).tolist()
        total_voxels = seg.size
        
        # 클래스별 통계
        stats = {
            'patient': patient_dir,
            'shape': seg.shape,
            'total_voxels': total_voxels,
            'unique_labels_orig': unique_labels_orig,
            'class_counts': {},
            'class_ratios': {}
        }
        
        print(f"\n[{idx+1}] 환자: {patient_dir}")
        print(f"  Shape: {seg.shape}")
        print(f"  총 복셀 수: {total_voxels:,}")
        print(f"  원본 라벨 값: {unique_labels_orig}")
        
        # 각 라벨 값 확인
        for label_val in [0, 1, 2, 4]:  # BraTS 원본 라벨: 0, 1, 2, 4
            count = np.sum(seg == label_val)
            ratio = count / total_voxels * 100
            stats['class_counts'][label_val] = count
            stats['class_ratios'][label_val] = ratio
            
            if label_val == 0:
                label_name = "Background"
            elif label_val == 1:
                label_name = "NCR/NET"
            elif label_val == 2:
                label_name = "ED"
            elif label_val == 4:
                label_name = "ET (원본)"
            else:
                label_name = f"Unknown({label_val})"
            
            print(f"    라벨 {label_val} ({label_name}): {count:,} 복셀 ({ratio:.2f}%)")
        
        # 포그라운드 합계 (1 + 2 + 4)
        fg_count = (np.sum(seg == 1) + np.sum(seg == 2) + np.sum(seg == 4))
        fg_ratio = fg_count / total_voxels * 100
        stats['fg_count'] = fg_count
        stats['fg_ratio'] = fg_ratio
        
        print(f"    포그라운드 합계 (1+2+4): {fg_count:,} 복셀 ({fg_ratio:.2f}%)")
        print(f"    참고: BraTS 라벨에는 정상 뇌 조직이 라벨링되지 않음")
        print(f"    참고: 라벨 0=배경, 라벨 1,2,4=종양 영역만 존재")
        
        all_stats.append(stats)
    
    # 전체 통계
    print("\n" + "="*80)
    print("전체 평균 통계")
    print("="*80)
    
    avg_fg_ratio = np.mean([s['fg_ratio'] for s in all_stats])
    
    print(f"평균 포그라운드 비율 (전체 볼륨 대비): {avg_fg_ratio:.2f}%")
    
    # 클래스별 평균
    print("\n클래스별 평균:")
    for label_val in [0, 1, 2, 4]:
        avg_count = np.mean([s['class_counts'].get(label_val, 0) for s in all_stats])
        avg_ratio = np.mean([s['class_ratios'].get(label_val, 0) for s in all_stats])
        print(f"  라벨 {label_val}: 평균 {avg_count:.0f} 복셀 ({avg_ratio:.2f}%)")
    
    # 이상치 확인
    print("\n" + "="*80)
    print("이상치 확인")
    print("="*80)
    
    fg_ratios = [s['fg_ratio'] for s in all_stats]
    min_fg = min(fg_ratios)
    max_fg = max(fg_ratios)
    
    print(f"포그라운드 비율 범위: {min_fg:.2f}% ~ {max_fg:.2f}%")
    
    if avg_fg_ratio < 1.0:
        print(f"\n⚠️  참고: 평균 포그라운드 비율이 매우 낮습니다 ({avg_fg_ratio:.2f}%)")
        print("   가능한 원인:")
        print("   1. 전체 볼륨(240x240x155)에 배경이 대부분 포함됨")
        print("   2. 뇌 조직 자체가 볼륨의 일부분만 차지함")
        print("   3. 정상적인 현상일 수 있음 (BraTS 볼륨은 뇌 전체가 아닌 ROI 포함)")
        print("   4. BraTS 라벨에는 정상 뇌 조직이 라벨링되지 않아 포그라운드 비율이 낮음")


def main():
    parser = argparse.ArgumentParser(description='BraTS 라벨 통계 확인')
    parser.add_argument('--data_dir', type=str, 
                        default=os.environ.get('BRATS_DATA_DIR', '/home/work/3D_/BT/BRATS2021'),
                        help='BraTS 데이터셋 루트 디렉토리')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='확인할 최대 샘플 수')
    args = parser.parse_args()
    
    check_label_statistics(args.data_dir, args.max_samples)


if __name__ == '__main__':
    main()


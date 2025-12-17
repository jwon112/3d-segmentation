#!/usr/bin/env python3
"""
HDF5 압축 레벨 벤치마크 스크립트

여러 압축 레벨을 테스트하여 파일 크기와 로딩 시간을 측정합니다.

Usage:
    python benchmark_compression.py --sample_file /path/to/sample/preprocessed.h5 --levels 1 2 3 4 5 6
"""

import os
import time
import h5py
import numpy as np
import torch
from pathlib import Path


def benchmark_compression_levels(sample_file_path, compression_levels=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    여러 압축 레벨을 테스트하여 최적값 찾기
    
    Args:
        sample_file_path: 테스트할 샘플 HDF5 파일 경로 (압축 없이 저장된 파일)
        compression_levels: 테스트할 압축 레벨 리스트
    """
    # 샘플 파일 로드 (압축 없음)
    print(f"Loading sample file: {sample_file_path}")
    with h5py.File(sample_file_path, 'r') as f:
        image = f['image'][:]
        mask = f['mask'][:]
    
    print(f"Sample shape: image={image.shape}, mask={mask.shape}")
    uncompressed_size = image.nbytes + mask.nbytes
    print(f"Sample size (uncompressed): {uncompressed_size / 1024 / 1024:.2f} MB\n")
    
    results = []
    temp_dir = Path(sample_file_path).parent / 'benchmark_temp'
    temp_dir.mkdir(exist_ok=True)
    
    print("="*100)
    print(f"{'Compression':<25} | {'Size (MB)':<12} | {'Ratio':<8} | {'Load (ms)':<15} | {'Save (s)':<10}")
    print("="*100)
    
    for comp_level in compression_levels:
        if comp_level is None:
            comp_name = "None (no compression)"
            comp_kwargs = {'compression': None}
        else:
            comp_name = f"gzip level {comp_level}"
            comp_kwargs = {'compression': 'gzip', 'compression_opts': comp_level}
        
        # 테스트 파일 경로
        test_file = temp_dir / f"test_level_{comp_level if comp_level is not None else 'none'}.h5"
        
        # 저장 시간 측정
        save_start = time.time()
        with h5py.File(test_file, 'w') as f:
            f.create_dataset('image', data=image, **comp_kwargs)
            f.create_dataset('mask', data=mask, **comp_kwargs)
        save_time = time.time() - save_start
        
        # 파일 크기
        file_size = test_file.stat().st_size
        file_size_mb = file_size / 1024 / 1024
        
        # 로딩 시간 측정 (여러 번 평균)
        load_times = []
        for _ in range(5):  # 5번 평균
            load_start = time.time()
            with h5py.File(test_file, 'r') as f:
                loaded_image = torch.from_numpy(f['image'][:]).float()
                loaded_mask = torch.from_numpy(f['mask'][:]).long()
            load_time = time.time() - load_start
            load_times.append(load_time)
        
        avg_load_time = np.mean(load_times)
        std_load_time = np.std(load_times)
        
        # 압축률
        compression_ratio = uncompressed_size / file_size
        
        results.append({
            'compression': comp_name,
            'level': comp_level,
            'file_size_mb': file_size_mb,
            'compression_ratio': compression_ratio,
            'save_time': save_time,
            'load_time_avg': avg_load_time,
            'load_time_std': std_load_time,
        })
        
        print(f"{comp_name:<25} | {file_size_mb:>10.2f} | {compression_ratio:>6.2f}x | "
              f"{avg_load_time*1000:>6.2f}±{std_load_time*1000:>5.2f} | {save_time:>8.2f}")
    
    # 정리
    import shutil
    shutil.rmtree(temp_dir)
    
    return results


def find_optimal_compression(results):
    """
    결과를 분석하여 최적 압축 레벨 찾기
    
    기준: 로딩 시간과 파일 크기의 균형
    """
    print("\n" + "="*100)
    print("최적 압축 레벨 분석")
    print("="*100)
    
    # 로딩 시간 기준 정렬
    results_sorted = sorted(results, key=lambda x: x['load_time_avg'])
    
    print("\n로딩 시간 기준 (빠른 순 Top 5):")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"  {i}. {r['compression']:<25} - {r['load_time_avg']*1000:>6.2f}ms, "
              f"{r['file_size_mb']:>6.2f} MB, {r['compression_ratio']:>5.2f}x")
    
    # 파일 크기와 로딩 시간의 균형점 찾기
    # (로딩 시간 * 가중치 + 파일 크기 * 가중치) 최소화
    best_balanced = None
    best_score = float('inf')
    
    for r in results:
        # 로딩 시간을 ms로, 파일 크기를 MB로 정규화
        # 가중치: 로딩 시간 70%, 파일 크기 30%
        # 로딩 시간이 더 중요하므로 가중치를 높게 설정
        score = r['load_time_avg'] * 1000 * 0.7 + r['file_size_mb'] * 0.3
        if score < best_score:
            best_score = score
            best_balanced = r
    
    print(f"\n균형점 (로딩 70% + 크기 30%):")
    print(f"  Compression: {best_balanced['compression']}")
    print(f"  로딩 시간: {best_balanced['load_time_avg']*1000:.2f}ms ± {best_balanced['load_time_std']*1000:.2f}ms")
    print(f"  파일 크기: {best_balanced['file_size_mb']:.2f} MB")
    print(f"  압축률: {best_balanced['compression_ratio']:.2f}x")
    print(f"  저장 시간: {best_balanced['save_time']:.2f}s")
    
    return best_balanced


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HDF5 압축 레벨 벤치마크 - 최적 압축 레벨 찾기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 테스트 (레벨 1-6)
  python benchmark_compression.py --sample_file data/BRATS2021_preprocessed/BraTS2021_00000.h5
  
  # 특정 레벨만 테스트
  python benchmark_compression.py --sample_file data/BRATS2021_preprocessed/BraTS2021_00000.h5 --levels 1 2 3
  
  # 압축 없음도 포함하여 테스트
  python benchmark_compression.py --sample_file data/BRATS2021_preprocessed/BraTS2021_00000.h5 --levels 0 1 2 3 4
        """
    )
    parser.add_argument('--sample_file', type=str, required=True,
                        help='테스트할 샘플 HDF5 파일 경로 (압축 없이 저장된 파일)')
    parser.add_argument('--levels', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6],
                        help='테스트할 압축 레벨 (기본: 1-6, 0은 압축 없음을 의미)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sample_file):
        print(f"Error: File not found: {args.sample_file}")
        return
    
    # 0을 None으로 변환 (압축 없음)
    compression_levels = []
    for level in args.levels:
        if level == 0:
            compression_levels.append(None)
        else:
            compression_levels.append(level)
    
    # None이 없으면 추가 (비교를 위해)
    if None not in compression_levels:
        compression_levels = [None] + compression_levels
    
    print("="*100)
    print("HDF5 압축 레벨 벤치마크")
    print("="*100)
    print()
    
    results = benchmark_compression_levels(args.sample_file, compression_levels)
    
    best = find_optimal_compression(results)
    
    print(f"\n{'='*100}")
    print("권장 사항")
    print(f"{'='*100}")
    print(f"최적 압축 레벨: {best['level']}")
    print(f"\n코드에 적용 (preprocess_brats.py):")
    if best['level'] is None:
        print(f"  f.create_dataset('image', data=image, compression=None)")
        print(f"  f.create_dataset('mask', data=mask, compression=None)")
    else:
        print(f"  f.create_dataset('image', data=image, compression='gzip', compression_opts={best['level']})")
        print(f"  f.create_dataset('mask', data=mask, compression='gzip', compression_opts={best['level']})")


if __name__ == '__main__':
    main()


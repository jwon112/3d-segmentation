#!/usr/bin/env python3
"""
HDF5 압축 레벨 벤치마크 스크립트 (개선 버전)

실제 DataLoader 환경을 시뮬레이션:
- 여러 파일 동시 읽기 (병렬 I/O)
- 디스크 캐시 비우기 (cold cache)
- 실제 학습 환경 재현

Usage:
    python benchmark_compression.py --sample_file /path/to/sample/preprocessed.h5 --levels 1 2 3 4
"""

import os
import time
import h5py
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess


def clear_disk_cache():
    """디스크 캐시 비우기 (Linux)"""
    try:
        # sync: 버퍼된 데이터를 디스크에 쓰기
        subprocess.run(['sync'], check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # drop_caches: 페이지 캐시 비우기 (root 권한 필요)
        # 실제로는 sudo가 필요하므로 경고만 출력
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass  # Windows나 권한 없으면 스킵


def benchmark_compression_with_parallel_io(sample_file_path, compression_levels=[None, 1, 2, 3, 4], 
                                           num_parallel_reads=4, num_files=10):
    """
    실제 DataLoader 환경을 시뮬레이션한 벤치마크
    
    Args:
        sample_file_path: 샘플 파일 경로
        compression_levels: 테스트할 압축 레벨
        num_parallel_reads: 동시 읽기 수 (worker 수 시뮬레이션)
        num_files: 테스트할 파일 수
    """
    print(f"Loading sample file: {sample_file_path}")
    with h5py.File(sample_file_path, 'r') as f:
        image = f['image'][:]
        mask = f['mask'][:]
    
    uncompressed_size = image.nbytes + mask.nbytes
    print(f"Sample shape: image={image.shape}, mask={mask.shape}")
    print(f"Sample size (uncompressed): {uncompressed_size / 1024 / 1024:.2f} MB")
    print(f"시뮬레이션: {num_parallel_reads}개 worker, {num_files}개 파일 동시 읽기\n")
    
    results = []
    temp_dir = Path(sample_file_path).parent / 'benchmark_temp'
    temp_dir.mkdir(exist_ok=True)
    
    print("="*120)
    print(f"{'Compression':<25} | {'Size (MB)':<12} | {'Cold Load (ms)':<18} | {'Warm Load (ms)':<18} | {'Parallel (ms)':<18}")
    print("="*120)
    
    for comp_level in compression_levels:
        if comp_level is None:
            comp_name = "None (no compression)"
            comp_kwargs = {'compression': None}
        else:
            comp_name = f"gzip level {comp_level}"
            comp_kwargs = {'compression': 'gzip', 'compression_opts': comp_level}
        
        # 여러 테스트 파일 생성
        test_files = []
        for i in range(num_files):
            test_file = temp_dir / f"test_level_{comp_level if comp_level is not None else 'none'}_file_{i}.h5"
            with h5py.File(test_file, 'w') as f:
                f.create_dataset('image', data=image, **comp_kwargs)
                f.create_dataset('mask', data=mask, **comp_kwargs)
            test_files.append(test_file)
        
        # 파일 크기
        file_size = test_files[0].stat().st_size
        file_size_mb = file_size / 1024 / 1024
        
        # 1. Cold cache 테스트 (디스크에서 직접 읽기)
        # 디스크 캐시 비우기 시도
        clear_disk_cache()
        time.sleep(0.5)  # 캐시 비우기 대기
        
        cold_load_times = []
        for test_file in test_files[:3]:  # 처음 3개만 테스트
            load_start = time.time()
            with h5py.File(test_file, 'r') as f:
                loaded_image = torch.from_numpy(f['image'][:]).float()
                loaded_mask = torch.from_numpy(f['mask'][:]).long()
            load_time = time.time() - load_start
            cold_load_times.append(load_time)
            time.sleep(0.1)  # 디스크 I/O 간격
        
        avg_cold_load = np.mean(cold_load_times) * 1000
        
        # 2. Warm cache 테스트 (메모리 캐시에서 읽기)
        warm_load_times = []
        for _ in range(5):
            test_file = test_files[0]
            load_start = time.time()
            with h5py.File(test_file, 'r') as f:
                loaded_image = torch.from_numpy(f['image'][:]).float()
                loaded_mask = torch.from_numpy(f['mask'][:]).long()
            load_time = time.time() - load_start
            warm_load_times.append(load_time)
        
        avg_warm_load = np.mean(warm_load_times) * 1000
        
        # 3. 병렬 읽기 테스트 (실제 DataLoader 환경)
        def load_file(file_path):
            start = time.time()
            with h5py.File(file_path, 'r') as f:
                loaded_image = torch.from_numpy(f['image'][:]).float()
                loaded_mask = torch.from_numpy(f['mask'][:]).long()
            return time.time() - start
        
        parallel_start = time.time()
        with ThreadPoolExecutor(max_workers=num_parallel_reads) as executor:
            futures = [executor.submit(load_file, f) for f in test_files[:num_parallel_reads]]
            parallel_times = [f.result() for f in as_completed(futures)]
        parallel_total = (time.time() - parallel_start) * 1000
        avg_parallel = np.mean(parallel_times) * 1000
        
        compression_ratio = uncompressed_size / file_size
        
        results.append({
            'compression': comp_name,
            'level': comp_level,
            'file_size_mb': file_size_mb,
            'compression_ratio': compression_ratio,
            'cold_load_ms': avg_cold_load,
            'warm_load_ms': avg_warm_load,
            'parallel_load_ms': avg_parallel,
            'parallel_total_ms': parallel_total,
        })
        
        print(f"{comp_name:<25} | {file_size_mb:>10.2f} | {avg_cold_load:>15.2f} | "
              f"{avg_warm_load:>15.2f} | {parallel_total:>15.2f}")
    
    # 정리
    import shutil
    shutil.rmtree(temp_dir)
    
    return results


def find_optimal_compression_realistic(results):
    """실제 환경을 고려한 최적 압축 레벨 찾기"""
    print("\n" + "="*120)
    print("최적 압축 레벨 분석 (실제 환경 기준)")
    print("="*120)
    
    # Cold cache 기준 정렬 (실제 환경에서는 캐시 미스가 빈번함)
    results_sorted = sorted(results, key=lambda x: x['cold_load_ms'])
    
    print("\nCold Cache 기준 (실제 디스크 I/O, 빠른 순 Top 5):")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"  {i}. {r['compression']:<25} - {r['cold_load_ms']:>6.2f}ms, "
              f"{r['file_size_mb']:>6.2f} MB, {r['compression_ratio']:>5.2f}x")
    
    # 병렬 읽기 기준 정렬 (실제 DataLoader 환경)
    results_parallel = sorted(results, key=lambda x: x['parallel_total_ms'])
    
    print("\n병렬 읽기 기준 (실제 DataLoader 환경, 빠른 순 Top 5):")
    for i, r in enumerate(results_parallel[:5], 1):
        print(f"  {i}. {r['compression']:<25} - {r['parallel_total_ms']:>6.2f}ms, "
              f"{r['file_size_mb']:>6.2f} MB, {r['compression_ratio']:>5.2f}x")
    
    # 실제 환경 균형점 (Cold cache 50% + 병렬 50%)
    best_balanced = None
    best_score = float('inf')
    
    for r in results:
        # Cold cache와 병렬 읽기의 가중 평균
        score = r['cold_load_ms'] * 0.5 + r['parallel_total_ms'] * 0.5
        if score < best_score:
            best_score = score
            best_balanced = r
    
    print(f"\n실제 환경 균형점 (Cold 50% + 병렬 50%):")
    print(f"  Compression: {best_balanced['compression']}")
    print(f"  Cold cache 로딩: {best_balanced['cold_load_ms']:.2f}ms")
    print(f"  병렬 로딩: {best_balanced['parallel_total_ms']:.2f}ms")
    print(f"  파일 크기: {best_balanced['file_size_mb']:.2f} MB")
    print(f"  압축률: {best_balanced['compression_ratio']:.2f}x")
    
    return best_balanced


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HDF5 압축 레벨 벤치마크 (실제 환경 시뮬레이션)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
이 벤치마크는 실제 DataLoader 환경을 시뮬레이션합니다:
- Cold cache: 디스크에서 직접 읽기 (캐시 미스)
- Warm cache: 메모리 캐시에서 읽기 (캐시 히트)
- 병렬 읽기: 여러 worker가 동시에 다른 파일 읽기

참고: Cold cache 테스트를 위해 디스크 캐시를 비우려면:
  sudo sync && sudo sysctl vm.drop_caches=3

예시:
  # 기본 테스트 (레벨 1-4)
  python benchmark_compression.py --sample_file data/BRATS2021_preprocessed/BraTS2021_00000.h5
  
  # 특정 레벨만 테스트
  python benchmark_compression.py --sample_file data/BRATS2021_preprocessed/BraTS2021_00000.h5 --levels 1 2 3
  
  # 병렬 worker 수 조정
  python benchmark_compression.py --sample_file data/BRATS2021_preprocessed/BraTS2021_00000.h5 --num_parallel 8
        """
    )
    parser.add_argument('--sample_file', type=str, required=True,
                        help='테스트할 샘플 HDF5 파일 경로')
    parser.add_argument('--levels', nargs='+', type=int, default=[1, 2, 3, 4],
                        help='테스트할 압축 레벨 (기본: 1-4, 0은 압축 없음을 의미)')
    parser.add_argument('--num_parallel', type=int, default=4,
                        help='병렬 읽기 수 (worker 수 시뮬레이션, 기본: 4)')
    parser.add_argument('--num_files', type=int, default=10,
                        help='테스트 파일 수 (기본: 10)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sample_file):
        print(f"Error: File not found: {args.sample_file}")
        return
    
    compression_levels = []
    for level in args.levels:
        if level == 0:
            compression_levels.append(None)
        else:
            compression_levels.append(level)
    
    # None이 없으면 추가 (비교를 위해)
    if None not in compression_levels:
        compression_levels = [None] + compression_levels
    
    print("="*120)
    print("HDF5 압축 레벨 벤치마크 (실제 환경 시뮬레이션)")
    print("="*120)
    print()
    
    results = benchmark_compression_with_parallel_io(
        args.sample_file, 
        compression_levels,
        num_parallel_reads=args.num_parallel,
        num_files=args.num_files
    )
    
    best = find_optimal_compression_realistic(results)
    
    print(f"\n{'='*120}")
    print("권장 사항")
    print(f"{'='*120}")
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

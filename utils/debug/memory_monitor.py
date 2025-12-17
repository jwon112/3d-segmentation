#!/usr/bin/env python3
"""
메모리 사용 패턴 모니터링 유틸리티

SHM vs Application RAM 사용량을 추적하여
num_workers, max_cache_size, prefetch_factor의 영향을 분석
"""

import os
import psutil
import subprocess
from typing import Dict, Optional
import torch


def get_shm_usage() -> Dict[str, float]:
    """SHM (Shared Memory) 사용량 확인"""
    try:
        # /dev/shm 디렉토리 크기 확인
        result = subprocess.run(
            ['df', '-h', '/dev/shm'],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 4:
                total = parts[1]  # 총 크기
                used = parts[2]   # 사용 중
                available = parts[3]  # 사용 가능
                use_percent = parts[4].rstrip('%') if len(parts) > 4 else '0'
                
                return {
                    'total': total,
                    'used': used,
                    'available': available,
                    'use_percent': float(use_percent)
                }
    except Exception as e:
        pass
    
    return {'total': 'N/A', 'used': 'N/A', 'available': 'N/A', 'use_percent': 0.0}


def get_process_memory(pid: Optional[int] = None) -> Dict[str, float]:
    """프로세스 메모리 사용량 (Application RAM)"""
    if pid is None:
        pid = os.getpid()
    
    try:
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        
        # RSS (Resident Set Size): 실제 물리 메모리 사용량
        rss_gb = mem_info.rss / (1024 ** 3)
        
        # VMS (Virtual Memory Size): 가상 메모리 사용량
        vms_gb = mem_info.vms / (1024 ** 3)
        
        # 메모리 퍼센트
        mem_percent = process.memory_percent()
        
        return {
            'rss_gb': rss_gb,
            'vms_gb': vms_gb,
            'percent': mem_percent,
            'pid': pid
        }
    except Exception as e:
        return {'rss_gb': 0.0, 'vms_gb': 0.0, 'percent': 0.0, 'pid': pid}


def get_system_memory() -> Dict[str, float]:
    """시스템 전체 메모리 사용량"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024 ** 3),
        'available_gb': mem.available / (1024 ** 3),
        'used_gb': mem.used / (1024 ** 3),
        'percent': mem.percent
    }


def get_all_worker_memory() -> Dict[str, Dict[str, float]]:
    """모든 DataLoader worker 프로세스의 메모리 사용량"""
    current_pid = os.getpid()
    parent = psutil.Process(current_pid)
    
    worker_memories = {}
    
    try:
        # 자식 프로세스 찾기 (DataLoader workers)
        children = parent.children(recursive=True)
        
        for child in children:
            try:
                mem_info = child.memory_info()
                rss_gb = mem_info.rss / (1024 ** 3)
                worker_memories[f'worker_{child.pid}'] = {
                    'rss_gb': rss_gb,
                    'pid': child.pid
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        pass
    
    return worker_memories


def print_memory_summary(title: str = "Memory Summary"):
    """메모리 사용량 요약 출력"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # 시스템 메모리
    sys_mem = get_system_memory()
    print(f"\n[System Memory]")
    print(f"  Total: {sys_mem['total_gb']:.2f} GB")
    print(f"  Used: {sys_mem['used_gb']:.2f} GB ({sys_mem['percent']:.1f}%)")
    print(f"  Available: {sys_mem['available_gb']:.2f} GB")
    
    # SHM
    shm = get_shm_usage()
    print(f"\n[Shared Memory (/dev/shm)]")
    print(f"  Total: {shm['total']}")
    print(f"  Used: {shm['used']} ({shm['use_percent']:.1f}%)")
    print(f"  Available: {shm['available']}")
    
    # 메인 프로세스
    main_mem = get_process_memory()
    print(f"\n[Main Process (PID {main_mem['pid']})]")
    print(f"  RSS (Application RAM): {main_mem['rss_gb']:.2f} GB")
    print(f"  VMS: {main_mem['vms_gb']:.2f} GB")
    print(f"  Percent: {main_mem['percent']:.1f}%")
    
    # Worker 프로세스들
    workers = get_all_worker_memory()
    if workers:
        print(f"\n[DataLoader Workers]")
        total_worker_rss = 0.0
        for worker_name, worker_mem in workers.items():
            print(f"  {worker_name} (PID {worker_mem['pid']}): {worker_mem['rss_gb']:.2f} GB")
            total_worker_rss += worker_mem['rss_gb']
        print(f"  Total Workers RSS: {total_worker_rss:.2f} GB")
    
    # Application RAM 추정 (시스템 메모리 - SHM 사용량)
    # 주의: 이것은 근사치입니다
    print(f"\n[Estimated Application RAM Usage]")
    print(f"  Main Process: {main_mem['rss_gb']:.2f} GB")
    if workers:
        print(f"  All Workers: {total_worker_rss:.2f} GB")
        print(f"  Total (Main + Workers): {main_mem['rss_gb'] + total_worker_rss:.2f} GB")
    
    print(f"{'='*60}\n")


def estimate_memory_usage(
    num_workers: int,
    prefetch_factor: int,
    max_cache_size: int,
    batch_size: int = 8,
    volume_size_mb: float = 136.0,  # 전처리된 볼륨 크기 (MB)
) -> Dict[str, float]:
    """
    메모리 사용량 추정
    
    Args:
        num_workers: DataLoader worker 수
        prefetch_factor: 각 worker가 미리 로드하는 배치 수
        max_cache_size: 데이터셋 캐시 크기 (볼륨 수)
        batch_size: 배치 크기
        volume_size_mb: 볼륨 크기 (MB)
    
    Returns:
        추정된 메모리 사용량 (GB)
    """
    # 각 worker의 메모리 사용량 추정
    # 1. 캐시된 볼륨: max_cache_size * volume_size_mb
    cache_memory_mb = max_cache_size * volume_size_mb
    
    # 2. Prefetch된 배치: prefetch_factor * batch_size * volume_size_mb
    prefetch_memory_mb = prefetch_factor * batch_size * volume_size_mb
    
    # 3. Worker 오버헤드 (프로세스, Python 인터프리터 등): 약 500MB
    worker_overhead_mb = 500.0
    
    # 각 worker의 총 메모리
    per_worker_memory_mb = cache_memory_mb + prefetch_memory_mb + worker_overhead_mb
    per_worker_memory_gb = per_worker_memory_mb / 1024.0
    
    # 전체 worker 메모리
    total_workers_memory_gb = num_workers * per_worker_memory_gb
    
    # 메인 프로세스 메모리 (모델, 메인 프로세스 오버헤드 등): 약 2GB
    main_process_memory_gb = 2.0
    
    # 총 Application RAM 사용량
    total_app_ram_gb = main_process_memory_gb + total_workers_memory_gb
    
    return {
        'per_worker_gb': per_worker_memory_gb,
        'total_workers_gb': total_workers_memory_gb,
        'main_process_gb': main_process_memory_gb,
        'total_app_ram_gb': total_app_ram_gb,
        'cache_memory_gb': (cache_memory_mb * num_workers) / 1024.0,
        'prefetch_memory_gb': (prefetch_memory_mb * num_workers) / 1024.0,
    }


def print_memory_estimation(
    num_workers: int,
    prefetch_factor: int,
    max_cache_size: int,
    batch_size: int = 8,
):
    """메모리 사용량 추정 출력"""
    est = estimate_memory_usage(num_workers, prefetch_factor, max_cache_size, batch_size)
    
    print(f"\n{'='*60}")
    print(f"Memory Usage Estimation")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  num_workers: {num_workers}")
    print(f"  prefetch_factor: {prefetch_factor}")
    print(f"  max_cache_size: {max_cache_size}")
    print(f"  batch_size: {batch_size}")
    print(f"\nEstimated Memory Usage (Application RAM):")
    print(f"  Per Worker: {est['per_worker_gb']:.2f} GB")
    print(f"    - Cache: {est['cache_memory_gb'] / num_workers:.2f} GB")
    print(f"    - Prefetch: {est['prefetch_memory_gb'] / num_workers:.2f} GB")
    print(f"    - Overhead: ~0.5 GB")
    print(f"  Total Workers ({num_workers}): {est['total_workers_gb']:.2f} GB")
    print(f"  Main Process: {est['main_process_gb']:.2f} GB")
    print(f"  Total Application RAM: {est['total_app_ram_gb']:.2f} GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 예제: 현재 설정으로 메모리 추정
    print_memory_estimation(
        num_workers=4,
        prefetch_factor=8,
        max_cache_size=50,
        batch_size=8
    )
    
    # 실제 메모리 사용량 확인
    print_memory_summary("Current Memory Usage")


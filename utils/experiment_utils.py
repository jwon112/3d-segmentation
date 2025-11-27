#!/usr/bin/env python3
"""
Experiment Utilities
유틸리티 함수들: distributed training, seed 설정, sliding window, FLOPs 계산, 모델 팩토리
"""

import os
import copy
import sys
import torch
import numpy as np
import random
import math
from datetime import timedelta
from typing import Dict, Optional, Tuple

from models.architecture.cascade.roi_model import (
    build_roi_mobileunetr3d,
    build_roi_unet3d_small,
)

# Global input size configuration
# 2D models use 256x256 for stable down/upsampling (avoid odd sizes)
INPUT_SIZE_2D = (256, 256)
# 3D models keep dataset-native depth; adjust if needed
INPUT_SIZE_3D = (240, 240, 155)

# Distributed training helpers
def setup_distributed():
    import torch.distributed as dist
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # NCCL timeout 설정 (기본 10분 -> 30분으로 증가)
        timeout = os.environ.get('NCCL_TIMEOUT', '1800')  # 30분 (초 단위)
        dist.init_process_group(
            backend='nccl', 
            init_method='env://',
            timeout=timedelta(seconds=int(timeout))
        )
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1

def cleanup_distributed():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int):
    return rank == 0

def set_seed(seed):
    """랜덤 시드 설정 (완전한 재현성을 위한 전역 시드 고정)"""
    # Python 내장 random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch CUDA (모든 GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN 결정성 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # PyTorch deterministic 알고리즘 강제 (선택적)
    # 주의: 완전한 재현성을 위해 필요하지만, 성능 저하가 있을 수 있음 (약 10-20%)
    # 대부분의 연구에서는 이 설정 없이도 충분히 재현 가능한 결과를 얻음
    # 필요시 주석 해제: torch.use_deterministic_algorithms(True, warn_only=True)
    # try:
    #     torch.use_deterministic_algorithms(True, warn_only=True)
    # except Exception:
    #     # PyTorch 버전이 낮거나 지원하지 않는 경우 무시
    #     pass
    # Python hash seed (딕셔너리 순서 등에 영향)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _make_blend_weights_3d(patch_size):
    """Create separable cosine blending weights for 3D patches (H,W,D layout).
    Returns tensor of shape (1, 1, H, W, D).
    """
    ph, pw, pd = patch_size
    def one_dim(n):
        t = torch.arange(n, dtype=torch.float32)
        # raised cosine from 0..pi
        w = 0.5 * (1 - torch.cos(math.pi * (t + 0.5) / n))
        # avoid zeros on borders completely
        return w.clamp_min(1e-4)
    wh = one_dim(ph)
    ww = one_dim(pw)
    wd = one_dim(pd)
    w3 = wh.view(ph, 1, 1) * ww.view(1, pw, 1) * wd.view(1, 1, pd)
    w3 = w3 / w3.max()
    return w3.view(1, 1, ph, pw, pd)


def sliding_window_inference_3d(model, volume, patch_size=(128, 128, 128), overlap=0.5, device='cuda', model_name='mobile_unetr_3d'):
    """Naive sliding-window inference for tensors shaped (1, C, H, W, D).
    Aggregates logits with cosine blending. Returns logits of shape (1, C_out, H, W, D).
    """
    assert volume.dim() == 5 and volume.size(0) == 1, "Expect (1, C, H, W, D)"
    _, C, H, W, D = volume.shape
    ph, pw, pd = patch_size

    # strides
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))
    sd = max(1, int(pd * (1 - overlap)))

    # prepare accumulators
    with torch.no_grad():
        # run one pass to get out channels
        # crop a minimal patch within bounds
        y0 = min(0, H - ph)
        x0 = min(0, W - pw)
        z0 = min(0, D - pd)
        patch = volume[:, :, 0:ph, 0:pw, 0:pd].to(device)
        logits0 = model(patch)
        C_out = logits0.size(1)

    out = torch.zeros((1, C_out, H, W, D), dtype=torch.float32, device=device)
    wsum = torch.zeros((1, 1, H, W, D), dtype=torch.float32, device=device)
    wpatch = _make_blend_weights_3d(patch_size).to(device)

    def ranges(L, p, s):
        if L <= p:
            return [0]
        xs = list(range(0, L - p + 1, s))
        if xs[-1] != L - p:
            xs.append(L - p)
        return xs

    hs = ranges(H, ph, sh)
    ws = ranges(W, pw, sw)
    ds = ranges(D, pd, sd)

    model.eval()
    with torch.no_grad():
        for hy in hs:
            for wx in ws:
                for dz in ds:
                    patch = volume[:, :, hy:hy+ph, wx:wx+pw, dz:dz+pd].to(device)
                    logits = model(patch)
                    out[:, :, hy:hy+ph, wx:wx+pw, dz:dz+pd] += logits * wpatch
                    wsum[:, :, hy:hy+ph, wx:wx+pw, dz:dz+pd] += wpatch

    out = out / wsum.clamp_min(1e-6)
    return out

def calculate_flops(model, input_size=(1, 4, 64, 64, 64)):
    """모델의 FLOPs 계산"""
    try:
        from thop import profile
        # unwrap DDP if needed
        real_model = model.module if hasattr(model, 'module') else model
        # Profile on a deepcopy to avoid mutating the live model with thop buffers
        device = next(real_model.parameters()).device
        model_copy = copy.deepcopy(real_model).to(device).eval()
        dummy_input = torch.randn(input_size).to(device)
        flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
        return flops
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return 0
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return 0

def calculate_pam(model, input_size=(1, 4, 64, 64, 64), mode='inference', stage_wise=True, device='cuda', num_runs=10):
    """
    Peak Activation Memory (PAM) 계산
    
    Args:
        model: PyTorch 모델 (DDP wrapped 가능)
        input_size: 입력 텐서 크기 (batch_size=1로 고정)
        mode: 'train' 또는 'inference'
        stage_wise: True면 stage별 PAM도 측정 (기본값: True)
        device: 디바이스 ('cuda' 또는 'cpu')
        num_runs: 측정 반복 횟수 (기본 5회, 변동성 감소를 위해)
    
    Returns:
        - 전체 PAM (bytes, batch_size=1 기준) - 여러 번 측정한 값들의 리스트
        - stage별 PAM dict (stage_wise=True일 때)
            - key: stage 이름 (예: 'stem_flair', 'branch_flair', 'down3', 'up1', 'outc')
            - value: 해당 stage의 PAM 리스트 (bytes, num_runs만큼)
    
    Note:
        - batch_size=1로 고정하여 측정 (모든 모델을 동일한 조건에서 비교)
        - Train 모드에서는 backward pass까지 포함하여 측정
        - Inference 모드에서는 forward pass만 측정
        - 여러 번 측정하여 변동성을 줄이고, model_comparison에서 평균/표준편차 계산
        - Stage별 측정은 forward hook을 사용하여 각 stage의 출력 텐서 메모리 측정
    """
    if not torch.cuda.is_available() or device == 'cpu':
        return [], {}
    
    try:
        # unwrap DDP if needed
        real_model = model.module if hasattr(model, 'module') else model
        device_obj = torch.device(device)
        
        # 입력 seed 고정 (재현성 향상)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Warmup pass (CUDA 커널 초기화)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)
        dummy_input_warmup = torch.randn(input_size, dtype=torch.float32).to(device_obj)
        if mode == 'train':
            real_model.train()
            with torch.enable_grad():
                output_warmup = real_model(dummy_input_warmup)
                if isinstance(output_warmup, tuple):
                    loss_warmup = output_warmup[0].sum()
                else:
                    loss_warmup = output_warmup.sum()
                loss_warmup.backward()
                real_model.zero_grad()
        else:
            real_model.eval()
            with torch.no_grad():
                _ = real_model(dummy_input_warmup)
        
        # 메모리 정리 후 측정 시작
        del dummy_input_warmup
        if mode == 'train':
            del output_warmup, loss_warmup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)
        
        # Stage별 측정을 위한 stage 감지 및 hook 준비
        stage_modules = {}
        if stage_wise:
            # Stage 패턴: stem_*, branch_*, down*, up*, outc, cross_attn
            stage_patterns = ['stem_', 'branch_', 'down', 'up', 'outc', 'cross_attn']
            
            for name, module in real_model.named_modules():
                # Stage 패턴이 포함된 모듈 찾기
                if any(pattern in name for pattern in stage_patterns):
                    # 최상위 레벨의 stage만 선택 (중첩된 모듈 제외)
                    # 예: 'stem_flair'는 선택, 'stem_flair.conv1'은 제외
                    parts = name.split('.')
                    if len(parts) <= 2:  # 최상위 또는 한 단계 아래만
                        # 부모가 stage가 아닌 경우만 선택
                        is_top_level = True
                        if len(parts) == 2:
                            parent_name = parts[0]
                            if any(p in parent_name for p in stage_patterns):
                                is_top_level = False
                        
                        if is_top_level:
                            stage_modules[name] = module
        
        # 여러 번 측정하여 변동성 감소
        peak_memories = []
        stage_memories_dict = {name: [] for name in stage_modules.keys()} if stage_wise else {}
        
        for run_idx in range(num_runs):
            # 각 측정마다 메모리 초기화
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_obj)
            
            # Stage별 메모리 초기화
            if stage_wise:
                for stage_name in stage_memories_dict.keys():
                    stage_memories_dict[stage_name] = []
            
            # Forward hook 등록 (stage별 메모리 측정)
            hooks = []
            if stage_wise and stage_modules:
                def make_stage_hook(stage_name):
                    def hook_fn(module, input, output):
                        # Output 텐서의 메모리 크기 계산
                        if isinstance(output, torch.Tensor):
                            output_mem = output.element_size() * output.nelement()
                        elif isinstance(output, tuple):
                            output_mem = sum(
                                t.element_size() * t.nelement() 
                                if isinstance(t, torch.Tensor) else 0 
                                for t in output
                            )
                        else:
                            output_mem = 0
                        
                        stage_memories_dict[stage_name].append(output_mem)
                    return hook_fn
                
                for name, module in stage_modules.items():
                    hook = module.register_forward_hook(make_stage_hook(name))
                    hooks.append(hook)
            
            # Dummy input 생성 (동일한 seed로 인해 동일한 값)
            dummy_input = torch.randn(input_size, dtype=torch.float32).to(device_obj)
            
            # 모드에 따라 forward/backward 실행
            if mode == 'train':
                real_model.train()
                # Forward pass
                output = real_model(dummy_input)
                # Dummy loss for backward
                if isinstance(output, tuple):
                    loss = output[0].sum()
                else:
                    loss = output.sum()
                # Backward pass
                loss.backward()
            else:  # inference
                real_model.eval()
                with torch.no_grad():
                    output = real_model(dummy_input)
            
            # 동기화 후 peak memory 측정
            torch.cuda.synchronize(device_obj)
            peak_memory_bytes = torch.cuda.max_memory_allocated(device_obj)
            peak_memories.append(peak_memory_bytes)
            
            # Hook 제거
            for hook in hooks:
                hook.remove()
            
            # Gradient 정리 (train 모드인 경우)
            if mode == 'train':
                real_model.zero_grad()
            
            # 메모리 정리
            del dummy_input, output
            if mode == 'train':
                del loss
            torch.cuda.empty_cache()
        
        # Stage별 메모리 결과 정리 (각 run별로 리스트로 저장)
        stage_memories = {}
        if stage_wise and stage_memories_dict:
            # 각 stage별로 num_runs만큼의 측정값이 있어야 함
            for stage_name, mem_list in stage_memories_dict.items():
                if len(mem_list) == num_runs:
                    stage_memories[stage_name] = mem_list
                elif len(mem_list) > 0:
                    # 일부만 측정된 경우 (예: 조건부 실행 등)
                    stage_memories[stage_name] = mem_list
        
        return peak_memories, stage_memories
    
    except Exception as e:
        print(f"Error calculating PAM: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

def calculate_inference_latency(model, input_size=(1, 4, 64, 64, 64), device='cuda', num_warmup=10, num_runs=100):
    """
    Inference Latency 계산 (ms 단위)
    
    Args:
        model: PyTorch 모델 (DDP wrapped 가능)
        input_size: 입력 텐서 크기 (batch_size=1로 고정)
        device: 디바이스 ('cuda'만 지원)
        num_warmup: Warmup 실행 횟수 (CUDA 커널 초기화, 기본 10회)
        num_runs: 측정 반복 횟수 (기본 100회, 통계적 신뢰성 확보)
    
    Returns:
        - Latency 리스트 (ms 단위) - 여러 번 측정한 값들
        - 통계 정보 dict (mean, std, min, max, median, p50, p95, p99)
    
    Note:
        - batch_size=1로 고정하여 측정 (모든 모델을 동일한 조건에서 비교)
        - CUDA의 경우 torch.cuda.Event를 사용하여 정확한 측정
        - PAM과 동일하게 dummy input 사용 (torch.randn)
        - 여러 번 측정하여 변동성을 줄이고, model_comparison에서 평균/표준편차 계산
    """
    # device가 torch.device 객체일 수 있으므로 type을 확인
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if not torch.cuda.is_available() or device_type != 'cuda':
        print(f"Warning: CUDA not available or device != 'cuda' (got {device_type}), skipping latency measurement")
        return [], {}
    
    try:
        # unwrap DDP if needed
        real_model = model.module if hasattr(model, 'module') else model
        # device가 이미 torch.device 객체면 그대로 사용, 아니면 변환
        device_obj = device if isinstance(device, torch.device) else torch.device(device)
        
        # 모델을 eval 모드로 설정
        real_model.eval()
        
        # 입력 seed 고정 (재현성, PAM과 동일)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Warmup (CUDA 커널 초기화, PAM과 동일한 방식)
        torch.cuda.empty_cache()
        dummy_input_warmup = torch.randn(input_size, dtype=torch.float32).to(device_obj)
        
        # Warmup 중 모델이 제대로 실행되는지 확인
        with torch.no_grad():
            try:
                output_warmup = real_model(dummy_input_warmup)
                if output_warmup is None:
                    print(f"Warning: Model returned None during warmup, input_size={input_size}")
                    return [], {}
            except Exception as e:
                print(f"Error during warmup: {e}, input_size={input_size}")
                import traceback
                traceback.print_exc()
                return [], {}
            
            for _ in range(num_warmup - 1):  # 첫 번째는 이미 실행했으므로
                _ = real_model(dummy_input_warmup)
        
        # CUDA 동기화 (warmup 완료)
        torch.cuda.synchronize(device_obj)
        
        del dummy_input_warmup
        torch.cuda.empty_cache()
        
        # 실제 측정 (각 측정마다 새로운 입력 생성하여 변동성 반영)
        import time
        latencies = []
        for run_idx in range(num_runs):
            # Dummy input 생성 (각 측정마다 다른 값, 실제 inference 변동성 반영)
            # Seed는 warmup에서만 고정하고, 실제 측정에서는 변동성 허용
            dummy_input = torch.randn(input_size, dtype=torch.float32).to(device_obj)
            
            # CUDA Event를 사용한 정확한 시간 측정
            # 동기화 후 Event 생성 (더 정확한 측정을 위해)
            torch.cuda.synchronize(device_obj)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Event 기록 시작
            start_event.record()
            with torch.no_grad():
                _ = real_model(dummy_input)
            end_event.record()
            
            # Event가 완료될 때까지 대기 (중요!)
            torch.cuda.synchronize(device_obj)
            
            # elapsed_time은 end_event가 완료된 후에만 호출해야 함
            latency_ms = start_event.elapsed_time(end_event)  # milliseconds
            
            # 측정값이 유효한지 확인 (0이거나 음수면 문제)
            # torch.cuda.Event가 0을 반환하는 경우가 있으므로, 대안으로 time.perf_counter() 사용
            if latency_ms <= 0 or latency_ms > 100000:  # 비정상적으로 큰 값도 체크
                # 대안: time.perf_counter() 사용 (CUDA 동기화와 함께)
                torch.cuda.synchronize(device_obj)
                start_time = time.perf_counter()
                with torch.no_grad():
                    _ = real_model(dummy_input)
                torch.cuda.synchronize(device_obj)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000  # ms로 변환
                
                # 여전히 0이면 경고 출력 (디버깅용)
                if latency_ms <= 0:
                    print(f"Warning: Latency measurement still 0 at run {run_idx}, input_size={input_size}")
            
            latencies.append(latency_ms)
            
            # 메모리 정리
            del dummy_input
            torch.cuda.empty_cache()
        
        # 통계 정보 계산
        latencies_array = np.array(latencies)
        stats = {
            'mean': float(np.mean(latencies_array)),
            'std': float(np.std(latencies_array)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'median': float(np.median(latencies_array)),
            'p50': float(np.percentile(latencies_array, 50)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99))
        }
        
        return latencies, stats
    
    except Exception as e:
        print(f"Error calculating inference latency: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

def get_model(model_name, n_channels=4, n_classes=4, dim='3d', patch_size=None, use_pretrained=False, norm: str = 'bn'):
    """모델 생성 함수
    
    Args:
        model_name: 모델 이름 (예: 'dualbranch_01_unet_s', 'unet3d_m', 'dualbranch_01_unet_xs')
        n_channels: 입력 채널 수
        n_classes: 출력 클래스 수
        dim: '2d' 또는 '3d'
        patch_size: 하이퍼파라미터 (None이면 모델별 기본값 사용)
        use_pretrained: pretrained 가중치 사용 여부 (MobileUNETR의 경우)
    
    Raises:
        ValueError: 모델 이름이 알려지지 않았거나 비어있는 경우, 또는 파라미터가 유효하지 않은 경우
        ImportError: 모델 모듈을 import할 수 없는 경우
        RuntimeError: 모델 생성 중 오류가 발생한 경우
    """
    # 파라미터 검증
    if not model_name:
        raise ValueError("model_name cannot be empty")
    if not isinstance(model_name, str):
        raise ValueError(f"model_name must be a string, got {type(model_name)}")
    if n_channels <= 0:
        raise ValueError(f"n_channels must be positive, got {n_channels}")
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}")
    if dim not in ['2d', '3d']:
        raise ValueError(f"dim must be '2d' or '3d', got '{dim}'")
    if norm not in ['bn', 'in', 'gn', 'ln']:
        raise ValueError(f"norm must be one of ['bn', 'in', 'gn', 'ln'], got '{norm}'")
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from models.channel_configs import parse_model_size
    except ImportError as e:
        raise ImportError(f"Failed to import parse_model_size: {e}")
    
    # 지원하는 모델 목록 정의 (패턴 기반)
    SUPPORTED_MODEL_PATTERNS = [
        'unet3d_', 'unet3d_stride_', 'unetr', 'swin_unetr', 'mobile_unetr', 'mobile_unetr_3d',
        'dualbranch_01_unet_', 'dualbranch_02_unet_', 'dualbranch_03_unet_', 'dualbranch_04_unet_',
        'dualbranch_05_unet_', 'dualbranch_06_unet_', 'dualbranch_07_unet_', 'dualbranch_10_unet_',
        'dualbranch_11_unet_', 'dualbranch_13_unet_', 'dualbranch_14_',
        'dualbranch_16_shufflenet_hybrid_', 'dualbranch_mobilenetv2_dilated_',
        'dualbranch_mobilenetv2_dilated_fixed_',
        'dualbranch_16_shufflenet_hybrid_ln_', 'dualbranch_17_shufflenet_pamlite_',
        'dualbranch_17_shufflenet_pamlite_v3_', 'dualbranch_18_shufflenet_v1_',
        'dualbranch_18_shufflenet_v1_stage3fused_',
        'dualbranch_19_shufflenet_v2_stage3fused_',
        'quadbranch_unet_', 'quadbranch_channel_centralized_concat_',
        'quadbranch_channel_distributed_concat_', 'quadbranch_channel_distributed_conv_',
        'quadbranch_spatial_centralized_concat_', 'quadbranch_spatial_distributed_concat_',
        'quadbranch_spatial_distributed_conv_',
        # 특정 모델 이름 (패턴이 아닌 정확한 이름)
        'unet3d_2modal_s', 'unet3d_4modal_s', 'dualbranch_2modal_unet_s',
        'quadbranch_4modal_unet_s', 'quadbranch_4modal_attention_unet_s'
    ]
    
    # 모델 이름 검증
    is_supported = False
    for pattern in SUPPORTED_MODEL_PATTERNS:
        if model_name.startswith(pattern) or model_name == pattern:
            is_supported = True
            break
    
    if not is_supported:
        raise ValueError(
            f"Unknown model: '{model_name}'\n"
            f"Supported model patterns: {', '.join(SUPPORTED_MODEL_PATTERNS)}\n"
            f"Note: Most models support size suffixes: _xs, _s, _m, _l (e.g., 'dualbranch_18_shufflenet_v1_s')"
        )
    
    # Helper function for consistent error handling
    def _create_model_with_error_handling(model_name, create_func, *args, **kwargs):
        """모델 생성 시 일관된 에러 처리"""
        try:
            return create_func(*args, **kwargs)
        except ImportError as e:
            raise ImportError(
                f"Failed to import model class for '{model_name}': {e}\n"
                f"Please check if the required model module exists and is properly configured."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create model '{model_name}': {e}\n"
                f"Please check model configuration and parameters."
            )
    
    # Import models (필요한 경우에만, lazy import)
    UNETR_Simplified = None
    SwinUNETR_Simplified = None
    MobileUNETR = None
    MobileUNETR_3D_Wrapper = None
    
    # 2D 입력인 경우 3D로 확장 (unsqueeze depth dimension)
    if model_name.startswith('unet3d_') and not model_name.startswith('unet3d_stride_'):
        if dim == '2d':
            # 2D 데이터는 depth 차원 추가가 필요
            pass
        try:
            base_name, size = parse_model_size(model_name)
        except Exception as e:
            raise ValueError(f"Failed to parse model size from '{model_name}': {e}")
        
        def _create_unet3d():
            from models.model_3d_unet import UNet3D
            return UNet3D(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size)
        return _create_model_with_error_handling(model_name, _create_unet3d)
    elif model_name.startswith('unet3d_stride_'):
        # UNet3D variant with stride-2 conv downsampling - Support xs, s, m, l sizes
        try:
            base_name, size = parse_model_size(model_name)
        except Exception as e:
            raise ValueError(f"Failed to parse model size from '{model_name}': {e}")
        
        def _create_unet3d_stride():
            from models.model_3d_unet_stride import UNet3D_Stride
            return UNet3D_Stride(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size)
        return _create_model_with_error_handling(model_name, _create_unet3d_stride)
    elif model_name == 'unetr':
        # dim에 따라 img_size 설정
        if dim == '2d':
            img_size = INPUT_SIZE_2D
            # patch_size가 지정되지 않으면 기본값 사용 (2D)
            if patch_size is None:
                patch_size = (16, 16)  # 논문 권장값 (16, 16, 16)의 2D 버전
        else:
            img_size = (240, 240, 155)
            # patch_size가 지정되지 않으면 기본값 사용 (3D)
            if patch_size is None:
                patch_size = (16, 16, 16)  # 논문 권장값
        
        def _create_unetr():
            from models import UNETR_Simplified
            return UNETR_Simplified(
                img_size=img_size, 
                patch_size=patch_size,
                in_channels=n_channels, 
                out_channels=n_classes
            )
        return _create_model_with_error_handling(model_name, _create_unetr)
    elif model_name == 'swin_unetr':
        # dim에 따라 img_size 설정
        if dim == '2d':
            img_size = INPUT_SIZE_2D
            # patch_size가 지정되지 않으면 기본값 사용 (2D)
            if patch_size is None:
                patch_size = (4, 4)
        else:
            img_size = (240, 240, 155)
            # patch_size가 지정되지 않으면 기본값 사용 (3D)
            if patch_size is None:
                patch_size = (4, 4, 4)
        
        def _create_swin_unetr():
            from models import SwinUNETR_Simplified
            return SwinUNETR_Simplified(
                img_size=img_size, 
                patch_size=patch_size,
                in_channels=n_channels, 
                out_channels=n_classes
            )
        return _create_model_with_error_handling(model_name, _create_swin_unetr)
    elif model_name == 'mobile_unetr':
        # MobileUNETR는 2D 전용 모델
        if dim != '2d':
            raise ValueError("MobileUNETR is only supported for 2D data (dim='2d')")
        
        # img_size 설정 (2D만) - 전역 설정 사용
        img_size = INPUT_SIZE_2D
        if patch_size is None:
            patch_size = (16, 16)  # 2D에서 권장값
        
        def _create_mobile_unetr():
            from models import MobileUNETR
            return MobileUNETR(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=n_channels,
                out_channels=n_classes,
                use_pretrained=use_pretrained
            )
        return _create_model_with_error_handling(model_name, _create_mobile_unetr)
    elif model_name == 'mobile_unetr_3d':
        # MobileUNETR 3D 모델
        if dim != '3d':
            raise ValueError("MobileUNETR 3D is only supported for 3D data (dim='3d')")
        
        # img_size 설정 (3D)
        img_size = INPUT_SIZE_3D
        if patch_size is None:
            patch_size = (2, 2, 2)  # 3D에서 권장값
        
        def _create_mobile_unetr_3d():
            from models.mobileunetr_3d import MobileUNETR_3D_Wrapper
            return MobileUNETR_3D_Wrapper(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=n_channels,
                out_channels=n_classes
            )
        return _create_model_with_error_handling(model_name, _create_mobile_unetr_3d)
    elif model_name.startswith('dualbranch_01_unet_'):
        # Dual-branch 3D UNet (v0.1) - Support xs, s, m, l sizes
        try:
            base_name, size = parse_model_size(model_name)
        except Exception as e:
            raise ValueError(f"Failed to parse model size from '{model_name}': {e}")
        
        def _create_dualbranch_01():
            from models.dualbranch_basic import DualBranchUNet3D
            return DualBranchUNet3D(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
        return _create_model_with_error_handling(model_name, _create_dualbranch_01)
    elif model_name.startswith('dualbranch_02_unet_'):
        # Dual-branch 3D UNet (v0.2) - stride-2 convolutional downsampling - Support xs, s, m, l sizes
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_basic import DualBranchUNet3D_Stride
        return DualBranchUNet3D_Stride(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    elif model_name.startswith('dualbranch_03_unet_'):
        # Dual-branch 3D UNet (v0.3) - dilated conv for FLAIR branch - Support xs, s, m, l sizes
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_basic import DualBranchUNet3D_StrideDilated
        return DualBranchUNet3D_StrideDilated(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    elif model_name.startswith('dualbranch_04_unet_'):
        # Dual-branch UNet with RepLK (13x13x13) for FLAIR branch - Support xs, s, m, l sizes
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_replk import DualBranchUNet3D_StrideLK
        return DualBranchUNet3D_StrideLK(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    elif model_name.startswith('dualbranch_05_unet_'):
        # Dual-branch UNet with RepLK + FFN2 (expansion_ratio=2) - Support xs, s, m, l sizes
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_replk import DualBranchUNet3D_StrideLK_FFN2
        return DualBranchUNet3D_StrideLK_FFN2(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    elif model_name.startswith('dualbranch_06_unet_'):
        # Dual-branch UNet with RepLK + FFN2 + MViT Stage 4,5 - Support xs, s, m, l sizes
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_replk import DualBranchUNet3D_StrideLK_FFN2_MViT
        return DualBranchUNet3D_StrideLK_FFN2_MViT(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    elif model_name.startswith('dualbranch_07_unet_'):
        # Dual-branch UNet with RepLK + FFN2 + MViT Stage 5만 - Support xs, s, m, l sizes
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_replk import DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5
        return DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    elif model_name.startswith('dualbranch_13_unet_'):
        # Dual-branch UNet with MobileViT extended to FLAIR branch Stage 3,4 + MViT Stage5 - Support xs, s, m, l sizes
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_mvit import DualBranchUNet3D_MViT_Extended
        return DualBranchUNet3D_MViT_Extended(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    # PAM comparison experiments - different backbones (dualbranch_14)
    elif model_name.startswith('dualbranch_14_'):
        # Extract backbone type and size: dualbranch_14_mobilenetv2_expand2_s -> mobilenetv2_expand2, s
        parts = model_name.split('_', 2)  # ['dualbranch', '14', 'mobilenetv2_expand2_s']
        if len(parts) >= 3:
            backbone_and_size = parts[2]  # 'mobilenetv2_expand2_s'
            # Extract size suffix
            if backbone_and_size.endswith('_xs'):
                backbone = backbone_and_size[:-3]
                size = 'xs'
            elif backbone_and_size.endswith('_s'):
                backbone = backbone_and_size[:-2]
                size = 's'
            elif backbone_and_size.endswith('_m'):
                backbone = backbone_and_size[:-2]
                size = 'm'
            elif backbone_and_size.endswith('_l'):
                backbone = backbone_and_size[:-2]
                size = 'l'
            else:
                raise ValueError(f"Invalid model name: {model_name}")
            
            # Map backbone to class name
            backbone_map = {
                'mobilenetv2_expand2': 'DualBranchUNet3D_MobileNetV2_Expand2',
                'ghostnet': 'DualBranchUNet3D_GhostNet',
                'dilated': 'DualBranchUNet3D_Dilated',
                'convnext': 'DualBranchUNet3D_ConvNeXt',
                'shufflenetv2': 'DualBranchUNet3D_ShuffleNetV2',
                'shufflenetv2_crossattn': 'DualBranchUNet3D_ShuffleNetV2_CrossAttn',
                'shufflenetv2_dilated': 'DualBranchUNet3D_ShuffleNetV2_Dilated',
                'shufflenetv2_lk': 'DualBranchUNet3D_ShuffleNetV2_LK',
            }
            
            if backbone in backbone_map:
                class_name = backbone_map[backbone]
                from models import dualbranch_14_unet
                model_class = getattr(dualbranch_14_unet, class_name)
                return model_class(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
            else:
                raise ValueError(f"Unknown backbone in dualbranch_14: {backbone}")
    elif model_name.startswith('dualbranch_mobilenetv2_dilated_fixed_'):
        # Stage3-fused MobileNetV2 dual-branch UNet with fixed decoder channels
        size = model_name.split('dualbranch_mobilenetv2_dilated_fixed_')[-1]
        from models.dualbranch_mobile import DualBranchUNet3D_MobileNetV2
        return DualBranchUNet3D_MobileNetV2(
            n_channels=n_channels,
            n_classes=n_classes,
            norm=norm,
            size=size,
            fixed_decoder=True,
        )
    elif model_name.startswith('dualbranch_mobilenetv2_dilated_'):
        # Stage3-fused MobileNetV2 dual-branch UNet (standard decoder)
        size = model_name.split('dualbranch_mobilenetv2_dilated_')[-1]
        from models.dualbranch_mobile import DualBranchUNet3D_MobileNetV2
        return DualBranchUNet3D_MobileNetV2(
            n_channels=n_channels,
            n_classes=n_classes,
            norm=norm,
            size=size,
            fixed_decoder=False,
        )
    elif model_name.startswith('dualbranch_16_shufflenet_hybrid_'):
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_16_unet import DualBranchUNet3D_ShuffleHybrid
        return DualBranchUNet3D_ShuffleHybrid(
            n_channels=n_channels,
            n_classes=n_classes,
            norm=norm,
            size=size,
            log_hybrid_stats=True,
        )
    elif model_name.startswith('dualbranch_18_shufflenet_v1_stage3fused_'):
        # Dual-branch UNet with ShuffleNet V1 - Stage 3 fused at down4 (4-stage structure) - Support xs, s, m, l sizes
        # Support variants: _fixed_decoder_*, _half_decoder_*, or default
        try:
            # Check for fixed_decoder or half_decoder suffix
            fixed_decoder = False
            half_decoder = False
            
            if '_fixed_decoder_' in model_name:
                fixed_decoder = True
                # Extract size after _fixed_decoder_
                size_part = model_name.split('_fixed_decoder_')[-1]
                size = size_part.split('_')[0] if '_' in size_part else size_part
            elif '_half_decoder_' in model_name:
                half_decoder = True
                # Extract size after _half_decoder_
                size_part = model_name.split('_half_decoder_')[-1]
                size = size_part.split('_')[0] if '_' in size_part else size_part
            else:
                # Default: extract size normally
                base_name, size = parse_model_size(model_name)
        except Exception as e:
            raise ValueError(f"Failed to parse model size from '{model_name}': {e}")
        
        def _create_shufflenet_v1_stage3fused():
            from models.dualbranch_shufflenet import DualBranchUNet3D_ShuffleNetV1_Stage3Fused
            return DualBranchUNet3D_ShuffleNetV1_Stage3Fused(
                n_channels=n_channels, 
                n_classes=n_classes, 
                norm=norm, 
                size=size,
                fixed_decoder=fixed_decoder,
                half_decoder=half_decoder
            )
        return _create_model_with_error_handling(model_name, _create_shufflenet_v1_stage3fused)
    elif model_name.startswith('dualbranch_18_shufflenet_v1_'):
        # Dual-branch UNet with ShuffleNet V1 + SE blocks - Support xs, s, m, l sizes
        try:
            base_name, size = parse_model_size(model_name)
        except Exception as e:
            raise ValueError(f"Failed to parse model size from '{model_name}': {e}")
        
        def _create_shufflenet_v1():
            from models.dualbranch_shufflenet import DualBranchUNet3D_ShuffleNetV1
            return DualBranchUNet3D_ShuffleNetV1(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
        return _create_model_with_error_handling(model_name, _create_shufflenet_v1)
    elif model_name.startswith('dualbranch_19_shufflenet_v2_stage3fused_'):
        # Dual-branch UNet with ShuffleNet V2 - Stage 3 fused at down4 (4-stage structure) - Support xs, s, m, l sizes
        # Support variants: _fixed_decoder_*, _half_decoder_*, or default
        try:
            # Check for fixed_decoder or half_decoder suffix
            fixed_decoder = False
            half_decoder = False
            
            if '_fixed_decoder_' in model_name:
                fixed_decoder = True
                # Extract size after _fixed_decoder_
                size_part = model_name.split('_fixed_decoder_')[-1]
                size = size_part.split('_')[0] if '_' in size_part else size_part
            elif '_half_decoder_' in model_name:
                half_decoder = True
                # Extract size after _half_decoder_
                size_part = model_name.split('_half_decoder_')[-1]
                size = size_part.split('_')[0] if '_' in size_part else size_part
            else:
                # Default: extract size normally
                base_name, size = parse_model_size(model_name)
        except Exception as e:
            raise ValueError(f"Failed to parse model size from '{model_name}': {e}")
        
        def _create_shufflenet_v2_stage3fused():
            from models.dualbranch_shufflenet_v2 import DualBranchUNet3D_ShuffleNetV2_Stage3Fused
            return DualBranchUNet3D_ShuffleNetV2_Stage3Fused(
                n_channels=n_channels, 
                n_classes=n_classes, 
                norm=norm, 
                size=size,
                fixed_decoder=fixed_decoder,
                half_decoder=half_decoder
            )
        return _create_model_with_error_handling(model_name, _create_shufflenet_v2_stage3fused)
    elif model_name.startswith('dualbranch_16_shufflenet_hybrid_ln_'):
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_16_unet import DualBranchUNet3D_ShuffleHybrid_AllLN
        return DualBranchUNet3D_ShuffleHybrid_AllLN(
            n_channels=n_channels,
            n_classes=n_classes,
            norm=norm,
            size=size,
            log_hybrid_stats=True,
        )
    elif model_name.startswith('dualbranch_17_shufflenet_pamlite_'):
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_17_unet import DualBranchUNet3D_ShufflePamLite
        return DualBranchUNet3D_ShufflePamLite(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    elif model_name.startswith('dualbranch_17_shufflenet_pamlite_v3_'):
        base_name, size = parse_model_size(model_name)
        from models.dualbranch_17_unet import DualBranchUNet3D_ShufflePamLiteV3
        return DualBranchUNet3D_ShufflePamLiteV3(n_channels=n_channels, n_classes=n_classes, norm=norm, size=size)
    # 모달리티 비교 실험 모델들
    elif model_name == 'unet3d_2modal_s':
        # 단일 분기, 2채널 (t1ce, flair) concat
        from models.model_3d_unet_modal_comparison import UNet3D_2Modal_Small
        return UNet3D_2Modal_Small(n_classes=n_classes, norm=norm)
    elif model_name == 'unet3d_4modal_s':
        # 단일 분기, 4채널 (t1, t1ce, t2, flair) concat
        from models.model_3d_unet_modal_comparison import UNet3D_4Modal_Small
        return UNet3D_4Modal_Small(n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_2modal_unet_s':
        # 2개 분기 (t1ce, flair) - dualbranch_01_unet_s와 동일 (MaxPool 기반)
        from models.dualbranch_basic import DualBranchUNet3D
        return DualBranchUNet3D(n_channels=n_channels, n_classes=n_classes, norm=norm, size='s')
    elif model_name == 'quadbranch_4modal_unet_s':
        # 4개 분기 (t1, t1ce, t2, flair) - 어텐션 없음
        from models.model_3d_unet_modal_comparison import QuadBranchUNet3D_4Modal_Small
        return QuadBranchUNet3D_4Modal_Small(n_classes=n_classes, norm=norm)
    elif model_name == 'quadbranch_4modal_attention_unet_s':
        # 4개 분기 (t1, t1ce, t2, flair) - 채널 어텐션 포함
        from models.model_3d_unet_modal_comparison import QuadBranchUNet3D_4Modal_Attention_Small
        return QuadBranchUNet3D_4Modal_Attention_Small(n_classes=n_classes, norm=norm)
    # Quad-Branch UNet models (new implementation)
    elif model_name.startswith('quadbranch_unet_'):
        # 기본 Quad-Branch UNet (attention 없음)
        base_name, size = parse_model_size(model_name)
        from models.quadbranch_unet import QuadBranchUNet3D
        return QuadBranchUNet3D(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size)
    elif model_name.startswith('quadbranch_channel_centralized_concat_'):
        # Channel Attention (Centralized, Concat only)
        base_name, size = parse_model_size(model_name)
        from models.quadbranch_unet_attention import QuadBranchUNet3D_Channel_Centralized_Concat
        return QuadBranchUNet3D_Channel_Centralized_Concat(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size, reduction=8)
    elif model_name.startswith('quadbranch_channel_distributed_concat_'):
        # Channel Attention (Distributed, Concat only)
        base_name, size = parse_model_size(model_name)
        from models.quadbranch_unet_attention import QuadBranchUNet3D_Channel_Distributed_Concat
        return QuadBranchUNet3D_Channel_Distributed_Concat(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size, reduction=16)
    elif model_name.startswith('quadbranch_channel_distributed_conv_'):
        # Channel Attention (Distributed, Conv fusion)
        base_name, size = parse_model_size(model_name)
        from models.quadbranch_unet_attention import QuadBranchUNet3D_Channel_Distributed_Conv
        return QuadBranchUNet3D_Channel_Distributed_Conv(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size, reduction=16)
    elif model_name.startswith('quadbranch_spatial_centralized_concat_'):
        # Spatial Attention (Centralized, Concat only)
        base_name, size = parse_model_size(model_name)
        from models.quadbranch_unet_attention import QuadBranchUNet3D_Spatial_Centralized_Concat
        return QuadBranchUNet3D_Spatial_Centralized_Concat(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size)
    elif model_name.startswith('quadbranch_spatial_distributed_concat_'):
        # Spatial Attention (Distributed, Concat only)
        base_name, size = parse_model_size(model_name)
        from models.quadbranch_unet_attention import QuadBranchUNet3D_Spatial_Distributed_Concat
        return QuadBranchUNet3D_Spatial_Distributed_Concat(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size)
    elif model_name.startswith('quadbranch_spatial_distributed_conv_'):
        # Spatial Attention (Distributed, Conv fusion)
        base_name, size = parse_model_size(model_name)
        from models.quadbranch_unet_attention import QuadBranchUNet3D_Spatial_Distributed_Conv
        return QuadBranchUNet3D_Spatial_Distributed_Conv(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False, size=size)
    # 이 코드는 실행되지 않아야 함 (검증 단계에서 이미 처리됨)
    # 하지만 방어적 프로그래밍을 위해 남겨둠
    raise RuntimeError(
        f"Internal error: Model '{model_name}' passed validation but was not handled. "
        f"This should not happen. Please report this issue."
    )


def get_roi_model(
    model_name: str,
    n_channels: int = 7,
    n_classes: int = 2,
    roi_model_cfg: Optional[Dict] = None,
) -> torch.nn.Module:
    """ROI 탐지 모델 생성 (Cascade 1단계용)."""
    if not model_name:
        raise ValueError("ROI model name cannot be empty")
    cfg = roi_model_cfg.copy() if roi_model_cfg else {}
    img_size = tuple(cfg.pop('img_size', (64, 64, 64)))
    patch_size = tuple(cfg.pop('patch_size', (2, 2, 2)))
    norm = cfg.pop('norm', 'bn')
    model_name = model_name.lower()

    if model_name.startswith('roi_mobileunetr3d'):
        return build_roi_mobileunetr3d(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=n_channels,
            out_channels=n_classes,
        )
    if model_name == 'roi_unet3d_small':
        base_channels = cfg.pop('base_channels', 16)
        depth = cfg.pop('depth', 4)
        bilinear = cfg.pop('bilinear', True)
        return build_roi_unet3d_small(
            in_channels=n_channels,
            out_channels=n_classes,
            norm=norm,
            base_channels=base_channels,
            depth=depth,
            bilinear=bilinear,
        )

    raise ValueError(f"Unknown ROI model '{model_name}'.")


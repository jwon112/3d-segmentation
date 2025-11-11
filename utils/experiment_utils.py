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

def get_model(model_name, n_channels=4, n_classes=4, dim='3d', patch_size=None, use_pretrained=False, norm: str = 'bn'):
    """모델 생성 함수
    
    Args:
        model_name: 모델 이름
        n_channels: 입력 채널 수
        n_classes: 출력 클래스 수
        dim: '2d' 또는 '3d'
        patch_size: 하이퍼파라미터 (None이면 모델별 기본값 사용)
        use_pretrained: pretrained 가중치 사용 여부 (MobileUNETR의 경우)
    """
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import baseline models
    from baseline import (
        UNETR_Simplified, 
        SwinUNETR_Simplified,
        MobileUNETR,
    )
    from baseline.mobileunetr_3d import MobileUNETR_3D_Wrapper
    
    # 2D 입력인 경우 3D로 확장 (unsqueeze depth dimension)
    if model_name == 'unet3d_s':
        if dim == '2d':
            # 2D 데이터는 depth 차원 추가가 필요
            pass
        from baseline.model_3d_unet import UNet3D_Small
        return UNet3D_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'unet3d_m':
        if dim == '2d':
            # 2D 데이터는 depth 차원 추가가 필요
            pass
        from baseline.model_3d_unet import UNet3D_Medium
        return UNet3D_Medium(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    elif model_name == 'unet3d_stride_s':
        # UNet3D variant with stride-2 conv downsampling (Small channels)
        from baseline.model_3d_unet_stride import UNet3D_Stride_Small
        return UNet3D_Stride_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'unet3d_stride_m':
        # UNet3D variant with stride-2 conv downsampling (Medium channels)
        from baseline.model_3d_unet_stride import UNet3D_Stride_Medium
        return UNet3D_Stride_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm, bilinear=False)
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
        return UNETR_Simplified(
            img_size=img_size, 
            patch_size=patch_size,
            in_channels=n_channels, 
            out_channels=n_classes
        )
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
        return SwinUNETR_Simplified(
            img_size=img_size, 
            patch_size=patch_size,
            in_channels=n_channels, 
            out_channels=n_classes
        )
    elif model_name == 'mobile_unetr':
        # MobileUNETR는 2D 전용 모델
        if dim != '2d':
            raise ValueError("MobileUNETR is only supported for 2D data (dim='2d')")
        
        # img_size 설정 (2D만) - 전역 설정 사용
        img_size = INPUT_SIZE_2D
        if patch_size is None:
            patch_size = (16, 16)  # 2D에서 권장값
        
        return MobileUNETR(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=n_channels,
            out_channels=n_classes,
            use_pretrained=use_pretrained
        )
    elif model_name == 'mobile_unetr_3d':
        # MobileUNETR 3D 모델
        if dim != '3d':
            raise ValueError("MobileUNETR 3D is only supported for 3D data (dim='3d')")
        
        # img_size 설정 (3D)
        img_size = INPUT_SIZE_3D
        if patch_size is None:
            patch_size = (2, 2, 2)  # 3D에서 권장값
        
        return MobileUNETR_3D_Wrapper(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=n_channels,
            out_channels=n_classes
        )
    elif model_name == 'dualbranch_01_unet_s':
        # Dual-branch 3D UNet (v0.1) - Small channels. Expect exactly 2 input channels (FLAIR, t1ce)
        from baseline.dualbranch_01_unet import DualBranchUNet3D_Small
        return DualBranchUNet3D_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_01_unet_m':
        # Dual-branch 3D UNet (v0.1) - Medium channels. Expect exactly 2 input channels (FLAIR, t1ce)
        from baseline.dualbranch_01_unet import DualBranchUNet3D_Medium
        return DualBranchUNet3D_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_02_unet_s':
        # Dual-branch 3D UNet (v0.2) - stride-2 convolutional downsampling (Small channels)
        from baseline.dualbranch_02_unet import DualBranchUNet3D_Stride_Small
        return DualBranchUNet3D_Stride_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_02_unet_m':
        # Dual-branch 3D UNet (v0.2) - stride-2 convolutional downsampling (Medium channels)
        from baseline.dualbranch_02_unet import DualBranchUNet3D_Stride_Medium
        return DualBranchUNet3D_Stride_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_03_unet_s':
        # Dual-branch 3D UNet (v0.3) - dilated conv for FLAIR branch (Small channels)
        from baseline.dualbranch_03_unet import DualBranchUNet3D_StrideDilated_Small
        return DualBranchUNet3D_StrideDilated_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_03_unet_m':
        # Dual-branch 3D UNet (v0.3) - dilated conv for FLAIR branch (Medium channels)
        from baseline.dualbranch_03_unet import DualBranchUNet3D_StrideDilated_Medium
        return DualBranchUNet3D_StrideDilated_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_04_unet_s':
        from baseline.dualbranch_04_unet import DualBranchUNet3D_StrideLK_Small
        return DualBranchUNet3D_StrideLK_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_04_unet_m':
        from baseline.dualbranch_04_unet import DualBranchUNet3D_StrideLK_Medium
        return DualBranchUNet3D_StrideLK_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_05_unet_s':
        from baseline.dualbranch_05_unet import DualBranchUNet3D_StrideLK_FFN2_Small
        return DualBranchUNet3D_StrideLK_FFN2_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_05_unet_m':
        from baseline.dualbranch_05_unet import DualBranchUNet3D_StrideLK_FFN2_Medium
        return DualBranchUNet3D_StrideLK_FFN2_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_06_unet_s':
        from baseline.dualbranch_06_unet import DualBranchUNet3D_StrideLK_FFN2_MViT_Small
        return DualBranchUNet3D_StrideLK_FFN2_MViT_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_06_unet_m':
        from baseline.dualbranch_06_unet import DualBranchUNet3D_StrideLK_FFN2_MViT_Medium
        return DualBranchUNet3D_StrideLK_FFN2_MViT_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_07_unet_s':
        from baseline.dualbranch_07_unet import DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Small
        return DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_07_unet_m':
        from baseline.dualbranch_07_unet import DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Medium
        return DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_08_unet_s':
        from baseline.dualbranch_08_unet import DualBranchUNet3D_LK_MobileNet_MViT_Small
        return DualBranchUNet3D_LK_MobileNet_MViT_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_08_unet_m':
        from baseline.dualbranch_08_unet import DualBranchUNet3D_LK_MobileNet_MViT_Medium
        return DualBranchUNet3D_LK_MobileNet_MViT_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_09_unet_s':
        from baseline.dualbranch_09_unet import DualBranchUNet3D_LK7x7_MobileNet_MViT_Small
        return DualBranchUNet3D_LK7x7_MobileNet_MViT_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_09_unet_m':
        from baseline.dualbranch_09_unet import DualBranchUNet3D_LK7x7_MobileNet_MViT_Medium
        return DualBranchUNet3D_LK7x7_MobileNet_MViT_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_10_unet_s':
        from baseline.dualbranch_10_unet import DualBranchUNet3D_DilatedMobile_Small
        return DualBranchUNet3D_DilatedMobile_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_10_unet_m':
        from baseline.dualbranch_10_unet import DualBranchUNet3D_DilatedMobile_Medium
        return DualBranchUNet3D_DilatedMobile_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_11_unet_s':
        from baseline.dualbranch_11_unet import DualBranchUNet3D_Dilated123_Mobile_Small
        return DualBranchUNet3D_Dilated123_Mobile_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_11_unet_m':
        from baseline.dualbranch_11_unet import DualBranchUNet3D_Dilated123_Mobile_Medium
        return DualBranchUNet3D_Dilated123_Mobile_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_12_unet_s':
        from baseline.dualbranch_12_unet import DualBranchUNet3D_MobileNet_MViT_Small
        return DualBranchUNet3D_MobileNet_MViT_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_12_unet_m':
        from baseline.dualbranch_12_unet import DualBranchUNet3D_MobileNet_MViT_Medium
        return DualBranchUNet3D_MobileNet_MViT_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_13_unet_s':
        from baseline.dualbranch_13_unet import DualBranchUNet3D_MViT_Extended_Small
        return DualBranchUNet3D_MViT_Extended_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_13_unet_m':
        from baseline.dualbranch_13_unet import DualBranchUNet3D_MViT_Extended_Medium
        return DualBranchUNet3D_MViT_Extended_Medium(n_channels=n_channels, n_classes=n_classes, norm=norm)
    # 모달리티 비교 실험 모델들
    elif model_name == 'unet3d_2modal_s':
        # 단일 분기, 2채널 (t1ce, flair) concat
        from baseline.model_3d_unet_modal_comparison import UNet3D_2Modal_Small
        return UNet3D_2Modal_Small(n_classes=n_classes, norm=norm)
    elif model_name == 'unet3d_4modal_s':
        # 단일 분기, 4채널 (t1, t1ce, t2, flair) concat
        from baseline.model_3d_unet_modal_comparison import UNet3D_4Modal_Small
        return UNet3D_4Modal_Small(n_classes=n_classes, norm=norm)
    elif model_name == 'dualbranch_2modal_unet_s':
        # 2개 분기 (t1ce, flair) - dualbranch_01_unet_s와 동일 (MaxPool 기반)
        from baseline.dualbranch_01_unet import DualBranchUNet3D_Small
        return DualBranchUNet3D_Small(n_channels=n_channels, n_classes=n_classes, norm=norm)
    elif model_name == 'quadbranch_4modal_unet_s':
        # 4개 분기 (t1, t1ce, t2, flair) - 어텐션 없음
        from baseline.model_3d_unet_modal_comparison import QuadBranchUNet3D_4Modal_Small
        return QuadBranchUNet3D_4Modal_Small(n_classes=n_classes, norm=norm)
    elif model_name == 'quadbranch_4modal_attention_unet_s':
        # 4개 분기 (t1, t1ce, t2, flair) - 채널 어텐션 포함
        from baseline.model_3d_unet_modal_comparison import QuadBranchUNet3D_4Modal_Attention_Small
        return QuadBranchUNet3D_4Modal_Attention_Small(n_classes=n_classes, norm=norm)
    else:
        raise ValueError(f"Unknown model: {model_name}")


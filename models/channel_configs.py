"""
Channel Configuration for Model Variants
중앙 집중식 채널 설정 관리

채널 구조:
- XS (Extra Small): 가장 작은 모델
- S (Small): 작은 모델
- M (Medium): 중간 모델
- L (Large): 큰 모델

각 버전은 2배씩 채널이 증가합니다.
"""

from typing import Dict, List, Tuple


# ============================================================================
# UNet Channel Configurations
# ============================================================================

UNET_CHANNELS = {
    'xs': {
        'enc': [16, 32, 64, 128, 256],  # encoder channels
        'bottleneck': 256,  # bottleneck channel
    },
    's': {
        'enc': [32, 64, 128, 256, 512],  # encoder channels
        'bottleneck': 512,  # bottleneck channel
    },
    'm': {
        'enc': [64, 128, 256, 512, 1024],  # encoder channels
        'bottleneck': 1024,  # bottleneck channel
    },
    'l': {
        'enc': [128, 256, 512, 1024, 2048],  # encoder channels
        'bottleneck': 2048,  # bottleneck channel
    },
}


# ============================================================================
# Dual-Branch UNet Channel Configurations
# ============================================================================

DUALBRANCH_CHANNELS = {
    'xs': {
        'stem': 8,           # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 16,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 32,       # Stage 3: branch channels (each branch) - stride=2
        'branch4': 64,       # Stage 4: branch channels (each branch) - stride=2
        'branch5': 128,      # Stage 5: branch channels (each branch) - stride=2 (per-branch bottleneck)
        'down6': 512,        # Stage 6: fused branch channels - stride=2 (input: branch5, output: branch5*4)
        'out': 16,           # Output channels (decoder final)
    },
    's': {
        'stem': 16,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 32,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 64,       # Stage 3: branch channels (each branch) - stride=2
        'branch4': 128,      # Stage 4: branch channels (each branch) - stride=2
        'branch5': 256,      # Stage 5: branch channels (each branch) - stride=2 (per-branch bottleneck)
        'down6': 1024,       # Stage 6: fused branch channels - stride=2 (input: branch5, output: branch5*4)
        'out': 32,           # Output channels (decoder final)
    },
    'm': {
        'stem': 32,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 64,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 128,      # Stage 3: branch channels (each branch) - stride=2
        'branch4': 256,      # Stage 4: branch channels (each branch) - stride=2
        'branch5': 512,      # Stage 5: branch channels (each branch) - stride=2 (per-branch bottleneck)
        'down6': 2048,       # Stage 6: fused branch channels - stride=2 (input: branch5, output: branch5*4)
        'out': 64,           # Output channels (decoder final)
    },
    'l': {
        'stem': 64,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 128,      # Stage 2: branch channels (each branch) - stride=2
        'branch3': 256,      # Stage 3: branch channels (each branch) - stride=2
        'branch4': 512,      # Stage 4: branch channels (each branch) - stride=2
        'branch5': 1024,     # Stage 5: branch channels (each branch) - stride=2 (per-branch bottleneck)
        'down6': 4096,       # Stage 6: fused branch channels - stride=2 (input: branch5, output: branch5*4)
        'out': 128,          # Output channels (decoder final)
    },
}


# ============================================================================
# Quad-Branch UNet Channel Configurations
# ============================================================================

QUADBRANCH_CHANNELS = {
    'xs': {
        'stem': 8,           # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 16,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 32,       # Stage 3: branch channels (each branch) - stride=2
        'branch4': 64,       # Stage 4: branch channels (each branch) - stride=2
        'down5': 256,        # Stage 5: single branch channels - stride=2 (input: branch4*4=256, output: 256)
        'out': 16,           # Output channels (decoder final)
    },
    's': {
        'stem': 16,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 32,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 64,       # Stage 3: branch channels (each branch) - stride=2
        'branch4': 128,      # Stage 4: branch channels (each branch) - stride=2
        'down5': 512,        # Stage 5: single branch channels - stride=2 (input: branch4*4=512, output: 512)
        'out': 32,           # Output channels (decoder final)
    },
    'm': {
        'stem': 32,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 64,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 128,      # Stage 3: branch channels (each branch) - stride=2
        'branch4': 256,      # Stage 4: branch channels (each branch) - stride=2
        'down5': 1024,       # Stage 5: single branch channels - stride=2 (input: branch4*4=1024, output: 1024)
        'out': 64,           # Output channels (decoder final)
    },
    'l': {
        'stem': 64,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 128,      # Stage 2: branch channels (each branch) - stride=2
        'branch3': 256,      # Stage 3: branch channels (each branch) - stride=2
        'branch4': 512,      # Stage 4: branch channels (each branch) - stride=2
        'down5': 2048,       # Stage 5: single branch channels - stride=2 (input: branch4*4=2048, output: 2048)
        'out': 128,          # Output channels (decoder final)
    },
}


# ============================================================================
# Dual-Branch UNet Channel Configurations (Stage 3 fused at down4, 4-stage structure)
# ============================================================================

DUALBRANCH_CHANNELS_STAGE3_FUSED = {
    'xs': {
        'stem': 8,           # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 16,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 32,       # Stage 3: branch channels (each branch) - stride=2
        'down4': 128,        # Stage 4: single branch channels (bottleneck) - stride=2 (input: branch3*2=64, output: 128 = branch3*4)
        'out': 16,           # Output channels (decoder final)
    },
    's': {
        'stem': 16,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 32,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 64,       # Stage 3: branch channels (each branch) - stride=2
        'down4': 256,        # Stage 4: single branch channels (bottleneck) - stride=2 (input: branch3*2=128, output: 256 = branch3*4)
        'out': 32,           # Output channels (decoder final)
    },
    'm': {
        'stem': 32,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 64,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 128,      # Stage 3: branch channels (each branch) - stride=2
        'down4': 512,        # Stage 4: single branch channels (bottleneck) - stride=2 (input: branch3*2=256, output: 512 = branch3*4)
        'out': 64,           # Output channels (decoder final)
    },
    'l': {
        'stem': 64,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 128,      # Stage 2: branch channels (each branch) - stride=2
        'branch3': 256,      # Stage 3: branch channels (each branch) - stride=2
        'down4': 1024,       # Stage 4: single branch channels (bottleneck) - stride=2 (input: branch3*2=512, output: 1024 = branch3*4)
        'out': 128,          # Output channels (decoder final)
    },
}

# Dual-Branch UNet with Fixed Decoder Channels (Stage 3 Fused, 디코더 채널 고정)
# 디코더 채널을 고정된 값으로 유지하여 매우 가볍고 일관된 구조
# up1은 버퍼 역할로 2배 크게, up2부터 고정 채널 사용
DUALBRANCH_CHANNELS_STAGE3_FUSED_FIXED_DECODER = {
    # -------------------------------------------------------
    # XS: Decoder Fixed at 16 (Ultra Light)
    # -------------------------------------------------------
    'xs': {
        'stem': 8,       # Total: 16
        'branch2': 16,   # Total: 32
        'branch3': 32,   # Total: 64
        'down4': 32,     # Bottleneck Output (Project to 32, Inverted Bottleneck 유지)
        # Decoder (bilinear=False): up1은 절반으로 줄이고, up2/up3는 채널 유지하여 skip과 1:1 매칭
        'up1': 16,       # down4(32) -> ConvTranspose3d(16) + skip_compress(16) -> concat(32) -> Out: 16
        'up2': 16,       # up1(16) -> ConvTranspose3d(16) + skip_compress(16) -> concat(32) -> Out: 16
        'up3': 16,       # up2(16) -> ConvTranspose3d(16) + skip(16) -> concat(32) -> Out: 16
        'out': 16,       # Final
    },
    # -------------------------------------------------------
    # S: Decoder Fixed at 32 (Balanced)
    # -------------------------------------------------------
    's': {
        'stem': 16,      # Total: 32
        'branch2': 32,   # Total: 64
        'branch3': 64,   # Total: 128
        'down4': 64,     # Bottleneck Output (Project to 64, Inverted Bottleneck 유지)
        # Decoder (bilinear=False): up1은 절반으로 줄이고, up2/up3는 채널 유지하여 skip과 1:1 매칭
        'up1': 32,       # down4(64) -> ConvTranspose3d(32) + skip_compress(32) -> concat(64) -> Out: 32
        'up2': 32,       # up1(32) -> ConvTranspose3d(32) + skip_compress(32) -> concat(64) -> Out: 32
        'up3': 32,       # up2(32) -> ConvTranspose3d(32) + skip(32) -> concat(64) -> Out: 32
        'out': 32,
    },
    # -------------------------------------------------------
    # M: Decoder Fixed at 64 (Performance)
    # -------------------------------------------------------
    'm': {
        'stem': 32,      # Total: 64
        'branch2': 64,   # Total: 128
        'branch3': 128,  # Total: 256
        'down4': 128,    # Bottleneck Output (Project to 128, Inverted Bottleneck 유지)
        # Decoder (bilinear=False): up1은 절반으로 줄이고, up2/up3는 채널 유지하여 skip과 1:1 매칭
        'up1': 64,       # down4(128) -> ConvTranspose3d(64) + skip_compress(64) -> concat(128) -> Out: 64
        'up2': 64,       # up1(64) -> ConvTranspose3d(64) + skip_compress(64) -> concat(128) -> Out: 64
        'up3': 64,       # up2(64) -> ConvTranspose3d(64) + skip(64) -> concat(128) -> Out: 64
        'out': 64,
    },
    # -------------------------------------------------------
    # L: Decoder Fixed at 128 (High Performance)
    # -------------------------------------------------------
    'l': {
        'stem': 64,      # Total: 128
        'branch2': 128,  # Total: 256
        'branch3': 256,  # Total: 512
        'down4': 256,    # Bottleneck Output (Project to 256, Inverted Bottleneck 유지)
        # Decoder (bilinear=False): up1은 절반으로 줄이고, up2/up3는 채널 유지하여 skip과 1:1 매칭
        'up1': 128,      # down4(256) -> ConvTranspose3d(128) + skip_compress(128) -> concat(256) -> Out: 128
        'up2': 128,      # up1(128) -> ConvTranspose3d(128) + skip_compress(128) -> concat(256) -> Out: 128
        'up3': 128,      # up2(128) -> ConvTranspose3d(128) + skip(128) -> concat(256) -> Out: 128
        'out': 128,
    },
}

# Dual-Branch UNet with Half Decoder Channels (Stage 3 Fused, 디코더 채널을 인코더의 절반으로)
# 인코더 stage 채널 = branch * 2 (두 분기 concat), 따라서 디코더 채널 = branch (인코더의 절반)
DUALBRANCH_CHANNELS_STAGE3_FUSED_HALF_DECODER = {
    'xs': {
        'stem': 8,           # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 16,       # Stage 2: branch channels (each branch) - stride=2, stage2 채널 = 16*2=32
        'branch3': 32,       # Stage 3: branch channels (each branch) - stride=2, stage3 채널 = 32*2=64
        'down4': 128,        # Stage 4: single branch channels (bottleneck) - stride=2 (input: branch3*2=64, output: 128 = branch3*4)
        'up1': 16,           # Decoder up1 출력: stage3 채널 64의 절반 = 32, 하지만 디코더 입력도 절반이므로 16
        'up2': 8,            # Decoder up2 출력: stage2 채널 32의 절반 = 16, 하지만 디코더 입력도 절반이므로 8
        'up3': 4,            # Decoder up3 출력: stage1 채널 16의 절반 = 8, 하지만 디코더 입력도 절반이므로 4
        'out': 4,            # Output channels (OutConv3D input = up3 output)
    },
    's': {
        'stem': 16,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 32,       # Stage 2: branch channels (each branch) - stride=2, stage2 채널 = 32*2=64
        'branch3': 64,       # Stage 3: branch channels (each branch) - stride=2, stage3 채널 = 64*2=128
        'down4': 256,        # Stage 4: single branch channels (bottleneck) - stride=2 (input: branch3*2=128, output: 256 = branch3*4)
        'up1': 32,           # Decoder up1 출력: stage3 채널 128의 절반 = 64, 하지만 디코더 입력도 절반이므로 32
        'up2': 16,           # Decoder up2 출력: stage2 채널 64의 절반 = 32, 하지만 디코더 입력도 절반이므로 16
        'up3': 8,            # Decoder up3 출력: stage1 채널 32의 절반 = 16, 하지만 디코더 입력도 절반이므로 8
        'out': 8,            # Output channels (OutConv3D input = up3 output)
    },
    'm': {
        'stem': 32,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 64,       # Stage 2: branch channels (each branch) - stride=2, stage2 채널 = 64*2=128
        'branch3': 128,      # Stage 3: branch channels (each branch) - stride=2, stage3 채널 = 128*2=256
        'down4': 512,        # Stage 4: single branch channels (bottleneck) - stride=2 (input: branch3*2=256, output: 512 = branch3*4)
        'up1': 64,           # Decoder up1 출력: stage3 채널 256의 절반 = 128, 하지만 디코더 입력도 절반이므로 64
        'up2': 32,           # Decoder up2 출력: stage2 채널 128의 절반 = 64, 하지만 디코더 입력도 절반이므로 32
        'up3': 16,           # Decoder up3 출력: stage1 채널 64의 절반 = 32, 하지만 디코더 입력도 절반이므로 16
        'out': 16,           # Output channels (OutConv3D input = up3 output)
    },
    'l': {
        'stem': 64,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 128,      # Stage 2: branch channels (each branch) - stride=2, stage2 채널 = 128*2=256
        'branch3': 256,      # Stage 3: branch channels (each branch) - stride=2, stage3 채널 = 256*2=512
        'down4': 1024,       # Stage 4: single branch channels (bottleneck) - stride=2 (input: branch3*2=512, output: 1024 = branch3*4)
        'up1': 128,          # Decoder up1 출력: stage3 채널 512의 절반 = 256, 하지만 디코더 입력도 절반이므로 128
        'up2': 64,           # Decoder up2 출력: stage2 채널 256의 절반 = 128, 하지만 디코더 입력도 절반이므로 64
        'up3': 32,           # Decoder up3 출력: stage1 채널 128의 절반 = 64, 하지만 디코더 입력도 절반이므로 32
        'out': 32,           # Output channels (OutConv3D input = up3 output)
    },
}


# ============================================================================
# Dual-Branch UNet Channel Configurations (Stage 4 fused, Stage 5 single branch)
# ============================================================================

DUALBRANCH_CHANNELS_STAGE4_FUSED = {
    'xs': {
        'stem': 8,           # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 16,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 32,       # Stage 3: branch channels (each branch) - stride=2
        'branch4': 64,       # Stage 4: branch channels (each branch) - stride=2
        'down5': 256,        # Stage 5: single branch channels - stride=2 (input: branch4*2=128, output: 256 = branch4*4)
        'out': 16,           # Output channels (decoder final)
    },
    's': {
        'stem': 16,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 32,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 64,       # Stage 3: branch channels (each branch) - stride=2
        'branch4': 128,      # Stage 4: branch channels (each branch) - stride=2
        'down5': 512,        # Stage 5: single branch channels - stride=2 (input: branch4*2=256, output: 512 = branch4*4)
        'out': 32,           # Output channels (decoder final)
    },
    'm': {
        'stem': 32,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 64,       # Stage 2: branch channels (each branch) - stride=2
        'branch3': 128,      # Stage 3: branch channels (each branch) - stride=2
        'branch4': 256,      # Stage 4: branch channels (each branch) - stride=2
        'down5': 1024,       # Stage 5: single branch channels - stride=2 (input: branch4*2=512, output: 1024 = branch4*4)
        'out': 64,           # Output channels (decoder final)
    },
    'l': {
        'stem': 64,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 128,      # Stage 2: branch channels (each branch) - stride=2
        'branch3': 256,      # Stage 3: branch channels (each branch) - stride=2
        'branch4': 512,      # Stage 4: branch channels (each branch) - stride=2
        'down5': 2048,       # Stage 5: single branch channels - stride=2 (input: branch4*2=1024, output: 2048 = branch4*4)
        'out': 128,          # Output channels (decoder final)
    },
}

# Dual-Branch UNet with Half Decoder Channels (디코더 채널을 인코더의 절반으로)
# 인코더 stage 채널 = branch * 2 (두 분기 concat), 따라서 디코더 채널 = branch (인코더의 절반)
DUALBRANCH_CHANNELS_STAGE4_FUSED_HALF_DECODER = {
    'xs': {
        'stem': 8,           # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 16,       # Stage 2: branch channels (each branch) - stride=2, stage2 채널 = 16*2=32
        'branch3': 32,       # Stage 3: branch channels (each branch) - stride=2, stage3 채널 = 32*2=64
        'branch4': 64,       # Stage 4: branch channels (each branch) - stride=2, stage4 채널 = 64*2=128
        'down5': 256,        # Stage 5: single branch channels - stride=2 (input: branch4*2=128, output: 256 = branch4*4)
        'up1': 32,           # Decoder up1 출력: stage4 채널 128의 절반 = 64, 하지만 디코더 입력도 절반이므로 32
        'up2': 16,           # Decoder up2 출력: stage3 채널 64의 절반 = 32, 하지만 디코더 입력도 절반이므로 16
        'up3': 8,            # Decoder up3 출력: stage2 채널 32의 절반 = 16, 하지만 디코더 입력도 절반이므로 8
        'up4': 4,            # Decoder up4 출력: stage1 채널 16의 절반 = 8, 하지만 디코더 입력도 절반이므로 4
        'out': 4,            # Output channels (OutConv3D input = up4 output)
    },
    's': {
        'stem': 16,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 32,       # Stage 2: branch channels (each branch) - stride=2, stage2 채널 = 32*2=64
        'branch3': 64,       # Stage 3: branch channels (each branch) - stride=2, stage3 채널 = 64*2=128
        'branch4': 128,      # Stage 4: branch channels (each branch) - stride=2, stage4 채널 = 128*2=256
        'down5': 512,        # Stage 5: single branch channels - stride=2 (input: branch4*2=256, output: 512 = branch4*4)
        'up1': 64,           # Decoder up1 출력: stage4 채널 256의 절반 = 128, 하지만 디코더 입력도 절반이므로 64
        'up2': 32,           # Decoder up2 출력: stage3 채널 128의 절반 = 64, 하지만 디코더 입력도 절반이므로 32
        'up3': 16,           # Decoder up3 출력: stage2 채널 64의 절반 = 32, 하지만 디코더 입력도 절반이므로 16
        'up4': 8,            # Decoder up4 출력: stage1 채널 32의 절반 = 16, 하지만 디코더 입력도 절반이므로 8
        'out': 8,            # Output channels (OutConv3D input = up4 output)
    },
    'm': {
        'stem': 32,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 64,       # Stage 2: branch channels (each branch) - stride=2, stage2 채널 = 64*2=128
        'branch3': 128,      # Stage 3: branch channels (each branch) - stride=2, stage3 채널 = 128*2=256
        'branch4': 256,      # Stage 4: branch channels (each branch) - stride=2, stage4 채널 = 256*2=512
        'down5': 1024,       # Stage 5: single branch channels - stride=2 (input: branch4*2=512, output: 1024 = branch4*4)
        'up1': 128,          # Decoder up1 출력: stage4 채널 512의 절반 = 256, 하지만 디코더 입력도 절반이므로 128
        'up2': 64,           # Decoder up2 출력: stage3 채널 256의 절반 = 128, 하지만 디코더 입력도 절반이므로 64
        'up3': 32,           # Decoder up3 출력: stage2 채널 128의 절반 = 64, 하지만 디코더 입력도 절반이므로 32
        'up4': 16,           # Decoder up4 출력: stage1 채널 64의 절반 = 32, 하지만 디코더 입력도 절반이므로 16
        'out': 16,           # Output channels (OutConv3D input = up4 output)
    },
    'l': {
        'stem': 64,          # Stage 1: stem channels (each branch) - no downsampling
        'branch2': 128,      # Stage 2: branch channels (each branch) - stride=2, stage2 채널 = 128*2=256
        'branch3': 256,      # Stage 3: branch channels (each branch) - stride=2, stage3 채널 = 256*2=512
        'branch4': 512,      # Stage 4: branch channels (each branch) - stride=2, stage4 채널 = 512*2=1024
        'down5': 2048,       # Stage 5: single branch channels - stride=2 (input: branch4*2=1024, output: 2048 = branch4*4)
        'up1': 256,          # Decoder up1 출력: stage4 채널 1024의 절반 = 512, 하지만 디코더 입력도 절반이므로 256
        'up2': 128,          # Decoder up2 출력: stage3 채널 512의 절반 = 256, 하지만 디코더 입력도 절반이므로 128
        'up3': 64,           # Decoder up3 출력: stage2 채널 256의 절반 = 128, 하지만 디코더 입력도 절반이므로 64
        'up4': 32,           # Decoder up4 출력: stage1 채널 128의 절반 = 64, 하지만 디코더 입력도 절반이므로 32
        'out': 32,           # Output channels (OutConv3D input = up4 output)
    },
}


def get_unet_channels(size: str) -> Dict:
    """
    Get UNet channel configuration for given size.
    
    Args:
        size: 'xs', 's', 'm', or 'l'
    
    Returns:
        Dictionary with channel configuration
    """
    if size not in UNET_CHANNELS:
        raise ValueError(f"Unknown size: {size}. Must be one of {list(UNET_CHANNELS.keys())}")
    return UNET_CHANNELS[size]


def get_dualbranch_channels(size: str) -> Dict:
    """
    Get Dual-Branch UNet channel configuration for given size.
    
    Args:
        size: 'xs', 's', 'm', or 'l'
    
    Returns:
        Dictionary with channel configuration
    """
    if size not in DUALBRANCH_CHANNELS:
        raise ValueError(f"Unknown size: {size}. Must be one of {list(DUALBRANCH_CHANNELS.keys())}")
    return DUALBRANCH_CHANNELS[size]


def get_dualbranch_channels_stage4_fused(size: str, half_decoder: bool = False) -> Dict:
    """
    Get Dual-Branch UNet channel configuration for given size (Stage 4 fused, Stage 5 single branch).
    
    This configuration is for models where:
    - Stage 1-4: Dual-branch structure (each branch independently)
    - Stage 5: Single branch (Stage 4 outputs are concatenated before Stage 5)
    
    Args:
        size: 'xs', 's', 'm', or 'l'
        half_decoder: If True, decoder channels are half of encoder channels (default: False)
    
    Returns:
        Dictionary with channel configuration
        Note: Stage 5 input channels = branch4 * 2 (after concatenation)
    """
    if half_decoder:
        if size not in DUALBRANCH_CHANNELS_STAGE4_FUSED_HALF_DECODER:
            raise ValueError(f"Unknown size: {size}. Must be one of {list(DUALBRANCH_CHANNELS_STAGE4_FUSED_HALF_DECODER.keys())}")
        return DUALBRANCH_CHANNELS_STAGE4_FUSED_HALF_DECODER[size]
    else:
        if size not in DUALBRANCH_CHANNELS_STAGE4_FUSED:
            raise ValueError(f"Unknown size: {size}. Must be one of {list(DUALBRANCH_CHANNELS_STAGE4_FUSED.keys())}")
        return DUALBRANCH_CHANNELS_STAGE4_FUSED[size]


def get_dualbranch_channels_stage3_fused(size: str, half_decoder: bool = False, fixed_decoder: bool = False) -> Dict:
    """
    Get Dual-Branch UNet channel configuration for given size (Stage 3 fused at down4, 4-stage structure).
    
    This configuration is for models where:
    - Stage 1-3: Dual-branch structure (each branch independently)
    - Stage 4: Single branch bottleneck (down4, Stage 3 outputs are concatenated before down4)
    
    Structure: stem → branch2 (dual) → branch3 (dual) → down4 (fused bottleneck) → out
    
    Args:
        size: 'xs', 's', 'm', or 'l'
        half_decoder: If True, decoder channels are half of encoder channels (default: False)
        fixed_decoder: If True, decoder channels are fixed at a constant value (default: False)
    
    Returns:
        Dictionary with channel configuration
        Note: down4 input channels = branch3 * 2 (after concatenation)
    """
    if fixed_decoder:
        if size not in DUALBRANCH_CHANNELS_STAGE3_FUSED_FIXED_DECODER:
            raise ValueError(f"Unknown size: {size}. Must be one of {list(DUALBRANCH_CHANNELS_STAGE3_FUSED_FIXED_DECODER.keys())}")
        return DUALBRANCH_CHANNELS_STAGE3_FUSED_FIXED_DECODER[size]
    elif half_decoder:
        if size not in DUALBRANCH_CHANNELS_STAGE3_FUSED_HALF_DECODER:
            raise ValueError(f"Unknown size: {size}. Must be one of {list(DUALBRANCH_CHANNELS_STAGE3_FUSED_HALF_DECODER.keys())}")
        return DUALBRANCH_CHANNELS_STAGE3_FUSED_HALF_DECODER[size]
    else:
        if size not in DUALBRANCH_CHANNELS_STAGE3_FUSED:
            raise ValueError(f"Unknown size: {size}. Must be one of {list(DUALBRANCH_CHANNELS_STAGE3_FUSED.keys())}")
        return DUALBRANCH_CHANNELS_STAGE3_FUSED[size]


def get_quadbranch_channels(size: str) -> Dict:
    """
    Get Quad-Branch UNet channel configuration for given size.
    
    This configuration is for models with 4 modality branches:
    - Stage 1: 4 modality-specific stems (each branch independently)
    - Stage 2-4: 4 modality-specific branches (each branch independently)
    - Stage 5+: Single fused branch (Stage 4 outputs are concatenated before Stage 5)
    
    Args:
        size: 'xs', 's', 'm', or 'l'
    
    Returns:
        Dictionary with channel configuration
        Note: Stage 5 input channels = branch4 * 4 (after concatenation of 4 branches)
    """
    if size not in QUADBRANCH_CHANNELS:
        raise ValueError(f"Unknown size: {size}. Must be one of {list(QUADBRANCH_CHANNELS.keys())}")
    return QUADBRANCH_CHANNELS[size]


def get_activation_type(size: str) -> str:
    """모델 크기에 따라 활성화 함수 타입을 반환합니다.
    
    Args:
        size: 모델 크기 ('xs', 's', 'm', 'l')
    
    Returns:
        활성화 함수 타입 ('hardswish' for xs/s, 'gelu' for m/l)
    """
    size = size.lower()
    if size in ('xs', 's'):
        return 'hardswish'
    elif size in ('m', 'l'):
        return 'gelu'
    else:
        # 기본값: hardswish
        return 'hardswish'


def parse_model_size(model_name: str) -> Tuple[str, str]:
    """
    Parse model name to extract base name and size.
    
    Args:
        model_name: Model name (e.g., 'dualbranch_01_unet_s', 'unet3d_m')
    
    Returns:
        Tuple of (base_name, size) where size is 'xs', 's', 'm', or 'l'
    
    Examples:
        'dualbranch_01_unet_s' -> ('dualbranch_01_unet', 's')
        'unet3d_m' -> ('unet3d', 'm')
        'dualbranch_14_dilated_xs' -> ('dualbranch_14_dilated', 'xs')
    """
    size_suffixes = ['_xs', '_s', '_m', '_l']
    
    for suffix in size_suffixes:
        if model_name.endswith(suffix):
            base_name = model_name[:-len(suffix)]
            size = suffix[1:]  # Remove leading underscore
            return base_name, size
    
    # Default to 's' if no size suffix found
    return model_name, 's'


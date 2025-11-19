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


def get_dualbranch_channels_stage4_fused(size: str) -> Dict:
    """
    Get Dual-Branch UNet channel configuration for given size (Stage 4 fused, Stage 5 single branch).
    
    This configuration is for models where:
    - Stage 1-4: Dual-branch structure (each branch independently)
    - Stage 5: Single branch (Stage 4 outputs are concatenated before Stage 5)
    
    Args:
        size: 'xs', 's', 'm', or 'l'
    
    Returns:
        Dictionary with channel configuration
        Note: Stage 5 input channels = branch4 * 2 (after concatenation)
    """
    if size not in DUALBRANCH_CHANNELS_STAGE4_FUSED:
        raise ValueError(f"Unknown size: {size}. Must be one of {list(DUALBRANCH_CHANNELS_STAGE4_FUSED.keys())}")
    return DUALBRANCH_CHANNELS_STAGE4_FUSED[size]


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


"""
Experiment Configuration
실험 설정 및 모델 리스트 관리
"""

from typing import List, Dict, Optional


# ============================================================================
# Model Configuration
# ============================================================================

# Size suffix를 지원하는 모델 prefix들 (xs, s, m, l 모두 지원)
SIZE_SUFFIX_MODELS = {
    'unet3d_': ['xs', 's', 'm', 'l'],
    'unet3d_stride_': ['xs', 's', 'm', 'l'],
    'dualbranch_01_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_02_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_03_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_04_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_05_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_06_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_07_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_10_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_11_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_13_unet_': ['xs', 's', 'm', 'l'],
    'dualbranch_mobilenetv2_dilated_': ['xs', 's', 'm', 'l'],
    'dualbranch_mobilenetv2_dilated_fixed_': ['xs', 's', 'm', 'l'],
    'dualbranch_16_shufflenet_hybrid_': ['xs', 's', 'm', 'l'],
    'dualbranch_16_shufflenet_hybrid_ln_': ['xs', 's', 'm', 'l'],
    'dualbranch_17_shufflenet_pamlite_': ['xs', 's', 'm', 'l'],
    'dualbranch_17_shufflenet_pamlite_v3_': ['xs', 's', 'm', 'l'],
    'dualbranch_18_shufflenet_v1_': ['xs', 's', 'm', 'l'],
    'dualbranch_18_shufflenet_v1_stage3fused_': ['xs', 's', 'm', 'l'],
    'dualbranch_18_shufflenet_v1_stage3fused_fixed_decoder_': ['xs', 's', 'm', 'l'],
    'dualbranch_18_shufflenet_v1_stage3fused_half_decoder_': ['xs', 's', 'm', 'l'],
    'dualbranch_19_shufflenet_v2_stage3fused_': ['xs', 's', 'm', 'l'],
    'dualbranch_19_shufflenet_v2_stage3fused_fixed_decoder_': ['xs', 's', 'm', 'l'],
    'dualbranch_19_shufflenet_v2_stage3fused_half_decoder_': ['xs', 's', 'm', 'l'],
    'quadbranch_unet_': ['xs', 's', 'm', 'l'],
    'quadbranch_channel_centralized_concat_': ['xs', 's', 'm', 'l'],
    'quadbranch_channel_distributed_concat_': ['xs', 's', 'm', 'l'],
    'quadbranch_channel_distributed_conv_': ['xs', 's', 'm', 'l'],
    'quadbranch_spatial_centralized_concat_': ['xs', 's', 'm', 'l'],
    'quadbranch_spatial_distributed_concat_': ['xs', 's', 'm', 'l'],
    'quadbranch_spatial_distributed_conv_': ['xs', 's', 'm', 'l'],
    'cascade_shufflenet_v2_': ['xs', 's', 'm', 'l'],
    'cascade_shufflenet_v2_p3d_': ['xs', 's', 'm', 'l'],
    'cascade_shufflenet_v2_mvit_': ['xs', 's', 'm', 'l'],
}

# Size suffix를 지원하는 dualbranch_14 backbone들
DUALBRANCH_14_BACKBONES = [
    'mobilenetv2_expand2', 'ghostnet', 'dilated', 'convnext', 
    'shufflenetv2', 'shufflenetv2_crossattn', 'shufflenetv2_dilated', 'shufflenetv2_lk'
]

# Size suffix를 지원하지 않는 모델들 (고정 이름)
FIXED_NAME_MODELS = [
    'unetr', 'swin_unetr', 'mobile_unetr', 'mobile_unetr_3d',
    'unet3d_2modal_s', 'unet3d_4modal_s', 'dualbranch_2modal_unet_s',
    'quadbranch_4modal_unet_s', 'quadbranch_4modal_attention_unet_s'
]

# 4개 모달리티를 사용하는 모델들
MODELS_WITH_4_MODALITIES = [
    'unet3d_4modal_s', 'quadbranch_4modal_unet_s', 'quadbranch_4modal_attention_unet_s'
]


# ============================================================================
# Model List Generation
# ============================================================================

def get_all_available_models() -> List[str]:
    """모든 사용 가능한 모델 리스트 생성"""
    available_models = []
    
    # Size suffix를 지원하는 모델들 생성
    for prefix, sizes in SIZE_SUFFIX_MODELS.items():
        for size in sizes:
            available_models.append(f"{prefix}{size}")
    
    # dualbranch_14 모델들 생성
    for backbone in DUALBRANCH_14_BACKBONES:
        for size in ['xs', 's', 'm', 'l']:
            available_models.append(f"dualbranch_14_{backbone}_{size}")
    
    # 고정 이름 모델들 추가
    available_models.extend(FIXED_NAME_MODELS)
    
    return available_models


def validate_and_filter_models(models: Optional[List[str]]) -> List[str]:
    """사용자가 지정한 모델들을 검증하고 필터링"""
    if models is None:
        return get_all_available_models()
    
    available_models = []
    for model_name in models:
        # 고정 이름 모델인지 확인
        if model_name in FIXED_NAME_MODELS:
            available_models.append(model_name)
            continue
        
        # Size suffix를 지원하는 모델 prefix인지 확인
        is_valid = False
        for prefix, sizes in SIZE_SUFFIX_MODELS.items():
            if model_name.startswith(prefix):
                # Size suffix 추출 (fixed_decoder, half_decoder 같은 중간 suffix 고려)
                suffix = model_name[len(prefix):]
                # _fixed_decoder_ 또는 _half_decoder_ 같은 중간 suffix가 있으면 제거
                if '_fixed_decoder_' in suffix:
                    suffix = suffix.split('_fixed_decoder_')[-1]
                elif '_half_decoder_' in suffix:
                    suffix = suffix.split('_half_decoder_')[-1]
                # 최종 size suffix 추출
                if '_' in suffix:
                    suffix = suffix.split('_')[0]
                if suffix in sizes:
                    is_valid = True
                    break
        
        # dualbranch_14 모델인지 확인
        if not is_valid and model_name.startswith('dualbranch_14_'):
            parts = model_name.split('_', 2)
            if len(parts) >= 3:
                backbone_and_size = parts[2]
                for size in ['xs', 's', 'm', 'l']:
                    if backbone_and_size.endswith(f'_{size}'):
                        backbone = backbone_and_size[:-len(f'_{size}')]
                        if backbone in DUALBRANCH_14_BACKBONES:
                            is_valid = True
                            break
        
        if is_valid:
            available_models.append(model_name)
        else:
            print(f"Warning: Invalid model name '{model_name}' will be skipped.")
    
    return available_models


def get_n_channels_for_model(model_name: str) -> int:
    """모델 이름에 따라 필요한 입력 채널 수 반환
    
    Note: Cascade 모델의 경우 실제 입력은 7채널(4 MRI + 3 CoordConv)이지만,
    데이터로더에서 자동으로 처리되므로 여기서는 4채널로 반환합니다.
    """
    if model_name in MODELS_WITH_4_MODALITIES or model_name.startswith('quadbranch_') or model_name.startswith('cascade_'):
        return 4
    return 2


def get_model_config(model_name: str) -> Dict:
    """모델 설정 정보 반환"""
    return {
        'n_channels': get_n_channels_for_model(model_name),
        'use_4modalities': get_n_channels_for_model(model_name) == 4,
    }


# ============================================================================
# ROI Model Configuration
# ============================================================================
ROI_MODEL_CONFIGS: Dict[str, Dict] = {
    'roi_mobileunetr3d_tiny': {
        'img_size': (64, 64, 64),
        'patch_size': (2, 2, 2),
        'norm': 'bn',
    },
    'roi_unet3d_small': {
        'norm': 'bn',
        'base_channels': 16,
        'depth': 4,
    },
}
DEFAULT_ROI_MODEL = 'roi_mobileunetr3d_tiny'


def get_available_roi_models() -> List[str]:
    return list(ROI_MODEL_CONFIGS.keys())


def validate_roi_model(model_name: Optional[str]) -> str:
    if model_name is None:
        return DEFAULT_ROI_MODEL
    if model_name not in ROI_MODEL_CONFIGS:
        raise ValueError(f"Unknown ROI model '{model_name}'. Available: {get_available_roi_models()}")
    return model_name


def get_roi_model_config(model_name: Optional[str]) -> Dict:
    name = validate_roi_model(model_name)
    return ROI_MODEL_CONFIGS[name]
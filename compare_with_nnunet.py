#!/usr/bin/env python3
"""
nnUNet과 우리 unet3d_s 모델의 파라미터 및 FLOPs 비교 스크립트

사용법:
    python compare_with_nnunet.py --data_path /home/nas/vision_data/BRATS/BRATS2021/BraTS2021_Training_Data
"""

import argparse
import torch
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.experiment_utils import get_model, calculate_flops, set_seed, INPUT_SIZE_3D


def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_layer_wise_parameters(nnunet_model, our_model):
    """레이어별 파라미터 수 비교 분석"""
    print("\n" + "=" * 80)
    print("레이어별 파라미터 수 분석")
    print("=" * 80)
    
    def get_layer_params(model):
        """모델의 레이어별 파라미터 수 추출"""
        params_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_dict[name] = param.numel()
        return params_dict
    
    nnunet_params = get_layer_params(nnunet_model)
    our_params = get_layer_params(our_model)
    
    # 전체 파라미터 수
    nnunet_total = sum(nnunet_params.values())
    our_total = sum(our_params.values())
    
    print(f"\n전체 파라미터 수:")
    print(f"  nnUNet: {nnunet_total:,}")
    print(f"  우리:   {our_total:,}")
    print(f"  차이:   {nnunet_total - our_total:,} ({((nnunet_total - our_total) / nnunet_total * 100):+.2f}%)")
    
    # 카테고리별 그룹화
    def categorize_params(name):
        """파라미터 이름으로 카테고리 분류"""
        name_lower = name.lower()
        if 'encoder' in name_lower or ('enc' in name_lower and 'dec' not in name_lower):
            return 'Encoder'
        elif 'decoder' in name_lower or 'dec' in name_lower:
            return 'Decoder'
        elif 'bottleneck' in name_lower:
            return 'Bottleneck'
        elif 'out' in name_lower or 'head' in name_lower or 'outc' in name_lower:
            return 'Output Head'
        elif 'ds' in name_lower or 'deep' in name_lower:
            return 'Deep Supervision'
        else:
            return 'Other'
    
    # 카테고리별 집계
    nnunet_by_category = {}
    our_by_category = {}
    
    for name, count in nnunet_params.items():
        cat = categorize_params(name)
        nnunet_by_category[cat] = nnunet_by_category.get(cat, 0) + count
    
    for name, count in our_params.items():
        cat = categorize_params(name)
        our_by_category[cat] = our_by_category.get(cat, 0) + count
    
    print(f"\n카테고리별 파라미터 수:")
    print(f"{'카테고리':<20} {'nnUNet':<20} {'우리':<20} {'차이':<20} {'차이(%)':<15}")
    print("-" * 95)
    
    all_categories = set(nnunet_by_category.keys()) | set(our_by_category.keys())
    for cat in sorted(all_categories):
        nnunet_count = nnunet_by_category.get(cat, 0)
        our_count = our_by_category.get(cat, 0)
        diff = nnunet_count - our_count
        diff_pct = (diff / nnunet_count * 100) if nnunet_count > 0 else 0
        print(f"{cat:<20} {nnunet_count:<20,} {our_count:<20,} {diff:<20,} ({diff_pct:+.2f}%)")
    
    # 주요 차이점 레이어 찾기
    print(f"\n주요 차이점 레이어 (상위 10개):")
    print(f"{'레이어 이름':<50} {'nnUNet':<15} {'우리':<15} {'차이':<15}")
    print("-" * 95)
    
    # 공통 레이어 비교
    common_layers = set(nnunet_params.keys()) & set(our_params.keys())
    layer_diffs = []
    for name in common_layers:
        diff = nnunet_params[name] - our_params[name]
        if abs(diff) > 0:
            layer_diffs.append((name, nnunet_params[name], our_params[name], diff))
    
    # nnUNet에만 있는 레이어
    nnunet_only = set(nnunet_params.keys()) - set(our_params.keys())
    for name in nnunet_only:
        layer_diffs.append((name, nnunet_params[name], 0, nnunet_params[name]))
    
    # 우리 모델에만 있는 레이어
    our_only = set(our_params.keys()) - set(nnunet_params.keys())
    for name in our_only:
        layer_diffs.append((name, 0, our_params[name], -our_params[name]))
    
    # 차이 절댓값 기준 정렬
    layer_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    
    for name, nnunet_count, our_count, diff in layer_diffs[:10]:
        print(f"{name:<50} {nnunet_count:<15,} {our_count:<15,} {diff:<15,}")


def create_nnunet_with_default_config(n_channels=4, n_classes=4, device='cuda'):
    """plans 파일 없이 기본 설정으로 nnUNet 모델 생성
    
    BraTS2021에 맞는 기본 설정 사용:
    - 5 stages (encoder 4 + bottleneck 1)
    - Features per stage: [32, 64, 128, 256, 320]
    - N conv per stage: [2, 2, 2, 2, 2]
    - InstanceNorm3d
    - LeakyReLU
    - Deep Supervision
    
    Args:
        n_channels: 입력 채널 수
        n_classes: 출력 클래스 수
        device: 디바이스
    
    Returns:
        (model, config_manager): 모델과 설정 정보 (config_manager는 None)
    """
    try:
        # nnUNet 모듈 import
        nnunet_path = os.path.join(os.path.dirname(__file__), '..', 'nnUNet')
        sys.path.insert(0, nnunet_path)
        
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        
        # BraTS2021에 맞는 기본 설정
        # nnUNet의 기본 3D UNet 설정
        n_stages = 5  # Encoder 4 stages + Bottleneck 1 stage
        features_per_stage = [32, 64, 128, 256, 320]  # 최대 320으로 제한
        n_conv_per_stage = [2, 2, 2, 2, 2]  # 각 stage마다 2개 conv
        n_conv_per_stage_decoder = [2, 2, 2, 2]  # Decoder는 4개 stage
        
        # 커널 크기 (모두 3x3x3)
        kernel_sizes = [[3, 3, 3]] * n_stages
        
        # Strides (첫 번째는 1, 나머지는 2)
        strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
        
        arch_kwargs = {
            'n_stages': n_stages,
            'features_per_stage': features_per_stage,
            'conv_op': 'torch.nn.modules.conv.Conv3d',
            'kernel_sizes': kernel_sizes,
            'strides': strides,
            'n_conv_per_stage': n_conv_per_stage,
            'n_conv_per_stage_decoder': n_conv_per_stage_decoder,
            'conv_bias': True,
            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': 'torch.nn.LeakyReLU',
            'nonlin_kwargs': {'inplace': True},
        }
        
        arch_class_name = 'dynamic_network_architectures.architectures.unet.PlainConvUNet'
        arch_kwargs_req_import = ['conv_op', 'norm_op', 'dropout_op', 'nonlin']
        
        print(f"  기본 설정:")
        print(f"    - Network: PlainConvUNet")
        print(f"    - Stages: {n_stages}")
        print(f"    - Features per stage: {features_per_stage}")
        print(f"    - N conv per stage: {n_conv_per_stage}")
        print(f"    - Normalization: InstanceNorm3d")
        print(f"    - Activation: LeakyReLU")
        print(f"    - Deep Supervision: True")
        
        network = get_network_from_plans(
            arch_class_name,
            arch_kwargs,
            arch_kwargs_req_import,
            n_channels,
            n_classes,
            allow_init=True,
            deep_supervision=True
        )
        
        network = network.to(device)
        network.eval()
        
        # 가짜 config_manager 생성 (구조 정보만 포함)
        class FakeConfigManager:
            def __init__(self, arch_kwargs):
                self.architecture_kwargs = {
                    'network_class_name': arch_class_name,
                    'arch_kwargs': arch_kwargs,
                    '_kw_requires_import': arch_kwargs_req_import
                }
        
        config_manager = FakeConfigManager(arch_kwargs)
        
        return network, config_manager
        
    except Exception as e:
        print(f"  ⚠️  기본 설정으로 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_nnunet_architecture_info(nnunet_model, config_manager):
    """nnUNet 모델의 아키텍처 정보 추출
    
    Args:
        nnunet_model: nnUNet 모델 객체
        config_manager: ConfigurationManager 객체
    
    Returns:
        dict: 아키텍처 정보
    """
    arch_info = {}
    
    try:
        # ConfigurationManager에서 아키텍처 정보 추출
        arch_kwargs_dict = config_manager.architecture_kwargs
        
        # arch_kwargs가 중첩된 경우 처리 (실제 ConfigurationManager)
        if 'arch_kwargs' in arch_kwargs_dict:
            arch_kwargs = arch_kwargs_dict['arch_kwargs']
            arch_info['network_class_name'] = arch_kwargs_dict.get('network_class_name', '')
        else:
            # FakeConfigManager의 경우 직접 arch_kwargs 사용
            arch_kwargs = arch_kwargs_dict
            arch_info['network_class_name'] = arch_kwargs.get('network_class_name', '')
        
        arch_info['features_per_stage'] = arch_kwargs.get('features_per_stage', [])
        arch_info['n_stages'] = arch_kwargs.get('n_stages', 0)
        arch_info['n_conv_per_stage'] = arch_kwargs.get('n_conv_per_stage', [])
        arch_info['n_conv_per_stage_decoder'] = arch_kwargs.get('n_conv_per_stage_decoder', [])
        arch_info['kernel_sizes'] = arch_kwargs.get('kernel_sizes', [])
        arch_info['strides'] = arch_kwargs.get('strides', [])
        arch_info['norm_op'] = arch_kwargs.get('norm_op', '')
        arch_info['norm_op_kwargs'] = arch_kwargs.get('norm_op_kwargs', {})
        arch_info['nonlin'] = arch_kwargs.get('nonlin', '')
        arch_info['nonlin_kwargs'] = arch_kwargs.get('nonlin_kwargs', {})
        arch_info['conv_bias'] = arch_kwargs.get('conv_bias', True)
        
        # Deep Supervision 확인 (모델 출력이 리스트인지 확인)
        if nnunet_model is not None:
            try:
                # nnUNet은 입력 크기가 2^n 배수여야 함 (64×64×64 사용)
                dummy_input = torch.randn(1, 4, 64, 64, 64)
                with torch.no_grad():
                    dummy_output = nnunet_model(dummy_input)
                arch_info['deep_supervision'] = isinstance(dummy_output, (list, tuple))
                if arch_info['deep_supervision']:
                    arch_info['deep_supervision_outputs'] = len(dummy_output)
            except:
                arch_info['deep_supervision'] = None
                arch_info['deep_supervision_outputs'] = None
        else:
            arch_info['deep_supervision'] = None
            arch_info['deep_supervision_outputs'] = None
        
    except Exception as e:
        print(f"  ⚠️  아키텍처 정보 추출 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return arch_info


def extract_our_model_architecture_info(model):
    """우리 모델의 아키텍처 정보 추출
    
    Args:
        model: 우리 UNet3D 모델 객체 또는 PlainConvUNet 객체
    
    Returns:
        dict: 아키텍처 정보
    """
    arch_info = {}
    
    try:
        # PlainConvUNet 객체인지 확인
        model_type_name = type(model).__name__
        is_plainconvunet = 'PlainConvUNet' in model_type_name or hasattr(model, 'encoder') and hasattr(model, 'decoder')
        
        if is_plainconvunet:
            # PlainConvUNet 객체인 경우 (외부 패키지에서 import한 경우)
            # encoder.stages를 통해 정보 추출
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'stages'):
                features_per_stage = []
                n_conv_per_stage = []
                
                # Encoder stages에서 features 추출
                for stage in model.encoder.stages:
                    # 각 stage의 첫 번째 conv 블록에서 out_channels 확인
                    if hasattr(stage, 'convs') and len(stage.convs) > 0:
                        first_conv = stage.convs[0]
                        if hasattr(first_conv, 'conv') and len(first_conv.conv) > 0:
                            out_channels = first_conv.conv[0].out_channels
                            features_per_stage.append(out_channels)
                            # conv 개수 확인
                            n_conv = len(stage.convs)
                            n_conv_per_stage.append(n_conv)
                
                # Bottleneck 확인
                if hasattr(model, 'bottleneck') and hasattr(model.bottleneck, 'convs'):
                    if len(model.bottleneck.convs) > 0:
                        first_conv = model.bottleneck.convs[0]
                        if hasattr(first_conv, 'conv') and len(first_conv.conv) > 0:
                            bottleneck_channels = first_conv.conv[0].out_channels
                            features_per_stage.append(bottleneck_channels)
                            n_conv = len(model.bottleneck.convs)
                            n_conv_per_stage.append(n_conv)
                
                arch_info['features_per_stage'] = features_per_stage
                arch_info['n_stages'] = len(features_per_stage)
                
                # Decoder stages에서 conv 개수 추출
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'stages'):
                    n_conv_per_stage_decoder = []
                    for stage in model.decoder.stages:
                        if hasattr(stage, 'convs'):
                            n_conv_per_stage_decoder.append(len(stage.convs))
                    arch_info['n_conv_per_stage_decoder'] = n_conv_per_stage_decoder
                else:
                    arch_info['n_conv_per_stage_decoder'] = [2] * (arch_info['n_stages'] - 1)
                
                arch_info['n_conv_per_stage'] = n_conv_per_stage
                
                # Normalization 확인 (encoder의 첫 번째 stage에서)
                if len(model.encoder.stages) > 0:
                    first_stage = model.encoder.stages[0]
                    if hasattr(first_stage, 'convs') and len(first_stage.convs) > 0:
                        first_conv_block = first_stage.convs[0]
                        if hasattr(first_conv_block, 'conv') and len(first_conv_block.conv) > 1:
                            norm_layer = first_conv_block.conv[1]
                            norm_type = type(norm_layer).__name__
                            if 'InstanceNorm' in norm_type:
                                arch_info['norm_op'] = 'torch.nn.modules.instancenorm.InstanceNorm3d'
                            elif 'BatchNorm' in norm_type:
                                arch_info['norm_op'] = 'torch.nn.modules.batchnorm.BatchNorm3d'
                            else:
                                arch_info['norm_op'] = norm_type
                            arch_info['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
                
                # Activation 확인
                if len(model.encoder.stages) > 0:
                    first_stage = model.encoder.stages[0]
                    if hasattr(first_stage, 'convs') and len(first_stage.convs) > 0:
                        first_conv_block = first_stage.convs[0]
                        if hasattr(first_conv_block, 'conv') and len(first_conv_block.conv) > 2:
                            act_layer = first_conv_block.conv[2]
                            act_type = type(act_layer).__name__
                            if 'LeakyReLU' in act_type:
                                arch_info['nonlin'] = 'torch.nn.LeakyReLU'
                            elif 'ReLU' in act_type:
                                arch_info['nonlin'] = 'torch.nn.ReLU'
                            else:
                                arch_info['nonlin'] = act_type
                            arch_info['nonlin_kwargs'] = {'inplace': True}
                
                # Conv bias 확인
                if len(model.encoder.stages) > 0:
                    first_stage = model.encoder.stages[0]
                    if hasattr(first_stage, 'convs') and len(first_stage.convs) > 0:
                        first_conv_block = first_stage.convs[0]
                        if hasattr(first_conv_block, 'conv') and len(first_conv_block.conv) > 0:
                            first_conv = first_conv_block.conv[0]
                            if hasattr(first_conv, 'bias') and first_conv.bias is not None:
                                arch_info['conv_bias'] = True
                            else:
                                arch_info['conv_bias'] = False
                
                # Deep Supervision 확인 (forward pass로 확인)
                try:
                    dummy_input = torch.randn(1, 4, 64, 64, 64)
                    with torch.no_grad():
                        output = model(dummy_input)
                    arch_info['deep_supervision'] = isinstance(output, (list, tuple))
                    if arch_info['deep_supervision']:
                        arch_info['deep_supervision_outputs'] = len(output)
                    else:
                        arch_info['deep_supervision_outputs'] = 1
                except:
                    arch_info['deep_supervision'] = model.deep_supervision if hasattr(model, 'deep_supervision') else False
                    arch_info['deep_supervision_outputs'] = 4 if arch_info['deep_supervision'] else 1
                
                # 커널 크기와 strides는 기본값 사용 (PlainConvUNet은 보통 3x3x3, stride는 첫 번째가 1, 나머지가 2)
                arch_info['kernel_sizes'] = [[3, 3, 3]] * arch_info['n_stages']
                arch_info['strides'] = [[1, 1, 1]] + [[2, 2, 2]] * (arch_info['n_stages'] - 1)
                
        else:
            # 기존 UNet3D 객체인 경우
            # 채널 정보 추출
            from models.channel_configs import get_unet_channels
            channels = get_unet_channels(model.size if hasattr(model, 'size') else 's')
            enc_channels = channels['enc']
            arch_info['features_per_stage'] = enc_channels  # [32, 64, 128, 256, 320]
            arch_info['n_stages'] = len(arch_info['features_per_stage'])  # 5
            arch_info['bottleneck_channel'] = channels['bottleneck']
            
            # 각 stage의 conv 개수 (DoubleConv3D = 2개 conv)
            arch_info['n_conv_per_stage'] = [2] * arch_info['n_stages']  # [2, 2, 2, 2, 2]
            arch_info['n_conv_per_stage_decoder'] = [2] * (arch_info['n_stages'] - 1)  # Decoder는 4개 stage
            
            # 커널 크기 (DoubleConv3D는 모두 3x3x3)
            arch_info['kernel_sizes'] = [[3, 3, 3]] * arch_info['n_stages']
            
            # Strides (Down3D는 MaxPool3d(2) 사용)
            arch_info['strides'] = [[1, 1, 1]] + [[2, 2, 2]] * (arch_info['n_stages'] - 2)
            
            # 정규화
            arch_info['norm_op'] = 'InstanceNorm3d' if (hasattr(model, 'norm') and model.norm == 'in') else 'BatchNorm3d'
            arch_info['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
            
            # 활성화 함수 (DoubleConv3D는 LeakyReLU 사용 - nnUNet과 동일)
            arch_info['nonlin'] = 'torch.nn.LeakyReLU'
            arch_info['nonlin_kwargs'] = {'inplace': True}
            
            # Conv bias (DoubleConv3D는 bias=True - nnUNet과 동일)
            arch_info['conv_bias'] = True
            
            # Deep Supervision
            arch_info['deep_supervision'] = model.deep_supervision if hasattr(model, 'deep_supervision') else False
            if arch_info['deep_supervision']:
                arch_info['deep_supervision_outputs'] = 4  # [main_output, ds2, ds3, ds4]
            else:
                arch_info['deep_supervision_outputs'] = 1
            
            # Bilinear upsampling 여부
            arch_info['bilinear'] = model.bilinear if hasattr(model, 'bilinear') else False
        
    except Exception as e:
        print(f"  ⚠️  아키텍처 정보 추출 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return arch_info


def analyze_architectural_differences(nnunet_info, our_info):
    """구조적 차이점 분석
    
    Args:
        nnunet_info: nnUNet 아키텍처 정보
        our_info: 우리 모델 아키텍처 정보
    
    Returns:
        dict: 차이점 분석 결과
    """
    differences = {
        'features_per_stage': None,
        'n_stages': None,
        'n_conv_per_stage': None,
        'kernel_sizes': None,
        'norm_op': None,
        'nonlin': None,
        'deep_supervision': None,
        'conv_bias': None,
    }
    
    print("\n" + "=" * 80)
    print("구조적 차이점 분석")
    print("=" * 80)
    
    # Features per stage 비교
    if nnunet_info.get('features_per_stage') and our_info.get('features_per_stage'):
        nnunet_features = nnunet_info['features_per_stage']
        our_features = our_info['features_per_stage']
        if nnunet_features != our_features:
            differences['features_per_stage'] = {
                'nnunet': nnunet_features,
                'ours': our_features,
                'match': False
            }
            print(f"\n❌ Features per stage 불일치:")
            print(f"   nnUNet: {nnunet_features}")
            print(f"   우리:   {our_features}")
        else:
            differences['features_per_stage'] = {'match': True}
            print(f"\n✅ Features per stage 일치: {nnunet_features}")
    
    # N stages 비교
    if nnunet_info.get('n_stages') and our_info.get('n_stages'):
        nnunet_stages = nnunet_info['n_stages']
        our_stages = our_info['n_stages']
        if nnunet_stages != our_stages:
            differences['n_stages'] = {
                'nnunet': nnunet_stages,
                'ours': our_stages,
                'match': False
            }
            print(f"\n❌ Stage 수 불일치:")
            print(f"   nnUNet: {nnunet_stages}")
            print(f"   우리:   {our_stages}")
        else:
            differences['n_stages'] = {'match': True}
            print(f"\n✅ Stage 수 일치: {nnunet_stages}")
    
    # N conv per stage 비교
    if nnunet_info.get('n_conv_per_stage') and our_info.get('n_conv_per_stage'):
        nnunet_conv = nnunet_info['n_conv_per_stage']
        our_conv = our_info['n_conv_per_stage']
        if nnunet_conv != our_conv:
            differences['n_conv_per_stage'] = {
                'nnunet': nnunet_conv,
                'ours': our_conv,
                'match': False
            }
            print(f"\n❌ Conv per stage 불일치:")
            print(f"   nnUNet: {nnunet_conv}")
            print(f"   우리:   {our_conv}")
        else:
            differences['n_conv_per_stage'] = {'match': True}
            print(f"\n✅ Conv per stage 일치: {nnunet_conv}")
    
    # Normalization 비교
    if nnunet_info.get('norm_op') and our_info.get('norm_op'):
        nnunet_norm = nnunet_info['norm_op']
        our_norm = our_info['norm_op']
        # InstanceNorm 확인
        nnunet_is_in = 'InstanceNorm' in nnunet_norm or 'instancenorm' in nnunet_norm.lower()
        our_is_in = 'InstanceNorm' in our_norm or 'instancenorm' in our_norm.lower()
        if nnunet_is_in != our_is_in:
            differences['norm_op'] = {
                'nnunet': nnunet_norm,
                'ours': our_norm,
                'match': False
            }
            print(f"\n❌ Normalization 불일치:")
            print(f"   nnUNet: {nnunet_norm}")
            print(f"   우리:   {our_norm}")
        else:
            differences['norm_op'] = {'match': True}
            print(f"\n✅ Normalization 일치: {nnunet_norm}")
    
    # Activation function 비교
    if nnunet_info.get('nonlin') and our_info.get('nonlin'):
        nnunet_nonlin = nnunet_info['nonlin']
        our_nonlin = our_info['nonlin']
        # LeakyReLU vs ReLU 확인
        nnunet_is_leaky = 'LeakyReLU' in nnunet_nonlin or 'leakyrelu' in nnunet_nonlin.lower()
        our_is_leaky = 'LeakyReLU' in our_nonlin or 'leakyrelu' in our_nonlin.lower()
        if nnunet_is_leaky != our_is_leaky:
            differences['nonlin'] = {
                'nnunet': nnunet_nonlin,
                'ours': our_nonlin,
                'match': False
            }
            print(f"\n❌ Activation function 불일치:")
            print(f"   nnUNet: {nnunet_nonlin} (LeakyReLU)")
            print(f"   우리:   {our_nonlin} (ReLU)")
            print(f"   ⚠️  권장: ReLU를 LeakyReLU로 변경 고려")
        else:
            differences['nonlin'] = {'match': True}
            print(f"\n✅ Activation function 일치: {nnunet_nonlin}")
    
    # Deep Supervision 비교
    if nnunet_info.get('deep_supervision') is not None and our_info.get('deep_supervision') is not None:
        nnunet_ds = nnunet_info['deep_supervision']
        our_ds = our_info['deep_supervision']
        if nnunet_ds != our_ds:
            differences['deep_supervision'] = {
                'nnunet': nnunet_ds,
                'ours': our_ds,
                'match': False
            }
            print(f"\n❌ Deep Supervision 불일치:")
            print(f"   nnUNet: {nnunet_ds}")
            print(f"   우리:   {our_ds}")
        else:
            differences['deep_supervision'] = {'match': True}
            print(f"\n✅ Deep Supervision 일치: {nnunet_ds}")
            if nnunet_ds:
                nnunet_ds_out = nnunet_info.get('deep_supervision_outputs', 0)
                our_ds_out = our_info.get('deep_supervision_outputs', 0)
                if nnunet_ds_out != our_ds_out:
                    print(f"   ⚠️  Deep Supervision 출력 개수 불일치:")
                    print(f"      nnUNet: {nnunet_ds_out}")
                    print(f"      우리:   {our_ds_out}")
    
    # Conv bias 비교
    if nnunet_info.get('conv_bias') is not None and our_info.get('conv_bias') is not None:
        nnunet_bias = nnunet_info['conv_bias']
        our_bias = our_info['conv_bias']
        if nnunet_bias != our_bias:
            differences['conv_bias'] = {
                'nnunet': nnunet_bias,
                'ours': our_bias,
                'match': False
            }
            print(f"\n❌ Conv bias 불일치:")
            print(f"   nnUNet: {nnunet_bias}")
            print(f"   우리:   {our_bias}")
            print(f"   ⚠️  권장: conv_bias를 {nnunet_bias}로 변경 고려")
        else:
            differences['conv_bias'] = {'match': True}
            print(f"\n✅ Conv bias 일치: {nnunet_bias}")
    
    return differences


def compare_model_outputs(nnunet_model, our_model, input_size, device='cuda'):
    """실제 모델 출력 비교
    
    Args:
        nnunet_model: nnUNet 모델
        our_model: 우리 모델
        input_size: 입력 크기 (B, C, H, W, D)
        device: 디바이스
    
    Returns:
        dict: 출력 비교 결과
    """
    print("\n" + "=" * 80)
    print("모델 출력 비교")
    print("=" * 80)
    
    comparison = {}
    
    try:
        # 동일한 입력 생성
        dummy_input = torch.randn(*input_size).to(device)
        
        # nnUNet forward pass
        nnunet_model.eval()
        with torch.no_grad():
            nnunet_output = nnunet_model(dummy_input)
        
        # 우리 모델 forward pass
        our_model.eval()
        with torch.no_grad():
            our_output = our_model(dummy_input)
        
        # 출력 타입 확인
        nnunet_is_list = isinstance(nnunet_output, (list, tuple))
        our_is_list = isinstance(our_output, (list, tuple))
        
        comparison['nnunet_output_type'] = 'list' if nnunet_is_list else 'tensor'
        comparison['our_output_type'] = 'list' if our_is_list else 'tensor'
        
        # Deep Supervision 출력 개수
        if nnunet_is_list:
            comparison['nnunet_output_count'] = len(nnunet_output)
            nnunet_main_output = nnunet_output[0]
        else:
            comparison['nnunet_output_count'] = 1
            nnunet_main_output = nnunet_output
        
        if our_is_list:
            comparison['our_output_count'] = len(our_output)
            our_main_output = our_output[0]
        else:
            comparison['our_output_count'] = 1
            our_main_output = our_output
        
        # Main output shape 비교
        nnunet_shape = nnunet_main_output.shape
        our_shape = our_main_output.shape
        
        comparison['nnunet_shape'] = nnunet_shape
        comparison['our_shape'] = our_shape
        comparison['shape_match'] = nnunet_shape == our_shape
        
        print(f"\n출력 타입:")
        print(f"   nnUNet: {comparison['nnunet_output_type']} ({comparison['nnunet_output_count']} outputs)")
        print(f"   우리:   {comparison['our_output_type']} ({comparison['our_output_count']} outputs)")
        
        print(f"\nMain output shape:")
        print(f"   nnUNet: {nnunet_shape}")
        print(f"   우리:   {our_shape}")
        
        if comparison['shape_match']:
            print(f"   ✅ Shape 일치")
        else:
            print(f"   ❌ Shape 불일치")
        
        # 출력 통계 비교
        if comparison['shape_match']:
            nnunet_mean = nnunet_main_output.mean().item()
            nnunet_std = nnunet_main_output.std().item()
            our_mean = our_main_output.mean().item()
            our_std = our_main_output.std().item()
            
            comparison['nnunet_stats'] = {'mean': nnunet_mean, 'std': nnunet_std}
            comparison['our_stats'] = {'mean': our_mean, 'std': our_std}
            
            print(f"\n출력 통계:")
            print(f"   nnUNet - Mean: {nnunet_mean:.6f}, Std: {nnunet_std:.6f}")
            print(f"   우리   - Mean: {our_mean:.6f}, Std: {our_std:.6f}")
            
            mean_diff = abs(nnunet_mean - our_mean)
            std_diff = abs(nnunet_std - our_std)
            
            print(f"   Mean 차이: {mean_diff:.6f}")
            print(f"   Std 차이:  {std_diff:.6f}")
            
            # Deep Supervision 출력 비교
            if nnunet_is_list and our_is_list:
                print(f"\nDeep Supervision 출력 비교:")
                for i in range(min(len(nnunet_output), len(our_output))):
                    nnunet_ds_shape = nnunet_output[i].shape
                    our_ds_shape = our_output[i].shape
                    match = nnunet_ds_shape == our_ds_shape
                    status = "✅" if match else "❌"
                    print(f"   DS{i}: nnUNet {nnunet_ds_shape} vs 우리 {our_ds_shape} {status}")
        
    except Exception as e:
        print(f"\n⚠️  출력 비교 중 오류: {e}")
        import traceback
        traceback.print_exc()
        comparison['error'] = str(e)
    
    return comparison


def create_nnunet_model(data_path, n_channels=4, n_classes=4, device='cuda'):
    """nnUNet 모델 생성 (plans 파일에서)
    
    Args:
        data_path: BraTS2021 데이터셋 경로
        n_channels: 입력 채널 수
        n_classes: 출력 클래스 수
        device: 디바이스
    """
    try:
        # nnUNet 모듈 import
        nnunet_path = os.path.join(os.path.dirname(__file__), '..', 'nnUNet')
        sys.path.insert(0, nnunet_path)
        
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
        
        # 환경 변수 확인
        nnunet_preprocessed = os.environ.get('nnUNet_preprocessed', None)
        if nnunet_preprocessed is None:
            print("⚠️  nnUNet_preprocessed 환경 변수가 설정되지 않았습니다.")
            print("   plans 파일을 찾을 수 없어 nnUNet 모델을 생성할 수 없습니다.")
            return None
        
        # 데이터셋 이름 찾기 (data_path에서 추출하거나 자동 감지)
        # BraTS2021의 경우 일반적으로 DatasetXXX_BraTS2021 형태
        # 또는 plans 파일을 직접 찾기
        print(f"nnUNet_preprocessed 경로: {nnunet_preprocessed}")
        
        # plans 파일 찾기 시도 (일반적인 BraTS2021 데이터셋 이름들)
        possible_dataset_names = [
            'Dataset001_BraTS2021',
            'Dataset002_BraTS2021', 
            'BraTS2021',
            'BRATS2021'
        ]
        
        plans_file = None
        dataset_name = None
        
        for ds_name in possible_dataset_names:
            test_plans = join(nnunet_preprocessed, ds_name, 'nnUNetPlans.json')
            if isfile(test_plans):
                plans_file = test_plans
                dataset_name = ds_name
                break
        
        if plans_file is None:
            # 모든 데이터셋 폴더 확인
            if os.path.isdir(nnunet_preprocessed):
                for item in os.listdir(nnunet_preprocessed):
                    test_plans = join(nnunet_preprocessed, item, 'nnUNetPlans.json')
                    if isfile(test_plans):
                        plans_file = test_plans
                        dataset_name = item
                        print(f"  발견된 데이터셋: {dataset_name}")
                        break
        
        if plans_file is None or not isfile(plans_file):
            print("⚠️  nnUNet plans 파일을 찾을 수 없습니다.")
            print(f"   예상 경로: {nnunet_preprocessed}/*/nnUNetPlans.json")
            print("   plans 파일이 없으면 nnUNet으로 데이터셋을 먼저 전처리해야 합니다.")
            return None
        
        print(f"  Plans 파일 발견: {plans_file}")
        
        # Plans 로드
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        
        # dataset.json 로드
        dataset_json_file = join(nnunet_preprocessed, dataset_name, 'dataset.json')
        if not isfile(dataset_json_file):
            print(f"⚠️  dataset.json을 찾을 수 없습니다: {dataset_json_file}")
            return None
        
        dataset_json = load_json(dataset_json_file)
        
        # 3d_fullres configuration 사용
        configuration_name = '3d_fullres'
        if configuration_name not in plans['configurations']:
            print(f"⚠️  Configuration '{configuration_name}'을 찾을 수 없습니다.")
            print(f"   사용 가능한 configurations: {list(plans['configurations'].keys())}")
            # 첫 번째 configuration 사용
            configuration_name = list(plans['configurations'].keys())[0]
            print(f"   대신 '{configuration_name}'을 사용합니다.")
        
        config_manager = plans_manager.get_configuration(configuration_name)
        
        # 입력 채널 수 결정
        num_input_channels = determine_num_input_channels(plans_manager, config_manager, dataset_json)
        print(f"  입력 채널 수: {num_input_channels}")
        
        # 출력 채널 수 (label_manager에서)
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_output_channels = label_manager.num_segmentation_heads
        print(f"  출력 채널 수: {num_output_channels}")
        
        # 네트워크 생성
        architecture_kwargs = config_manager.architecture_kwargs
        network = get_network_from_plans(
            architecture_kwargs['network_class_name'],
            architecture_kwargs['arch_kwargs'],
            architecture_kwargs.get('_kw_requires_import', []),
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=True  # nnUNet은 기본적으로 Deep Supervision 사용
        )
        
        print(f"  ✅ nnUNet 모델 생성 완료")
        return network
        
    except ImportError as e:
        print(f"⚠️  nnUNet 모듈 import 실패: {e}")
        print("   nnUNet이 설치되어 있는지 확인하세요.")
        return None
    except Exception as e:
        print(f"⚠️  nnUNet 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(data_path, device='cuda'):
    """우리 unet3d_s 모델과 nnUNet 구조 비교"""
    
    print("=" * 80)
    print("nnUNet vs 우리 unet3d_s 모델 비교")
    print("=" * 80)
    
    # 시드 설정
    set_seed(24)
    
    # 입력 채널 수 (BraTS2021은 4채널: T1, T1CE, T2, FLAIR)
    n_channels = 4
    n_classes = 4  # Background, Edema, Non-enhancing, Enhancing
    
    # 입력 크기 정의
    # nnUNet과의 공정한 FLOPs 비교를 위해 여기서는 64×64×64로 고정
    # (실제 학습/추론에서는 INPUT_SIZE_3D를 사용)
    input_size = (1, n_channels, 64, 64, 64)
    
    # 우리 모델 생성 (unet3d_s, InstanceNorm, Deep Supervision)
    print("\n[1] 우리 unet3d_s 모델 생성 중...")
    our_model = get_model(
        model_name='unet3d_s',
        n_channels=n_channels,
        n_classes=n_classes,
        norm='in',  # InstanceNorm
        dim='3d',
        coord_type='none'
    )
    our_model = our_model.to(device)
    our_model.eval()
    
    # 파라미터 수 계산
    our_params = count_parameters(our_model)
    print(f"  파라미터 수: {our_params:,}")
    
    # FLOPs 계산 (nnUNet과 동일하게 64×64×64 기준)
    print(f"  입력 크기 (FLOPs 기준): {input_size}")
    try:
        our_flops = calculate_flops(our_model, input_size=input_size)
        print(f"  FLOPs: {our_flops:,}")
    except Exception as e:
        print(f"  FLOPs 계산 실패: {e}")
        our_flops = 0
    
    # 우리 모델 구조 정보 추출
    print("\n[2] 우리 모델 구조 정보 추출 중...")
    our_arch_info = extract_our_model_architecture_info(our_model)
    print(f"  Features per stage: {our_arch_info.get('features_per_stage', [])}")
    print(f"  N stages: {our_arch_info.get('n_stages', 0)}")
    print(f"  N conv per stage: {our_arch_info.get('n_conv_per_stage', [])}")
    print(f"  Normalization: {our_arch_info.get('norm_op', '')}")
    print(f"  Activation: {our_arch_info.get('nonlin', '')}")
    print(f"  Deep Supervision: {our_arch_info.get('deep_supervision', False)}")
    
    # nnUNet 실제 모델 생성 시도
    print("\n[3] nnUNet 실제 모델 생성 시도...")
    nnunet_model = None
    config_manager = None
    
    try:
        # nnUNet 모듈 import
        nnunet_path = os.path.join(os.path.dirname(__file__), '..', 'nnUNet')
        sys.path.insert(0, nnunet_path)
        
        # 필요한 패키지 확인 및 안내
        try:
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
            from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
        except ImportError as e:
            missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
            print(f"⚠️  필요한 패키지가 설치되지 않았습니다: {missing_module}")
            print("\n" + "=" * 80)
            print("nnUNet 로드를 위한 패키지 설치 안내")
            print("=" * 80)
            print("\n다음 명령어 중 하나를 실행하세요:")
            print("\n[방법 1] 필요한 패키지만 설치:")
            print("  pip install batchgenerators dynamic-network-architectures")
            print("\n[방법 2] nnUNet 전체 설치 (권장):")
            print(f"  cd {nnunet_path}")
            print("  pip install -e .")
            print("\n[방법 3] 최소 필수 패키지만:")
            print("  pip install 'batchgenerators>=0.25.1' 'dynamic-network-architectures>=0.4.1,<0.5'")
            print("\n" + "=" * 80)
            print("패키지 설치 후 다시 실행하세요.")
            print("=" * 80)
            raise
        
        # 환경 변수 확인 및 자동 탐색
        nnunet_preprocessed = os.environ.get('nnUNet_preprocessed', None)
        if nnunet_preprocessed is None:
            print("⚠️  nnUNet_preprocessed 환경 변수가 설정되지 않았습니다.")
            print("   일반적인 경로를 자동으로 탐색합니다...")
            
            # 일반적인 nnUNet_preprocessed 경로들
            possible_paths = [
                os.path.expanduser('~/nnUNet_preprocessed'),
                os.path.expanduser('~/data/nnUNet_preprocessed'),
                '/data/nnUNet_preprocessed',
                '/home/nas/nnUNet_preprocessed',
                '/home/nas/data/nnUNet_preprocessed',
            ]
            
            for path in possible_paths:
                if os.path.isdir(path):
                    print(f"  발견된 경로: {path}")
                    nnunet_preprocessed = path
                    # 임시로 환경 변수 설정 (이 세션에서만)
                    os.environ['nnUNet_preprocessed'] = path
                    break
            
            if nnunet_preprocessed is None:
                print("   plans 파일을 찾을 수 없습니다.")
                print("   기본 설정으로 nnUNet 모델을 생성합니다...")
                # plans 파일 없이 기본 설정으로 모델 생성
                nnunet_model, config_manager = create_nnunet_with_default_config(
                    n_channels, n_classes, device
                )
                if nnunet_model is not None:
                    print("  ✅ 기본 설정으로 nnUNet 모델 생성 완료")
                else:
                    print("\n환경 변수를 설정하려면:")
                    print("  export nnUNet_preprocessed=/path/to/nnUNet_preprocessed")
                    print("\n또는 plans 파일이 있는 경로를 직접 지정하세요.")
            else:
                print(f"nnUNet_preprocessed 경로: {nnunet_preprocessed}")
        
        # plans 파일 찾기 또는 기본 설정으로 모델 생성
        if nnunet_preprocessed and (nnunet_model is None or config_manager is None):
            plans_file = None
            dataset_name = None
            
            possible_dataset_names = [
                'Dataset001_BraTS2021',
                'Dataset002_BraTS2021', 
                'BraTS2021',
                'BRATS2021'
            ]
            
            for ds_name in possible_dataset_names:
                test_plans = join(nnunet_preprocessed, ds_name, 'nnUNetPlans.json')
                if isfile(test_plans):
                    plans_file = test_plans
                    dataset_name = ds_name
                    break
            
            if plans_file is None:
                if os.path.isdir(nnunet_preprocessed):
                    for item in os.listdir(nnunet_preprocessed):
                        test_plans = join(nnunet_preprocessed, item, 'nnUNetPlans.json')
                        if isfile(test_plans):
                            plans_file = test_plans
                            dataset_name = item
                            print(f"  발견된 데이터셋: {dataset_name}")
                            break
            
            if plans_file and isfile(plans_file):
                print(f"  Plans 파일 발견: {plans_file}")
                
                # Plans 로드
                plans = load_json(plans_file)
                plans_manager = PlansManager(plans)
                
                # dataset.json 로드
                dataset_json_file = join(nnunet_preprocessed, dataset_name, 'dataset.json')
                if isfile(dataset_json_file):
                    dataset_json = load_json(dataset_json_file)
                    
                    # 3d_fullres configuration 사용
                    configuration_name = '3d_fullres'
                    if configuration_name not in plans['configurations']:
                        configuration_name = list(plans['configurations'].keys())[0]
                        print(f"  Configuration '{configuration_name}' 사용")
                    
                    config_manager = plans_manager.get_configuration(configuration_name)
                    
                    # 입력 채널 수 결정
                    num_input_channels = determine_num_input_channels(plans_manager, config_manager, dataset_json)
                    print(f"  입력 채널 수: {num_input_channels}")
                    
                    # 출력 채널 수
                    label_manager = plans_manager.get_label_manager(dataset_json)
                    num_output_channels = label_manager.num_segmentation_heads
                    print(f"  출력 채널 수: {num_output_channels}")
                    
                    # 네트워크 생성
                    architecture_kwargs = config_manager.architecture_kwargs
                    nnunet_model = get_network_from_plans(
                        architecture_kwargs['network_class_name'],
                        architecture_kwargs['arch_kwargs'],
                        architecture_kwargs.get('_kw_requires_import', []),
                        num_input_channels,
                        num_output_channels,
                        allow_init=True,
                        deep_supervision=True
                    )
                    
                    nnunet_model = nnunet_model.to(device)
                    nnunet_model.eval()
                    print(f"  ✅ nnUNet 모델 생성 완료")
                    
    except Exception as e:
        print(f"  ⚠️  nnUNet 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # nnUNet 모델이 생성된 경우 상세 비교 수행
    if nnunet_model is not None and config_manager is not None:
        # nnUNet 파라미터 및 FLOPs 계산
        nnunet_params = count_parameters(nnunet_model)
        # nnUNet은 입력 크기가 2^n 배수여야 하므로 FLOPs 계산용으로 64×64×64 사용
        nnunet_flops_input_size = (1, n_channels, 64, 64, 64)
        try:
            nnunet_flops = calculate_flops(nnunet_model, input_size=nnunet_flops_input_size)
        except Exception as e:
            print(f"  ⚠️  nnUNet FLOPs 계산 실패: {e}")
            nnunet_flops = 0
        
        print(f"\n[4] nnUNet 모델 정보:")
        print(f"  파라미터 수: {nnunet_params:,}")
        print(f"  FLOPs: {nnunet_flops:,}")
        
        # nnUNet 구조 정보 추출
        print("\n[5] nnUNet 구조 정보 추출 중...")
        nnunet_arch_info = extract_nnunet_architecture_info(nnunet_model, config_manager)
        print(f"  Features per stage: {nnunet_arch_info.get('features_per_stage', [])}")
        print(f"  N stages: {nnunet_arch_info.get('n_stages', 0)}")
        print(f"  N conv per stage: {nnunet_arch_info.get('n_conv_per_stage', [])}")
        print(f"  Normalization: {nnunet_arch_info.get('norm_op', '')}")
        print(f"  Activation: {nnunet_arch_info.get('nonlin', '')}")
        print(f"  Deep Supervision: {nnunet_arch_info.get('deep_supervision', False)}")
        
        # 실제 모델 출력 비교 (구조 분석 전에 실행하여 Deep Supervision 정보 업데이트)
        # nnUNet은 입력 크기가 2^n 배수여야 하므로 64×64×64 사용
        print("\n[6] 실제 모델 출력 비교...")
        dummy_input_size = (1, n_channels, 64, 64, 64)
        output_comparison = compare_model_outputs(nnunet_model, our_model, dummy_input_size, device)
        
        # 출력 비교 결과를 사용하여 Deep Supervision 정보 업데이트
        if output_comparison:
            nnunet_output_type = output_comparison.get('nnunet_output_type', '')
            our_output_type = output_comparison.get('our_output_type', '')
            nnunet_is_list = 'list' in nnunet_output_type.lower() or 'tuple' in nnunet_output_type.lower()
            our_is_list = 'list' in our_output_type.lower() or 'tuple' in our_output_type.lower()
            
            if nnunet_is_list:
                nnunet_arch_info['deep_supervision'] = True
                nnunet_arch_info['deep_supervision_outputs'] = output_comparison.get('nnunet_output_count', 0)
            if our_is_list:
                our_arch_info['deep_supervision'] = True
                our_arch_info['deep_supervision_outputs'] = output_comparison.get('our_output_count', 0)
        
        # 구조적 차이점 분석 (Deep Supervision 정보 업데이트 후)
        print("\n[7] 구조적 차이점 분석...")
        differences = analyze_architectural_differences(nnunet_arch_info, our_arch_info)
        
        # 파라미터 및 FLOPs 비교
        print("\n[8] 파라미터 및 FLOPs 비교:")
        param_diff = abs(our_params - nnunet_params)
        param_diff_pct = ((our_params - nnunet_params) / nnunet_params * 100) if nnunet_params > 0 else 0
        print(f"  파라미터 차이: {param_diff:,} ({param_diff_pct:+.2f}%)")
        
        if abs(param_diff_pct) < 1.0:
            print("  ✅ 파라미터 수가 거의 동일합니다! (1% 이내)")
        else:
            print("  ⚠️  파라미터 수에 차이가 있습니다.")
        
        if nnunet_flops > 0:
            flops_diff = abs(our_flops - nnunet_flops)
            flops_diff_pct = ((our_flops - nnunet_flops) / nnunet_flops * 100) if nnunet_flops > 0 else 0
            print(f"  FLOPs 차이: {flops_diff:,} ({flops_diff_pct:+.2f}%)")
            
            if abs(flops_diff_pct) < 1.0:
                print("  ✅ FLOPs가 거의 동일합니다! (1% 이내)")
            else:
                print("  ⚠️  FLOPs에 차이가 있습니다.")
                print(f"  ⚠️  참고: 입력 크기가 다릅니다!")
                print(f"      - 우리 모델 FLOPs: {input_size[2]}×{input_size[3]}×{input_size[4]} 입력으로 계산")
                print(f"      - nnUNet FLOPs: 64×64×64 입력으로 계산")
                print(f"      - 공정한 비교를 위해 동일한 입력 크기로 재계산이 필요합니다.")
        
        # 동일한 입력 크기로 FLOPs 재계산
        print("\n[9] 동일한 입력 크기로 FLOPs 재계산...")
        comparison_sizes = [
            (1, n_channels, 64, 64, 64),
            (1, n_channels, 128, 128, 128),
        ]
        print(f"{'입력 크기':<20} {'nnUNet FLOPs':<25} {'우리 FLOPs':<25} {'차이(%)':<15}")
        print("-" * 85)
        for comp_size in comparison_sizes:
            try:
                nnunet_flops_comp = calculate_flops(nnunet_model, input_size=comp_size)
                our_flops_comp = calculate_flops(our_model, input_size=comp_size)
                diff_pct_comp = ((our_flops_comp - nnunet_flops_comp) / nnunet_flops_comp * 100) if nnunet_flops_comp > 0 else 0
                size_str = f"{comp_size[2]}×{comp_size[3]}×{comp_size[4]}"
                print(f"{size_str:<20} {nnunet_flops_comp:<25,.0f} {our_flops_comp:<25,.0f} {diff_pct_comp:+.2f}%")
            except Exception as e:
                size_str = f"{comp_size[2]}×{comp_size[3]}×{comp_size[4]}"
                print(f"{size_str:<20} Error: {str(e)[:50]}")
        
        # 레이어별 파라미터 분석
        print("\n[10] 레이어별 파라미터 수 분석...")
        analyze_layer_wise_parameters(nnunet_model, our_model)
        
        # 최종 요약 리포트
        print("\n" + "=" * 80)
        print("최종 비교 요약")
        print("=" * 80)
        
        # 구조 비교 테이블
        print("\n구조 비교 테이블:")
        print("-" * 80)
        print(f"{'항목':<30} {'nnUNet':<25} {'우리 모델':<25}")
        print("-" * 80)
        
        # Features per stage
        nnunet_features = str(nnunet_arch_info.get('features_per_stage', []))
        our_features = str(our_arch_info.get('features_per_stage', []))
        feat_diff = differences.get('features_per_stage') if isinstance(differences, dict) else None
        features_match = (isinstance(feat_diff, dict) and feat_diff.get('match', False))
        status = "✅" if features_match else "❌"
        print(f"{'Features per stage':<30} {nnunet_features:<25} {our_features:<25} {status}")
        
        # N stages
        nnunet_stages = str(nnunet_arch_info.get('n_stages', 0))
        our_stages = str(our_arch_info.get('n_stages', 0))
        stages_diff = differences.get('n_stages') if isinstance(differences, dict) else None
        stages_match = (isinstance(stages_diff, dict) and stages_diff.get('match', False))
        status = "✅" if stages_match else "❌"
        print(f"{'N stages':<30} {nnunet_stages:<25} {our_stages:<25} {status}")
        
        # N conv per stage
        nnunet_conv = str(nnunet_arch_info.get('n_conv_per_stage', []))
        our_conv = str(our_arch_info.get('n_conv_per_stage', []))
        conv_diff = differences.get('n_conv_per_stage') if isinstance(differences, dict) else None
        conv_match = (isinstance(conv_diff, dict) and conv_diff.get('match', False))
        status = "✅" if conv_match else "❌"
        print(f"{'N conv per stage':<30} {nnunet_conv:<25} {our_conv:<25} {status}")
        
        # Normalization
        nnunet_norm = nnunet_arch_info.get('norm_op', '')
        our_norm = our_arch_info.get('norm_op', '')
        nnunet_norm_str = nnunet_norm.split('.')[-1] if nnunet_norm else ''
        our_norm_str = our_norm.split('.')[-1] if our_norm else ''
        norm_diff = differences.get('norm_op')
        norm_match = (norm_diff is not None and isinstance(norm_diff, dict) and norm_diff.get('match', False)) if norm_diff else False
        status = "✅" if norm_match else "❌"
        print(f"{'Normalization':<30} {nnunet_norm_str:<25} {our_norm_str:<25} {status}")
        
        # Activation
        nnunet_act = nnunet_arch_info.get('nonlin', '')
        our_act = our_arch_info.get('nonlin', '')
        nnunet_act_str = nnunet_act.split('.')[-1] if nnunet_act else ''
        our_act_str = our_act.split('.')[-1] if our_act else ''
        act_diff = differences.get('nonlin')
        act_match = (act_diff is not None and isinstance(act_diff, dict) and act_diff.get('match', False)) if act_diff else False
        status = "✅" if act_match else "❌"
        print(f"{'Activation':<30} {nnunet_act_str:<25} {our_act_str:<25} {status}")
        
        # Deep Supervision
        nnunet_ds = str(nnunet_arch_info.get('deep_supervision', False))
        our_ds = str(our_arch_info.get('deep_supervision', False))
        ds_diff = differences.get('deep_supervision')
        if ds_diff is not None and isinstance(ds_diff, dict):
            ds_match = ds_diff.get('match', False)
        else:
            ds_match = False
        status = "✅" if ds_match else "❌"
        print(f"{'Deep Supervision':<30} {nnunet_ds:<25} {our_ds:<25} {status}")
        
        # Conv bias
        nnunet_bias = str(nnunet_arch_info.get('conv_bias', True))
        our_bias = str(our_arch_info.get('conv_bias', False))
        bias_match = differences.get('conv_bias', {}).get('match', False)
        status = "✅" if bias_match else "❌"
        print(f"{'Conv bias':<30} {nnunet_bias:<25} {our_bias:<25} {status}")
        
        print("-" * 80)
        
        # 파라미터 및 FLOPs 비교
        print("\n파라미터 및 FLOPs 비교:")
        print("-" * 80)
        print(f"{'항목':<30} {'nnUNet':<25} {'우리 모델':<25} {'차이':<15}")
        print("-" * 80)
        print(f"{'파라미터 수':<30} {nnunet_params:>24,} {our_params:>24,} {param_diff_pct:>+13.2f}%")
        if nnunet_flops > 0:
            print(f"{'FLOPs':<30} {nnunet_flops:>24,} {our_flops:>24,} {flops_diff_pct:>+13.2f}%")
        print("-" * 80)
        
        # 차이점 요약
        all_match = True
        for key, value in differences.items():
            if value is not None and isinstance(value, dict) and not value.get('match', True):
                all_match = False
                break
        
        print("\n차이점 요약:")
        if all_match and abs(param_diff_pct) < 1.0:
            print("  ✅ 전체적으로 nnUNet과 거의 동일한 구현입니다!")
        else:
            print("  ⚠️  다음 차이점이 발견되었습니다:")
            if differences.get('nonlin', {}).get('match') == False:
                print("     - 활성화 함수: 우리는 ReLU 사용, nnUNet은 LeakyReLU 사용")
                print("       → 권장: ReLU를 LeakyReLU로 변경 고려")
            if differences.get('conv_bias', {}).get('match') == False:
                print("     - Conv bias: 우리는 False, nnUNet은 True")
                print("       → 권장: conv_bias 설정 확인 필요")
            if differences.get('features_per_stage', {}).get('match') == False:
                print("     - Features per stage 불일치")
            if differences.get('n_stages', {}).get('match') == False:
                print("     - Stage 수 불일치")
            if differences.get('n_conv_per_stage', {}).get('match') == False:
                print("     - Conv per stage 불일치")
            if not output_comparison.get('shape_match', True):
                print("     - 출력 shape 불일치")
        
        # 권장 사항
        print("\n권장 사항:")
        recommendations = []
        if differences.get('nonlin', {}).get('match') == False:
            recommendations.append("1. 활성화 함수를 LeakyReLU로 변경 (models/model_3d_unet.py의 DoubleConv3D)")
        if differences.get('conv_bias', {}).get('match') == False:
            recommendations.append("2. Conv3d의 bias 파라미터를 True로 변경 (models/model_3d_unet.py)")
        if abs(param_diff_pct) > 1.0:
            recommendations.append("3. 파라미터 수 차이 원인 확인 필요")
        
        if recommendations:
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print("  모든 항목이 nnUNet과 일치합니다!")
        
        return {
            'our_params': our_params,
            'our_flops': our_flops,
            'nnunet_params': nnunet_params,
            'nnunet_flops': nnunet_flops,
            'our_arch_info': our_arch_info,
            'nnunet_arch_info': nnunet_arch_info,
            'differences': differences,
            'output_comparison': output_comparison
        }
    else:
        print("\n⚠️  nnUNet 모델을 직접 생성할 수 없습니다.")
        print("  우리 모델의 구조가 nnUNet과 일치하는지 확인:")
        print(f"    - Encoder 채널: [32, 64, 128, 256, 320]")
        print(f"    - Bottleneck: 320")
        print(f"    - Normalization: InstanceNorm")
        print(f"    - Deep Supervision: 사용")
        print(f"    - Activation: ReLU (nnUNet은 LeakyReLU)")
        
        return {
            'our_params': our_params,
            'our_flops': our_flops,
            'nnunet_params': None,
            'nnunet_flops': None,
            'our_arch_info': our_arch_info,
            'nnunet_arch_info': None,
            'differences': None,
            'output_comparison': None
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nnUNet과 우리 unet3d_s 모델 비교')
    parser.add_argument('--data_path', type=str, 
                       default='/home/nas/vision_data/BRATS/BRATS2021/BraTS2021_Training_Data',
                       help='BraTS2021 데이터셋 경로')
    parser.add_argument('--device', type=str, default='cuda', help='디바이스 (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
        args.device = 'cpu'
    
    compare_models(args.data_path, device=args.device)

#!/usr/bin/env python3
"""
nnUNet과 우리 모델의 차이점 심도 분석 스크립트

분석 항목:
1. 동일한 입력 크기로 FLOPs 재계산
2. 파라미터 수 차이의 레이어별 원인 분석
3. Deep Supervision 비교 로직 개선
4. 모델 구조의 미세한 차이점 분석
"""

import torch
import sys
import os
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.experiment_utils import get_model, calculate_flops
from compare_with_nnunet import (
    create_nnunet_with_default_config,
    count_parameters,
    extract_nnunet_architecture_info,
    extract_our_model_architecture_info
)


def analyze_layer_wise_parameters(nnunet_model, our_model):
    """레이어별 파라미터 수 비교 분석"""
    print("\n" + "=" * 80)
    print("레이어별 파라미터 수 분석")
    print("=" * 80)
    
    def get_layer_params(model, prefix=''):
        """모델의 레이어별 파라미터 수 추출"""
        params_dict = OrderedDict()
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
        if 'encoder' in name_lower or 'enc' in name_lower:
            return 'Encoder'
        elif 'decoder' in name_lower or 'dec' in name_lower:
            return 'Decoder'
        elif 'bottleneck' in name_lower:
            return 'Bottleneck'
        elif 'out' in name_lower or 'head' in name_lower:
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
    print(f"{'카테고리':<20} {'nnUNet':<20} {'우리':<20} {'차이':<20}")
    print("-" * 80)
    
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


def compare_flops_with_same_input_size(nnunet_model, our_model, input_sizes):
    """동일한 입력 크기로 FLOPs 비교"""
    print("\n" + "=" * 80)
    print("동일한 입력 크기로 FLOPs 비교")
    print("=" * 80)
    
    print(f"\n{'입력 크기':<25} {'nnUNet FLOPs':<25} {'우리 FLOPs':<25} {'차이':<25} {'차이(%)':<15}")
    print("-" * 115)
    
    for input_size in input_sizes:
        try:
            nnunet_flops = calculate_flops(nnunet_model, input_size=input_size)
            our_flops = calculate_flops(our_model, input_size=input_size)
            
            diff = our_flops - nnunet_flops
            diff_pct = (diff / nnunet_flops * 100) if nnunet_flops > 0 else 0
            
            size_str = f"{input_size[2]}×{input_size[3]}×{input_size[4]}"
            print(f"{size_str:<25} {nnunet_flops:<25,.0f} {our_flops:<25,.0f} {diff:<25,.0f} {diff_pct:+.2f}%")
        except Exception as e:
            print(f"{str(input_size):<25} Error: {e}")


def analyze_model_structure_differences(nnunet_model, our_model):
    """모델 구조의 미세한 차이점 분석"""
    print("\n" + "=" * 80)
    print("모델 구조 미세 차이점 분석")
    print("=" * 80)
    
    # 모델 구조 출력
    print("\n[nnUNet 모델 구조]")
    print(nnunet_model)
    
    print("\n[우리 모델 구조]")
    print(our_model)
    
    # Forward pass로 실제 구조 확인
    print("\n[Forward pass 구조 확인]")
    dummy_input = torch.randn(1, 4, 64, 64, 64).cuda()
    
    nnunet_model.eval()
    our_model.eval()
    
    with torch.no_grad():
        nnunet_output = nnunet_model(dummy_input)
        our_output = our_model(dummy_input)
    
    print(f"\nnnUNet 출력 타입: {type(nnunet_output)}")
    if isinstance(nnunet_output, (list, tuple)):
        print(f"  출력 개수: {len(nnunet_output)}")
        for i, out in enumerate(nnunet_output):
            print(f"    Output {i}: {out.shape}")
    
    print(f"\n우리 모델 출력 타입: {type(our_output)}")
    if isinstance(our_output, (list, tuple)):
        print(f"  출력 개수: {len(our_output)}")
        for i, out in enumerate(our_output):
            print(f"    Output {i}: {out.shape}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_channels = 4
    n_classes = 4
    
    print("=" * 80)
    print("nnUNet vs 우리 모델 차이점 심도 분석")
    print("=" * 80)
    
    # 모델 생성
    print("\n[1] 모델 생성 중...")
    nnunet_model = create_nnunet_with_default_config(n_channels, n_classes, device)
    our_model = get_model(
        model_name='unet3d_s',
        n_channels=n_channels,
        n_classes=n_classes,
        norm='in',
        dim='3d',
        coord_type='none'
    ).to(device)
    
    nnunet_model.eval()
    our_model.eval()
    
    # 1. 레이어별 파라미터 분석
    analyze_layer_wise_parameters(nnunet_model, our_model)
    
    # 2. 동일한 입력 크기로 FLOPs 비교
    input_sizes = [
        (1, 4, 64, 64, 64),
        (1, 4, 128, 128, 128),
        (1, 4, 240, 240, 155),  # 실제 데이터셋 크기
    ]
    compare_flops_with_same_input_size(nnunet_model, our_model, input_sizes)
    
    # 3. 모델 구조 미세 차이점 분석
    analyze_model_structure_differences(nnunet_model, our_model)
    
    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()

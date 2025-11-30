"""
MobileViT Attention Weight Analysis Utilities

MobileViT 모델의 attention 가중치를 수집하고 분석하는 유틸리티 함수들
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os


def analyze_mvit_attention_weights(
    attention_weights_list: List[Dict],
    results_dir: Optional[str] = None,
    model_name: str = "model",
) -> Dict:
    """
    MobileViT attention weights를 분석하고 통계를 계산합니다.
    
    Args:
        attention_weights_list: 각 샘플의 attention weights 리스트
            각 요소는 {'mvit_attn': [layer1_attn, layer2_attn, ...]} 형태
            layer_attn: (B, num_heads, num_patches, num_patches) 또는 (B, num_patches, num_patches)
        results_dir: 결과 저장 디렉토리 (None이면 저장 안 함)
        model_name: 모델 이름 (파일명에 사용)
    
    Returns:
        분석 결과 딕셔너리
    """
    if len(attention_weights_list) == 0:
        print("Warning: No attention weights to analyze")
        return {}
    
    # 첫 번째 샘플로 구조 파악
    first_attn = attention_weights_list[0]
    if 'mvit_attn' not in first_attn:
        print("Warning: 'mvit_attn' key not found in attention weights")
        return {}
    
    mvit_attn_list = first_attn['mvit_attn']
    num_layers = len(mvit_attn_list)
    
    # 각 레이어별 attention weight 분석
    layer_stats = []
    all_layer_entropies = []
    
    for layer_idx, layer_attn in enumerate(mvit_attn_list):
        # layer_attn shape: (B, num_heads, num_patches, num_patches) 또는 (B, num_patches, num_patches)
        if isinstance(layer_attn, torch.Tensor):
            layer_attn = layer_attn.detach().cpu().numpy()
        
        # 모든 샘플의 해당 레이어 attention 수집
        layer_attns_all_samples = []
        layer_entropies_all_samples = []
        
        for sample_attn in attention_weights_list:
            if 'mvit_attn' in sample_attn and len(sample_attn['mvit_attn']) > layer_idx:
                attn = sample_attn['mvit_attn'][layer_idx]
                if isinstance(attn, torch.Tensor):
                    attn = attn.detach().cpu().numpy()
                
                # (B, num_heads, num_patches, num_patches) 또는 (B, num_patches, num_patches)
                if len(attn.shape) == 4:
                    # Multi-head: (B, num_heads, num_patches, num_patches)
                    # Head별로 평균
                    attn = attn[0].mean(axis=0)  # (num_patches, num_patches)
                elif len(attn.shape) == 3:
                    # (B, num_patches, num_patches)
                    attn = attn[0]  # (num_patches, num_patches)
                elif len(attn.shape) == 2:
                    # (num_patches, num_patches)
                    attn = attn
                else:
                    continue
                
                layer_attns_all_samples.append(attn)
                
                # Attention entropy 계산 (attention이 얼마나 집중되어 있는지)
                # Entropy가 낮으면 특정 패치에 집중, 높으면 균등하게 분산
                attn_flat = attn.flatten()
                attn_flat = attn_flat / (attn_flat.sum() + 1e-8)  # Normalize
                entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
                layer_entropies_all_samples.append(entropy)
        
        if len(layer_attns_all_samples) == 0:
            continue
        
        # 통계 계산
        layer_attns_array = np.array(layer_attns_all_samples)  # (n_samples, num_patches, num_patches)
        num_patches = layer_attns_array.shape[1]
        
        # 평균 attention matrix
        mean_attn = layer_attns_array.mean(axis=0)  # (num_patches, num_patches)
        
        # Attention이 가장 높은 패치들 (diagonal이 아닌 경우)
        # 각 패치가 다른 패치들에 얼마나 attention을 주는지
        mean_attn_per_patch = mean_attn.mean(axis=1)  # (num_patches,)
        
        # Entropy 통계
        mean_entropy = np.mean(layer_entropies_all_samples)
        std_entropy = np.std(layer_entropies_all_samples)
        
        layer_stats.append({
            'layer': layer_idx,
            'num_patches': num_patches,
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy,
            'max_attention_patch': np.argmax(mean_attn_per_patch),
            'mean_attention_value': mean_attn.mean(),
            'std_attention_value': mean_attn.std(),
        })
        
        all_layer_entropies.append(layer_entropies_all_samples)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"MobileViT Attention Analysis - {model_name}")
    print(f"{'='*60}")
    print(f"Total samples analyzed: {len(attention_weights_list)}")
    print(f"Number of transformer layers: {num_layers}")
    print(f"\nLayer-wise Attention Statistics:")
    print(f"{'Layer':<8} {'Num Patches':<12} {'Mean Entropy':<15} {'Std Entropy':<15}")
    print(f"{'-'*60}")
    for stat in layer_stats:
        print(f"{stat['layer']:<8} {stat['num_patches']:<12} {stat['mean_entropy']:<15.4f} {stat['std_entropy']:<15.4f}")
    
    # Entropy 해석
    print(f"\nAttention Entropy Interpretation:")
    print(f"  - Low entropy (< 2.0): Attention이 특정 패치에 집중 (좋은 학습 신호)")
    print(f"  - High entropy (> 4.0): Attention이 균등하게 분산 (학습 부족 가능성)")
    print(f"  - Medium entropy (2.0-4.0): 적절한 attention 분산")
    
    # CSV 저장
    if results_dir:
        # Layer별 통계 저장
        stats_df = pd.DataFrame(layer_stats)
        csv_path = os.path.join(results_dir, f'mvit_attention_stats_{model_name}.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"\nAttention statistics saved to: {csv_path}")
        
        # Entropy 분포 시각화
        if len(all_layer_entropies) > 0:
            fig, axes = plt.subplots(1, min(3, num_layers), figsize=(5 * min(3, num_layers), 4))
            if num_layers == 1:
                axes = [axes]
            
            for layer_idx, entropies in enumerate(all_layer_entropies[:3]):  # 최대 3개 레이어만
                if layer_idx >= len(axes):
                    break
                ax = axes[layer_idx]
                ax.hist(entropies, bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Attention Entropy')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Layer {layer_idx} Attention Entropy Distribution')
                ax.axvline(np.mean(entropies), color='r', linestyle='--', label=f'Mean: {np.mean(entropies):.2f}')
                ax.legend()
            
            plt.tight_layout()
            entropy_plot_path = os.path.join(results_dir, f'mvit_attention_entropy_{model_name}.png')
            plt.savefig(entropy_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Attention entropy distribution saved to: {entropy_plot_path}")
    
    print(f"{'='*60}\n")
    
    return {
        'layer_stats': layer_stats,
        'all_layer_entropies': all_layer_entropies,
        'num_samples': len(attention_weights_list),
        'num_layers': num_layers,
    }


def check_mvit_attention_learning(
    attention_weights_list: List[Dict],
    threshold_entropy: float = 4.0,
) -> Tuple[bool, str]:
    """
    MobileViT attention이 제대로 학습되었는지 확인합니다.
    
    Args:
        attention_weights_list: analyze_mvit_attention_weights와 동일한 형식
        threshold_entropy: Entropy 임계값 (이보다 높으면 학습 부족으로 판단)
    
    Returns:
        (is_learning_well, message) 튜플
    """
    if len(attention_weights_list) == 0:
        return False, "No attention weights provided"
    
    # 첫 번째 샘플로 구조 파악
    first_attn = attention_weights_list[0]
    if 'mvit_attn' not in first_attn:
        return False, "Invalid attention weights format"
    
    mvit_attn_list = first_attn['mvit_attn']
    num_layers = len(mvit_attn_list)
    
    # 각 레이어별 entropy 계산
    layer_entropies = []
    for layer_idx in range(num_layers):
        entropies = []
        for sample_attn in attention_weights_list:
            if 'mvit_attn' in sample_attn and len(sample_attn['mvit_attn']) > layer_idx:
                attn = sample_attn['mvit_attn'][layer_idx]
                if isinstance(attn, torch.Tensor):
                    attn = attn.detach().cpu().numpy()
                
                # Shape 정규화
                if len(attn.shape) == 4:
                    attn = attn[0].mean(axis=0)
                elif len(attn.shape) == 3:
                    attn = attn[0]
                
                if len(attn.shape) != 2:
                    continue
                
                # Entropy 계산
                attn_flat = attn.flatten()
                attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
                entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
                entropies.append(entropy)
        
        if len(entropies) > 0:
            layer_entropies.append(np.mean(entropies))
    
    if len(layer_entropies) == 0:
        return False, "Failed to calculate entropies"
    
    # 모든 레이어의 평균 entropy
    mean_entropy = np.mean(layer_entropies)
    max_entropy = np.max(layer_entropies)
    
    # 판단
    if mean_entropy > threshold_entropy:
        return False, f"Attention may not be learning well (mean entropy: {mean_entropy:.2f} > {threshold_entropy})"
    elif max_entropy > threshold_entropy:
        return True, f"Attention is learning, but some layers need improvement (max entropy: {max_entropy:.2f})"
    else:
        return True, f"Attention is learning well (mean entropy: {mean_entropy:.2f} <= {threshold_entropy})"


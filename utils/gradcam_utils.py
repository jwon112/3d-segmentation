#!/usr/bin/env python3
"""
Grad-CAM utility functions for model interpretability
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

# Import Grad-CAM from visualization package
from visualization.gradcam_3d import GradCAM3D, GradCAM3DVisualizer


def find_target_layer_for_gradcam(model, model_name):
    """
    모델 구조를 직접 탐색하여 적절한 Grad-CAM target layer를 찾습니다.
    
    Grad-CAM에 적합한 레이어 조건:
    1. Conv 레이어를 포함하는 모듈
    2. Encoder의 깊은 부분 (bottleneck 근처)
    3. 출력 feature map이 너무 작지 않은 레이어 (최소 4x4x4 이상)
    
    Args:
        model: PyTorch 모델
        model_name: 모델 이름 (예: 'unet3d_s', 'dualbranch_01_unet_s')
    
    Returns:
        target_layer_name: 레이어 이름 (str) 또는 None (찾지 못한 경우)
    """
    # Unwrap DDP if needed
    real_model = model.module if hasattr(model, 'module') else model
    
    # 모든 레이어 이름과 모듈 수집
    all_layers = list(real_model.named_modules())
    all_layer_names = [name for name, _ in all_layers]
    
    # 우선순위 1: 명시적인 bottleneck 레이어
    for name, module in all_layers:
        if 'bottleneck' in name.lower() and name:  # 빈 문자열 제외
            # Conv 레이어를 포함하는지 확인
            if _has_conv_layer(module):
                return name
    
    # 우선순위 2: Encoder의 깊은 down 레이어 (down5, down6, down4 순서)
    down_layers = []
    for name, module in all_layers:
        if name and 'down' in name.lower() and _has_conv_layer(module):
            # 숫자 추출하여 정렬
            import re
            numbers = re.findall(r'\d+', name)
            if numbers:
                down_layers.append((int(numbers[-1]), name))
            else:
                down_layers.append((0, name))
    
    if down_layers:
        # 숫자가 큰 것 우선 (down6 > down5 > down4)
        down_layers.sort(reverse=True)
        return down_layers[0][1]
    
    # 우선순위 3: Transformer 모델의 경우 마지막 transformer block
    if 'transformer' in model_name.lower() or 'unetr' in model_name.lower():
        # transformer_blocks는 ModuleList이므로 마지막 블록 찾기
        for name, module in all_layers:
            if 'transformer_blocks' in name:
                # ModuleList의 마지막 블록 찾기
                if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                    # 마지막 블록의 첫 번째 서브모듈 찾기
                    last_block = module[-1]
                    for sub_name, sub_module in last_block.named_modules():
                        if sub_name and _has_conv_layer(sub_module):
                            return f"{name}.{len(module)-1}.{sub_name}" if sub_name else f"{name}.{len(module)-1}"
                elif hasattr(module, '__len__') and len(module) > 0:
                    # 다른 형태의 리스트인 경우
                    return f"{name}.{len(module)-1}"
    
    # 우선순위 4: enc4, enc3 등 encoder 레이어
    enc_layers = []
    for name, module in all_layers:
        if name and ('enc' in name.lower() or 'encoder' in name.lower()) and _has_conv_layer(module):
            # 숫자 추출
            import re
            numbers = re.findall(r'\d+', name)
            if numbers:
                enc_layers.append((int(numbers[-1]), name))
            else:
                enc_layers.append((0, name))
    
    if enc_layers:
        enc_layers.sort(reverse=True)
        return enc_layers[0][1]
    
    # 우선순위 5: 일반적인 패턴으로 찾기 (down, encoder 등)
    for name, module in all_layers:
        if name and _has_conv_layer(module):
            # 너무 깊지 않은 레이어 선택 (최상위 레이어 제외)
            depth = name.count('.')
            if 1 <= depth <= 3:  # 적절한 깊이
                if any(keyword in name.lower() for keyword in ['down', 'enc', 'encoder', 'block']):
                    return name
    
    # 우선순위 6: 마지막 수단 - Conv 레이어를 가진 첫 번째 적절한 깊이의 레이어
    for name, module in all_layers:
        if name and _has_conv_layer(module):
            depth = name.count('.')
            if 1 <= depth <= 4:
                return name
    
    return None


def _has_conv_layer(module):
    """
    모듈이 Conv 레이어를 포함하는지 확인합니다.
    """
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                          torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
        return True
    
    # 재귀적으로 서브모듈 확인 (너무 깊지 않게, 최대 2단계)
    for child in module.children():
        if isinstance(child, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                             torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            return True
    
    return False


def generate_gradcam_for_model(model, test_loader, device, model_name, results_dir, 
                                num_samples=3, target_layer=None):
    """
    모델에 대해 Grad-CAM을 생성하고 결과를 저장합니다.
    
    Args:
        model: 학습된 PyTorch 모델
        test_loader: 테스트 데이터 로더
        device: 디바이스
        model_name: 모델 이름
        results_dir: 결과 저장 디렉토리
        num_samples: Grad-CAM을 생성할 샘플 수
        target_layer: 타겟 레이어 이름 (None이면 자동으로 찾음)
    
    Returns:
        bool: 성공 여부
    """
    try:
        # Unwrap DDP if needed
        real_model = model.module if hasattr(model, 'module') else model
        real_model.eval()
        
        # 타겟 레이어 찾기
        if target_layer is None:
            target_layer = find_target_layer_for_gradcam(real_model, model_name)
        
        if target_layer is None:
            print(f"Warning: Could not find suitable target layer for {model_name}. Skipping Grad-CAM.")
            return False
        
        print(f"Using target layer for Grad-CAM: {target_layer}")
        
        # Grad-CAM 저장 디렉토리 생성
        gradcam_dir = os.path.join(results_dir, 'gradcam', model_name)
        os.makedirs(gradcam_dir, exist_ok=True)
        
        visualizer = GradCAM3DVisualizer(real_model, device)
        
        sample_count = 0
        for i, batch in enumerate(test_loader):
            if sample_count >= num_samples:
                break
            
            try:
                # 데이터 준비
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs = batch.get('image', batch.get('inputs'))
                    labels = batch.get('segmentation', batch.get('labels'))
                
                if inputs is None or labels is None:
                    continue
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 2D 모델의 경우 depth 차원 추가
                if model_name != 'mobile_unetr' and len(inputs.shape) == 4:
                    inputs = inputs.unsqueeze(2)
                    labels = labels.unsqueeze(2)
                
                # 3D 볼륨만 처리 (batch_size=1)
                if inputs.dim() != 5 or inputs.size(0) != 1:
                    continue
                
                # 예측
                with torch.no_grad():
                    outputs = real_model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    # 3D 세그멘테이션에서는 각 클래스별로 평균 확률 계산
                    avg_probs = probs.mean(dim=(2, 3, 4))  # [B, C]
                    predicted_class = torch.argmax(avg_probs, dim=1).item()
                
                # Grad-CAM 생성
                gradcam = GradCAM3D(real_model, target_layer)
                cam = gradcam.generate_cam(inputs, class_idx=predicted_class)
                gradcam.remove_hooks()
                
                # NumPy 변환
                image_np = inputs[0].cpu().detach().numpy()
                seg_np = labels[0].cpu().detach().numpy()
                
                # 샘플별 디렉토리 생성
                sample_dir = os.path.join(gradcam_dir, f'sample_{sample_count:03d}')
                os.makedirs(sample_dir, exist_ok=True)
                
                # 2D 슬라이스 시각화 저장
                slice_path = os.path.join(sample_dir, 'gradcam_slices.png')
                modality_names = ['FLAIR', 'T1CE'] if image_np.shape[0] == 2 else ['FLAIR', 'T1', 'T1CE', 'T2']
                visualizer.visualize_slices(
                    image_np, cam, seg_np,
                    slice_indices=None,  # 자동으로 중간 슬라이스 선택
                    modality_names=modality_names,
                    save_path=slice_path,
                    show=False
                )
                
                # 3D 볼륨 시각화 저장
                volume_path = os.path.join(sample_dir, 'gradcam_3d.html')
                visualizer.visualize_3d_volume(
                    image_np, cam, seg_np, 
                    threshold=0.3,
                    save_path=volume_path,
                    title=f'3D Grad-CAM - {model_name} - Sample {sample_count}'
                )
                
                sample_count += 1
                print(f"  Grad-CAM generated for sample {sample_count}/{num_samples}")
                
            except Exception as e:
                print(f"  Warning: Failed to generate Grad-CAM for sample {i}: {e}")
                continue
        
        if sample_count > 0:
            print(f"Grad-CAM results saved to: {gradcam_dir}")
            return True
        else:
            print(f"Warning: No Grad-CAM samples generated for {model_name}")
            return False
            
    except Exception as e:
        print(f"Warning: Failed to generate Grad-CAM for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


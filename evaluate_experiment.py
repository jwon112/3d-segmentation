#!/usr/bin/env python3
"""
평가 전용 스크립트 - 저장된 체크포인트를 로드하여 평가만 수행

Usage:
    python evaluate_experiment.py --results_dir experiment_result/integrated_experiment_results_YYYYMMDD_HHMMSS --models unet3d_2modal_s unet3d_4modal_s
    python evaluate_experiment.py --results_dir experiment_result/integrated_experiment_results_YYYYMMDD_HHMMSS --seeds 24 42
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from refactored modules
from utils.experiment_utils import (
    get_model, calculate_flops, calculate_pam, calculate_inference_latency,
    setup_distributed, cleanup_distributed, is_main_process,
    INPUT_SIZE_2D, INPUT_SIZE_3D
)
from utils.runner import evaluate_model
from utils.runner.cascade_evaluation import evaluate_segmentation_with_roi, load_roi_model_from_checkpoint
from dataloaders import get_data_loaders
from visualization import create_comprehensive_analysis, create_interactive_3d_plot
from utils.gradcam_utils import generate_gradcam_for_model

def load_checkpoint_and_evaluate(results_dir, model_name, seed, data_path, dim='3d', 
                                 dataset_version='brats2018', batch_size=1, num_workers=0,
                                 device='cuda', distributed=False, rank=0, world_size=1,
                                 fold_idx=None, use_5fold=False, preprocessed_base_dir=None):
    """저장된 best.pt 체크포인트를 로드하여 평가, 기록, 시각화 파이프라인 수행
    
    비정상적으로 중단된 학습이나 여러 실험의 best.pt를 모아서 평가할 때 사용
    """
    
    # 체크포인트 로드 (5-fold 지원) - 먼저 체크포인트를 로드해서 coord_type과 modalities 수 감지
    if use_5fold and fold_idx is not None:
        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_fold_{fold_idx}_best.pth")
    else:
        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_best.pth")
    
    if not os.path.exists(ckpt_path):
        if is_main_process(rank):
            print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping...")
        return None
    
    if is_main_process(rank):
        print(f"Loading best checkpoint from {ckpt_path}")
    
    # 체크포인트를 먼저 로드해서 coord_type과 modalities 수 자동 감지
    state = torch.load(ckpt_path, map_location=device)
    
    # Cascade 모델의 경우 첫 번째 레이어에서 입력 채널 수 감지
    coord_type = 'none'  # 기본값
    detected_n_modalities = 2  # 기본값 (T1CE, FLAIR)
    
    if model_name.startswith('cascade_'):
        # Cascade 모델의 첫 번째 레이어 찾기 (stem.conv1.0.conv2d.weight 등)
        detected_channels = None
        detected_key = None
        for key in state.keys():
            if 'weight' in key and ('stem.conv1' in key or 'stem' in key or 'conv1' in key):
                if 'conv2d.weight' in key or 'weight' in key:
                    # shape: [out_channels, in_channels, ...]
                    detected_channels = state[key].shape[1]
                    detected_key = key
                    break
        
        if detected_channels is not None:
            if is_main_process(rank):
                print(f"[Debug] Found first layer: {detected_key}, input channels: {detected_channels}")
            
            # 채널 수에 따라 coord_type과 modalities 수 추론
            # 4 channels = 4 modalities, no coords
            # 7 channels = 2 modalities + 3 simple coords
            # 11 channels = 2 modalities + 9 hybrid coords
            # 13 channels = 4 modalities + 9 hybrid coords
            if detected_channels == 4:
                coord_type = 'none'
                detected_n_modalities = 4
                if is_main_process(rank):
                    print(f"Detected {detected_channels} input channels -> 4 modalities, coord_type='none'")
            elif detected_channels == 7:
                coord_type = 'simple'
                detected_n_modalities = 2
                if is_main_process(rank):
                    print(f"Detected {detected_channels} input channels -> 2 modalities + 3 simple coords")
            elif detected_channels == 11:
                coord_type = 'hybrid'
                detected_n_modalities = 2
                if is_main_process(rank):
                    print(f"Detected {detected_channels} input channels -> 2 modalities + 9 hybrid coords")
            elif detected_channels == 13:
                coord_type = 'hybrid'
                detected_n_modalities = 4
                if is_main_process(rank):
                    print(f"Detected {detected_channels} input channels -> 4 modalities + 9 hybrid coords")
            else:
                if is_main_process(rank):
                    print(f"Warning: Unexpected input channels {detected_channels}. Using defaults (2 modalities, coord_type='none')")
        else:
            if is_main_process(rank):
                print(f"[Debug] Could not find first layer in checkpoint. Available keys (first 10): {list(state.keys())[:10]}")
    
    # 감지된 modalities 수로 n_channels 및 use_4modalities 업데이트
    # 주의: get_model은 n_channels에 modalities 수만 전달하고, coord_type을 통해 coord channels를 별도로 처리합니다
    n_channels = detected_n_modalities
    use_4modalities = (detected_n_modalities == 4)
    
    if is_main_process(rank):
        print(f"[Debug] Creating model with n_channels={n_channels}, coord_type={coord_type}, use_4modalities={use_4modalities}")
        if model_name.startswith('cascade_'):
            if coord_type == 'hybrid':
                expected_total = n_channels + 9
            elif coord_type == 'simple':
                expected_total = n_channels + 3
            else:
                expected_total = n_channels
            print(f"[Debug] Expected total input channels: {expected_total} (n_image_channels={n_channels} + n_coord_channels={9 if coord_type == 'hybrid' else 3 if coord_type == 'simple' else 0})")
    
    # 데이터 로더 생성 (체크포인트에서 감지한 use_4modalities 사용)
    # preprocessed_base_dir이 제공되면 버전별 디렉토리로 변환
    # 만약 preprocessed_base_dir이 fold_0 같은 fold 경로로 끝나면 fold_split_dir로 인식
    preprocessed_dir = None
    fold_split_dir = None
    detected_fold_idx = fold_idx
    
    if preprocessed_base_dir:
        # fold_0, fold_1 등의 패턴 확인
        import re
        fold_pattern = re.search(r'/fold_(\d+)$', preprocessed_base_dir)
        if fold_pattern:
            # fold 경로로 지정된 경우: fold_split_dir과 fold_idx 자동 감지
            detected_fold_idx = int(fold_pattern.group(1))
            # 부모 디렉토리를 fold_split_dir로 사용
            fold_split_dir = os.path.dirname(preprocessed_base_dir)
            if is_main_process(rank):
                print(f"[Debug] Detected fold path: {preprocessed_base_dir}")
                print(f"[Debug] Using fold_split_dir: {fold_split_dir}, fold_idx: {detected_fold_idx}")
        else:
            # 일반 경로: 버전별 디렉토리 사용
            preprocessed_dir = os.path.join(preprocessed_base_dir, dataset_version.upper())
            if is_main_process(rank):
                print(f"[Debug] preprocessed_base_dir: {preprocessed_base_dir}")
                print(f"[Debug] Using preprocessed directory: {preprocessed_dir}")
    else:
        if is_main_process(rank):
            print(f"[Debug] preprocessed_base_dir is None or empty")
    
    train_loader, val_loader, test_loader, _, _, _ = get_data_loaders(
        data_dir=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=None,
        dim=dim,
        dataset_version=dataset_version,
        seed=seed,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        use_4modalities=use_4modalities,  # 체크포인트에서 감지한 값 사용
        use_5fold=use_5fold,
        fold_idx=detected_fold_idx,
        fold_split_dir=fold_split_dir,
        preprocessed_dir=preprocessed_dir
    )
    
    # 모델 생성 (coord_type 전달)
    # get_model 내부에서 coord_type에 따라 n_coord_channels를 계산하고 n_image_channels + n_coord_channels로 모델을 생성합니다
    model = get_model(model_name, n_channels=n_channels, n_classes=4, dim=dim, coord_type=coord_type)
    
    # DDP wrap if distributed
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = model.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        model = model.to(device)
    
    # 체크포인트 로드
    real_model = model.module if hasattr(model, 'module') else model
    real_model.load_state_dict(state, strict=False)
    
    if is_main_process(rank):
        print(f"Successfully loaded checkpoint from {ckpt_path}")
    
    # RepLK 모델의 경우 deploy 모드로 전환
    # Check if model name starts with any RepLK model prefix (supports all sizes: xs, s, m, l)
    replk_model_prefixes = [
        'dualbranch_04_unet_',
        'dualbranch_05_unet_',
        'dualbranch_06_unet_',
        'dualbranch_07_unet_',
    ]
    if any(model_name.startswith(prefix) for prefix in replk_model_prefixes):
        if hasattr(real_model, 'switch_to_deploy'):
            if is_main_process(rank):
                print(f"Switching RepLK blocks to deploy mode...")
            real_model.switch_to_deploy()
    
    # 파라미터 및 FLOPs 계산
    # Cascade 모델의 경우 실제 입력 채널 수는 n_channels + n_coord_channels
    if model_name.startswith('cascade_'):
        if coord_type == 'hybrid':
            actual_input_channels = n_channels + 9
        elif coord_type == 'simple':
            actual_input_channels = n_channels + 3
        else:  # 'none'
            actual_input_channels = n_channels
    else:
        actual_input_channels = n_channels
    
    total_params = sum(p.numel() for p in real_model.parameters())
    if dim == '2d':
        flops = calculate_flops(model, input_size=(1, actual_input_channels, *INPUT_SIZE_2D))
        input_size = (1, actual_input_channels, *INPUT_SIZE_2D)
    else:
        flops = calculate_flops(model, input_size=(1, actual_input_channels, *INPUT_SIZE_3D))
        input_size = (1, actual_input_channels, *INPUT_SIZE_3D)
    
    # PAM 계산 (rank 0에서만, batch_size=1로 고정, 여러 번 측정)
    pam_train_list = []
    pam_inference_list = []
    pam_train_stages = {}
    pam_inference_stages = {}
    if is_main_process(rank):
        try:
            pam_train_list, pam_train_stages = calculate_pam(
                model, input_size=input_size, mode='train', stage_wise=True, device=device
            )
            pam_inference_list, pam_inference_stages = calculate_pam(
                model, input_size=input_size, mode='inference', stage_wise=True, device=device
            )
            if pam_train_stages:
                print(f"PAM (Train) by stage:")
                for stage_name, mem_list in sorted(pam_train_stages.items()):
                    if mem_list:
                        mem_mean = sum(mem_list) / len(mem_list)
                        print(f"  {stage_name}: {mem_mean / 1024**2:.2f} MB")
            if pam_inference_stages:
                print(f"PAM (Inference) by stage:")
                for stage_name, mem_list in sorted(pam_inference_stages.items()):
                    if mem_list:
                        mem_mean = sum(mem_list) / len(mem_list)
                        print(f"  {stage_name}: {mem_mean / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Warning: Failed to calculate PAM: {e}")
            pam_train_list = []
            pam_inference_list = []
            pam_train_stages = {}
            pam_inference_stages = {}
    
    # Inference Latency 계산 (rank 0에서만, batch_size=1로 고정, 여러 번 측정)
    inference_latency_list = []
    inference_latency_stats = {}
    if is_main_process(rank):
        try:
            inference_latency_list, inference_latency_stats = calculate_inference_latency(
                model, input_size=input_size, device=device, num_warmup=10, num_runs=100
            )
            if inference_latency_list:
                latency_mean = inference_latency_stats['mean']
                latency_std = inference_latency_stats['std']
                print(f"Inference Latency (batch_size=1): {latency_mean:.2f} ± {latency_std:.2f} ms (mean ± std of {len(inference_latency_list)} runs)")
        except Exception as e:
            print(f"Warning: Failed to calculate inference latency: {e}")
            inference_latency_list = []
            inference_latency_stats = {}
    
    if is_main_process(rank):
        print(f"Parameters: {total_params:,}")
        print(f"FLOPs: {flops:,}")
        if pam_train_list:
            pam_train_mean = sum(pam_train_list) / len(pam_train_list)
            print(f"PAM (Train, batch_size=1): {pam_train_mean / 1024**2:.2f} MB (mean of {len(pam_train_list)} runs)")
        if pam_inference_list:
            pam_inference_mean = sum(pam_inference_list) / len(pam_inference_list)
            print(f"PAM (Inference, batch_size=1): {pam_inference_mean / 1024**2:.2f} MB (mean of {len(pam_inference_list)} runs)")
    
    # Test set 평가
    # Cascade 모델은 ROI 기반 평가 사용
    if model_name.startswith('cascade_'):
        if is_main_process(rank):
            print(f"\n[Cascade Model] Using ROI-based evaluation for {model_name}")
        
        # ROI 모델 경로 찾기 (여러 ROI 모델 이름 시도)
        # 일반적으로 사용되는 ROI 모델 이름들
        possible_roi_model_names = ['roi_unet3d_small', 'roi_mobileunetr3d_tiny', 'roi_mobileunetr3d_small']
        roi_model_name = None
        roi_weight_path = None
        
        # 1. results_dir에서 ROI 모델 체크포인트 찾기
        for candidate_roi_name in possible_roi_model_names:
            roi_checkpoint_patterns = [
                os.path.join(results_dir, f"{candidate_roi_name}_seed_{seed}_best.pth"),
                os.path.join(results_dir, f"{candidate_roi_name}_seed_{seed}_fold_{fold_idx}_best.pth") if use_5fold and fold_idx is not None else None,
            ]
            for pattern in roi_checkpoint_patterns:
                if pattern and os.path.exists(pattern):
                    roi_weight_path = pattern
                    roi_model_name = candidate_roi_name
                    break
            if roi_weight_path:
                break
        
        # 2. 기본 경로 시도 (여러 ROI 모델 이름)
        if not roi_weight_path:
            for candidate_roi_name in possible_roi_model_names:
                default_path = f"models/weights/cascade/roi_model/{candidate_roi_name}/seed_{seed}/weights/best.pth"
                if os.path.exists(default_path):
                    roi_weight_path = default_path
                    roi_model_name = candidate_roi_name
                    break
        
        # 기본값 설정 (경로를 찾지 못한 경우)
        if not roi_model_name:
            roi_model_name = 'roi_unet3d_small'  # 기본값
        
        if roi_weight_path and os.path.exists(roi_weight_path):
            try:
                if is_main_process(rank):
                    print(f"[Cascade] Loading ROI model from {roi_weight_path}")
                
                # coord_type에 따라 include_coords 결정
                include_coords = (coord_type != 'none')
                
                roi_model, detected_include_coords = load_roi_model_from_checkpoint(
                    roi_model_name,
                    roi_weight_path,
                    device,
                    include_coords=include_coords,
                )
                
                # coord_type에 따라 coord_encoding_type 결정
                if coord_type == 'hybrid':
                    coord_encoding_type = 'hybrid'
                elif coord_type == 'simple':
                    coord_encoding_type = 'simple'
                else:
                    coord_encoding_type = 'simple'  # 기본값
                
                # Cascade ROI 기반 평가
                real_model = model.module if hasattr(model, 'module') else model
                # preprocessed_dir 설정 (brats2024 등 전처리된 데이터 사용 시)
                # fold_split_dir이 있으면 fold 디렉토리에서 로드, 없으면 일반 preprocessed_dir 사용
                cascade_preprocessed_dir = None
                cascade_fold_split_dir = fold_split_dir
                cascade_fold_idx = detected_fold_idx if fold_split_dir else (fold_idx if use_5fold else None)
                
                if preprocessed_base_dir and not fold_split_dir:
                    # fold_split_dir이 없으면 일반 경로 사용
                    if use_5fold:
                        # 5-fold 모드: fold_split_dir 사용
                        if fold_idx is not None:
                            cascade_preprocessed_dir = os.path.join(preprocessed_base_dir, f'{dataset_version.upper()}_5fold_splits', f'fold_{fold_idx}')
                        else:
                            cascade_preprocessed_dir = os.path.join(preprocessed_base_dir, f'{dataset_version.upper()}_5fold_splits')
                    else:
                        # 일반 모드: 버전별 디렉토리 사용
                        cascade_preprocessed_dir = os.path.join(preprocessed_base_dir, dataset_version.upper())
                
                metrics = evaluate_segmentation_with_roi(
                    seg_model=real_model,
                    roi_model=roi_model,
                    data_dir=data_path,
                    dataset_version=dataset_version,
                    seed=seed,
                    roi_resize=(64, 64, 64),
                    crop_size=(96, 96, 96),
                    include_coords=include_coords,
                    coord_encoding_type=coord_encoding_type,
                    use_5fold=use_5fold,
                    fold_idx=cascade_fold_idx,
                    fold_split_dir=cascade_fold_split_dir,
                    crops_per_center=1,
                    crop_overlap=0.5,
                    use_blending=True,
                    results_dir=results_dir,
                    model_name=model_name,
                    preprocessed_dir=cascade_preprocessed_dir,  # 전처리된 데이터 디렉토리 (fold_split_dir이 없을 때만 사용)
                )
                
                if is_main_process(rank):
                    print(f"[Cascade] Evaluation completed successfully")
            except Exception as e:
                if is_main_process(rank):
                    print(f"[Cascade] Error during ROI-based evaluation: {e}")
                    print(f"[Cascade] Falling back to standard evaluation...")
                    import traceback
                    traceback.print_exc()
                # Fallback to standard evaluation
                metrics = evaluate_model(
                    model, test_loader, device, model_name, 
                    distributed=distributed, world_size=world_size,
                    sw_patch_size=(128, 128, 128), sw_overlap=0.5, 
                    results_dir=results_dir,
                    coord_type=coord_type
                )
        else:
            # ROI 모델을 찾지 못한 경우 에러 발생 (fallback 하지 않음)
            error_msg = f"[Cascade] Error: ROI model not found for cascade model {model_name}.\n"
            error_msg += f"  Tried paths:\n"
            for pattern in roi_checkpoint_patterns:
                if pattern:
                    error_msg += f"    - {pattern}\n"
            if 'default_path' in locals():
                error_msg += f"    - {default_path}\n"
            error_msg += f"\n  Cascade models require ROI-based evaluation.\n"
            error_msg += f"  Please ensure the ROI model checkpoint exists at one of the above paths."
            
            if is_main_process(rank):
                print(error_msg)
            raise FileNotFoundError(error_msg)
    else:
        # 일반 모델은 기존 방식 사용
        metrics = evaluate_model(
            model, test_loader, device, model_name, 
            distributed=distributed, world_size=world_size,
            sw_patch_size=(128, 128, 128), sw_overlap=0.5, 
            results_dir=results_dir,
            coord_type=coord_type  # 체크포인트에서 감지한 coord_type 전달
        )
    
    # Grad-CAM 생성 (rank 0에서만, 3D 모델만)
    # TODO: Grad-CAM 기능은 아직 안정화되지 않아 주석 처리됨
    # if is_main_process(rank) and dim == '3d':
    #     try:
    #         print(f"\nGenerating Grad-CAM visualizations for {model_name}...")
    #         generate_gradcam_for_model(
    #             model=model,
    #             test_loader=test_loader,
    #             device=device,
    #             model_name=model_name,
    #             results_dir=results_dir,
    #             num_samples=3,  # 각 모델당 3개 샘플
    #             target_layer=None  # 자동으로 찾음
    #         )
    #     except Exception as e:
    #         print(f"Warning: Failed to generate Grad-CAM for {model_name}: {e}")
    #         import traceback
    #         traceback.print_exc()
    
    # 결과 반환 (각 run마다 하나의 행만 생성, PAM과 Latency는 평균값 사용)
    # PAM과 Latency 평균값 계산
    pam_train_mean = sum(pam_train_list) / len(pam_train_list) if pam_train_list else 0
    pam_inference_mean = sum(pam_inference_list) / len(pam_inference_list) if pam_inference_list else 0
    latency_mean = sum(inference_latency_list) / len(inference_latency_list) if inference_latency_list else 0
    
    result = {
        'dataset': dataset_version,
        'seed': seed,
        'model_name': model_name,
        'total_params': total_params,
        'flops': flops,
        'pam_train': pam_train_mean,  # bytes, batch_size=1 기준 (평균값)
        'pam_inference': pam_inference_mean,  # bytes, batch_size=1 기준 (평균값)
        'inference_latency_ms': latency_mean,  # milliseconds, batch_size=1 기준 (평균값)
        'test_dice': metrics['dice'],
        'test_wt': metrics.get('wt', None),
        'test_tc': metrics.get('tc', None),
        'test_et': metrics.get('et', None),
        'test_hd95_mean': metrics.get('hd95_mean', None),
        'test_hd95_wt': metrics.get('hd95_wt', None),
        'test_hd95_tc': metrics.get('hd95_tc', None),
        'test_hd95_et': metrics.get('hd95_et', None),
        'val_dice': None,  # 평가 전용이므로 val_dice는 없음
        'val_wt': None,
        'val_tc': None,
        'val_et': None,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'best_epoch': None  # 평가 전용이므로 best_epoch는 없음
    }
    
    if is_main_process(rank):
        print(f"Test Dice: {metrics['dice']:.4f} (WT {metrics['wt']:.4f} | TC {metrics['tc']:.4f} | ET {metrics['et']:.4f})")
        if all(v is not None for v in [metrics.get('hd95_wt'), metrics.get('hd95_tc'), metrics.get('hd95_et')]):
            print(f"HD95 (mm): WT {metrics['hd95_wt']:.2f} | TC {metrics['hd95_tc']:.2f} | ET {metrics['hd95_et']:.2f}")
        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    
    # Stage별 PAM 결과도 함께 반환
    stage_pam_results = []
    if is_main_process(rank):
        if pam_train_stages:
            for stage_name, mem_list in pam_train_stages.items():
                if mem_list:
                    mem_mean = sum(mem_list) / len(mem_list)
                    mem_std = (sum((x - mem_mean) ** 2 for x in mem_list) / len(mem_list)) ** 0.5 if len(mem_list) > 1 else 0.0
                    stage_pam_results.append({
                        'dataset': dataset_version,
                        'seed': seed,
                        'model_name': model_name,
                        'mode': 'train',
                        'stage_name': stage_name,
                        'pam_mean': mem_mean,
                        'pam_std': mem_std,
                        'num_runs': len(mem_list)
                    })
        
        if pam_inference_stages:
            for stage_name, mem_list in pam_inference_stages.items():
                if mem_list:
                    mem_mean = sum(mem_list) / len(mem_list)
                    mem_std = (sum((x - mem_mean) ** 2 for x in mem_list) / len(mem_list)) ** 0.5 if len(mem_list) > 1 else 0.0
                    stage_pam_results.append({
                        'dataset': dataset_version,
                        'seed': seed,
                        'model_name': model_name,
                        'mode': 'inference',
                        'stage_name': stage_name,
                        'pam_mean': mem_mean,
                        'pam_std': mem_std,
                        'num_runs': len(mem_list)
                    })
    
    # 결과와 stage별 PAM 결과 반환
    return ([result], stage_pam_results) if is_main_process(rank) else ([], [])


def run_evaluation(results_dir, data_path, models=None, seeds=None, dim='3d', 
                   dataset_version='brats2018', batch_size=1, num_workers=0,
                   use_5fold=False, fold_idx=None, preprocessed_base_dir=None):
    """평가 실행 (Early stopping으로 중단된 경우에도 사용 가능)"""
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return None, pd.DataFrame()
    
    # Distributed setup
    distributed, rank, local_rank, world_size = setup_distributed()
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process(rank):
            print(f"\nUsing DDP with world_size={world_size}, local_rank={local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
    
    if is_main_process(rank):
        print(f"\n[Evaluation Mode] Loading best.pt checkpoints for evaluation, recording, and visualization")
    
    # 사용 가능한 체크포인트 찾기 (5-fold 지원)
    all_checkpoints = []
    for file in os.listdir(results_dir):
        if file.endswith('_best.pth'):
            # 파일명 형식: {model_name}_seed_{seed}_best.pth 또는 {model_name}_seed_{seed}_fold_{fold_idx}_best.pth
            if '_fold_' in file:
                # 5-fold 형식: {model_name}_seed_{seed}_fold_{fold_idx}_best.pth
                parts = file.replace('_best.pth', '').split('_seed_')
                if len(parts) == 2:
                    model_name = parts[0]
                    seed_fold = parts[1].split('_fold_')
                    if len(seed_fold) == 2:
                        seed = int(seed_fold[0])
                        fold = int(seed_fold[1])
                        all_checkpoints.append((model_name, seed, fold))
            else:
                # 일반 형식: {model_name}_seed_{seed}_best.pth
                parts = file.replace('_best.pth', '').split('_seed_')
                if len(parts) == 2:
                    model_name = parts[0]
                    seed = int(parts[1])
                    all_checkpoints.append((model_name, seed, None))
    
    if not all_checkpoints:
        print(f"Error: No checkpoint files found in {results_dir}")
        return None, pd.DataFrame()
    
    # 필터링
    if models:
        all_checkpoints = [(m, s, f) for m, s, f in all_checkpoints if m in models]
    if seeds:
        all_checkpoints = [(m, s, f) for m, s, f in all_checkpoints if s in seeds]
    if use_5fold and fold_idx is not None:
        all_checkpoints = [(m, s, f) for m, s, f in all_checkpoints if f == fold_idx]
    
    if not all_checkpoints:
        print(f"Error: No matching checkpoints found after filtering")
        return None, pd.DataFrame()
    
    if is_main_process(rank):
        print(f"\nFound {len(all_checkpoints)} checkpoint(s) to evaluate:")
        for model_name, seed, fold in all_checkpoints:
            if fold is not None:
                print(f"  - {model_name} (seed {seed}, fold {fold})")
            else:
                print(f"  - {model_name} (seed {seed})")
    
    # 평가 실행
    all_results = []
    all_stage_pam_results = []  # Stage별 PAM 결과 저장용
    for model_name, seed, fold in all_checkpoints:
        try:
            if is_main_process(rank):
                print(f"\n{'='*60}")
                if fold is not None:
                    print(f"Evaluating {model_name.upper()} (seed {seed}, fold {fold})...")
                else:
                    print(f"Evaluating {model_name.upper()} (seed {seed})...")
                print(f"{'='*60}")
            
            results, stage_pam_results = load_checkpoint_and_evaluate(
                results_dir=results_dir,
                model_name=model_name,
                seed=seed,
                data_path=data_path,
                dim=dim,
                dataset_version=dataset_version,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                fold_idx=fold,
                use_5fold=(fold is not None),
                preprocessed_base_dir=preprocessed_base_dir
            )
            
            if results and is_main_process(rank):
                all_results.extend(results)  # 여러 결과를 모두 추가
                all_stage_pam_results.extend(stage_pam_results)  # Stage별 PAM 결과 추가
                
        except Exception as e:
            if is_main_process(rank):
                if fold is not None:
                    print(f"Error evaluating {model_name} (seed {seed}, fold {fold}): {e}")
                else:
                    print(f"Error evaluating {model_name} (seed {seed}): {e}")
                import traceback
                traceback.print_exc()
            continue
    
    # 결과를 DataFrame으로 변환
    if not all_results:
        print("Warning: No results were collected.")
        return results_dir, pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    
    # CSV로 저장 (기존 파일이 있으면 백업)
    if is_main_process(rank) or not distributed:
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        if os.path.exists(csv_path):
            backup_path = os.path.join(results_dir, f"evaluation_results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.rename(csv_path, backup_path)
            print(f"Backed up existing results to {backup_path}")
        
        results_df.to_csv(csv_path, index=False)
        print(f"\nEvaluation results saved to: {csv_path}")
    
    # Stage별 PAM 결과 저장
    if all_stage_pam_results and (is_main_process(rank) or not distributed):
        stage_pam_df = pd.DataFrame(all_stage_pam_results)
        stage_pam_csv_path = os.path.join(results_dir, "stage_wise_pam_results.csv")
        stage_pam_df.to_csv(stage_pam_csv_path, index=False)
        print(f"Stage-wise PAM results saved to: {stage_pam_csv_path}")
    
    # 시각화 생성
    if is_main_process(rank) or not distributed:
        print("\nCreating visualization charts...")
        try:
            create_comprehensive_analysis(results_df, None, results_dir)
            create_interactive_3d_plot(results_df, results_dir)
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    # 결과 출력
    if (is_main_process(rank) or not distributed) and not results_df.empty:
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        for _, row in results_df.iterrows():
            print(f"{row['model_name']:30} (seed {row['seed']:3d}) | "
                  f"Test Dice: {row['test_dice']:.4f} | "
                  f"WT: {row['test_wt']:.4f} | TC: {row['test_tc']:.4f} | ET: {row['test_et']:.4f}")
    
    # cleanup
    if distributed:
        cleanup_distributed()
    
    return results_dir, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate saved checkpoints')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Results directory containing checkpoint files (e.g., experiment_result/integrated_experiment_results_YYYYMMDD_HHMMSS)')
    parser.add_argument('--data_path', type=str, default='/home/work/3D_/BT/',
                       help='Common path to BraTS dataset root (default: /home/work/3D_/BT/)')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                       help='Specific models to evaluate (default: all found checkpoints)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help='Specific seeds to evaluate (default: all found seeds)')
    parser.add_argument('--dim', type=str, default='3d', choices=['2d', '3d'],
                       help='Data dimension: 2d or 3d (default: 3d)')
    parser.add_argument('--dataset_version', type=str, default='brats2018',
                       choices=['brats2017', 'brats2018', 'brats2019', 'brats2020', 'brats2021', 'brats2023', 'brats2024'],
                       help='Dataset version (default: brats2018)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers (default: 0 for evaluation)')
    parser.add_argument('--use_5fold', action='store_true', default=False,
                       help='Use 5-fold cross-validation mode (default: False)')
    parser.add_argument('--fold_idx', type=int, default=None,
                       help='Specific fold index to evaluate (only used with --use_5fold, default: None for all folds)')
    parser.add_argument('--preprocessed_base_dir', type=str, default='/home/work/3D_/processed_data',
                       help='Base directory containing preprocessed H5 files (default: /home/work/3D_/processed_data). If provided, will be used for datasets that require preprocessed data (e.g., brats2024)')
    
    args = parser.parse_args()
    
    print("Starting evaluation of saved checkpoints...")
    print(f"Results directory: {args.results_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Preprocessed base dir: {args.preprocessed_base_dir}")
    print(f"Models: {args.models if args.models else 'All found checkpoints'}")
    print(f"Seeds: {args.seeds if args.seeds else 'All found seeds'}")
    print(f"Dimension: {args.dim}")
    print(f"Dataset version: {args.dataset_version}")
    
    try:
        results_dir, results_df = run_evaluation(
            results_dir=args.results_dir,
            data_path=args.data_path,
            models=args.models,
            seeds=args.seeds,
            dim=args.dim,
            dataset_version=args.dataset_version,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_5fold=args.use_5fold,
            fold_idx=args.fold_idx,
            preprocessed_base_dir=args.preprocessed_base_dir
        )
        
        if results_dir and results_df is not None and not results_df.empty:
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved in: {results_dir}")
        else:
            print(f"\nEvaluation completed with no results.")
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()


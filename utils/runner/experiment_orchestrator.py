"""
Experiment Orchestrator
통합 실험 실행 함수
"""

import os
import torch
import pandas as pd
from datetime import datetime

from utils.experiment_utils import (
    setup_distributed, cleanup_distributed, is_main_process,
    set_seed, calculate_flops, calculate_pam, calculate_inference_latency, get_model,
    INPUT_SIZE_2D, INPUT_SIZE_3D
)
from dataloaders import get_data_loaders
from visualization import create_comprehensive_analysis, create_interactive_3d_plot
from utils.experiment_config import (
    validate_and_filter_models,
    get_model_config,
)
from utils.result_utils import (
    create_result_dict,
    create_stage_pam_result,
    create_epoch_result_dict,
    save_results_to_csv
)
from utils.runner.training import train_model
from utils.runner.evaluation import evaluate_model
from utils.runner.cascade_evaluation import load_roi_model_from_checkpoint, evaluate_segmentation_with_roi


def run_integrated_experiment(data_path, epochs=10, batch_size=1, seeds=[24], models=None, datasets=None, dim='2d', use_pretrained=False, use_nnunet_loss=True, num_workers: int = 2, dataset_version='brats2018', use_5fold=False, use_mri_augmentation=False, cascade_infer_cfg=None, cascade_model_cfg=None, train_crops_per_center=1, train_crop_overlap=0.5, anisotropy_augment: bool = False, coord_type: str = 'none', use_4modalities: bool = False, preprocessed_base_dir=None):
    """3D Segmentation 통합 실험 실행
    
    Args:
        data_path: 데이터셋 루트 디렉토리 경로 (기본: 'data')
        epochs: 훈련 에포크 수
        batch_size: 배치 크기
        use_nnunet_loss: If True, use nnU-Net style loss (Soft Dice with Squared Pred, Dice 70% + CE 30%)
        seeds: 실험 시드 리스트
        models: 사용할 모델 리스트 (기본: ['unet3d', 'unetr', 'swin_unetr', 'mobile_unetr'])
        datasets: 사용할 데이터셋 리스트 (기본: ['brats2021'])
                  지원: 'brats2021', 'auto' (자동 선택)
        dim: 데이터 차원 '2d' 또는 '3d' (기본: '2d')
        use_pretrained: pretrained 가중치 사용 여부 (기본: False, scratch 학습)
        dataset_version: 데이터셋 버전 'brats2021' 또는 'brats2018' (기본: 'brats2018')
        use_5fold: 5-fold cross-validation 사용 여부
    """
    
    # coord_type에 따라 include_coords와 coord_encoding_type 결정
    if coord_type == 'none':
        include_coords = False
        coord_encoding_type = 'simple'  # 사용 안 하지만 기본값
    elif coord_type == 'simple':
        include_coords = True
        coord_encoding_type = 'simple'
    elif coord_type == 'hybrid':
        include_coords = True
        coord_encoding_type = 'hybrid'
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}. Must be 'none', 'simple', or 'hybrid'")
    
    # 실험 결과 저장 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiment_result/integrated_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Distributed setup (먼저 rank 확인)
    distributed, rank, local_rank, world_size = setup_distributed()
    
    if is_main_process(rank):
        print(f"Train data augmentation: {'MRI augmentations' if use_mri_augmentation else 'None'}")
        print(f"Anisotropy augmentation: {'On' if anisotropy_augment else 'Off'}")
        print(f"Coordinate encoding type: {coord_type} (include_coords={include_coords}, encoding={coord_encoding_type})")
    
    # 사용 가능한 모델들 검증 및 필터링
    available_models = validate_and_filter_models(models)
    
    # 결과 저장용
    all_results = []
    all_epochs_results = []
    all_stage_pam_results = []  # Stage별 PAM 결과 저장용
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process(rank):
            print(f"\nUsing DDP with world_size={world_size}, local_rank={local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
    
    # 데이터셋 경로 확인 (dataset_version에 따라)
    # use_5fold를 사용할 때는 전처리된 데이터를 사용하므로 원본 경로 체크 건너뛰기
    if use_5fold:
        # 5-fold 모드에서는 전처리된 데이터를 사용하므로 원본 경로 체크 건너뛰기
        dataset_dir = None
    else:
        # 일반 모드: 원본 데이터셋 경로 확인
        if dataset_version == 'brats2017':
            dataset_dir = os.path.join(data_path, 'BRATS2017', 'Brats17TrainingData')
        elif dataset_version == 'brats2018':
            dataset_dir = os.path.join(data_path, 'BRATS2018', 'MICCAI_BraTS_2018_Data_Training')
        elif dataset_version == 'brats2019':
            # BRATS2019는 여러 가능한 경로 구조가 있을 수 있음
            possible_paths = [
                os.path.join(data_path, 'BRATS2019', 'HGG'),
                os.path.join(data_path, 'BRATS2019', 'MICCAI_BraTS_2019_Data_Training', 'HGG'),
            ]
            dataset_dir = None
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_dir = path
                    break
        elif dataset_version == 'brats2020':
            dataset_dir = os.path.join(data_path, 'BRATS2020', 'MICCAI_BraTS2020_TrainingData')
        elif dataset_version == 'brats2021':
            dataset_dir = os.path.join(data_path, 'BRATS2021', 'BraTS2021_Training_Data')
        elif dataset_version == 'brats2023':
            dataset_dir = os.path.join(data_path, 'BRATS2023', 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
        elif dataset_version == 'brats2024':
            # BRATS2024는 전처리된 데이터를 사용하므로 원본 경로 체크 건너뛰기
            dataset_dir = None
        else:
            raise ValueError(f"Unknown dataset_version: {dataset_version}")
        
        # 원본 경로가 필요한 경우에만 체크
        if dataset_dir is not None and not os.path.exists(dataset_dir):
            print(f"Warning: Dataset {dataset_version} not found at {dataset_dir}. Skipping...")
            return None, pd.DataFrame()
    
    if is_main_process(rank):
        print(f"\n{'#'*80}")
        print(f"Dataset Version: {dataset_version.upper()}")
        print(f"{'#'*80}")
    
    # preprocessed_base_dir에서 fold 경로 감지 (use_5fold와 무관)
    # fold_0, fold_1 등의 패턴 확인
    detected_fold_split_dir = None
    detected_fold_idx = None
    if preprocessed_base_dir:
        import re
        fold_pattern = re.search(r'/fold_(\d+)$', preprocessed_base_dir)
        if fold_pattern:
            # fold 경로로 지정된 경우: fold_split_dir과 fold_idx 자동 감지
            detected_fold_idx = int(fold_pattern.group(1))
            # 부모 디렉토리를 fold_split_dir로 사용
            detected_fold_split_dir = os.path.dirname(preprocessed_base_dir)
            if is_main_process(rank):
                print(f"[Debug] Detected fold path: {preprocessed_base_dir}")
                print(f"[Debug] Using fold_split_dir: {detected_fold_split_dir}, fold_idx: {detected_fold_idx}")
        
    # 5-fold CV 또는 일반 실험
    if use_5fold:
        fold_list = list(range(5))
        if is_main_process(rank):
            print(f"\n{'='*60}")
            print(f"5-Fold Cross-Validation Mode")
            print(f"{'='*60}")
        # Fold별 디렉토리 경로 설정
        # fold_split_dir이 이미 감지된 경우 사용, 아니면 기본 경로에서 찾기
        if detected_fold_split_dir:
            fold_split_dir = detected_fold_split_dir
        else:
            # 1. preprocessed_base_dir 우선 사용
            if preprocessed_base_dir:
                search_base_dir = preprocessed_base_dir
            else:
                # 2. 기본값: /home/work/3D_/processed_data
                search_base_dir = '/home/work/3D_/processed_data'
            
            # 특정 버전의 5fold_splits 디렉토리만 찾기
            fold_split_dir = os.path.join(search_base_dir, f'{dataset_version.upper()}_5fold_splits')
            if not os.path.exists(fold_split_dir):
                fold_split_dir = None
                if is_main_process(rank):
                    print(f"Error: Fold split directory not found: {fold_split_dir}")
                    print(f"Expected path: {os.path.join(search_base_dir, f'{dataset_version.upper()}_5fold_splits')}")
                    print(f"Please run prepare_5fold_splits.py first to create fold directories.")
        
        if fold_split_dir and is_main_process(rank):
            print(f"Using fold split directory: {fold_split_dir}")
    else:
        # use_5fold=False여도 fold 경로가 감지되면 해당 fold만 사용
        if detected_fold_split_dir:
            fold_list = [detected_fold_idx]  # 감지된 fold만 사용
            fold_split_dir = detected_fold_split_dir
            if is_main_process(rank):
                print(f"[Debug] use_5fold=False but fold path detected. Using fold {detected_fold_idx} only.")
        else:
            fold_list = [None]  # 일반 모드에서는 fold 없음
            fold_split_dir = None
    
    # 각 시드별로 실험
    for seed in seeds:
        # 각 fold별로 실험 (5-fold CV인 경우)
        for fold_idx in fold_list:
            if is_main_process(rank):
                if use_5fold:
                    print(f"\n{'='*60}")
                    print(f"Training 3D Segmentation Models - Dataset Version: {dataset_version.upper()}, Seed: {seed}, Fold: {fold_idx}")
                    print(f"{'='*60}")
                else:
                    print(f"\n{'='*60}")
                    print(f"Training 3D Segmentation Models - Dataset Version: {dataset_version.upper()}, Seed: {seed}")
                    print(f"{'='*60}")
            
            # 전역 seed 설정 (데이터 분할, 학습 모두에 적용)
            set_seed(seed)
            
            # 각 모델별로 실험
            for model_name in available_models:
                try:
                    if is_main_process(rank):
                        print(f"\nTraining {model_name.upper()}...")
                    
                    # 모델별 결정성 보장: 모델 초기화/샘플링 RNG 고정
                    set_seed(seed)
                    
                    # Cascade 모델인 경우 ROI 모델 정보 확인
                    roi_model_name_for_result = None
                    if model_name.startswith('cascade_') and cascade_model_cfg:
                        roi_model_name_for_result = cascade_model_cfg.get('roi_model_name')
                        if is_main_process(rank):
                            print(f"Cascade model detected. ROI model: {roi_model_name_for_result}")
                    
                    # 모델에 따라 use_4modalities 및 n_channels 결정
                    # 사용자가 명시적으로 지정한 경우 그 값을 사용, 아니면 모델 기본값 사용
                    model_config = get_model_config(model_name)
                    # use_4modalities가 명시적으로 전달된 경우 사용, 아니면 모델 기본값 사용
                    # (함수 파라미터 기본값이 False이므로, 명시적으로 True로 전달된 경우만 사용)
                    if use_4modalities or model_config['use_4modalities']:
                        # 사용자가 명시적으로 4개 모달리티를 요청했거나, 모델이 기본적으로 4개를 사용하는 경우
                        n_channels = 4
                        use_4modalities = True
                    else:
                        # 2개 모달리티 사용
                        n_channels = 2
                        use_4modalities = False
                    
                    # 데이터 로더 생성 (모델별로 use_4modalities 설정)
                    try:
                        train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = get_data_loaders(
                            data_dir=data_path,
                            batch_size=batch_size,
                            num_workers=num_workers,  # /dev/shm 2GB 환경에서 기본 2 권장
                            max_samples=None,  # 전체 데이터 사용
                            dim=dim,  # 2D 또는 3D
                            dataset_version=dataset_version,  # 데이터셋 버전
                            seed=seed,  # 데이터 분할을 위한 seed
                            distributed=distributed,
                            world_size=world_size,
                            rank=rank,
                            use_4modalities=use_4modalities,  # 모델에 따라 설정
                            use_5fold=use_5fold,  # 5-fold CV 사용 여부
                            fold_idx=fold_idx,  # fold 인덱스 (None이면 일반 분할)
                            fold_split_dir=fold_split_dir,  # Fold별 디렉토리 경로
                            use_mri_augmentation=use_mri_augmentation,
                            model_name=model_name,  # Cascade 모델 감지를 위해 전달
                            train_crops_per_center=train_crops_per_center,  # 학습 시 multi-crop 샘플링
                            train_crop_overlap=train_crop_overlap,
                            anisotropy_augment=anisotropy_augment,
                            coord_type=coord_type,  # 좌표 타입 전달
                            preprocessed_dir=preprocessed_base_dir,  # 전처리된 데이터 디렉토리
                        )
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"Error creating data loaders for {model_name}: {e}")
                            import traceback
                            traceback.print_exc()
                        continue

                    # 모델 생성 (데이터 로더 생성 후, DDP 초기화 전)
                    try:
                        if is_main_process(rank):
                            print(f"Creating model: {model_name}...")
                        # BRATS2024는 RC(Resection Cavity) 포함으로 5개 클래스, 다른 버전은 4개 클래스
                        n_classes = 5 if dataset_version == 'brats2024' else 4
                        if is_main_process(rank):
                            print(f"[Model Config] dataset_version={dataset_version}, n_classes={n_classes}")
                        model = get_model(model_name, n_channels=n_channels, n_classes=n_classes, dim=dim, use_pretrained=use_pretrained, coord_type=coord_type)
                        if is_main_process(rank):
                            print(f"Model {model_name} created successfully.")
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"Error creating model {model_name}: {e}")
                            import traceback
                            print("Full traceback:")
                            traceback.print_exc()
                        continue
                    # DDP wrap
                    if distributed:
                        from torch.nn.parallel import DistributedDataParallel as DDP
                        model = model.to(device)
                        # 일부 모델은 조건부 파라미터 사용으로 인해 find_unused_parameters=True 필요
                        # - GhostNet: 일부 파라미터가 사용되지 않을 수 있음
                        # - CascadeSwinUNETR: 조건부 skip connection으로 일부 파라미터가 사용되지 않을 수 있음
                        use_find_unused = (
                            'ghostnet' in model_name.lower() or 
                            'cascade_swin_unetr' in model_name.lower()
                        )
                        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=use_find_unused)
                    
                    # 모델 정보 출력 (rank 0에서만)
                    if is_main_process(rank):
                        print(f"\n=== {model_name.upper()} Model Information ===")
                    real_model = model.module if hasattr(model, 'module') else model
                    total_params = sum(p.numel() for p in real_model.parameters())
                    trainable_params = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
                    # 모델 출력 채널 수 확인 (디버깅용)
                    if hasattr(real_model, 'out_channels'):
                        actual_out_channels = real_model.out_channels
                    elif hasattr(real_model, 'num_classes'):
                        actual_out_channels = real_model.num_classes
                    else:
                        # 모델의 마지막 레이어에서 출력 채널 수 추출
                        try:
                            dummy_input = torch.randn(1, n_channels, 64, 64, 64).to(device)
                            with torch.no_grad():
                                dummy_output = real_model(dummy_input)
                            actual_out_channels = dummy_output.shape[1]
                        except:
                            actual_out_channels = "unknown"
                    if is_main_process(rank):
                        print(f"Dataset version: {dataset_version}")
                        print(f"Expected n_classes: {n_classes}")
                        print(f"Model output channels: {actual_out_channels}")
                        if actual_out_channels != n_classes and actual_out_channels != "unknown":
                            print(f"⚠️  WARNING: Model output channels ({actual_out_channels}) != expected n_classes ({n_classes})!")
                        print(f"Total parameters: {total_params:,}")
                        print(f"Trainable parameters: {trainable_params:,}")
                    
                    # 모델 크기 계산 (real_model 사용)
                    param_size = 0
                    buffer_size = 0
                    for param in real_model.parameters():
                        param_size += param.nelement() * param.element_size()
                    for buffer in real_model.buffers():
                        buffer_size += buffer.nelement() * buffer.element_size()
                    model_size_mb = (param_size + buffer_size) / 1024 / 1024
                    if is_main_process(rank):
                        print(f"Model size: {model_size_mb:.2f} MB")
                        print("=" * 50)
                    
                    # 입력 크기 설정 (PAM, Latency 공용)
                    # Cascade 모델은 n_channels(모달리티 수) + coord_channels
                    actual_n_channels = n_channels
                    if model_name.startswith('cascade_'):
                        if coord_type == 'none':
                            coord_channels = 0
                        elif coord_type == 'simple':
                            coord_channels = 3
                        elif coord_type == 'hybrid':
                            coord_channels = 9
                        else:
                            coord_channels = 0
                        actual_n_channels = n_channels + coord_channels  # 모달리티 수 + coord channels
                    
                    if dim == '2d':
                        input_size = (1, actual_n_channels, *INPUT_SIZE_2D)
                    else:
                        input_size = (1, actual_n_channels, *INPUT_SIZE_3D)

                    # PAM 계산 (모델 정보 출력 직후 바로 측정)
                    pam_train_list = []
                    pam_inference_list = []
                    pam_train_stages = {}
                    pam_inference_stages = {}
                    if is_main_process(rank):
                        if dim == '2d':
                            pam_input_size = (1, actual_n_channels, *INPUT_SIZE_2D)
                        else:
                            pam_input_size = (1, actual_n_channels, 128, 128, 128)
                        try:
                            pam_train_list, pam_train_stages = calculate_pam(
                                model, input_size=pam_input_size, mode='train', stage_wise=True, device=device
                            )
                            pam_inference_list, pam_inference_stages = calculate_pam(
                                model, input_size=pam_input_size, mode='inference', stage_wise=True, device=device
                            )
                            if pam_train_list:
                                pam_train_mean = sum(pam_train_list) / len(pam_train_list)
                                print(f"PAM (Train, batch_size=1): {pam_train_mean / 1024**2:.2f} MB (mean of {len(pam_train_list)} runs)")
                            if pam_inference_list:
                                pam_inference_mean = sum(pam_inference_list) / len(pam_inference_list)
                                print(f"PAM (Inference, batch_size=1): {pam_inference_mean / 1024**2:.2f} MB (mean of {len(pam_inference_list)} runs)")
                            if pam_train_stages:
                                print("PAM (Train) by stage:")
                                for stage_name, mem_list in sorted(pam_train_stages.items()):
                                    if mem_list:
                                        mem_mean = sum(mem_list) / len(mem_list)
                                        print(f"  {stage_name}: {mem_mean / 1024**2:.2f} MB")
                            if pam_inference_stages:
                                print("PAM (Inference) by stage:")
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
                    
                    # 체크포인트 저장 경로 (실험 결과 폴더 내부)
                    if use_5fold:
                        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_fold_{fold_idx}_best.pth")
                    else:
                        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_best.pth")
                    
                    # 훈련
                    train_result = train_model(
                        model, train_loader, val_loader, test_loader, epochs, device=device, model_name=model_name, seed=seed,
                        train_sampler=train_sampler, rank=rank,
                        sw_patch_size=(128, 128, 128), sw_overlap=0.5, dim=dim, use_nnunet_loss=use_nnunet_loss,
                        results_dir=results_dir, ckpt_path=ckpt_path, train_crops_per_center=train_crops_per_center,
                        dataset_version=dataset_version,
                        data_dir=data_path,  # Cascade 모델 validation용
                        cascade_infer_cfg=cascade_infer_cfg,  # Cascade 모델 validation용
                        coord_type=coord_type,  # Cascade 모델 validation용
                        preprocessed_dir=os.path.join(preprocessed_base_dir, dataset_version.upper()) if preprocessed_base_dir else None,  # Cascade 모델 validation용
                        use_5fold=use_5fold,  # Cascade 모델 validation용
                        fold_idx=fold_idx,  # Cascade 모델 validation용
                        fold_split_dir=fold_split_dir,  # Cascade 모델 validation용
                    )
                    # BRATS2024는 RC 포함, 다른 버전은 RC 없음
                    if dataset_version == 'brats2024' and len(train_result) >= 9:
                        train_losses, val_dices, epoch_results, best_epoch, best_val_dice, best_val_wt, best_val_tc, best_val_et, best_val_rc = train_result
                    else:
                        train_losses, val_dices, epoch_results, best_epoch, best_val_dice, best_val_wt, best_val_tc, best_val_et = train_result
                        best_val_rc = 0.0
                    
                    # FLOPs 계산 (모델이 device에 있는 상태에서)
                    if dim == '2d':
                        flops = calculate_flops(model, input_size=(1, actual_n_channels, *INPUT_SIZE_2D))
                    else:
                        flops = calculate_flops(model, input_size=(1, actual_n_channels, *INPUT_SIZE_3D))
                    if is_main_process(rank):
                        print(f"FLOPs: {flops:,}")
                    
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
                                print(f"  Min: {inference_latency_stats['min']:.2f} ms, Max: {inference_latency_stats['max']:.2f} ms")
                                print(f"  P50: {inference_latency_stats['p50']:.2f} ms, P95: {inference_latency_stats['p95']:.2f} ms, P99: {inference_latency_stats['p99']:.2f} ms")
                        except Exception as e:
                            print(f"Warning: Failed to calculate inference latency: {e}")
                            inference_latency_list = []
                            inference_latency_stats = {}
                    
                    # 최종 평가: Best 모델 로드 후 Test set 평가 (all ranks)
                    if is_main_process(rank):
                        print(f"\nLoading best model (epoch {best_epoch}, Val Dice: {best_val_dice:.4f}) for final test evaluation...")
                    # Best 체크포인트에서 모델 로드 (실험 결과 폴더 내부에서)
                    if use_5fold:
                        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_fold_{fold_idx}_best.pth")
                    else:
                        ckpt_path = os.path.join(results_dir, f"{model_name}_seed_{seed}_best.pth")
                    if os.path.exists(ckpt_path):
                        try:
                            real_model = model.module if hasattr(model, 'module') else model
                            state = torch.load(ckpt_path, map_location=device)
                            real_model.load_state_dict(state, strict=False)
                            if is_main_process(rank):
                                print(f"Loaded best checkpoint from {ckpt_path}")
                        except Exception as e:
                            if is_main_process(rank):
                                print(f"Warning: Failed to load checkpoint from {ckpt_path}: {e}")
                                print("Continuing with current model state...")
                    else:
                        if is_main_process(rank):
                            print(f"Warning: Checkpoint not found at {ckpt_path}, using current model state...")

                    # Switch RepLK blocks to deploy mode (before final test evaluation)
                    # RepLK blocks are fused into single 7x7x7 depthwise conv for efficient inference
                    # Check if model name starts with any RepLK model prefix (supports all sizes: xs, s, m, l)
                    replk_model_prefixes = [
                        'dualbranch_04_unet_',
                        'dualbranch_05_unet_',
                        'dualbranch_06_unet_',
                        'dualbranch_07_unet_',
                    ]
                    if any(model_name.startswith(prefix) for prefix in replk_model_prefixes):
                        real_model = model.module if hasattr(model, 'module') else model
                        if hasattr(real_model, 'switch_to_deploy'):
                            if is_main_process(rank):
                                print(f"Switching RepLK blocks to deploy mode (fusing branches)...")
                            real_model.switch_to_deploy()
                            # Recalculate parameters and FLOPs after deploy mode (fewer params/FLOPs due to fused branches)
                            total_params = sum(p.numel() for p in real_model.parameters())
                            if is_main_process(rank):
                                print(f"RepLK blocks switched to deploy mode.")
                                print(f"Parameters after deploy: {total_params:,} (branches fused)")
                            # Recalculate FLOPs for deploy mode
                            if dim == '2d':
                                flops = calculate_flops(model, input_size=(1, n_channels, *INPUT_SIZE_2D))
                            else:
                                flops = calculate_flops(model, input_size=(1, actual_n_channels, *INPUT_SIZE_3D))
                            if is_main_process(rank):
                                print(f"FLOPs after deploy: {flops:,}")
                            # Recalculate PAM after deploy mode (may be different due to fused branches)
                            if is_main_process(rank):
                                try:
                                    if dim == '2d':
                                        input_size_after_deploy = (1, actual_n_channels, *INPUT_SIZE_2D)
                                    else:
                                        input_size_after_deploy = (1, actual_n_channels, *INPUT_SIZE_3D)
                                    pam_inference_list_after_deploy, _ = calculate_pam(
                                        model, input_size=input_size_after_deploy, mode='inference', stage_wise=True, device=device
                                    )
                                    if pam_inference_list_after_deploy:
                                        pam_inference_mean_after_deploy = sum(pam_inference_list_after_deploy) / len(pam_inference_list_after_deploy)
                                        print(f"PAM (Inference after deploy, batch_size=1): {pam_inference_mean_after_deploy / 1024**2:.2f} MB (mean of {len(pam_inference_list_after_deploy)} runs)")
                                except Exception as e:
                                    print(f"Warning: Failed to recalculate PAM after deploy: {e}")

                    # Test set 평가
                    # evaluate_model이 cascade 모델을 자동으로 감지하여 ROI 기반 평가 수행
                    # cascade_infer_cfg에서 crops_per_center, crop_overlap, use_blending, batch_size, roi_batch_size 가져오기
                    eval_crops_per_center = cascade_infer_cfg.get('crops_per_center', 1) if cascade_infer_cfg else 1
                    eval_crop_overlap = cascade_infer_cfg.get('crop_overlap', 0.5) if cascade_infer_cfg else 0.5
                    eval_use_blending = cascade_infer_cfg.get('use_blending', True) if cascade_infer_cfg else True
                    eval_batch_size = cascade_infer_cfg.get('batch_size', 1) if cascade_infer_cfg else 1
                    eval_roi_batch_size = cascade_infer_cfg.get('roi_batch_size', None) if cascade_infer_cfg else None
                    
                    metrics = evaluate_model(
                        model,
                        test_loader,
                        device,
                        model_name,
                        distributed=distributed,
                        world_size=world_size,
                        sw_patch_size=(128, 128, 128),
                        sw_overlap=0.5,
                        results_dir=results_dir,
                        coord_type=coord_type,
                        dataset_version=dataset_version,
                        data_dir=data_path,  # Cascade 모델 평가용
                        seed=seed,  # Cascade 모델 평가용
                        use_5fold=use_5fold,  # Cascade 모델 평가용
                        fold_idx=fold_idx,  # Cascade 모델 평가용
                        fold_split_dir=fold_split_dir,  # Cascade 모델 평가용
                        preprocessed_dir=os.path.join(preprocessed_base_dir, dataset_version.upper()) if preprocessed_base_dir else None,  # Cascade 모델 평가용
                        crops_per_center=eval_crops_per_center,  # cascade_infer_cfg에서 가져온 값
                        crop_overlap=eval_crop_overlap,  # cascade_infer_cfg에서 가져온 값
                        use_blending=eval_use_blending,  # cascade_infer_cfg에서 가져온 값
                        batch_size=eval_batch_size,  # cascade_infer_cfg에서 가져온 값
                        roi_batch_size=eval_roi_batch_size,  # cascade_infer_cfg에서 가져온 값
                    )
                    
                    cascade_metrics = None
                    # Cascade 모델인 경우 cascade_metrics도 별도로 저장 (호환성 유지)
                    if model_name.startswith('cascade_'):
                        cascade_metrics = {
                            'mean': metrics.get('dice', 0.0),
                            'wt': metrics.get('wt', 0.0),
                            'tc': metrics.get('tc', 0.0),
                            'et': metrics.get('et', 0.0),
                        }
                        if dataset_version == 'brats2024' and 'rc' in metrics:
                            cascade_metrics['rc'] = metrics.get('rc', 0.0)
                        if is_main_process(rank):
                            if dataset_version == 'brats2024' and 'rc' in cascade_metrics:
                                print(
                                    f"Cascade ROI→Seg Dice: {cascade_metrics['mean']:.4f} "
                                    f"(WT {cascade_metrics['wt']:.4f} | TC {cascade_metrics['tc']:.4f} | ET {cascade_metrics['et']:.4f} | RC {cascade_metrics['rc']:.4f})"
                                )
                            else:
                                print(
                                    f"Cascade ROI→Seg Dice: {cascade_metrics['mean']:.4f} "
                                    f"(WT {cascade_metrics['wt']:.4f} | TC {cascade_metrics['tc']:.4f} | ET {cascade_metrics['et']:.4f})"
                                )
                    
                    # 결과 저장 (각 run마다 하나의 행만 생성, PAM과 Latency는 평균값 사용)
                    # PAM과 Latency 평균값 계산
                    pam_train_mean = sum(pam_train_list) / len(pam_train_list) if pam_train_list else 0
                    pam_inference_mean = sum(pam_inference_list) / len(pam_inference_list) if pam_inference_list else 0
                    latency_mean = sum(inference_latency_list) / len(inference_latency_list) if inference_latency_list else 0
                    
                    if is_main_process(rank):
                        # 메인 결과 저장
                        result = create_result_dict(
                            dataset_version=dataset_version,
                            seed=seed,
                            fold_idx=fold_idx if use_5fold else None,
                            model_name=model_name,
                            total_params=total_params,
                            flops=flops,
                            pam_train_mean=pam_train_mean,
                            pam_inference_mean=pam_inference_mean,
                            latency_mean=latency_mean,
                            metrics=metrics,
                            best_val_dice=best_val_dice,
                            best_val_wt=best_val_wt,
                            best_val_tc=best_val_tc,
                            best_val_et=best_val_et,
                            best_epoch=best_epoch,
                            cascade_metrics=cascade_metrics,
                            roi_model_name=roi_model_name_for_result,
                            coord_type=coord_type,  # 좌표 타입 추가
                            best_val_rc=best_val_rc if dataset_version == 'brats2024' else None
                        )
                        all_results.append(result)
                        
                        # Stage별 PAM 결과 저장
                        if pam_train_stages:
                            for stage_name, mem_list in pam_train_stages.items():
                                stage_result = create_stage_pam_result(
                                    dataset_version=dataset_version,
                                    seed=seed,
                                    fold_idx=fold_idx if use_5fold else None,
                                    model_name=model_name,
                                    mode='train',
                                    stage_name=stage_name,
                                    mem_list=mem_list
                                )
                                if stage_result:
                                    all_stage_pam_results.append(stage_result)
                        
                        if pam_inference_stages:
                            for stage_name, mem_list in pam_inference_stages.items():
                                stage_result = create_stage_pam_result(
                                    dataset_version=dataset_version,
                                    seed=seed,
                                    fold_idx=fold_idx if use_5fold else None,
                                    model_name=model_name,
                                    mode='inference',
                                    stage_name=stage_name,
                                    mem_list=mem_list
                                )
                                if stage_result:
                                    all_stage_pam_results.append(stage_result)
                    
                    # 모든 epoch 결과 저장 (test_dice는 최종 평가 값으로 업데이트)
                    if is_main_process(rank):
                        for epoch_result in epoch_results:
                            epoch_data = create_epoch_result_dict(
                                dataset_version=dataset_version,
                                seed=seed,
                                fold_idx=fold_idx if use_5fold else None,
                                model_name=model_name,
                                epoch_result=epoch_result,
                                best_epoch=best_epoch,
                                test_dice=metrics['dice']
                            )
                            all_epochs_results.append(epoch_data)
                    
                    if is_main_process(rank):
                        # precision과 recall이 None일 수 있으므로 안전하게 처리
                        prec_str = f"{metrics['precision']:.4f}" if metrics.get('precision') is not None else "N/A"
                        rec_str = f"{metrics['recall']:.4f}" if metrics.get('recall') is not None else "N/A"
                        
                        if dataset_version == 'brats2024' and 'rc' in metrics:
                            print(f"Final Val Dice: {best_val_dice:.4f} (WT {best_val_wt:.4f} | TC {best_val_tc:.4f} | ET {best_val_et:.4f} | RC {best_val_rc:.4f}) (epoch {best_epoch})")
                            print(f"Final Test Dice: {metrics['dice']:.4f} (WT {metrics['wt']:.4f} | TC {metrics['tc']:.4f} | ET {metrics['et']:.4f} | RC {metrics['rc']:.4f}) | Prec {prec_str} Rec {rec_str}")
                        else:
                            print(f"Final Val Dice: {best_val_dice:.4f} (WT {best_val_wt:.4f} | TC {best_val_tc:.4f} | ET {best_val_et:.4f}) (epoch {best_epoch})")
                            print(f"Final Test Dice: {metrics['dice']:.4f} (WT {metrics['wt']:.4f} | TC {metrics['tc']:.4f} | ET {metrics['et']:.4f}) | Prec {prec_str} Rec {rec_str}")
                
                except Exception as e:
                    # 모든 프로세스에서 에러 로깅 (디버깅을 위해)
                    import traceback
                    error_msg = f"[Rank {rank}] Error with {model_name}: {e}"
                    print(error_msg)
                    print(f"[Rank {rank}] Full traceback:")
                    traceback.print_exc()
                    # Main process에서만 추가 정보 출력
                    if is_main_process(rank):
                        print(f"\n{'='*60}")
                        print(f"FAILED: {model_name.upper()}")
                        print(f"Error: {e}")
                        print(f"{'='*60}\n")
                    continue
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    # 결과가 비어있는 경우 처리
    if results_df.empty:
        if is_main_process(rank) or not distributed:
            print("\n" + "="*80)
            print("WARNING: No results were collected. All experiments failed.")
            print("="*80)
            print("Please check the error messages above for details.")
            print("Common issues:")
            print("  - Model import errors (check if model file exists)")
            print("  - Invalid model name or parameters")
            print("  - Data loading errors")
            print("  - CUDA/device errors")
            print("="*80 + "\n")
        return results_dir, results_df
    
    # CSV로 저장
    save_results_to_csv(
        results_dir=results_dir,
        all_results=all_results,
        all_epochs_results=all_epochs_results,
        all_stage_pam_results=all_stage_pam_results,
        is_main_process=(is_main_process(0) or not distributed)
    )
    
    # Epochs DataFrame 생성 (분석용)
    epochs_df = pd.DataFrame(all_epochs_results) if all_epochs_results else None
    
    # 모델별 성능 비교 분석
    if is_main_process(0) or not distributed:
        print("\nCreating model comparison analysis...")
    
    # 모델별 평균 성능
    try:
        agg_dict = {
            'test_dice': ['mean', 'std'],
            'val_dice': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }
        # PAM이 있는 경우에만 추가
        if 'pam_train' in results_df.columns:
            agg_dict['pam_train'] = ['mean', 'std']
        if 'pam_inference' in results_df.columns:
            agg_dict['pam_inference'] = ['mean', 'std']
        # Inference Latency 추가
        if 'inference_latency_ms' in results_df.columns:
            agg_dict['inference_latency_ms'] = ['mean', 'std']
        
        model_comparison = results_df.groupby('model_name').agg(agg_dict).round(4)
        if is_main_process(0) or not distributed:
            comparison_path = os.path.join(results_dir, "model_comparison.csv")
            model_comparison.to_csv(comparison_path)
            print(f"Model comparison saved to: {comparison_path}")
    except KeyError as e:
        print(f"Warning: Could not create model comparison: {e}")
    
    # 시각화 생성
    if is_main_process(0) or not distributed:
        print("\nCreating visualization charts...")
        try:
            create_comprehensive_analysis(results_df, epochs_df, results_dir)
            create_interactive_3d_plot(results_df, results_dir)
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    # 결과 출력
    if (is_main_process(0) or not distributed) and not results_df.empty:
        print("\n" + "="*80)
        print("3D SEGMENTATION INTEGRATED EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        print("\n--- Model Performance Summary ---")
        for model_name in available_models:
            model_results = results_df[results_df['model_name'] == model_name]
            if not model_results.empty:
                print(f"\n{model_name.upper()} Model:")
                for _, row in model_results.iterrows():
                    print(f"  Seed {row['seed']:3d} | Test Dice: {row['test_dice']:.4f} | Val Dice: {row['val_dice']:.4f} | Params: {row['total_params']:,}")
        
        # 모델별 평균 성능
        print("\n--- Model-wise Average Performance ---")
        for model_name in available_models:
            model_results = results_df[results_df['model_name'] == model_name]
            if not model_results.empty:
                avg_dice = model_results['test_dice'].mean()
                avg_val_dice = model_results['val_dice'].mean()
                avg_params = model_results['total_params'].mean()
                print(f"{model_name.upper():12}: Test Dice {avg_dice:.4f} | Val Dice {avg_val_dice:.4f} | Avg Params {avg_params:,.0f}")
    
    # cleanup
    if distributed:
        cleanup_distributed()
    return results_dir, results_df


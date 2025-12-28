"""
Model Evaluation
모델 평가 관련 함수
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

from utils.experiment_utils import sliding_window_inference_3d
from metrics import calculate_wt_tc_et_dice  # , calculate_wt_tc_et_hd95  # HD95 계산 비활성화 (distributed timeout 방지)
from dataloaders import get_coord_map
from models.modules.se_modules import SEBlock3D
from models.modules.cbam_modules import CBAM3D, ChannelAttention3D
from utils.runner.cascade_evaluation import evaluate_segmentation_with_roi, load_roi_model_from_checkpoint


def evaluate_model(model, test_loader, device='cuda', model_name: str = 'model', distributed: bool = False, world_size: int = 1,
                   sw_patch_size=(128, 128, 128), sw_overlap=0.5, results_dir: str = None, coord_type: str = 'none', dataset_version: str = 'brats2021',
                   data_dir: str = None, seed: int = None, use_5fold: bool = False, fold_idx: int = None, 
                   fold_split_dir: str = None, preprocessed_dir: str = None):
    """모델 평가 함수
    
    Cascade 모델인 경우 자동으로 ROI 기반 평가를 수행합니다.
    Cascade 모델 평가를 위해서는 data_dir, seed 등의 추가 파라미터가 필요합니다.
    """
    # Cascade 모델인 경우 자동으로 ROI 기반 평가 수행
    if model_name.startswith('cascade_'):
        # ROI 기반 평가에 필요한 파라미터 확인
        if data_dir is None or seed is None:
            raise ValueError(
                f"Cascade models require data_dir and seed parameters for ROI-based evaluation. "
                f"Please provide data_dir and seed when calling evaluate_model for cascade models."
            )
        
        # ROI 모델 경로 찾기
        possible_roi_model_names = ['roi_unet3d_small', 'roi_mobileunetr3d_tiny', 'roi_mobileunetr3d_small']
        roi_model_name = None
        roi_weight_path = None
        
        # 1. results_dir에서 ROI 모델 체크포인트 찾기
        if results_dir:
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
        
        # 2. 기본 경로 시도
        if not roi_weight_path:
            for candidate_roi_name in possible_roi_model_names:
                default_path = f"models/weights/cascade/roi_model/{candidate_roi_name}/seed_{seed}/weights/best.pth"
                if os.path.exists(default_path):
                    roi_weight_path = default_path
                    roi_model_name = candidate_roi_name
                    break
        
        if not roi_weight_path or not os.path.exists(roi_weight_path):
            # ROI 모델을 찾지 못한 경우 에러 발생
            error_msg = f"[Cascade] Error: ROI model not found for cascade model {model_name}.\n"
            error_msg += f"  Cascade models require ROI-based evaluation.\n"
            error_msg += f"  Please ensure the ROI model checkpoint exists."
            raise FileNotFoundError(error_msg)
        
        # ROI 모델 로드
        # ROI 모델은 항상 4채널(no coords) 또는 2채널(no coords) 고정으로 사용
        roi_model, use_4modalities = load_roi_model_from_checkpoint(
            roi_model_name,
            roi_weight_path,
            device,
        )
        
        real_model = model.module if hasattr(model, 'module') else model
        
        # Cascade ROI 기반 평가 수행
        # ROI 모델은 항상 4채널(no coords) 고정
        # Segmentation 모델만 coord_type에 따라 동작
        cascade_metrics = evaluate_segmentation_with_roi(
            seg_model=real_model,
            roi_model=roi_model,
            data_dir=data_dir,
            dataset_version=dataset_version,
            seed=seed,
            roi_resize=(64, 64, 64),
            crop_size=(96, 96, 96),
            include_coords=False,  # ROI 모델은 항상 coords 사용 안 함
            coord_encoding_type='simple',  # ROI 모델은 coords 사용 안 하므로 의미 없음
            use_5fold=use_5fold,
            fold_idx=fold_idx,
            fold_split_dir=fold_split_dir,
            crops_per_center=1,
            crop_overlap=0.5,
            use_blending=True,
            results_dir=results_dir,
            model_name=model_name,
            preprocessed_dir=preprocessed_dir,
            roi_use_4modalities=use_4modalities,
        )
        
        # cascade_metrics를 evaluate_model과 동일한 형식으로 변환
        result = {
            'dice': cascade_metrics.get('mean', 0.0),
            'wt': cascade_metrics.get('wt', 0.0),
            'tc': cascade_metrics.get('tc', 0.0),
            'et': cascade_metrics.get('et', 0.0),
            'hd95_wt': None,
            'hd95_tc': None,
            'hd95_et': None,
            'hd95_mean': None,
            'precision': None,
            'recall': None,
        }
        if dataset_version == 'brats2024' and 'rc' in cascade_metrics:
            result['rc'] = cascade_metrics.get('rc', 0.0)
        
        return result
    
    model.eval()
    real_model = model.module if hasattr(model, 'module') else model
    test_dice = 0.0
    test_wt_sum = test_tc_sum = test_et_sum = test_rc_sum = 0.0
    n_te = 0
    precision_scores = []
    recall_scores = []
    # HD95 계산 비활성화 (distributed timeout 방지)
    # hd95_wt_sum = hd95_tc_sum = hd95_et_sum = 0.0
    # hd95_wt_count = hd95_tc_count = hd95_et_count = 0
    is_brats2024 = (dataset_version == 'brats2024')
    save_examples = results_dir is not None
    rank0 = True
    if distributed and hasattr(torch, 'distributed') and torch.distributed.is_available():
        if torch.distributed.is_initialized():
            rank0 = torch.distributed.get_rank() == 0
    save_examples = save_examples and rank0

    collect_se = (results_dir is not None) and rank0
    se_blocks = []
    se_excitation_data = {}
    if collect_se:
        for name, module in real_model.named_modules():
            if isinstance(module, SEBlock3D):
                se_blocks.append((name, module))
                se_excitation_data[name] = []
    
    collect_cbam = (results_dir is not None) and rank0
    cbam_blocks = []
    channel_attention_blocks = []  # 블록 내부의 ChannelAttention3D도 수집
    cbam_channel_data = {}
    cbam_spatial_data = {}
    channel_attention_data = {}  # 블록 내부 Channel Attention 가중치
    if collect_cbam:
        for name, module in real_model.named_modules():
            if isinstance(module, CBAM3D):
                cbam_blocks.append((name, module))
                cbam_channel_data[name] = []
                cbam_spatial_data[name] = []
            elif isinstance(module, ChannelAttention3D):
                # 블록 내부의 ChannelAttention3D도 수집 (예: ShuffleNetV1Unit3D 내부)
                channel_attention_blocks.append((name, module))
                channel_attention_data[name] = []
        if rank0:
            print(f"[CBAM Debug] Found {len(cbam_blocks)} CBAM blocks: {[name for name, _ in cbam_blocks]}")
            if channel_attention_blocks:
                print(f"[CBAM Debug] Found {len(channel_attention_blocks)} ChannelAttention3D blocks (inside units): {[name for name, _ in channel_attention_blocks]}")
    example_dir = None
    example_limit = 10
    examples_saved = 0
    if save_examples:
        example_dir = os.path.join(results_dir, f'qualitative_examples_{model_name}')
        os.makedirs(example_dir, exist_ok=True)
    
    # 모달리티별 어텐션 가중치 수집 (quadbranch_4modal_attention_unet_s만)
    collect_attention = (model_name == 'quadbranch_4modal_attention_unet_s')
    all_attention_weights = []  # 각 샘플별 어텐션 가중치 저장
    
    # MobileViT attention 가중치 수집 (모델에 MobileViT 블록이 있으면 자동 수집)
    collect_mvit_attention = (results_dir is not None) and rank0
    all_mvit_attention_weights = []  # 각 샘플별 MobileViT attention 가중치 저장
    mvit_blocks_found = []
    if collect_mvit_attention:
        # MobileViT 블록이 모델에 있는지 확인
        try:
            from models.modules.mvit_modules import MobileViT3DBlock, MobileViT3DBlockV3
            for name, module in real_model.named_modules():
                if isinstance(module, (MobileViT3DBlock, MobileViT3DBlockV3)):
                    mvit_blocks_found.append((name, module))
            
            # 모델의 forward 메서드가 return_attention을 지원하는지 확인
            import inspect
            sig = inspect.signature(real_model.forward)
            has_return_attention = 'return_attention' in sig.parameters
            
            if len(mvit_blocks_found) > 0 and has_return_attention:
                if rank0:
                    print(f"[MobileViT Debug] Found {len(mvit_blocks_found)} MobileViT blocks: {[name for name, _ in mvit_blocks_found]}")
                    print(f"[MobileViT Debug] Model supports return_attention parameter")
            elif len(mvit_blocks_found) > 0 and not has_return_attention:
                # MobileViT 블록은 있지만 return_attention을 지원하지 않음
                collect_mvit_attention = False
                if rank0:
                    print(f"[MobileViT Debug] Found {len(mvit_blocks_found)} MobileViT blocks but model does not support return_attention")
            else:
                collect_mvit_attention = False
        except ImportError:
            collect_mvit_attention = False
        except Exception as e:
            if rank0:
                print(f"[MobileViT Debug] Error checking for MobileViT blocks: {e}")
            collect_mvit_attention = False
    
    with torch.no_grad():
        seg_cmap = ListedColormap(['black', '#ff0000', '#00ff00', '#0000ff'])
        cm_accum = np.zeros((4, 4), dtype=np.int64)
        for batch_data in test_loader:
            # 포그라운드 좌표가 포함될 수 있으므로 처리
            if len(batch_data) == 3:
                inputs, labels, _ = batch_data  # fg_coords_dict 무시
            else:
                inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 2D/3D 분기: 2D 모델은 그대로, 3D 모델은 depth 차원 추가
            if model_name not in ['mobile_unetr', 'mobile_unetr_3d'] and len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(2)
                labels = labels.unsqueeze(2)
            
            # 3D 테스트: 슬라이딩 윈도우 추론
            # 모든 3D 모델은 전체 볼륨을 처리하기 위해 슬라이딩 윈도우 사용
            # 단, CBAM/SE 가중치 수집을 위해서는 전체 볼륨에 대해 직접 forward 호출 필요
            if inputs.dim() == 5 and inputs.size(0) == 1:  # 3D 볼륨
                # CBAM 또는 SE 가중치 수집이 필요한 경우, 전체 볼륨에 대해 직접 forward 호출
                if (collect_cbam and len(cbam_blocks) > 0) or (collect_se and len(se_blocks) > 0):
                    try:
                        # 전체 볼륨에 대해 직접 forward (가중치 수집을 위해)
                        logits = model(inputs)
                    except RuntimeError as e:
                        # 메모리 부족 시 슬라이딩 윈도우 사용 (가중치 수집 불가)
                        if "out of memory" in str(e).lower():
                            if rank0:
                                print(f"Warning: OOM during CBAM/SE weight collection, using sliding window (weights may not be collected)")
                            # Cascade 모델인 경우 좌표 추가
                            if model_name.startswith('cascade_') and coord_type != 'none':
                                coord_map = get_coord_map(inputs.shape[2:], device=device, encoding_type=coord_type)
                                coord_map = coord_map.unsqueeze(0)
                                inputs = torch.cat([inputs, coord_map], dim=1)
                            logits = sliding_window_inference_3d(
                                model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name, coord_type=coord_type
                            )
                        else:
                            raise
                elif collect_attention:
                    # 어텐션 가중치 수집을 위해 슬라이딩 윈도우에서 직접 호출
                    # 슬라이딩 윈도우는 어텐션 가중치를 평균내야 하므로, 여기서는 전체 볼륨에 대해 직접 호출
                    real_model = model.module if hasattr(model, 'module') else model
                    if hasattr(real_model, 'forward') and 'return_attention' in real_model.forward.__code__.co_varnames:
                        # 전체 볼륨에 대해 직접 forward (슬라이딩 윈도우 없이, 메모리 허용 시)
                        # 실제로는 슬라이딩 윈도우를 사용하되, 각 패치의 어텐션을 평균내야 함
                        # 간단하게 전체 볼륨에 대해 직접 호출 (메모리 허용 시)
                        try:
                            logits, attention_dict = real_model(inputs, return_attention=True)
                            # 어텐션 가중치 저장 (평균 가중치 사용)
                            avg_weights = attention_dict['average'].cpu().numpy()  # [B, 4]
                            all_attention_weights.append(avg_weights[0])  # 첫 번째 샘플 (batch_size=1)
                        except RuntimeError as e:
                            # 메모리 부족 시 슬라이딩 윈도우 사용 (어텐션 수집 불가)
                            if "out of memory" in str(e).lower():
                                print(f"Warning: OOM during attention collection, using sliding window without attention")
                                # Cascade 모델인 경우 좌표 추가
                                if model_name.startswith('cascade_') and coord_type != 'none':
                                    coord_map = get_coord_map(inputs.shape[2:], device=device, encoding_type=coord_type)
                                    coord_map = coord_map.unsqueeze(0)
                                    inputs = torch.cat([inputs, coord_map], dim=1)
                                logits = sliding_window_inference_3d(
                                    model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name, coord_type=coord_type
                                )
                            else:
                                raise
                    else:
                        # Cascade 모델인 경우 좌표 추가
                        if model_name.startswith('cascade_') and coord_type != 'none':
                            coord_map = get_coord_map(inputs.shape[2:], device=device, encoding_type=coord_type)
                            coord_map = coord_map.unsqueeze(0)
                            inputs = torch.cat([inputs, coord_map], dim=1)
                        logits = sliding_window_inference_3d(
                            model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name, coord_type=coord_type
                        )
                elif collect_mvit_attention:
                    # MobileViT attention 가중치 수집 (이미 위에서 모듈 존재 및 return_attention 지원 확인됨)
                    try:
                        # 전체 볼륨에 대해 직접 forward (메모리 허용 시)
                        logits, attention_dict = real_model(inputs, return_attention=True)
                        # MobileViT attention 가중치 저장
                        all_mvit_attention_weights.append(attention_dict)
                    except RuntimeError as e:
                        # 메모리 부족 시 슬라이딩 윈도우 사용 (attention 수집 불가)
                        if "out of memory" in str(e).lower():
                            if rank0:
                                print(f"Warning: OOM during MobileViT attention collection, using sliding window without attention")
                            # Cascade 모델인 경우 좌표 추가
                            if model_name.startswith('cascade_') and coord_type != 'none':
                                coord_map = get_coord_map(inputs.shape[2:], device=device, encoding_type=coord_type)
                                coord_map = coord_map.unsqueeze(0)
                                inputs = torch.cat([inputs, coord_map], dim=1)
                            logits = sliding_window_inference_3d(
                                model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name, coord_type=coord_type
                            )
                        else:
                            raise
                else:
                    # Cascade 모델인 경우 좌표 추가
                    if model_name.startswith('cascade_') and coord_type != 'none':
                        # 전체 볼륨에 좌표 추가
                        coord_map = get_coord_map(inputs.shape[2:], device=device, encoding_type=coord_type)
                        coord_map = coord_map.unsqueeze(0)  # (1, C_coord, H, W, D)
                        inputs = torch.cat([inputs, coord_map], dim=1)  # (1, C+C_coord, H, W, D)
                    logits = sliding_window_inference_3d(
                        model, inputs, patch_size=sw_patch_size, overlap=sw_overlap, device=device, model_name=model_name, coord_type=coord_type
                    )
            else:
                if collect_attention:
                    real_model = model.module if hasattr(model, 'module') else model
                    if hasattr(real_model, 'forward') and 'return_attention' in real_model.forward.__code__.co_varnames:
                        logits, attention_dict = real_model(inputs, return_attention=True)
                        avg_weights = attention_dict['average'].cpu().numpy()  # [B, 4]
                        for i in range(avg_weights.shape[0]):
                            all_attention_weights.append(avg_weights[i])
                    else:
                        logits = model(inputs)
                else:
                    logits = model(inputs)
            
            # Dice score 계산 (WT/TC/ET, RC for BRATS2024)
            dice_scores = calculate_wt_tc_et_dice(logits, labels, dataset_version=dataset_version)
            mean_dice = dice_scores.mean()
            bsz = inputs.size(0)
            test_dice += mean_dice.item() * bsz
            test_wt_sum += float(dice_scores[0].item()) * bsz
            test_tc_sum += float(dice_scores[1].item()) * bsz
            test_et_sum += float(dice_scores[2].item()) * bsz
            if is_brats2024 and len(dice_scores) >= 4:
                test_rc_sum += float(dice_scores[3].item()) * bsz
            n_te += bsz
            
            # Precision, Recall 계산 (클래스별)
            pred = torch.argmax(logits, dim=1)
            # HD95 계산 비활성화 (distributed timeout 방지)
            # hd95_batch = calculate_wt_tc_et_hd95(pred, labels)
            # if hd95_batch.size > 0:
            #     for hd_wt, hd_tc, hd_et in hd95_batch:
            #         if np.isfinite(hd_wt):
            #             hd95_wt_sum += float(hd_wt)
            #             hd95_wt_count += 1
            #         if np.isfinite(hd_tc):
            #             hd95_tc_sum += float(hd_tc)
            #             hd95_tc_count += 1
            #         if np.isfinite(hd_et):
            #             hd95_et_sum += float(hd_et)
            #             hd95_et_count += 1

            if collect_se and se_blocks:
                for block_name, block_module in se_blocks:
                    excitation = getattr(block_module, 'last_excitation', None)
                    if excitation is not None:
                        se_excitation_data[block_name].append(excitation.clone())
            
            # Collect CBAM weights (독립적으로 실행)
            if collect_cbam and cbam_blocks:
                for block_name, block_module in cbam_blocks:
                    channel_weights = getattr(block_module, 'last_channel_weights', None)
                    spatial_weights = getattr(block_module, 'last_spatial_weights', None)
                    if channel_weights is not None:
                        cbam_channel_data[block_name].append(channel_weights.clone())
                    else:
                        if rank0 and len(cbam_channel_data[block_name]) == 0:  # 첫 번째 배치에서만 경고
                            print(f"[CBAM Debug] Warning: No channel weights found for {block_name}")
                    if spatial_weights is not None:
                        cbam_spatial_data[block_name].append(spatial_weights.clone())
                    else:
                        if rank0 and len(cbam_spatial_data[block_name]) == 0:  # 첫 번째 배치에서만 경고
                            print(f"[CBAM Debug] Warning: No spatial weights found for {block_name}")
            
            # Collect ChannelAttention3D weights from inside blocks (e.g., ShuffleNetV1Unit3D)
            if collect_cbam and channel_attention_blocks:
                for block_name, block_module in channel_attention_blocks:
                    channel_weights = getattr(block_module, 'last_channel_weights', None)
                    if channel_weights is not None:
                        channel_attention_data[block_name].append(channel_weights.clone())
            pred_np = pred.cpu().numpy()
            labels_np = labels.cpu().numpy()
            # Accumulate 4-class confusion matrix (0..3)
            cm_batch = confusion_matrix(labels_np.flatten(), pred_np.flatten(), labels=list(range(4)))
            cm_accum += cm_batch
            
            for class_id in range(4):
                pred_class = (pred_np == class_id)
                true_class = (labels_np == class_id)
                
                if true_class.sum() > 0:
                    precision = (pred_class & true_class).sum() / pred_class.sum() if pred_class.sum() > 0 else 0
                    recall = (pred_class & true_class).sum() / true_class.sum()
                else:
                    precision = recall = 0
                    
                precision_scores.append(precision)
                recall_scores.append(recall)

            # 예시 이미지 저장
            if save_examples and examples_saved < example_limit:
                inputs_np = inputs.detach().cpu().numpy()
                for bi in range(bsz):
                    if examples_saved >= example_limit:
                        break
                    input_sample = inputs_np[bi]  # (C, D, H, W) or (C, H, W)
                    label_sample = labels_np[bi]  # (D, H, W) or (H, W)
                    pred_sample = pred_np[bi]     # (D, H, W) or (H, W)

                    # 3D 볼륨인 경우: (C, H, W, D) 및 (H, W, D)
                    if input_sample.ndim == 4:  # (C, H, W, D)
                        C, H, W, D = input_sample.shape
                        # FLAIR 채널 선택: 2채널이면 1번(T1CE, FLAIR 중 FLAIR), 4채널이면 3번(T1, T1CE, T2, FLAIR 중 FLAIR)
                        channel_idx = C - 1  # 항상 마지막 채널이 FLAIR
                        
                        # 마스크가 있는 슬라이스 찾기 (배경이 아닌 픽셀이 있는 슬라이스)
                        valid_slices = []
                        if label_sample.ndim == 3:  # (H, W, D)
                            for d_idx in range(D):
                                label_slice_2d = label_sample[:, :, d_idx]
                                if (label_slice_2d > 0).any():
                                    valid_slices.append(d_idx)
                        else:
                            # 2D인 경우
                            if (label_sample > 0).any():
                                valid_slices = [0]
                        
                        # 마스크가 있는 슬라이스가 없으면 스킵
                        if not valid_slices:
                            continue
                        
                        # 마스크가 있는 슬라이스 중에서 선택 (중간 부분 우선)
                        if len(valid_slices) > 1:
                            mid_idx = len(valid_slices) // 2
                            slice_idx = valid_slices[mid_idx]
                        else:
                            slice_idx = valid_slices[0]
                        
                        # 이미지 슬라이스 추출: (C, H, W, D) -> (H, W)
                        image_slice = input_sample[channel_idx, :, :, slice_idx]
                        
                        # 라벨/예측 슬라이스 추출: (H, W, D) -> (H, W)
                        if label_sample.ndim == 3:
                            label_slice = label_sample[:, :, slice_idx]
                            pred_slice = pred_sample[:, :, slice_idx]
                        else:
                            label_slice = label_sample
                            pred_slice = pred_sample
                    
                    else:  # 2D: (C, H, W) 및 (H, W)
                        C, H, W = input_sample.shape
                        channel_idx = min(C - 1, 0)  # FLAIR 채널
                        image_slice = input_sample[channel_idx, :, :]
                        
                        # 2D인 경우에도 마스크 확인
                        if label_sample.ndim == 2:
                            if not (label_sample > 0).any():
                                continue  # 마스크가 없으면 스킵
                            label_slice = label_sample
                            pred_slice = pred_sample
                        else:
                            # 3D 라벨이지만 2D 입력인 경우 (이상한 케이스)
                            continue

                    # 이미지 정규화
                    img_min, img_max = image_slice.min(), image_slice.max()
                    if img_max > img_min:
                        image_display = (image_slice - img_min) / (img_max - img_min)
                    else:
                        image_display = image_slice

                    # 시각화: 원본 이미지 위에 마스크 오버레이
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Input: 원본 이미지만
                    axes[0].imshow(image_display, cmap='gray', origin='lower')
                    axes[0].set_title(f'Input (FLAIR, slice {slice_idx if input_sample.ndim == 4 else "2D"})')
                    axes[0].axis('off')

                    # Ground Truth: 원본 이미지 + 마스크 오버레이
                    axes[1].imshow(image_display, cmap='gray', origin='lower')
                    # 마스크가 있는 부분만 오버레이 (alpha=0.5)
                    mask_gt = label_slice > 0
                    if mask_gt.any():
                        im_gt = axes[1].imshow(label_slice, cmap=seg_cmap, alpha=0.5, vmin=0, vmax=3, origin='lower')
                    axes[1].set_title('Ground Truth (overlay)')
                    axes[1].axis('off')

                    # Prediction: 원본 이미지 + 예측 마스크 오버레이
                    axes[2].imshow(image_display, cmap='gray', origin='lower')
                    # 예측 마스크가 있는 부분만 오버레이 (alpha=0.5)
                    mask_pred = pred_slice > 0
                    if mask_pred.any():
                        im_pred = axes[2].imshow(pred_slice, cmap=seg_cmap, alpha=0.5, vmin=0, vmax=3, origin='lower')
                    axes[2].set_title('Prediction (overlay)')
                    axes[2].axis('off')

                    plt.tight_layout()
                    example_path = os.path.join(
                        example_dir, f"{model_name}_example_{examples_saved + 1:02d}.png"
                    )
                    plt.savefig(example_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    examples_saved += 1

    
    test_dice /= max(1, n_te)
    test_wt = test_wt_sum / max(1, n_te)
    test_tc = test_tc_sum / max(1, n_te)
    test_et = test_et_sum / max(1, n_te)
    test_rc = test_rc_sum / max(1, n_te) if is_brats2024 else 0.0
    # Reduce across processes if distributed
    if distributed and world_size > 1:
        import torch.distributed as dist
        if is_brats2024:
            td = torch.tensor([test_dice, test_wt, test_tc, test_et, test_rc], device=device)
            dist.all_reduce(td, op=dist.ReduceOp.SUM)
            td = td / world_size
            test_dice, test_wt, test_tc, test_et, test_rc = td.tolist()
        else:
            td = torch.tensor([test_dice, test_wt, test_tc, test_et], device=device)
            dist.all_reduce(td, op=dist.ReduceOp.SUM)
            td = td / world_size
            test_dice, test_wt, test_tc, test_et = td.tolist()
        # HD95 계산 비활성화 (distributed timeout 방지)
        # hd_tensor = torch.tensor(
        #     [
        #         hd95_wt_sum,
        #         hd95_wt_count,
        #         hd95_tc_sum,
        #         hd95_tc_count,
        #         hd95_et_sum,
        #         hd95_et_count,
        #     ],
        #     device=device,
        # )
        # dist.all_reduce(hd_tensor, op=dist.ReduceOp.SUM)
        # (
        #     hd95_wt_sum,
        #     hd95_wt_count,
        #     hd95_tc_sum,
        #     hd95_tc_count,
        #     hd95_et_sum,
        #     hd95_et_count,
        # ) = hd_tensor.tolist()
        # Confusion matrix reduction is non-trivial without custom gather; skip CM plot on non-main ranks
    # HD95 계산 비활성화 (distributed timeout 방지)
    hd95_wt = None  # (hd95_wt_sum / hd95_wt_count) if hd95_wt_count > 0 else None
    hd95_tc = None  # (hd95_tc_sum / hd95_tc_count) if hd95_tc_count > 0 else None
    hd95_et = None  # (hd95_et_sum / hd95_et_count) if hd95_et_count > 0 else None
    hd95_mean = None  # (total_sum / total_count) if total_count > 0 else None
    
    # Background 제외한 평균 (클래스 1, 2, 3만)
    avg_precision = np.mean(precision_scores[1::4])  # 클래스별로 평균
    avg_recall = np.mean(recall_scores[1::4])
    
    # 모달리티별 어텐션 가중치 분석 및 저장
    if collect_attention and len(all_attention_weights) > 0:
        try:
            attention_array = np.array(all_attention_weights)  # [n_samples, 4]
            modality_names = ['T1', 'T1CE', 'T2', 'FLAIR']
            
            # 평균 기여도 계산
            mean_contributions = attention_array.mean(axis=0)  # [4]
            std_contributions = attention_array.std(axis=0)  # [4]
            
            # 결과 출력
            print(f"\n{'='*60}")
            print(f"Modality Attention Analysis - {model_name}")
            print(f"{'='*60}")
            print(f"Total samples analyzed: {len(all_attention_weights)}")
            print(f"\nAverage Modality Contributions:")
            for i, mod_name in enumerate(modality_names):
                print(f"  {mod_name:6s}: {mean_contributions[i]:.4f} ± {std_contributions[i]:.4f}")
            
            # CSV 저장
            if results_dir:
                attention_df = pd.DataFrame(attention_array, columns=modality_names)
                attention_df['sample_id'] = range(len(attention_array))
                attention_df = attention_df[['sample_id'] + modality_names]
                
                csv_path = os.path.join(results_dir, f'modality_attention_{model_name}.csv')
                attention_df.to_csv(csv_path, index=False)
                print(f"\nAttention weights saved to: {csv_path}")
                
                # 요약 통계 저장
                summary_df = pd.DataFrame({
                    'modality': modality_names,
                    'mean_contribution': mean_contributions,
                    'std_contribution': std_contributions
                })
                summary_path = os.path.join(results_dir, f'modality_attention_summary_{model_name}.csv')
                summary_df.to_csv(summary_path, index=False)
                print(f"Attention summary saved to: {summary_path}")
            
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Failed to analyze/save attention weights: {e}")
    
    # MobileViT attention 가중치 분석 및 저장
    if collect_mvit_attention and len(all_mvit_attention_weights) > 0:
        try:
            from utils.mvit_attention_utils import analyze_mvit_attention_weights, check_mvit_attention_learning
            
            # Attention 분석
            analysis_result = analyze_mvit_attention_weights(
                all_mvit_attention_weights,
                results_dir=results_dir,
                model_name=model_name,
            )
            
            # Attention 학습 상태 확인
            is_learning, message = check_mvit_attention_learning(all_mvit_attention_weights)
            print(f"\nMobileViT Attention Learning Status: {message}")
            if not is_learning:
                print(f"⚠️  Warning: MobileViT attention may not be learning properly!")
                print(f"   Consider:")
                print(f"   - Checking learning rate")
                print(f"   - Increasing training epochs")
                print(f"   - Adjusting MobileViT hyperparameters (num_heads, num_layers, etc.)")
            
        except Exception as e:
            print(f"Warning: Failed to analyze/save MobileViT attention weights: {e}")
            import traceback
            traceback.print_exc()
    
    # Save confusion matrix heatmap on main (or non-distributed)
    try:
        if (not distributed) or (distributed and world_size > 0):
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_accum, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['BG', 'NCR/NET', 'ED', 'ET'],
                        yticklabels=['BG', 'NCR/NET', 'ED', 'ET'], ax=ax)
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            cm_path = os.path.join(results_dir or '.', f'confusion_matrix_{model_name}.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            if results_dir:
                print(f"Confusion matrix saved to: {cm_path}")
    except Exception as _e:
        print(f"Warning: failed to save confusion matrix: {_e}")

    # Save SE excitation statistics and histograms
    if collect_se and se_blocks:
        se_summary_rows = []
        for block_name, data_list in se_excitation_data.items():
            if not data_list:
                continue
            try:
                block_tensor = torch.cat(data_list, dim=0)
            except RuntimeError:
                block_tensor = torch.cat([d.cpu() for d in data_list], dim=0)
            block_np = block_tensor.numpy()
            channel_means = block_np.mean(axis=0)
            channel_stds = block_np.std(axis=0)
            for ch_idx, (m, s) in enumerate(zip(channel_means, channel_stds)):
                se_summary_rows.append({
                    'block_name': block_name,
                    'channel': ch_idx,
                    'mean_excitation': float(m),
                    'std_excitation': float(s)
                })
            if results_dir:
                safe_name = block_name.replace('.', '_')
                hist_path = os.path.join(
                    results_dir, f'se_excitation_hist_{model_name}_{safe_name}.png'
                )
                plt.figure(figsize=(6, 4))
                plt.hist(block_np.flatten(), bins=40, range=(0, 1), color='steelblue', alpha=0.8)
                plt.title(f'SE Excitation Histogram\n{model_name} - {block_name}')
                plt.xlabel('Excitation Weight')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
        if se_summary_rows and results_dir:
            se_df = pd.DataFrame(se_summary_rows)
            csv_path = os.path.join(results_dir, f'se_excitation_summary_{model_name}.csv')
            se_df.to_csv(csv_path, index=False)
            print(f"SE excitation summary saved to: {csv_path}")

    # Save CBAM attention statistics and histograms
    if collect_cbam and cbam_blocks:
        if rank0:
            print(f"[CBAM Debug] Saving CBAM statistics: {len(cbam_blocks)} blocks found")
            for block_name, data_list in cbam_channel_data.items():
                print(f"[CBAM Debug] Block {block_name}: {len(data_list)} channel weight samples collected")
            for block_name, data_list in cbam_spatial_data.items():
                print(f"[CBAM Debug] Block {block_name}: {len(data_list)} spatial weight samples collected")
        
        # Channel Attention weights
        cbam_channel_summary_rows = []
        for block_name, data_list in cbam_channel_data.items():
            if not data_list:
                if rank0:
                    print(f"[CBAM Debug] Warning: No channel data for {block_name}, skipping")
                continue
            try:
                block_tensor = torch.cat(data_list, dim=0)
            except RuntimeError:
                block_tensor = torch.cat([d.cpu() for d in data_list], dim=0)
            block_np = block_tensor.numpy()  # (N, C) where N is number of samples
            channel_means = block_np.mean(axis=0)
            channel_stds = block_np.std(axis=0)
            for ch_idx, (m, s) in enumerate(zip(channel_means, channel_stds)):
                cbam_channel_summary_rows.append({
                    'block_name': block_name,
                    'channel': ch_idx,
                    'mean_channel_weight': float(m),
                    'std_channel_weight': float(s)
                })
            if results_dir:
                safe_name = block_name.replace('.', '_')
                hist_path = os.path.join(
                    results_dir, f'cbam_channel_hist_{model_name}_{safe_name}.png'
                )
                plt.figure(figsize=(6, 4))
                plt.hist(block_np.flatten(), bins=40, range=(0, 1), color='coral', alpha=0.8)
                plt.title(f'CBAM Channel Attention Histogram\n{model_name} - {block_name}')
                plt.xlabel('Channel Attention Weight')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
        if cbam_channel_summary_rows and results_dir:
            cbam_channel_df = pd.DataFrame(cbam_channel_summary_rows)
            csv_path = os.path.join(results_dir, f'cbam_channel_summary_{model_name}.csv')
            cbam_channel_df.to_csv(csv_path, index=False)
            print(f"CBAM channel attention summary saved to: {csv_path}")
        
        # Spatial Attention weights
        cbam_spatial_summary_rows = []
        for block_name, data_list in cbam_spatial_data.items():
            if not data_list:
                if rank0:
                    print(f"[CBAM Debug] Warning: No spatial data for {block_name}, skipping")
                continue
            try:
                block_tensor = torch.cat(data_list, dim=0)
            except RuntimeError:
                block_tensor = torch.cat([d.cpu() for d in data_list], dim=0)
            block_np = block_tensor.numpy()  # (N, 1, D, H, W) where N is number of samples
            # Flatten spatial dimensions for statistics
            spatial_means = block_np.mean(axis=(0, 1))  # Average over batch and channel, shape: (D, H, W)
            spatial_stds = block_np.std(axis=(0, 1))
            spatial_flat_means = spatial_means.flatten()
            spatial_flat_stds = spatial_stds.flatten()
            for idx, (m, s) in enumerate(zip(spatial_flat_means, spatial_flat_stds)):
                cbam_spatial_summary_rows.append({
                    'block_name': block_name,
                    'spatial_idx': idx,
                    'mean_spatial_weight': float(m),
                    'std_spatial_weight': float(s)
                })
            if results_dir:
                safe_name = block_name.replace('.', '_')
                hist_path = os.path.join(
                    results_dir, f'cbam_spatial_hist_{model_name}_{safe_name}.png'
                )
                plt.figure(figsize=(6, 4))
                plt.hist(block_np.flatten(), bins=40, range=(0, 1), color='mediumseagreen', alpha=0.8)
                plt.title(f'CBAM Spatial Attention Histogram\n{model_name} - {block_name}')
                plt.xlabel('Spatial Attention Weight')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
        if cbam_spatial_summary_rows and results_dir:
            cbam_spatial_df = pd.DataFrame(cbam_spatial_summary_rows)
            csv_path = os.path.join(results_dir, f'cbam_spatial_summary_{model_name}.csv')
            cbam_spatial_df.to_csv(csv_path, index=False)
            print(f"CBAM spatial attention summary saved to: {csv_path}")
    
    # Save ChannelAttention3D weights from inside blocks (e.g., ShuffleNetV1Unit3D)
    if collect_cbam and channel_attention_blocks:
        if rank0:
            print(f"[CBAM Debug] Saving ChannelAttention3D statistics: {len(channel_attention_blocks)} blocks found")
            for block_name, data_list in channel_attention_data.items():
                print(f"[CBAM Debug] Block {block_name}: {len(data_list)} channel weight samples collected")
        
        # Channel Attention weights from inside blocks
        channel_attention_summary_rows = []
        for block_name, data_list in channel_attention_data.items():
            if not data_list:
                if rank0:
                    print(f"[CBAM Debug] Warning: No channel attention data for {block_name}, skipping")
                continue
            try:
                block_tensor = torch.cat(data_list, dim=0)
            except RuntimeError:
                block_tensor = torch.cat([d.cpu() for d in data_list], dim=0)
            block_np = block_tensor.numpy()  # (N, C) where N is number of samples
            channel_means = block_np.mean(axis=0)
            channel_stds = block_np.std(axis=0)
            for ch_idx, (m, s) in enumerate(zip(channel_means, channel_stds)):
                channel_attention_summary_rows.append({
                    'block_name': block_name,
                    'channel': ch_idx,
                    'mean_channel_weight': float(m),
                    'std_channel_weight': float(s)
                })
            if results_dir:
                safe_name = block_name.replace('.', '_')
                hist_path = os.path.join(
                    results_dir, f'channel_attention_hist_{model_name}_{safe_name}.png'
                )
                plt.figure(figsize=(6, 4))
                plt.hist(block_np.flatten(), bins=40, range=(0, 1), color='purple', alpha=0.8)
                plt.title(f'Channel Attention Histogram (Inside Blocks)\n{model_name} - {block_name}')
                plt.xlabel('Channel Attention Weight')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
        if channel_attention_summary_rows and results_dir:
            channel_attention_df = pd.DataFrame(channel_attention_summary_rows)
            csv_path = os.path.join(results_dir, f'channel_attention_summary_{model_name}.csv')
            channel_attention_df.to_csv(csv_path, index=False)
            print(f"Channel Attention (inside blocks) summary saved to: {csv_path}")

    result = {
        'dice': test_dice,
        'wt': test_wt,
        'tc': test_tc,
        'et': test_et,
        'hd95_wt': hd95_wt,
        'hd95_tc': hd95_tc,
        'hd95_et': hd95_et,
        'hd95_mean': hd95_mean,
        'precision': avg_precision,
        'recall': avg_recall
    }
    if is_brats2024:
        result['rc'] = test_rc
    return result


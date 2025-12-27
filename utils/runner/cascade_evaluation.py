"""
Cascade Evaluation
Cascade ROI→Seg 파이프라인 평가 관련 함수
"""

import os
import torch

from utils.experiment_utils import get_roi_model
from utils.experiment_config import get_roi_model_config
from utils.cascade_utils import run_cascade_inference
from dataloaders import get_brats_base_datasets
from metrics import calculate_wt_tc_et_dice


def evaluate_cascade_pipeline(roi_model, seg_model, base_dataset, device,
                              roi_resize=(64, 64, 64), crop_size=(96, 96, 96), include_coords=True,
                              coord_encoding_type='simple',
                              crops_per_center=1, crop_overlap=0.5, use_blending=True,
                              collect_attention=False, results_dir=None, model_name='model'):
    """
    Run cascade inference on base dataset and compute WT/TC/ET dice.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
        collect_attention: True면 MobileViT attention weights 수집 및 분석
        results_dir: 결과 저장 디렉토리 (attention 분석용)
        model_name: 모델 이름 (attention 분석용)
    """
    roi_model.eval()
    seg_model.eval()
    dice_rows = []
    all_attention_weights = [] if collect_attention else None
    
    for idx in range(len(base_dataset)):
        # base_dataset이 (image, mask) 또는 (image, mask, fg_coords_dict) 반환 가능
        loaded_data = base_dataset[idx]
        if len(loaded_data) == 3:
            image, target, _ = loaded_data  # fg_coords_dict는 evaluation에서는 사용 안 함
        else:
            image, target = loaded_data
        image = image.to(device)
        target = target.to(device)
        result = run_cascade_inference(
            roi_model=roi_model,
            seg_model=seg_model,
            image=image,
            device=device,
            roi_resize=roi_resize,
            crop_size=crop_size,
            include_coords=include_coords,
            coord_encoding_type=coord_encoding_type,
            crops_per_center=crops_per_center,
            crop_overlap=crop_overlap,
            use_blending=use_blending,
            return_attention=collect_attention,
        )
        
        # Attention weights 수집 (dice 계산 전에 먼저 수집하여, 에러가 발생해도 보존)
        if collect_attention and all_attention_weights is not None:
            if 'attention_weights' in result and result['attention_weights']:
                all_attention_weights.extend(result['attention_weights'])
                if idx == 0:  # 첫 번째 샘플만 출력
                    print(f"[Cascade Evaluation] Collected attention weights from sample {idx+1}/{len(base_dataset)}")
        
        # full_logits shape 처리: run_cascade_inference는 (C, H, W, D) 형태를 반환
        full_logits_raw = result['full_logits'].to(device)
        if full_logits_raw.dim() == 4:  # (C, H, W, D) - 정상
            full_logits = full_logits_raw.unsqueeze(0)  # (1, C, H, W, D)
        elif full_logits_raw.dim() == 5:  # 이미 (1, C, H, W, D) - 이미 배치 차원 있음
            full_logits = full_logits_raw
        elif full_logits_raw.dim() == 6:  # (1, 1, C, H, W, D) - 배치 차원 중복
            # 첫 번째 배치 차원 제거
            full_logits = full_logits_raw.squeeze(0)  # (1, C, H, W, D)
            if full_logits.dim() == 6:
                # 여전히 6차원이면 다시 squeeze
                full_logits = full_logits.squeeze(0)  # (C, H, W, D)
                full_logits = full_logits.unsqueeze(0)  # (1, C, H, W, D)
        else:
            # 예상치 못한 shape인 경우 처리
            raise ValueError(f"Unexpected full_logits shape: {full_logits_raw.shape}, expected (C, H, W, D), (1, C, H, W, D), or (1, 1, C, H, W, D)")
        
        # target이 배치 차원이 없으면 추가, 있으면 그대로 사용
        if target.dim() == 3:  # (H, W, D)
            target_batch = target.unsqueeze(0)  # (1, H, W, D)
        elif target.dim() == 4:  # 이미 (1, H, W, D)
            target_batch = target
        else:
            target_batch = target.unsqueeze(0)  # 안전장치
        
        # Shape 확인 (디버깅용)
        if idx == 0:  # 첫 번째 샘플만 출력
            print(f"[Cascade Evaluation] full_logits.shape={full_logits.shape}, target_batch.shape={target_batch.shape}")
        
        dice = calculate_wt_tc_et_dice(full_logits, target_batch).detach().cpu()
        dice_rows.append(dice)
    
    if not dice_rows:
        return {'wt': 0.0, 'tc': 0.0, 'et': 0.0, 'mean': 0.0}
    dice_tensor = torch.stack(dice_rows, dim=0)
    mean_scores = dice_tensor.mean(dim=0)
    
    # MobileViT attention 분석
    if collect_attention:
        if all_attention_weights and len(all_attention_weights) > 0:
            print(f"\n[Cascade Evaluation] Analyzing {len(all_attention_weights)} attention weight samples...")
            if results_dir:
                try:
                    from utils.mvit_attention_utils import analyze_mvit_attention_weights, check_mvit_attention_learning
                    
                    analysis_result = analyze_mvit_attention_weights(
                        all_attention_weights,
                        results_dir=results_dir,
                        model_name=model_name,
                    )
                    
                    is_learning, message = check_mvit_attention_learning(all_attention_weights)
                    print(f"\nMobileViT Attention Learning Status (Cascade): {message}")
                    if not is_learning:
                        print(f"⚠️  Warning: MobileViT attention may not be learning properly!")
                except Exception as e:
                    print(f"Warning: Failed to analyze/save MobileViT attention weights: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[Cascade Evaluation] Warning: results_dir is None, skipping attention analysis")
        else:
            print(f"[Cascade Evaluation] Warning: No attention weights collected (collect_attention={collect_attention}, len={len(all_attention_weights) if all_attention_weights else 0})")
    
    return {
        'wt': float(mean_scores[0].item()),
        'tc': float(mean_scores[1].item()),
        'et': float(mean_scores[2].item()),
        'mean': float(mean_scores.mean().item())
    }


def load_roi_model_from_checkpoint(roi_model_name, weight_path, device, include_coords=True):
    """Load ROI model weights for inference.
    
    Automatically detects the number of input channels from checkpoint and adjusts include_coords accordingly.
    
    Returns:
        model: Loaded ROI model
        include_coords: Detected or provided include_coords value
    """
    cfg = get_roi_model_config(roi_model_name)
    
    # Load checkpoint first to detect input channels
    state = torch.load(weight_path, map_location=device)
    
    # Detect input channels from checkpoint
    # Try common first layer names for ROI models
    detected_channels = None
    for key in state.keys():
        if 'weight' in key and ('enc_blocks.0.net.0' in key or 'patch_embed' in key or 'conv' in key):
            if 'enc_blocks.0.net.0.weight' in key:
                # ROICascadeUNet3D: enc_blocks.0.net.0.weight shape is [out_channels, in_channels, ...]
                detected_channels = state[key].shape[1]
                break
            elif 'patch_embed' in key and 'weight' in key:
                # MobileUNETR_3D: patch_embed.proj.weight shape is [out_channels, in_channels, ...]
                detected_channels = state[key].shape[1]
                break
    
    # Auto-detect include_coords based on detected channels
    if detected_channels is not None:
        if detected_channels == 4:
            include_coords = False
            print(f"Detected 4-channel ROI model (no CoordConv). Using include_coords=False")
        elif detected_channels == 7:
            include_coords = True
            print(f"Detected 7-channel ROI model (with CoordConv). Using include_coords=True")
        else:
            print(f"Warning: Unexpected input channels {detected_channels} in checkpoint. Using provided include_coords={include_coords}")
    
    model = get_roi_model(
        roi_model_name,
        n_channels=7 if include_coords else 4,
        n_classes=2,
        roi_model_cfg=cfg,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, include_coords


def evaluate_segmentation_with_roi(
    seg_model,
    roi_model,
    data_dir,
    dataset_version,
    seed,
    roi_resize=(64, 64, 64),
    crop_size=(96, 96, 96),
    include_coords=True,
    coord_encoding_type='simple',
    use_5fold=False,
    fold_idx=None,
    max_samples=None,
    crops_per_center=1,
    crop_overlap=0.5,
    use_blending=True,
    results_dir=None,
    model_name='model',
    preprocessed_dir=None,
):
    """
    Evaluate trained segmentation model with pre-trained ROI detector.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
        results_dir: 결과 저장 디렉토리 (MobileViT attention 분석용)
        model_name: 모델 이름 (MobileViT attention 분석용)
    """
    _, _, test_base = get_brats_base_datasets(
        data_dir=data_dir,
        dataset_version=dataset_version,
        max_samples=max_samples,
        seed=seed,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        use_4modalities=True,
        preprocessed_dir=preprocessed_dir,
    )
    seg_model.eval()
    roi_model.eval()
    
    # MobileViT attention 수집 여부 확인
    collect_attention = False
    if results_dir is not None:
        try:
            from models.modules.mvit_modules import MobileViT3DBlock, MobileViT3DBlockV3
            real_model = seg_model.module if hasattr(seg_model, 'module') else seg_model
            mvit_blocks_found = []
            for name, module in real_model.named_modules():
                if isinstance(module, (MobileViT3DBlock, MobileViT3DBlockV3)):
                    mvit_blocks_found.append((name, module))
            
            import inspect
            sig = inspect.signature(real_model.forward)
            has_return_attention = 'return_attention' in sig.parameters
            
            if len(mvit_blocks_found) > 0 and has_return_attention:
                collect_attention = True
                print(f"[Cascade MobileViT] Found {len(mvit_blocks_found)} MobileViT blocks, will collect attention weights")
        except Exception as e:
            print(f"[Cascade MobileViT] Error checking for MobileViT blocks: {e}")
    
    return evaluate_cascade_pipeline(
        roi_model=roi_model,
        seg_model=seg_model,
        base_dataset=test_base,
        device=next(seg_model.parameters()).device,
        roi_resize=roi_resize,
        crop_size=crop_size,
        include_coords=include_coords,
        coord_encoding_type=coord_encoding_type,
        crops_per_center=crops_per_center,
        crop_overlap=crop_overlap,
        use_blending=use_blending,
        collect_attention=collect_attention,
        results_dir=results_dir,
        model_name=model_name,
    )


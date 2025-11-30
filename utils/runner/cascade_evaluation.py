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
                              crops_per_center=1, crop_overlap=0.5, use_blending=True):
    """
    Run cascade inference on base dataset and compute WT/TC/ET dice.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
    """
    roi_model.eval()
    seg_model.eval()
    dice_rows = []
    for idx in range(len(base_dataset)):
        image, target = base_dataset[idx]
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
            crops_per_center=crops_per_center,
            crop_overlap=crop_overlap,
            use_blending=use_blending,
        )
        full_logits = result['full_logits'].unsqueeze(0).to(device)
        target_batch = target.unsqueeze(0)
        dice = calculate_wt_tc_et_dice(full_logits, target_batch).detach().cpu()
        dice_rows.append(dice)
    if not dice_rows:
        return {'wt': 0.0, 'tc': 0.0, 'et': 0.0, 'mean': 0.0}
    dice_tensor = torch.stack(dice_rows, dim=0)
    mean_scores = dice_tensor.mean(dim=0)
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
    use_5fold=False,
    fold_idx=None,
    max_samples=None,
    crops_per_center=1,
    crop_overlap=0.5,
    use_blending=True,
):
    """
    Evaluate trained segmentation model with pre-trained ROI detector.
    
    Args:
        crops_per_center: 각 중심당 crop 개수 (1=단일 crop, 2=2x2x2=8개, 3=3x3x3=27개)
        crop_overlap: crop 간 겹침 비율 (0.0 ~ 1.0)
        use_blending: True면 cosine blending, False면 voxel-wise max
    """
    _, _, test_base = get_brats_base_datasets(
        data_dir=data_dir,
        dataset_version=dataset_version,
        max_samples=max_samples,
        seed=seed,
        use_5fold=use_5fold,
        fold_idx=fold_idx,
        use_4modalities=True,
    )
    seg_model.eval()
    roi_model.eval()
    return evaluate_cascade_pipeline(
        roi_model=roi_model,
        seg_model=seg_model,
        base_dataset=test_base,
        device=next(seg_model.parameters()).device,
        roi_resize=roi_resize,
        crop_size=crop_size,
        include_coords=include_coords,
        crops_per_center=crops_per_center,
        crop_overlap=crop_overlap,
        use_blending=use_blending,
    )


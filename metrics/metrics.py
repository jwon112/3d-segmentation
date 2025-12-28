import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure


def calculate_dice_score(pred, target, smooth=1e-5, num_classes=4):
    """
    Dice score 계산 함수
    
    Args:
        pred: 모델 출력 logits (B, C, H, W) 또는 (B, C, H, W, D)
        target: Ground truth 라벨 (B, H, W) 또는 (B, H, W, D)
        smooth: Smoothing factor (기본값: 1e-5)
        num_classes: 클래스 수 (기본값: 4, 0=배경, 1-3=포그라운드)
    
    Returns:
        클래스별 Dice 점수 (num_classes,)
    """
    pred = torch.argmax(pred, dim=1)
    
    # num_classes를 명시적으로 지정 (예측이 모두 0이어도 클래스 0-3 모두 계산)
    # target에 실제로 나타나는 최대 클래스도 확인하여 더 큰 값 사용
    actual_num_classes = max(num_classes, target.max().item() + 1)
    
    if len(pred.shape) == 3:  # 2D: (B, H, W)
        target_one_hot = F.one_hot(target, num_classes=actual_num_classes).permute(0, 3, 1, 2).float()
        pred_one_hot = F.one_hot(pred, num_classes=actual_num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
        union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    else:  # 3D: (B, H, W, D)
        target_one_hot = F.one_hot(target, num_classes=actual_num_classes).permute(0, 4, 1, 2, 3).float()
        pred_one_hot = F.one_hot(pred, num_classes=actual_num_classes).permute(0, 4, 1, 2, 3).float()
        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3, 4))
        union = pred_one_hot.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    
    # Dice 계산: (2 * intersection + smooth) / (union + smooth)
    # 배치 차원 평균 (B, C) -> (C,)
    intersection_mean = intersection.mean(dim=0)  # (C,)
    union_mean = union.mean(dim=0)  # (C,)
    
    # Dice 계산
    dice = (2.0 * intersection_mean + smooth) / (union_mean + smooth)
    
    # GT에 클래스가 존재하는지 확인 (target에서 각 클래스가 나타나는지)
    # target_one_hot: (B, C, ...), sum over spatial dims -> (B, C)
    spatial_dims = tuple(range(2, target_one_hot.dim()))
    target_sum = target_one_hot.sum(dim=spatial_dims)  # (B, C)
    target_exists = (target_sum > 0).any(dim=0)  # (C,) - 배치 중 하나라도 존재하면 True
    
    # GT에 존재하지 않는 클래스 처리:
    # - GT에 없고 예측도 없으면: union=0, dice=smooth/smooth=1.0 (수동 계산과 일치)
    # - 하지만 의미 있는 메트릭이 아니므로, GT에 실제로 존재하는 클래스만 계산
    # - GT에 없는 클래스는 Dice를 0으로 설정 (평가에서 제외)
    dice = torch.where(target_exists, dice, torch.zeros_like(dice))
    
    return dice



def calculate_wt_tc_et_dice(logits, target, smooth: float = 1e-5, dataset_version: str = 'brats2021'):
    """Compute BraTS composite region Dice: WT, TC, ET (and RC for BRATS2024).

    For BRATS2021 and earlier:
        Assumes target labels are mapped to 0..3 with: 0=BG, 1=NCR/NET, 2=ED, 3=ET.
        WT = 1 ∪ 2 ∪ 3, TC = 1 ∪ 3, ET = 3.
    
    For BRATS2024:
        Assumes target labels are 0..4 with: 0=BG, 1=NETC, 2=SNFH, 3=ET, 4=RC.
        WT = 1 ∪ 2 ∪ 3 ∪ 4 (all tumor regions including RC), TC = 1 ∪ 3, ET = 3, RC = 4.

    Returns: 
        - For BRATS2024: tensor of shape (4,) -> [WT, TC, ET, RC]
        - For other versions: tensor of shape (3,) -> [WT, TC, ET]
    """
    # #region agent log
    log_path = r"d:\강의\성균관대\연구실\연구\3D segmentation\code\.cursor\debug.log"
    try:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "eval-check",
                "hypothesisId": "H3",
                "location": "metrics.py:76",
                "message": "calculate_wt_tc_et_dice entry",
                "data": {
                    "logits_shape": list(logits.shape),
                    "target_shape": list(target.shape),
                    "logits_num_classes": logits.shape[1],
                    "target_min": int(target.min().item()),
                    "target_max": int(target.max().item()),
                    "target_unique": [int(x) for x in torch.unique(target).cpu().tolist()],
                    "dataset_version": dataset_version
                },
                "timestamp": int(time.time() * 1000)
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
    
    # Argmax predictions
    pred = torch.argmax(logits, dim=1)  # (B, H, W[, D])
    
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "eval-check",
                "hypothesisId": "H3",
                "location": "metrics.py:100",
                "message": "After argmax",
                "data": {
                    "pred_shape": list(pred.shape),
                    "pred_min": int(pred.min().item()),
                    "pred_max": int(pred.max().item()),
                    "pred_unique": [int(x) for x in torch.unique(pred).cpu().tolist()],
                    "pred_class_counts": dict(zip(*[int(x) for x in torch.unique(pred, return_counts=True)[0].cpu().tolist()], 
                                                   [int(x) for x in torch.unique(pred, return_counts=True)[1].cpu().tolist()]))
                },
                "timestamp": int(time.time() * 1000)
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion

    # Binary masks for regions
    def to_bool(x):
        return x.to(dtype=torch.bool)

    if dataset_version == 'brats2024':
        # BRATS2024: 5 classes (0=BG, 1=NETC, 2=SNFH, 3=ET, 4=RC)
        # WT = 1 ∪ 2 ∪ 3 ∪ 4 (all tumor regions including RC)
        # TC = 1 ∪ 3 (NETC + ET, excluding RC and SNFH)
        # ET = 3 (Enhancing Tumor only)
        # RC = 4 (Resection Cavity only)
        pred_wt = to_bool((pred == 1) | (pred == 2) | (pred == 3) | (pred == 4))
        pred_tc = to_bool((pred == 1) | (pred == 3))
        pred_et = to_bool(pred == 3)
        pred_rc = to_bool(pred == 4)

        tgt_wt = to_bool((target == 1) | (target == 2) | (target == 3) | (target == 4))
        tgt_tc = to_bool((target == 1) | (target == 3))
        tgt_et = to_bool(target == 3)
        tgt_rc = to_bool(target == 4)
    else:
        # Other BRATS versions: 4 classes (0=BG, 1=NCR/NET, 2=ED, 3=ET)
        # WT = 1 ∪ 2 ∪ 3, TC = 1 ∪ 3, ET = 3
        pred_wt = to_bool((pred == 1) | (pred == 2) | (pred == 3))
        pred_tc = to_bool((pred == 1) | (pred == 3))
        pred_et = to_bool(pred == 3)
        pred_rc = None  # RC 없음

        tgt_wt = to_bool((target == 1) | (target == 2) | (target == 3))
        tgt_tc = to_bool((target == 1) | (target == 3))
        tgt_et = to_bool(target == 3)
        tgt_rc = None  # RC 없음

    def dice_bin(pb, tb):
        # Compute per-sample dice then average over batch
        # spatial_dims: 배치 차원(0)을 제외한 모든 공간 차원
        # pb/tb shape: (B, H, W[, D]) 또는 (H, W[, D])
        if pb.dim() == 0:
            # Scalar case (shouldn't happen, but handle gracefully)
            return torch.tensor(0.0, device=pb.device, dtype=pb.dtype)
        
        # pb와 tb의 차원이 같아야 함
        if pb.dim() != tb.dim():
            raise ValueError(f"pb and tb must have same number of dimensions, got pb.dim()={pb.dim()}, tb.dim()={tb.dim()}")
        
        # 공간 차원 계산: 배치 차원(0)을 제외한 모든 차원
        # (B, H, W, D) -> spatial_dims = (1, 2, 3)
        # (H, W, D) -> spatial_dims = (0, 1, 2)
        ndim = pb.dim()
        if ndim == 1:
            # 1D: (B,) 또는 (H,)
            spatial_dims = (0,)
        elif ndim >= 2:
            # 2D 이상: 배치 차원(0)을 제외한 나머지
            # range(1, ndim)은 1부터 ndim-1까지이므로 안전
            spatial_dims = tuple(range(1, ndim))
        else:
            # 0D는 이미 처리됨
            return torch.tensor(0.0, device=pb.device, dtype=pb.dtype)
        
        try:
            # 공간 차원에 대해 sum 연산 수행
            inter = (pb & tb).sum(dim=spatial_dims).float()
            union = pb.sum(dim=spatial_dims).float() + tb.sum(dim=spatial_dims).float()
            d = (2.0 * inter + smooth) / (union + smooth)
            
            # If tb is empty across sample, set dice to 0 (exclude meaningless perfect)
            has_t = (tb.sum(dim=spatial_dims) > 0)
            d = torch.where(has_t, d, torch.zeros_like(d))
            
            # 배치 차원이 있으면 평균, 없으면 그대로 반환
            if d.dim() > 0:
                return d.mean()
            else:
                return d
        except IndexError as e:
            # 디버깅 정보 출력
            import warnings
            warnings.warn(
                f"dice_bin IndexError: pb.shape={pb.shape}, tb.shape={tb.shape}, "
                f"pb.dim()={pb.dim()}, tb.dim()={tb.dim()}, spatial_dims={spatial_dims}, error={e}"
            )
            raise

    wt = dice_bin(pred_wt, tgt_wt)
    tc = dice_bin(pred_tc, tgt_tc)
    et = dice_bin(pred_et, tgt_et)
    
    if dataset_version == 'brats2024':
        # BRATS2024: RC도 계산
        rc = dice_bin(pred_rc, tgt_rc)
        return torch.stack([wt, tc, et, rc])
    else:
        # Other BRATS versions: WT, TC, ET만 반환
        return torch.stack([wt, tc, et])


def _compute_surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing: tuple) -> np.ndarray:
    """Compute pairwise surface distances between two binary masks."""
    if not mask_a.any() and not mask_b.any():
        return np.zeros(1, dtype=np.float32)

    structure = generate_binary_structure(mask_a.ndim, 1)

    def extract_surface(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return mask
        eroded = binary_erosion(mask, structure=structure, border_value=0)
        surface = mask ^ eroded
        if not surface.any():
            return mask
        return surface

    surface_a = extract_surface(mask_a)
    surface_b = extract_surface(mask_b)

    if not surface_a.any() or not surface_b.any():
        # One side empty - return max possible distance (diagonal of volume)
        diag = np.sqrt(np.sum((np.array(mask_a.shape) * np.array(spacing)) ** 2))
        return np.array([diag], dtype=np.float32)

    dt_b = distance_transform_edt(~surface_b, sampling=spacing)
    dt_a = distance_transform_edt(~surface_a, sampling=spacing)

    distances_a_to_b = dt_b[surface_a]
    distances_b_to_a = dt_a[surface_b]

    return np.concatenate([distances_a_to_b, distances_b_to_a]).astype(np.float32)


def _hd95_for_region(pred_region: np.ndarray, gt_region: np.ndarray, spacing: tuple) -> float:
    if not gt_region.any() and not pred_region.any():
        return 0.0
    if not gt_region.any() or not pred_region.any():
        diag = np.sqrt(np.sum((np.array(pred_region.shape) * np.array(spacing)) ** 2))
        return diag
    distances = _compute_surface_distances(pred_region, gt_region, spacing)
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def calculate_wt_tc_et_hd95(pred: torch.Tensor, target: torch.Tensor, spacing=None) -> np.ndarray:
    """
    Calculate 95th percentile Hausdorff Distance (HD95) for WT, TC, ET regions.

    Args:
        pred: Predicted labels (B, H, W[, D])
        target: Ground truth labels (B, H, W[, D])
        spacing: tuple of voxel spacings. If None, isotropic spacing of 1 is used.

    Returns:
        numpy array of shape (B, 3) containing HD95 for WT, TC, ET per sample.
    """
    if spacing is None:
        spacing = tuple([1.0] * (target.dim() - 1))
    elif isinstance(spacing, (int, float)):
        spacing = tuple([float(spacing)] * (target.dim() - 1))

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    batch_scores = []
    for pred_sample, target_sample in zip(pred_np, target_np):
        pred_wt = ((pred_sample == 1) | (pred_sample == 2) | (pred_sample == 3))
        pred_tc = ((pred_sample == 1) | (pred_sample == 3))
        pred_et = (pred_sample == 3)

        tgt_wt = ((target_sample == 1) | (target_sample == 2) | (target_sample == 3))
        tgt_tc = ((target_sample == 1) | (target_sample == 3))
        tgt_et = (target_sample == 3)

        wt_hd = _hd95_for_region(pred_wt, tgt_wt, spacing)
        tc_hd = _hd95_for_region(pred_tc, tgt_tc, spacing)
        et_hd = _hd95_for_region(pred_et, tgt_et, spacing)

        batch_scores.append([wt_hd, tc_hd, et_hd])

    return np.array(batch_scores, dtype=np.float32)


import torch
import torch.nn.functional as F


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



def calculate_wt_tc_et_dice(logits, target, smooth: float = 1e-5):
    """Compute BraTS composite region Dice: WT, TC, ET.

    Assumes target labels are mapped to 0..3 with: 0=BG, 1=NCR/NET, 2=ED, 3=ET.
    WT = 1 ∪ 2 ∪ 3, TC = 1 ∪ 3, ET = 3.

    Returns: tensor of shape (3,) -> [WT, TC, ET]
    """
    # Argmax predictions
    pred = torch.argmax(logits, dim=1)  # (B, H, W[, D])

    # Binary masks for regions
    def to_bool(x):
        return x.to(dtype=torch.bool)

    pred_wt = to_bool((pred == 1) | (pred == 2) | (pred == 3))
    pred_tc = to_bool((pred == 1) | (pred == 3))
    pred_et = to_bool(pred == 3)

    tgt_wt = to_bool((target == 1) | (target == 2) | (target == 3))
    tgt_tc = to_bool((target == 1) | (target == 3))
    tgt_et = to_bool(target == 3)

    def dice_bin(pb, tb):
        # Compute per-sample dice then average over batch
        spatial_dims = tuple(range(1, pb.dim()))
        inter = (pb & tb).sum(dim=spatial_dims).float()
        union = pb.sum(dim=spatial_dims).float() + tb.sum(dim=spatial_dims).float()
        d = (2.0 * inter + smooth) / (union + smooth)
        # If tb is empty across sample, set dice to 0 (exclude meaningless perfect)
        has_t = (tb.sum(dim=spatial_dims) > 0)
        d = torch.where(has_t, d, torch.zeros_like(d))
        return d.mean()

    wt = dice_bin(pred_wt, tgt_wt)
    tc = dice_bin(pred_tc, tgt_tc)
    et = dice_bin(pred_et, tgt_et)
    return torch.stack([wt, tc, et])


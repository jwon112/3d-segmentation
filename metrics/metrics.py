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
    
    # GT에 클래스가 존재하는지 확인 (target에서 각 클래스가 나타나는지)
    # target_one_hot: (B, C, ...), sum over spatial dims -> (B, C)
    spatial_dims = tuple(range(2, target_one_hot.dim()))
    target_sum = target_one_hot.sum(dim=spatial_dims)  # (B, C)
    target_exists = (target_sum > 0).any(dim=0).float()  # (C,) - 배치 중 하나라도 존재하면 1.0
    
    # Dice 계산: GT에 존재하지 않는 클래스는 union=0, intersection=0
    # -> (2*0 + smooth) / (0 + smooth) = 1.0 (잘못된 결과)
    # 이를 방지하기 위해: union이 0이면 Dice도 0으로 설정
    dice = (2.0 * intersection_mean + smooth) / (union_mean + smooth)
    
    # GT에 존재하지 않거나 예측도 없는 클래스(union=0)는 Dice를 0으로 설정
    dice = torch.where(union_mean > 0, dice, torch.zeros_like(dice))
    
    return dice



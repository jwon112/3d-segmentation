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
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean(dim=0)



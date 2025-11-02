import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-5):
    """Standard Dice Loss"""
    pred = F.softmax(pred, dim=1)
    if len(pred.shape) == 4:
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    else:
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def soft_dice_loss_with_squared_pred(pred, target, smooth=1e-5):
    """
    nnU-Net style Soft Dice Loss with Squared Prediction
    
    Squared prediction을 사용하여 작은 예측값에 더 큰 페널티를 줍니다.
    이는 클래스 불균형 문제에 더 강건합니다.
    
    Args:
        pred: 모델 출력 logits (B, C, H, W) 또는 (B, C, H, W, D)
        target: Ground truth 라벨 (B, H, W) 또는 (B, H, W, D)
        smooth: Smoothing factor
    
    Returns:
        Dice loss (scalar)
    """
    pred = F.softmax(pred, dim=1)
    pred = pred ** 2  # Squared prediction (nnU-Net style)
    
    if len(pred.shape) == 4:  # 2D: (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    else:  # 3D: (B, C, H, W, D)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target, alpha=0.5):
    """
    Standard combined loss: CE + Dice
    alpha: Cross Entropy weight (default 0.5 = 50% CE, 50% Dice)
    """
    ce_loss = F.cross_entropy(pred, target)
    d_loss = dice_loss(pred, target)
    return alpha * ce_loss + (1 - alpha) * d_loss


def combined_loss_nnunet_style(pred, target, alpha=0.3):
    """
    nnU-Net style combined loss
    
    - Uses Soft Dice Loss with Squared Prediction
    - Dice Loss 우선 (70%): alpha=0.3 means CE 30%, Dice 70%
    - This is more robust to class imbalance
    
    Args:
        pred: 모델 출력 logits
        target: Ground truth 라벨
        alpha: Cross Entropy weight (default 0.3 = 30% CE, 70% Dice)
    
    Returns:
        Combined loss (scalar)
    """
    ce_loss = F.cross_entropy(pred, target)
    d_loss = soft_dice_loss_with_squared_pred(pred, target)
    return alpha * ce_loss + (1 - alpha) * d_loss



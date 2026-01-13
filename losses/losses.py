import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """nnUNet 스타일 Robust Cross Entropy Loss
    
    target tensor의 차원과 타입을 자동으로 처리합니다.
    이는 nnUNet의 RobustCrossEntropyLoss와 동일한 구현입니다.
    
    Args:
        input: 모델 출력 logits (B, C, H, W) 또는 (B, C, H, W, D)
        target: Ground truth 라벨 (B, H, W) 또는 (B, H, W, D) 또는 (B, 1, H, W, D)
    
    Returns:
        Cross Entropy Loss (scalar)
    """
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target이 (B, 1, H, W, D) 형태인 경우 처리
        if target.ndim == input.ndim:
            assert target.shape[1] == 1, \
                f"Expected target shape[1] == 1 when target.ndim == input.ndim, got {target.shape[1]}"
            target = target[:, 0]  # (B, H, W, D)로 변환
        
        # long 타입으로 변환
        return super().forward(input, target.long())


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
    - Uses RobustCrossEntropyLoss for better compatibility
    - Dice Loss 우선 (70%): alpha=0.3 means CE 30%, Dice 70%
    - This is more robust to class imbalance
    
    Args:
        pred: 모델 출력 logits
        target: Ground truth 라벨
        alpha: Cross Entropy weight (default 0.3 = 30% CE, 70% Dice)
    
    Returns:
        Combined loss (scalar)
    """
    ce_loss = RobustCrossEntropyLoss()(pred, target)
    d_loss = soft_dice_loss_with_squared_pred(pred, target)
    return alpha * ce_loss + (1 - alpha) * d_loss


class DeepSupervisionWrapper(nn.Module):
    """Deep Supervision을 위한 Loss Wrapper
    
    여러 레벨의 출력에 대해 loss를 계산하고 가중 합을 반환합니다.
    nnUNet의 DeepSupervisionWrapper와 동일한 구현입니다.
    
    Args:
        loss: 기본 loss 함수 (예: combined_loss_nnunet_style)
        weight_factors: 각 레벨의 가중치 (기본값: [1.0, 0.5, 0.25, 0.125])
                       깊은 레벨일수록 낮은 가중치
    
    Example:
        >>> base_loss = combined_loss_nnunet_style
        >>> ds_loss = DeepSupervisionWrapper(base_loss, weight_factors=[1.0, 0.5, 0.25, 0.125])
        >>> outputs = [main_output, ds2, ds3, ds4]  # 모델 출력
        >>> loss = ds_loss(outputs, target)
    """
    def __init__(self, loss, weight_factors=None):
        super(DeepSupervisionWrapper, self).__init__()
        if weight_factors is None:
            # 기본 가중치: 깊은 레벨일수록 낮은 가중치
            weight_factors = [1.0, 0.5, 0.25, 0.125]
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, net_output: Union[Tuple, List], target: torch.Tensor):
        """
        Args:
            net_output: 모델 출력 리스트/튜플 (예: [main_output, ds2, ds3, ds4])
            target: Ground truth 라벨 (B, H, W, D)
        
        Returns:
            가중 합산된 loss
        """
        assert isinstance(net_output, (tuple, list)), \
            f"net_output must be tuple or list, got {type(net_output)}"
        
        # 각 레벨의 출력에 대해 loss 계산
        losses = []
        for i, output in enumerate(net_output):
            # 출력을 target 크기에 맞게 다운샘플링 (필요한 경우)
            if output.shape[2:] != target.shape[1:]:
                output = F.interpolate(
                    output, size=target.shape[1:],
                    mode='trilinear', align_corners=False
                )
            
            # Loss 계산
            loss = self.loss(output, target)
            
            # 가중치 적용
            if i < len(self.weight_factors):
                losses.append(self.weight_factors[i] * loss)
            else:
                # 가중치가 지정되지 않은 경우 기본값 1.0
                losses.append(loss)
        
        return sum(losses)

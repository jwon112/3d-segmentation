import torch
import torch.nn.functional as F


def calculate_dice_score(pred, target, smooth=1e-5):
    pred = torch.argmax(pred, dim=1)
    if len(pred.shape) == 3:
        num_classes = max(pred.max().item() + 1, target.max().item() + 1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pred_one_hot = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
        union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    else:
        num_classes = max(pred.max().item() + 1, target.max().item() + 1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        pred_one_hot = F.one_hot(pred, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3, 4))
        union = pred_one_hot.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean(dim=0)



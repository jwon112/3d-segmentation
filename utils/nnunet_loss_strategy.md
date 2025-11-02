# nnU-Net의 클래스 불균형 처리 전략

## nnU-Net의 손실 함수 전략

### 1. 기본 손실 함수 조합

nnU-Net은 **Dice Loss + Cross Entropy Loss**를 사용하지만, 일반적인 조합과는 다른 점이 있습니다:

#### 표준 접근 (현재 코드):
```python
ce_loss = F.cross_entropy(pred, target)  # 클래스 가중치 없음
dice_loss = dice_loss(pred, target)
combined_loss = 0.5 * ce_loss + 0.5 * dice_loss
```

#### nnU-Net의 접근:
```python
# 1. Soft Dice Loss (squared prediction 사용)
soft_dice_loss = soft_dice_loss_with_squared_pred(pred, target)

# 2. Cross Entropy Loss (클래스 가중치 또는 샘플 가중치)
ce_loss = cross_entropy_loss(pred, target, ...)

# 3. Deep Supervision (여러 레벨에서 loss 계산)
# 4. 각 레벨에서 loss를 평균내거나 가중합
```

### 2. Soft Dice Loss with Squared Prediction

nnU-Net은 **squared prediction**을 사용한 Dice Loss를 사용합니다:

```python
def soft_dice_loss_with_squared_pred(pred, target):
    """
    Soft Dice Loss with squared predictions
    - Squared prediction: pred^2 (not just pred)
    - 이는 작은 예측값에 더 큰 페널티를 줌
    """
    pred = F.softmax(pred, dim=1)
    pred = pred ** 2  # Squared prediction!
    
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
    
    intersection = (pred * target_one_hot).sum()
    union = pred.sum() + target_one_hot.sum()
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice
```

**효과:**
- 작은 예측값에 더 큰 페널티 → 포그라운드 학습 강화
- 클래스 불균형에 더 강건

### 3. Deep Supervision

nnU-Net은 **Deep Supervision**을 사용합니다:

```python
# 여러 디코더 레벨에서 loss 계산
losses = []
for decoder_output in decoder_outputs:  # 여러 레벨
    dice_loss = soft_dice_loss(decoder_output, target)
    ce_loss = cross_entropy_loss(decoder_output, target)
    losses.append(dice_loss + ce_loss)

# 가중합 (깊은 레벨일수록 낮은 가중치)
final_loss = sum(w * l for w, l in zip(weights, losses))
```

**효과:**
- 각 레벨에서 포그라운드를 학습
- Gradient가 더 잘 전파됨
- 포그라운드 학습 기회 증가

### 4. 클래스 가중치 사용 여부

nnU-Net은 **명시적인 클래스 가중치를 사용하지 않습니다!**

대신:
- **Dice Loss**가 클래스 불균형에 상대적으로 강건
- **Squared prediction**으로 작은 클래스에 더 큰 페널티
- **Sampling strategy** (이미 적용함)로 포그라운드 보장

### 5. 샘플링 전략 (이미 적용됨 ✅)

nnU-Net의 표준 샘플링:
- 1/3: 포그라운드 오버샘플링
- 2/3: 완전 무작위 샘플링

**현재 코드에서 이미 구현됨!**

## nnU-Net vs 현재 구현 비교

| 항목 | 현재 구현 | nnU-Net |
|------|-----------|---------|
| Dice Loss | Standard Dice Loss | **Soft Dice with Squared Pred** |
| Cross Entropy | Standard CE (no weights) | Standard CE (no weights) |
| Loss 조합 | 0.5 * CE + 0.5 * Dice | **Dice 우선** (CE는 보조) |
| Deep Supervision | ❌ 없음 | ✅ 있음 |
| 클래스 가중치 | ❌ 없음 | ❌ 없음 (Dice로 처리) |
| 샘플링 | ✅ nnU-Net 스타일 | ✅ nnU-Net 스타일 |

## nnU-Net 스타일로 개선하기

### 우선순위 1: Soft Dice Loss with Squared Prediction ⭐⭐⭐

```python
def soft_dice_loss_with_squared_pred(pred, target, smooth=1e-5):
    """
    nnU-Net 스타일 Soft Dice Loss
    - Squared prediction 사용
    """
    pred = F.softmax(pred, dim=1)
    pred = pred ** 2  # Squared!
    
    if len(pred.shape) == 4:  # 2D
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    else:  # 3D
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss_nnunet_style(pred, target, alpha=0.3):
    """
    nnU-Net 스타일 손실 함수
    - Dice Loss 우선 (70%)
    - CE Loss 보조 (30%)
    """
    dice_loss = soft_dice_loss_with_squared_pred(pred, target)
    ce_loss = F.cross_entropy(pred, target)
    return (1 - alpha) * dice_loss + alpha * ce_loss
```

### 우선순위 2: Dice Loss 비중 증가 ⭐⭐

```python
# 현재: alpha=0.5 (CE 50%, Dice 50%)
# nnU-Net 스타일: alpha=0.3 (CE 30%, Dice 70%)
combined_loss = 0.3 * ce_loss + 0.7 * dice_loss
```

### 우선순위 3: Deep Supervision (선택적) ⭐

- 모델 아키텍처 수정 필요
- 복잡도 증가
- 현재는 우선순위 낮음

## 권장 사항

### 즉시 적용 가능 (High Impact):

1. **Soft Dice Loss with Squared Prediction**
   - 가장 효과적
   - 코드 변경 최소
   - 클래스 불균형에 강건

2. **Dice Loss 비중 증가**
   - alpha=0.3으로 변경 (CE 30%, Dice 70%)
   - 간단한 변경
   - 효과 기대

### 추가 개선 (Medium Priority):

3. **Deep Supervision** (모델 수정 필요)
4. **클래스 가중치** (nnU-Net은 사용 안 하지만, 추가 실험 가능)

## 결론

**nnU-Net은 명시적인 클래스 가중치를 사용하지 않습니다!**

대신:
- ✅ **Soft Dice Loss with Squared Prediction** (가장 중요!)
- ✅ **Dice Loss 비중 우선** (CE는 보조)
- ✅ **Deep Supervision** (gradient 전파)
- ✅ **Sampling strategy** (이미 적용됨)

**현재 가장 효과적인 개선: Soft Dice Loss with Squared Prediction 구현**


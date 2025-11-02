# 배경 붕괴(Background Collapse) 문제 분석

## 현재 상황

### ✅ Dice 계산 함수: 정상 작동
- 간단한 케이스 검증: 통과
- 배경 제외 평균 계산: 정상
- GT에 없는 클래스 처리: 정상 (Dice = 0)

### ❌ 모델 예측: 배경만 예측 (Background Collapse)

#### 학습 데이터
- **예측**: 클래스 [0] (100% 배경)
- **GT**: 클래스 [0, 1, 2, 3] (배경 + 3개 포그라운드)
- **배경 제외 평균 Dice**: 0.0000

#### 검증 데이터
- **예측**: 클래스 [0] (100% 배경)
- **GT**: 클래스 [0, 1, 2, 3] (배경 + 3개 포그라운드)
- **배경 제외 평균 Dice**: 0.0000

## 문제 원인 분석

### 1. 극심한 클래스 불균형 (Class Imbalance)

**데이터 분포:**
- 배경 (클래스 0): ~99% (예: 8,870,695 / 8,928,000 = 99.36%)
- 포그라운드 (클래스 1,2,3): ~1% (예: 57,305 / 8,928,000 = 0.64%)

**영향:**
- Cross Entropy Loss에서 배경을 예측하면 99% 확률로 정답
- 포그라운드를 예측하면 1% 확률로만 정답
- 모델이 "안전하게" 배경만 예측하는 전략을 학습

### 2. 손실 함수에 클래스 가중치 부재

**현재 손실 함수 (`combined_loss`):**
```python
def combined_loss(pred, target, alpha=0.5):
    ce_loss = F.cross_entropy(pred, target)  # 클래스 가중치 없음
    d_loss = dice_loss(pred, target)
    return alpha * ce_loss + (1 - alpha) * d_loss
```

**문제점:**
- `F.cross_entropy(pred, target)`: 모든 클래스에 동일한 가중치
- 배경 클래스가 99%이므로, 배경을 예측하는 것이 loss를 최소화
- 포그라운드 클래스에 대한 페널티가 부족

### 3. 학습 과정

**학습 시나리오:**
1. 모델 초기화 → 무작위 예측
2. 첫 학습 스텝: 배경 예측 시 loss가 낮음
3. 모델이 배경 예측을 선호하도록 학습
4. 포그라운드 학습 기회가 줄어듦 (배경으로만 예측)
5. 배경만 예측하는 모델로 수렴

## 해결 방안

### 방안 1: 클래스 가중치 추가 (추천 ⭐⭐⭐)

**Cross Entropy Loss에 클래스 가중치 적용:**
```python
# 클래스 빈도 계산 (예시)
class_weights = compute_class_weights(train_loader)
# 예: [1.0, 50.0, 50.0, 50.0] (배경=1, 포그라운드=50)

ce_loss = F.cross_entropy(pred, target, weight=class_weights)
```

**효과:**
- 포그라운드 예측 오류에 대한 페널티 증가
- 배경 예측만으로는 loss 최소화 불가능
- 포그라운드 학습 강제

### 방안 2: Focal Loss 사용 (추천 ⭐⭐)

**Focal Loss는 어려운 샘플(포그라운드)에 더 많은 가중치를 부여:**
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce)
    focal = alpha * (1 - pt) ** gamma * ce
    return focal.mean()
```

**효과:**
- 어려운 샘플(포그라운드)에 더 많은 관심
- 배경이 쉬워서 포커스가 줄어듦
- 포그라운드 학습 향상

### 방안 3: Dice Loss 가중치 증가 (추천 ⭐)

**Dice Loss는 클래스 불균형에 상대적으로 강건:**
```python
# 현재: alpha=0.5 (CE와 Dice 50:50)
# 변경: alpha=0.3 (CE 30%, Dice 70%)
combined_loss = 0.3 * ce_loss + 0.7 * dice_loss
```

**효과:**
- Dice Loss 비중 증가로 클래스 불균형 완화
- 하지만 근본적인 해결책은 아님

### 방안 4: 샘플링 전략 개선 (이미 적용됨 ✅)

**nnU-Net 스타일 샘플링:**
- 1/3: 포그라운드 오버샘플링
- 2/3: 완전 무작위 샘플링

**현재 상태:**
- 이미 적용되어 있음
- 하지만 손실 함수 문제로 효과 제한적

## 권장 조치

### 즉시 적용 (High Priority)

1. **클래스 가중치 추가**
   ```python
   # 학습 데이터에서 클래스 빈도 계산
   # 가중치: 1.0 / 클래스 빈도 (정규화)
   class_weights = torch.tensor([1.0, 50.0, 50.0, 50.0], device=device)
   ce_loss = F.cross_entropy(pred, target, weight=class_weights)
   ```

2. **손실 함수 수정**
   - `losses/losses.py`에 가중치 옵션 추가
   - `integrated_experiment.py`에서 가중치 계산 및 적용

### 추가 개선 (Medium Priority)

3. **Focal Loss 실험**
   - Focal Loss 구현 및 실험
   - 클래스 가중치와 비교

4. **학습률 조정**
   - 초기 학습률 증가로 포그라운드 학습 촉진
   - Learning rate scheduler 조정

## 다음 단계

1. ✅ Dice 계산 함수 버그 수정 완료
2. ⏳ 클래스 가중치 추가 (다음 단계)
3. ⏳ 실제 학습 재실행 및 성능 확인
4. ⏳ 필요 시 Focal Loss 실험


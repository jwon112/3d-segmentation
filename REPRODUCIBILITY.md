# 재현성 보장 가이드

이 문서는 실험의 재현성을 보장하기 위한 설정 방법을 설명합니다.

## 현재 구현된 재현성 보장 기능

### 1. Seed 설정 (`utils/experiment_utils.py`)

`set_seed()` 함수가 다음을 설정합니다:
- Python `random` 모듈
- NumPy 랜덤 시드
- PyTorch CPU/CUDA 시드
- CuDNN deterministic 모드
- Python hash seed

### 2. DataLoader Worker Seed

각 DataLoader worker마다 고유한 seed를 사용합니다:
- Base seed + worker_id를 사용하여 각 worker가 다른 seed를 가집니다
- 이는 worker 간의 랜덤성 차이를 보장하면서도 재현성을 유지합니다

### 3. Generator 사용

모든 DataLoader에 `generator`를 설정하여 샘플링 순서를 고정합니다.

## 재현성 보장을 위한 추가 설정 (선택사항)

완전한 재현성이 필요한 경우, `set_seed()` 함수 호출 시 `use_deterministic_algorithms=True`를 설정할 수 있습니다:

```python
from utils.experiment_utils import set_seed

# 완전한 재현성을 위한 설정 (성능 저하 10-20% 가능)
set_seed(seed=42, use_deterministic_algorithms=True)
```

**주의사항:**
- `use_deterministic_algorithms=True`는 성능 저하를 일으킬 수 있습니다 (약 10-20%)
- 대부분의 경우 `False`로도 충분히 재현 가능한 결과를 얻을 수 있습니다
- 일부 연산은 deterministic 알고리즘을 지원하지 않을 수 있습니다

## 재현성에 영향을 줄 수 있는 요인

1. **하드웨어 차이**: 다른 GPU 모델이나 CUDA 버전은 약간의 수치적 차이를 만들 수 있습니다
2. **PyTorch 버전**: PyTorch 버전이 다르면 약간의 차이가 발생할 수 있습니다
3. **데이터 로딩 순서**: `num_workers`가 0이 아닌 경우, worker 초기화 순서에 따라 약간의 차이가 있을 수 있습니다
4. **부동소수점 연산**: GPU의 부동소수점 연산은 약간의 수치적 오차를 가질 수 있습니다

## 재현성 검증 방법

같은 seed로 여러 번 실행하여 결과를 비교하세요:

```python
# 같은 seed로 여러 번 실행
for run in range(3):
    set_seed(seed=42)
    # ... 학습 코드 ...
    # 결과 비교
```

## 권장 사항

1. **일반적인 경우**: `set_seed(seed=42)` (기본값, `use_deterministic_algorithms=False`)
2. **완전한 재현성이 필요한 경우**: `set_seed(seed=42, use_deterministic_algorithms=True)`
3. **결과 보고 시**: 사용한 seed 값과 `use_deterministic_algorithms` 설정을 명시하세요

## 수정 내역

- **2024-12-01**: DataLoader worker seed 개선 (각 worker마다 고유한 seed 사용)
- **2024-12-01**: 모든 DataLoader에 generator 추가
- **2024-12-01**: `set_seed()` 함수에 `use_deterministic_algorithms` 옵션 추가


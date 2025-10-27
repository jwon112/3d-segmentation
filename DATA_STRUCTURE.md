# 데이터 구조 설명

## H5 파일이란?

**HDF5 (Hierarchical Data Format Version 5)**는 과학 데이터를 저장하는 포맷입니다.

### 특징
- NumPy 배열과 호환
- 효율적 압축 및 I/O
- 계층 구조 지원
- 메타데이터 저장 가능

### 현재 데이터셋의 H5 구조
```python
# 각 H5 파일 내용
h5_file['image']  # Shape: (240, 240, 4) - 4개의 modality (T1, T1CE, T2, FLAIR)
h5_file['mask']   # Shape: (240, 240) - 세그멘테이션 마스크
```

## 현재 데이터셋 구조

### 통계
- **총 슬라이스**: 57,195개
- **볼륨 수**: 369개
- **평균 슬라이스/볼륨**: 약 155개

### 파일 구조
```
data/MICCAI_BraTS2020_TrainingData/
├── BraTS20 Training Metadata.csv          # 전체 메타데이터
├── BraTS20 Training Metadata.csv.backup   # 백업
└── BraTS2020_training_data/
    └── content/
        └── data/
            ├── volume_0_slice_0.h5
            ├── volume_0_slice_1.h5
            ├── ...
            ├── volume_1_slice_0.h5
            ├── volume_1_slice_1.h5
            ├── ...
            ├── volume_41_slice_0.h5
            ├── volume_41_slice_1.h5
            ├── ...
            ├── volume_100_slice_0.h5
            ├── volume_100_slice_1.h5
            └── ...
```

### H5 파일 내용 예시

```python
import h5py

with h5py.File('volume_41_slice_0.h5', 'r') as f:
    print(f'image shape: {f["image"].shape}')  # (240, 240, 4)
    print(f'mask shape: {f["mask"].shape}')    # (240, 240)
    
    # image는 4개의 modality
    # [:, :, 0]: T1
    # [:, :, 1]: T1CE (T1 + contrast)
    # [:, :, 2]: T2
    # [:, :, 3]: FLAIR
    
    # mask는 세그멘테이션 클래스
    # 0: Background
    # 1: NCR/NET (Necrotic core / Non-enhancing tumor)
    # 2: ED (Peritumoral edema)
    # 3: ET (Enhancing tumor)
```

## 데이터 로더 구조

### BratsDataset2D (2D 슬라이스)
- **입력**: H5 슬라이스 파일
- **출력**: `image (4, 64, 64)`, `mask (64, 64)`
- **용도**: 슬라이스별 세그멘테이션

### BratsDataset3D (3D 볼륨)
- **입력**: H5 슬라이스들을 모아서 3D 볼륨 생성
- **출력**: `image (4, 64, 64, 64)`, `mask (64, 64, 64)`
- **용도**: 전체 볼륨 세그멘테이션

### 사용 방법

```python
from data_loader import get_data_loaders

# 2D 데이터 사용 (슬라이스 단위)
train_loader, val_loader, test_loader = get_data_loaders(
    data_dir='data',
    batch_size=1,
    dim='2d'  # 2D 슬라이스
)

# 3D 데이터 사용 (볼륨 단위)
train_loader, val_loader, test_loader = get_data_loaders(
    data_dir='data',
    batch_size=1,
    dim='3d'  # 3D 볼륨
)
```

## 현재 문제점 및 해결방안

### 1. 3D 데이터 생성 방법
**현재**: 단일 슬라이스를 64번 복사해서 3D 생성
```python
image_3d = np.repeat(image[:, :, :, np.newaxis], 64, axis=3)
```
**문제**: 실제 3D 볼륨이 아님

**개선 방안**:
1. 여러 슬라이스를 모아서 실제 3D 볼륨 생성
2. 볼륨별로 슬라이스 그룹화
3. 각 볼륨의 슬라이스를 순차적으로 로드

### 2. 데이터 필터링
**현재**: 모든 슬라이스 로드
**개선**: 의미있는 슬라이스만 선택 (background 비율이 낮은 것만)

### 3. 볼륨 단위 로딩
**개선 방안**:
```python
# volume_41의 모든 슬라이스를 모아서 3D 볼륨 생성
slices = [f'volume_41_slice_{i}.h5' for i in range(155)]
volume = stack_slices(slices)  # (4, 240, 240, 155)
```

## 권장 개선사항

1. **볼륨 단위 로딩**: 같은 volume의 슬라이스들을 모아서 실제 3D 데이터 생성
2. **메타데이터 활용**: background 비율이 높은 슬라이스 제외
3. **데이터 증강**: 회전, 대칭 등
4. **캐싱**: 전처리된 데이터를 H5로 저장하여 재사용

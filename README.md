# 3D Segmentation Project

3D 뇌종양 세그멘테이션을 위한 다중 모델 비교 실험 시스템입니다.

## 📁 프로젝트 구조

```
3d_segmentation/
├── baseline/                    # Baseline 모델들
│   ├── __init__.py
│   ├── model_3d_unet.py        # 3D U-Net 모델
│   ├── model_unetr.py          # UNETR 모델
│   └── model_swin_unetr.py     # Swin UNETR 모델
├── train/                      # 훈련 스크립트
│   └── train_baseline.py       # Baseline 모델 훈련
├── visualization/              # 시각화 모듈
│   ├── __init__.py
│   ├── visualization_3d.py     # 3D 시각화 (다중 모델 지원)
│   └── visualization_dataframe.py # DataFrame 기반 시각화
├── baseline_results/           # 실험 결과 저장
├── data/                       # 데이터셋
├── integrated_experiment.py    # 통합 실험 스크립트
├── data_loader_kaggle.py      # 데이터 로더
└── requirements.txt            # 의존성 패키지
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n 3d_segmentation python=3.9
conda activate 3d_segmentation

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

#### 데이터셋 구조

**BRATS2018**:
```
{data_path}/
└── BRATS2018/
    └── MICCAI_BraTS_2018_Data_Training/
        ├── HGG/          # High-Grade Glioma (~210 samples)
        └── LGG/          # Low-Grade Glioma (~75 samples)
```

**BRATS2021**:
```
{data_path}/
└── BRATS2021/
    └── BraTS2021_Training_Data/  # ~1,251 samples
        ├── BraTS2021_00000/
        ├── BraTS2021_00001/
        └── ...
```

#### 경로 설정

- **서버**: `/home/work/3D_/BT/` (기본값)
- **로컬**: `C:\Users\user\Desktop\성균관대\3d_segmentation\data` (Windows) 또는 `/path/to/data` (Linux/Mac)

데이터셋은 공통 경로(`--data_path`) 아래에 `BRATS2018/` 또는 `BRATS2021/` 폴더로 구성되어 있어야 합니다.

### 3. 실험 실행

#### 기본 실험 (단일 시드)
```bash
python integrated_experiment.py --epochs 10 --batch_size 1 --seeds 24
```

#### 다중 시드 실험
```bash
python integrated_experiment.py --epochs 10 --batch_size 1 --seeds 24 42 123
```

#### 특정 모델만 실험
```bash
python integrated_experiment.py --epochs 10 --models unet3d unetr
```

#### BRATS2018 데이터셋 사용
```bash
python integrated_experiment.py --dataset_version brats2018 --epochs 10
```

#### BRATS2021 데이터셋 사용
```bash
python integrated_experiment.py --dataset_version brats2021 --epochs 10
```

#### 3D 모델 학습 (3D 데이터 사용)
```bash
python integrated_experiment.py --dim 3d --epochs 10 --batch_size 1
```

#### 분산 학습 (Multi-GPU)
```bash
# 2개 GPU 사용
torchrun --nproc_per_node=2 integrated_experiment.py --epochs 10 --batch_size 2

# 4개 GPU 사용
torchrun --nproc_per_node=4 integrated_experiment.py --epochs 10 --batch_size 4

# 특정 GPU만 사용 (예: GPU 0, 1만 사용)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 integrated_experiment.py --epochs 10

# 멀티 노드 (예: 2노드, 각각 4 GPU)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 --master_addr=<MASTER_IP> --master_port=29500 integrated_experiment.py --epochs 10
```

## 📝 실행 옵션

### 주요 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--data_path` | str | `/home/work/3D_/BT/` | 데이터셋 공통 경로 (서버/로컬 경로 지정) |
| `--dataset_version` | str | `brats2018` | 데이터셋 버전: `brats2018` 또는 `brats2021` |
| `--epochs` | int | `10` | 훈련 에포크 수 |
| `--batch_size` | int | `1` | 배치 크기 (분산 학습 시 GPU당 배치 크기) |
| `--seeds` | list[int] | `[24]` | 실험 시드 리스트 (예: `--seeds 24 42 123`) |
| `--models` | list[str] | `None` | 사용할 모델 리스트 (지정하지 않으면 모든 모델) |
| `--dim` | str | `2d` | 데이터 차원: `2d` 또는 `3d` |
| `--use_pretrained` | flag | `False` | Pretrained 가중치 사용 여부 |
| `--use_nnunet_loss` | flag | `True` | nnU-Net 스타일 loss 사용 (Dice 70% + CE 30%) |
| `--use_standard_loss` | flag | `False` | 표준 loss 사용 (Dice 50% + CE 50%) |
| `--num_workers` | int | `8` | DataLoader 워커 수 |
| `--sharing_strategy` | str | `file_descriptor` | PyTorch tensor sharing 전략: `file_descriptor` 또는 `file_system` |

### 모델 선택 옵션

지원되는 모델:
- `unet3d`: 3D U-Net
- `unetr`: UNETR
- `swin_unetr`: Swin UNETR
- `mobile_unetr`: Mobile UNETR
- `mobile_unetr_3d`: Mobile UNETR 3D

### 데이터셋 경로 설정

#### 서버 환경
```bash
# 기본 경로 사용 (서버)
python integrated_experiment.py --epochs 10

# 또는 명시적으로 지정
python integrated_experiment.py --data_path /home/work/3D_/BT/ --epochs 10
```

#### 로컬 환경
```bash
# Windows
python integrated_experiment.py --data_path "C:\Users\user\Desktop\성균관대\3d_segmentation\data" --epochs 10

# Linux/Mac
python integrated_experiment.py --data_path /path/to/data --epochs 10
```

## 🚀 분산 학습 설정

### 1. 단일 노드 멀티 GPU (Single Node Multi-GPU)

#### Linux/서버 환경 (torchrun 사용 - 권장)
```bash
# 4개 GPU 사용
torchrun --nproc_per_node=4 integrated_experiment.py \
    --epochs 10 \
    --batch_size 4 \
    --seeds 24

# 특정 GPU만 사용 (예: GPU 2, 3만 사용)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 integrated_experiment.py \
    --epochs 10 \
    --batch_size 2
```

#### Windows 환경 (libuv 오류 해결)
Windows에서는 PyTorch가 libuv 지원 없이 빌드된 경우 오류가 발생할 수 있습니다:

```bash
# 방법 1: 환경 변수로 libuv 비활성화
set USE_LIBUV=0
torchrun --nproc_per_node=4 integrated_experiment.py --epochs 10 --batch_size 4

# 방법 2: PowerShell에서
$env:USE_LIBUV=0
torchrun --nproc_per_node=4 integrated_experiment.py --epochs 10 --batch_size 4

# 방법 3: 한 줄로 실행
$env:USE_LIBUV=0; torchrun --nproc_per_node=4 integrated_experiment.py --epochs 10 --batch_size 4
```

**참고**: Windows에서는 `torchrun`이 완전히 지원되지 않을 수 있습니다. Linux/서버 환경에서 분산 학습을 권장합니다.

#### 환경 변수 직접 설정
```bash
# 4개 GPU 사용
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0
export LOCAL_RANK=0

# 각 GPU별로 실행 (스크립트로 자동화 권장)
python integrated_experiment.py --epochs 10
```

### 2. 멀티 노드 (Multi-Node)

#### 노드 0 (Master)
```bash
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=<MASTER_IP> \
    --master_port=29500 \
    integrated_experiment.py --epochs 10 --batch_size 4
```

#### 노드 1 (Worker)
```bash
torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=4 \
    --master_addr=<MASTER_IP> \
    --master_port=29500 \
    integrated_experiment.py --epochs 10 --batch_size 4
```

### 3. 분산 학습 주의사항

#### 배치 크기 설정
- **분산 학습 시**: `--batch_size`는 GPU당 배치 크기입니다
- **전체 배치 크기**: `batch_size × num_gpus`
- 예: `--batch_size 2` + 4 GPU = 전체 배치 크기 8

#### 메모리 최적화
```bash
# /dev/shm 공간 부족 시 file_system 전략 사용
python integrated_experiment.py \
    --sharing_strategy file_system \
    --num_workers 4 \
    --epochs 10
```

#### DataLoader 워커 수
- 분산 학습 시 각 GPU 프로세스마다 워커가 생성됩니다
- 메모리 부족 시 `--num_workers`를 줄이세요 (기본값: 8)

### 4. 분산 학습 체크리스트

- [ ] 모든 노드에서 동일한 코드와 데이터 경로 사용
- [ ] 네트워크 연결 확인 (NCCL 백엔드 사용)
- [ ] 방화벽 설정 확인 (master_port 개방)
- [ ] 각 노드의 GPU가 동일한 CUDA 버전 사용
- [ ] 공유 파일 시스템 접근 가능 (데이터셋 공유)
- [ ] **Windows 환경**: `USE_LIBUV=0` 환경 변수 설정 확인

### 5. Windows 환경 주의사항

Windows에서 분산 학습 시 다음 오류가 발생할 수 있습니다:

```
torch.distributed.DistStoreError: use_libuv was requested but PyTorch was built without libuv support
```

**해결 방법**:
1. 환경 변수 `USE_LIBUV=0` 설정 (위 참조)
2. 또는 Linux/서버 환경에서 실행 권장

**Windows 제한사항**:
- `torchrun`의 일부 기능이 제한될 수 있음
- 멀티 노드 분산 학습은 Linux 환경에서만 지원
- 단일 노드 멀티 GPU는 가능하지만, Linux 환경을 권장

## 🧠 지원 모델

### 1. 3D U-Net (UNet3D_Simplified)
- **특징**: 전통적인 U-Net 아키텍처의 3D 버전
- **장점**: 안정적이고 검증된 구조
- **단점**: 메모리 사용량이 높음

### 2. UNETR (UNETR_Simplified)
- **특징**: Vision Transformer 기반 3D 세그멘테이션
- **장점**: 긴 거리 의존성 학습 가능
- **단점**: 복잡한 구조로 인한 높은 계산 비용

### 3. Swin UNETR (SwinUNETR_Simplified)
- **특징**: Swin Transformer 기반 계층적 구조
- **장점**: 효율적인 계산과 좋은 성능
- **단점**: 구현 복잡도가 높음

## 📊 실험 결과

실험 결과는 `baseline_results/` 폴더에 저장됩니다:

- `integrated_experiment_results_YYYYMMDD_HHMMSS/`
  - `integrated_experiment_results.csv`: 모델별 성능 요약
  - `all_epochs_results.csv`: 에포크별 상세 결과
  - `model_comparison.csv`: 모델 비교 분석
  - `learning_curves.png`: 학습 곡선 차트
  - `model_comparison_chart.png`: 모델 성능 비교 차트
  - `parameter_efficiency.png`: 파라미터 효율성 분석
  - `interactive_3d_analysis.html`: 인터랙티브 3D 분석
  - `{model_name}_seed_{seed}_best.pth`: 각 모델별 최적 체크포인트 (실험 폴더 내부 저장)

## 🔧 주요 기능

### 1. 다중 모델 비교
- 3D U-Net, UNETR, Swin UNETR 모델 동시 훈련
- 모델별 성능 메트릭 비교 (Dice Score, Precision, Recall)
- 파라미터 수 및 FLOPs 효율성 분석

### 2. 다중 시드 실험
- 재현 가능한 실험을 위한 시드 설정
- 통계적 유의성 검증을 위한 다중 시드 평균
- 시드별 성능 분포 분석

### 3. 3D 시각화
- 슬라이스별 세그멘테이션 결과 시각화 (다중 모델 지원)
- 학습 곡선 및 성능 비교 차트
- 인터랙티브 3D 분석 플롯
- DataFrame 기반 실험 결과 분석

### 4. 자동화된 실험 관리
- 체크포인트 자동 저장
- 실험 결과 자동 정리
- 시각화 차트 자동 생성

## 📈 성능 메트릭

### 1. Dice Score
- 세그멘테이션 정확도 측정
- 클래스별 Dice Score 계산
- 평균 Dice Score로 전체 성능 평가

### 2. Precision & Recall
- 클래스별 정밀도와 재현율
- Background 클래스 제외한 평균
- 세그멘테이션 품질 상세 분석

### 3. 모델 효율성
- 파라미터 수 (Parameters)
- 연산량 (FLOPs)
- 모델 크기 (MB)

## 🛠️ 커스터마이징

### 새로운 모델 추가
1. `baseline/` 폴더에 새 모델 파일 생성
2. `baseline/__init__.py`에 모델 import 추가
3. `get_model()` 함수에 모델 케이스 추가

### 실험 설정 변경
- `integrated_experiment.py`의 기본 파라미터 수정
- 명령행 인자로 실시간 설정 변경 가능

### 시각화 커스터마이징
- `visualization/visualization_3d.py`: 다중 모델 3D 시각화
- `visualization/visualization_dataframe.py`: DataFrame 기반 분석 차트
- 새로운 분석 차트 추가 가능

## 📋 요구사항

### 하드웨어
- GPU: CUDA 지원 GPU (권장: RTX 3080 이상)
- RAM: 16GB 이상
- 저장공간: 50GB 이상

### 소프트웨어
- Python 3.9+
- PyTorch 1.12+
- CUDA 11.0+

### 주요 패키지
- torch, torchvision
- numpy, pandas
- matplotlib, seaborn
- plotly
- tqdm
- thop (FLOPs 계산용)

## 🐛 문제 해결

### 메모리 부족 오류
- `batch_size`를 1로 설정
- `max_samples` 파라미터로 데이터 크기 제한
- 모델 크기 축소 (Simplified 버전 사용)

### CUDA 오류
- CUDA 버전과 PyTorch 버전 호환성 확인
- `torch.cuda.is_available()` 확인

### 데이터 로딩 오류
- 데이터 경로 확인
- H5 파일 형식 확인
- 메타데이터 CSV 파일 존재 확인

## 📚 참고 문헌

1. **3D U-Net**: Çiçek, Ö., et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation."
2. **UNETR**: Hatamizadeh, A., et al. "UNETR: Transformers for 3D Medical Image Segmentation."
3. **Swin UNETR**: Hatamizadeh, A., et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images."

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

**Note**: 이 프로젝트는 연구 목적으로 개발되었으며, 실제 의료 진단에 사용해서는 안 됩니다.
# 3D Segmentation Project

3D 뇌종양 세그멘테이션을 위한 다중 모델 비교 실험 시스템입니다.

이 프로젝트는 크게 두 가지 파이프라인을 지원합니다.
- **기본 파이프라인**: nnU-Net 스타일 패치 샘플링 + 3D 슬라이딩 윈도우 기반 세그멘테이션
- **Cascade + CoordConv 파이프라인**: ROI 탐지 → 다중 크롭 → CoordConv(좌표맵) 포함 세그멘테이션 → 원본 공간으로 복원

## 📁 프로젝트 구조

```
3d_segmentation/
├── models/                             # 모델 아키텍처
│   ├── __init__.py
│   ├── channel_configs.py             # 중앙 집중식 채널 설정 (_xs, _s, _m, _l)
│   ├── model_3d_unet.py               # 3D U-Net (크기: xs, s, m, l)
│   ├── model_3d_unet_stride.py        # 3D U-Net Stride (크기: xs, s, m, l)
│   ├── model_3d_unet_modal_comparison.py  # 모달리티 비교 모델 (2modal, 4modal, quadbranch)
│   ├── model_unetr.py                 # UNETR 모델
│   ├── model_swin_unetr.py            # Swin UNETR 모델
│   ├── mobileunetr.py                 # Mobile UNETR (2D)
│   ├── mobileunetr_3d.py              # Mobile UNETR 3D
│   ├── dualbranch_basic.py            # Dual-Branch 기본 (MaxPool, Stride, Dilated)
│   ├── dualbranch_replk.py            # Dual-Branch RepLK (RepLK + MViT)
│   ├── dualbranch_mobile.py           # Dual-Branch MobileNetV2 (Shuffle-inspired)
│   ├── dualbranch_mvit.py             # Dual-Branch MobileViT Extended
│   ├── dualbranch_14_unet.py          # Dual-Branch Backbone 비교 (MobileNetV2, GhostNet, ShuffleNetV2, ConvNeXt 등)
│   └── modules/                       # 공통 모듈
│       ├── __init__.py
│       ├── replk_modules.py           # RepLK 관련 모듈
│       ├── mvit_modules.py            # MobileViT 관련 모듈
│       ├── ghostnet_modules.py        # GhostNet 관련 모듈
│       ├── shufflenet_modules.py      # ShuffleNetV2 관련 모듈
│       └── convnext_modules.py        # ConvNeXt 관련 모듈
├── utils/                              # 유틸리티 함수
│   ├── __init__.py
│   ├── experiment_utils.py            # 실험 유틸리티 (모델 생성, PAM 계산, 슬라이딩 윈도우 등)
│   ├── gradcam_utils.py               # Grad-CAM 유틸리티 (현재 비활성화)
│   ├── debug_*.py                     # 디버깅 스크립트들
│   └── *.md                           # 문서화 파일들
├── losses/                             # Loss 함수
│   ├── __init__.py
│   └── losses.py                      # Combined Loss, nnU-Net Style Loss
├── metrics/                            # 평가 메트릭
│   ├── __init__.py
│   └── metrics.py                     # Dice Score, WT/TC/ET 계산
├── visualization/                      # 시각화 모듈
│   ├── __init__.py
│   ├── visualization_3d.py            # 3D 시각화 (다중 모델 지원)
│   ├── visualization_dataframe.py     # DataFrame 기반 시각화 및 차트 생성
│   └── gradcam_3d.py                  # Grad-CAM 3D 시각화 (현재 비활성화)
├── dataloaders/                        # 새로운 데이터 로더 패키지 (권장 진입점)
│   ├── __init__.py                     # 공통 re-export (get_data_loaders 등)
│   ├── brats_base.py                   # BraTS 기본 Dataset 및 split 로직
│   ├── patch_3d.py                     # nnU-Net 스타일 3D 패치 데이터셋
│   ├── cascade.py                      # Cascade/CoordConv용 ROI·Seg 데이터셋 및 유틸
│   └── factory.py                      # 설정 기반 DataLoader 팩토리
├── baseline_results/                   # 실험 결과 저장
├── data/                               # 데이터셋
├── integrated_experiment.py            # 통합 실험 스크립트 (CLI 진입점)
├── experiment_runner.py                # 실험 실행 로직 (train_model, evaluate_model, run_integrated_experiment)
├── evaluate_experiment.py              # 체크포인트 평가 스크립트
├── train_roi.py                        # ROI 탐지 모델 전용 학습/평가 스크립트
├── data_loader.py                      # 하위 호환용 래퍼 (내부적으로 dataloaders 사용, 신규 코드는 dataloaders 사용 권장)
└── requirements.txt                    # 의존성 패키지
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
python integrated_experiment.py --epochs 10 --models unet3d_s dualbranch_01_unet_s
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

#### 5-Fold Cross-Validation
```bash
python integrated_experiment.py --use_5fold --epochs 10 --seeds 24
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

### 4. Cascade ROI → Segmentation 파이프라인

#### 1) ROI 탐지 모델 학습
`train_roi.py`는 ROI 모델만 독립적으로 학습/평가하고, 결과를 `models/weights/cascade/roi_model/<model_name>/seed_<seed>/` 아래에 저장합니다.
ROI 전용 아키텍처 코드는 `models/architecture/cascade/roi_model/`에 정리되어 있으며,
`roi_mobileunetr3d_*`, `roi_unet3d_*` 계열 모델을 선택적으로 사용할 수 있습니다.
`--use_mri_augmentation` 플래그를 켜면 ROI 리사이즈 볼륨에도 patch 데이터셋과 동일한 강도/플립 증강이 적용됩니다.

```bash
# 단일 GPU
python train_roi.py --data_path /path/to/data --dataset_version brats2018 --epochs 40

# 멀티 GPU (예: 2 GPU)
torchrun --nproc_per_node=2 train_roi.py --data_path /path/to/data --epochs 40 --batch_size 4
```

디렉터리 구조 (예시):
```
models/weights/cascade/roi_model/roi_mobileunetr3d_tiny/seed_24/
├── config.json          # 실행 설정
├── metrics.csv          # val/test Dice 기록
└── weights/
    └── best.pth         # 최고 성능 가중치
```

#### 2) Segmentation 학습 + Cascade 추론
Segmentation 학습은 기존 `integrated_experiment.py`로 진행합니다. 추론 단계에서 ROI 가중치를 불러와 ROI→Crop→Seg→Uncrop 파이프라인으로 평가하려면 아래 옵션을 추가합니다.

```bash
python integrated_experiment.py \
  --dim 3d \
  --models dualbranch_18_shufflenet_v1_s \
  --use_cascade_pipeline \
  --roi_model_name roi_mobileunetr3d_tiny \
  --roi_weight_path models/weights/cascade/roi_model/roi_mobileunetr3d_tiny/seed_24/weights/best.pth \
  --roi_resize 64 64 64 \
  --cascade_crop_size 96 96 96
```

`--use_cascade_pipeline`이 활성화되면 기본 슬라이딩 윈도우 평가 외에 ROI 기반 Cascade 평가 결과(`cascade_dice`, `cascade_wt/tc/et`)가 추가로 로그/CSV에 기록됩니다.

내부적으로는 `utils/cascade_utils.py`에 구현된 다음 로직이 사용됩니다.
- ROI 모델: 전체 뇌를 `roi_resize` 크기로 리사이즈 + CoordConv 좌표맵을 붙여 WT binary를 예측
- 중심 추출: connected components 기반으로 여러 WT 덩어리의 중심을 추출 (multi-focal 대응, 상위 N개만 사용)
- 세그멘테이션: 각 중심을 기준으로 `cascade_crop_size` 크기 3D 패치를 잘라 7채널 입력(4 MRI + 3 좌표맵)으로 세그멘테이션 모델 실행
- 병합: 각 패치의 logits를 원본 공간으로 되붙인 뒤, voxel-wise max로 병합하여 최종 전체 볼륨 mask 생성

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
| `--use_5fold` | flag | `False` | 5-fold cross-validation 사용 |
| `--use_cascade_pipeline` | flag | `False` | Pre-trained ROI detector를 사용한 Cascade 추론 활성화 |
| `--roi_model_name` | str | `roi_mobileunetr3d_tiny` | ROI 탐지 모델 아키텍처 이름 |
| `--roi_weight_path` | str | `None` | 사전 학습된 ROI 가중치(.pth) 경로 (Cascade 사용 시 필수) |
| `--roi_resize` | int×3 | `64 64 64` | ROI 입력 해상도 (D H W) |
| `--cascade_crop_size` | int×3 | `96 96 96` | 세그멘테이션용 3D 크롭 크기 (D H W) |

### 모델 선택 옵션

#### 기본 U-Net 모델 (크기: xs, s, m, l)
- `unet3d_xs`: 3D U-Net Extra Small
- `unet3d_s`: 3D U-Net Small
- `unet3d_m`: 3D U-Net Medium
- `unet3d_l`: 3D U-Net Large
- `unet3d_stride_xs`: 3D U-Net Stride Extra Small (Stride Conv downsampling)
- `unet3d_stride_s`: 3D U-Net Stride Small
- `unet3d_stride_m`: 3D U-Net Stride Medium
- `unet3d_stride_l`: 3D U-Net Stride Large

**크기별 채널 증가**: 각 크기가 2배씩 증가 (xs → s → m → l)

#### Transformer 기반 모델
- `unetr`: UNETR
- `swin_unetr`: Swin UNETR
- `mobile_unetr`: Mobile UNETR (2D 전용)
- `mobile_unetr_3d`: Mobile UNETR 3D

#### Dual-Branch 모델 (T1ce, FLAIR 이중 분기, 크기: xs, s, m, l)
- `dualbranch_01_unet_{xs|s|m|l}`: 기본 Dual-Branch (MaxPool)
- `dualbranch_02_unet_{xs|s|m|l}`: Stride Conv 버전
- `dualbranch_03_unet_{xs|s|m|l}`: Dilated Conv (FLAIR만)
- `dualbranch_04_unet_{xs|s|m|l}`: RepLK 13x13x13 (FLAIR만)
- `dualbranch_05_unet_{xs|s|m|l}`: RepLK + FFN2
- `dualbranch_06_unet_{xs|s|m|l}`: RepLK + MViT Stage 4,5
- `dualbranch_07_unet_{xs|s|m|l}`: RepLK + MViT Stage 5만
- `dualbranch_mobilenetv2_dilated_{xs|s|m|l}`: MobileNetV2 듀얼 분기 (Shuffle-inspired, Stage3 Fused)
- `dualbranch_mobilenetv2_dilated_fixed_{xs|s|m|l}`: MobileNetV2 듀얼 분기 (Fixed Decoder 변형)
- `dualbranch_13_unet_{xs|s|m|l}`: MobileViT Extended
- `dualbranch_14_mobilenetv2_expand2_{xs|s|m|l}`: MobileNetV2 (expand_ratio=2)
- `dualbranch_14_ghostnet_{xs|s|m|l}`: GhostNet
- `dualbranch_14_dilated_{xs|s|m|l}`: Dilated Conv (rate 1,2,5)
- `dualbranch_14_convnext_{xs|s|m|l}`: ConvNeXt
- `dualbranch_14_shufflenetv2_{xs|s|m|l}`: ShuffleNetV2
- `dualbranch_14_shufflenetv2_dilated_{xs|s|m|l}`: ShuffleNetV2 Dilated
- `dualbranch_14_shufflenetv2_lk_{xs|s|m|l}`: ShuffleNetV2 Large Kernel

**예시**: `dualbranch_01_unet_s`, `dualbranch_01_unet_m`, `dualbranch_14_ghostnet_l` 등

#### 모달리티 비교 모델
- `unet3d_2modal_s`: 단일 분기, 2채널 (T1ce, FLAIR) concat
- `unet3d_4modal_s`: 단일 분기, 4채널 (T1, T1ce, T2, FLAIR) concat
- `dualbranch_2modal_unet_s`: 2개 분기 (T1ce, FLAIR)
- `quadbranch_4modal_unet_s`: 4개 분기 (T1, T1ce, T2, FLAIR) - 어텐션 없음
- `quadbranch_4modal_attention_unet_s`: 4개 분기 + 채널 어텐션

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

보다 자세한 경로 설정 예시는 `PATH_CONFIGURATION.md`를, 다양한 사용 예시는 `USAGE_EXAMPLES.md`를 참고하세요.

## 🔄 전처리/후처리 파이프라인

### 전처리

1. **NIfTI 파일 로드**: BraTS 데이터셋에서 모달리티별 파일 자동 감지
2. **정규화**: 
   - 비영점 영역만 추출 (vol > 0)
   - 퍼센타일 클리핑: [0.5%, 99.5%]
   - Z-score 정규화: (clipped - mean) / std
   - 배경 영역은 0으로 유지
3. **라벨 매핑**: BraTS 원본 라벨 4 → 모델 라벨 3 (ET)
4. **데이터 분할**: 
   - 일반: train 80% / val 10% / test 10%
   - 5-Fold CV: 5개 fold로 분할 (각 fold 20%)
5. **패치 샘플링** (3D 학습 시):
   - 33.3%: 포그라운드 오버샘플링 (클래스 1,2,3 중심)
   - 66.7%: 완전 무작위 샘플링

### 후처리

1. **슬라이딩 윈도우 추론** (3D 평가 시):
   - 패치 크기: (128, 128, 128)
   - Overlap: 0.10 (검증/테스트)
   - Cosine blending으로 패치 경계 부드럽게 융합
2. **예측 생성**: Logits → Softmax → Argmax
3. **메트릭 계산**:
   - Dice Score (WT/TC/ET)
   - Precision/Recall (클래스별)
   - Confusion Matrix

## 🧠 지원 모델

### 1. 3D U-Net
- **기본 버전**: MaxPool 기반 downsampling
- **Stride 버전**: Stride-2 Conv 기반 downsampling
- **크기**: Extra Small, Small, Medium, Large (각 크기마다 채널이 2배씩 증가)
  - **xs**: 최소 채널 수
  - **s**: Small (기본)
  - **m**: Medium (Small의 2배)
  - **l**: Large (Medium의 2배)

### 2. UNETR
- **특징**: Vision Transformer 기반 3D 세그멘테이션
- **장점**: 긴 거리 의존성 학습 가능
- **단점**: 높은 계산 비용

### 3. Swin UNETR
- **특징**: Swin Transformer 기반 계층적 구조
- **장점**: 효율적인 계산과 좋은 성능

### 4. Mobile UNETR
- **2D 버전**: Mobile UNETR (2D 전용)
- **3D 버전**: Mobile UNETR 3D

### 5. Dual-Branch U-Net
- **구조**: T1ce와 FLAIR를 각각 독립적으로 처리 후 융합
- **Stage 1-4**: Dual-branch 구조 유지
- **Stage 5+**: 융합된 branch (MobileViT 또는 표준 UNet)
- **변형**: 다양한 backbone (RepLK, MobileNetV2, Dilated Conv, MobileViT, GhostNet, ShuffleNetV2, ConvNeXt 등)
- **크기**: 모든 모델이 xs, s, m, l 크기 지원 (채널 수 2배씩 증가)

### 6. Quad-Branch U-Net
- **구조**: T1, T1ce, T2, FLAIR를 각각 독립적으로 처리 후 융합
- **어텐션 버전**: 채널 어텐션으로 모달리티별 기여도 측정 가능

## 📊 실험 결과

실험 결과는 `baseline_results/` 폴더에 저장됩니다:

- `integrated_experiment_results_YYYYMMDD_HHMMSS/`
  - `integrated_experiment_results.csv`: 모델별 성능 요약 (PAM, Inference Latency 포함)
  - `all_epochs_results.csv`: 에포크별 상세 결과
  - `model_comparison.csv`: 모델 비교 분석 (평균/표준편차, PAM, Inference Latency 포함)
  - `stage_wise_pam_results.csv`: Stage별 PAM 분석 결과
  - `learning_curves.png`: 학습 곡선 차트
  - `model_comparison_chart.png`: 모델 성능 비교 차트 (PAM Train/Inference 포함)
  - `parameter_efficiency.png`: 파라미터 효율성 분석
  - `interactive_3d_analysis.html`: 인터랙티브 3D 분석
  - `{model_name}_seed_{seed}_best.pth`: 각 모델별 최적 체크포인트
  - `{model_name}_seed_{seed}_fold_{fold}_best.pth`: 5-fold CV 시 fold별 체크포인트
  - `gradcam/`: Grad-CAM 시각화 결과 (현재 비활성화)

## 🔧 주요 기능

### 1. 다중 모델 비교
- 다양한 아키텍처 동시 훈련 및 비교
- 모델별 성능 메트릭 비교 (Dice Score, Precision, Recall)
- 파라미터 수, FLOPs, PAM 효율성 분석

### 2. 다중 시드 실험
- 재현 가능한 실험을 위한 시드 설정
- 통계적 유의성 검증을 위한 다중 시드 평균
- 시드별 성능 분포 분석

### 3. 5-Fold Cross-Validation
- 신뢰성 향상을 위한 5-fold CV 지원
- 각 fold별 결과 저장 및 평균 계산

### 4. PAM (Peak Activation Memory) 측정
- Train/Inference 단계별 VRAM 사용량 측정
- 5회 측정 후 평균/표준편차 계산
- 모델별 메모리 효율성 비교
- **Stage-wise PAM**: 모델의 각 stage별 메모리 사용량 분석 (별도 CSV 저장)

### 5. 모달리티별 기여도 분석
- Quad-Branch 모델에서 채널 어텐션으로 모달리티별 기여도 측정
- 어텐션 가중치 시각화 및 저장

### 6. 3D 시각화
- 슬라이스별 세그멘테이션 결과 시각화 (다중 모델 지원)
- 학습 곡선 및 성능 비교 차트
- 인터랙티브 3D 분석 플롯
- DataFrame 기반 실험 결과 분석

### 7. 자동화된 실험 관리
- 체크포인트 자동 저장
- 실험 결과 자동 정리
- 시각화 차트 자동 생성

## 📈 성능 메트릭

### 1. Dice Score
- 세그멘테이션 정확도 측정
- BraTS 표준 평가 지표:
  - **WT (Whole Tumor)**: 클래스 1 ∪ 2 ∪ 3
  - **TC (Tumor Core)**: 클래스 1 ∪ 3
  - **ET (Enhancing Tumor)**: 클래스 3
- 클래스별 Dice Score 계산
- 평균 Dice Score로 전체 성능 평가

### 2. Precision & Recall
- 클래스별 정밀도와 재현율
- Background 클래스 제외한 평균
- 세그멘테이션 품질 상세 분석

### 3. 모델 효율성
- **파라미터 수** (Parameters)
- **연산량** (FLOPs)
- **PAM** (Peak Activation Memory): Train/Inference 단계별 VRAM 사용량
- **Inference Latency**: 추론 시간 (ms, batch_size=1 기준)

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

## 🛠️ 커스터마이징

### 새로운 모델 추가
1. `models/` 폴더에 새 모델 파일 생성
2. `models/__init__.py`에 모델 import 추가
3. `utils/experiment_utils.py`의 `get_model()` 함수에 모델 케이스 추가
4. 크기별 채널 설정이 필요한 경우 `models/channel_configs.py`에 추가
5. 공통 모듈은 `models/modules/` 폴더에 추가

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
- nibabel (NIfTI 파일 처리)

## 🐛 문제 해결

### 메모리 부족 오류
- `batch_size`를 1로 설정
- `max_samples` 파라미터로 데이터 크기 제한
- 모델 크기 축소 (Small 버전 사용)
- 슬라이딩 윈도우 overlap 줄이기

### CUDA 오류
- CUDA 버전과 PyTorch 버전 호환성 확인
- `torch.cuda.is_available()` 확인

### 데이터 로딩 오류
- 데이터 경로 확인
- NIfTI 파일 형식 확인
- 파일명 패턴 확인 (t1ce, flair, seg)
- `from dataloaders import get_data_loaders` 등 새 패키지 경로 사용 여부 확인 (`data_loader.py`는 하위 호환 래퍼)

#### 데이터 로더 사용 예시

새 코드에서는 `dataloaders` 패키지가 표준 진입점입니다.

```python
from dataloaders import get_data_loaders

train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = get_data_loaders(
    data_dir="/path/to/data",
    batch_size=1,
    num_workers=4,
    dim="3d",
    dataset_version="brats2018",
)
```

`data_loader.py`는 기존 코드와의 하위 호환을 위한 래퍼로만 남아 있으며, 새로운 스크립트에서는 `from dataloaders import ...` 사용을 권장합니다.

### NCCL Timeout 오류
- NCCL timeout 증가: `export NCCL_TIMEOUT=1800`
- 네트워크 연결 확인
- GPU 간 통신 속도 확인

## 📚 참고 문헌

1. **3D U-Net**: Çiçek, Ö., et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation."
2. **UNETR**: Hatamizadeh, A., et al. "UNETR: Transformers for 3D Medical Image Segmentation."
3. **Swin UNETR**: Hatamizadeh, A., et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images."
4. **nnU-Net**: Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

**Note**: 이 프로젝트는 연구 목적으로 개발되었으며, 실제 의료 진단에 사용해서는 안 됩니다.

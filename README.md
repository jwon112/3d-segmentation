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
├── baseline_results/           # 실험 결과 저장
├── data/                       # 데이터셋
├── integrated_experiment.py    # 통합 실험 스크립트
├── visualization_3d.py         # 3D 시각화 모듈
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

BraTS 2020 데이터셋을 `data/` 폴더에 준비합니다.

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
- 슬라이스별 세그멘테이션 결과 시각화
- 학습 곡선 및 성능 비교 차트
- 인터랙티브 3D 분석 플롯

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
- `visualization_3d.py`에서 차트 스타일 수정
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
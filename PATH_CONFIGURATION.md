# 경로 설정 가이드

## 경로 관련 주요 부분

### 1. 데이터 경로 설정

**파일**: `integrated_experiment.py`
**위치**: Line 447-448
```python
parser.add_argument('--data_path', type=str, default='data', 
                   help='Path to BraTS dataset root (default: data/)')
```

**설명**:
- 기본값: `'data'` (현재 디렉토리 기준)
- 변경 방법: `--data_path ./custom_data`
- 환경이 바뀔 경우: 상대 경로 또는 절대 경로 지정

### 2. 데이터셋 자동 감지 경로

**파일**: `integrated_experiment.py`
**위치**: Line 285-294
```python
if dataset_name == 'brats2021':
    dataset_dir = os.path.join(data_path, 'BraTS2021_Training_Data')
elif dataset_name == 'brats2020_kaggle':
    dataset_dir = os.path.join(data_path, 'MICCAI_BraTS2020_TrainingData')
```

**설명**:
- `data_path`: 사용자가 지정한 데이터 루트 경로
- 하위 디렉토리 구조가 고정적이므로 이 구조를 따라야 함

### 3. 데이터 로더 경로 처리

**파일**: `data_loader.py`
**위치**: Line 34-69
```python
# BraTS2021 데이터셋 확인
brats2021_dir = os.path.join(self.data_dir, 'BraTS2021_Training_Data')
if os.path.exists(brats2021_dir):
    # 로드

# BraTS2020 Kaggle 데이터셋 확인
training_dir = os.path.join(self.data_dir, 'MICCAI_BraTS2020_TrainingData')
```

**설명**:
- `self.data_dir`: 데이터 루트 경로
- 각 데이터셋의 하위 디렉토리를 자동 감지
- 우선순위: BraTS2021 > BraTS2020 Kaggle > BraTS2020 표준

### 4. 결과 저장 경로

**파일**: `integrated_experiment.py`
**위치**: Line 243-244
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"baseline_results/integrated_experiment_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
```

**설명**:
- 현재 작업 디렉토리 기준
- `baseline_results/` 하위에 저장
- 타임스탬프로 중복 방지

## 환경별 설정 예시

### 로컬 환경
```bash
python integrated_experiment.py --data_path ./data --epochs 10 --seeds 24
```

### 원격 서버
```bash
python integrated_experiment.py --data_path /data/project/brats --epochs 50 --seeds 24 42 123
```

### 다른 드라이브
```bash
python integrated_experiment.py --data_path D:/Projects/3d_segmentation/data --epochs 10
```

### 데이터셋 여러 개 사용
```bash
python integrated_experiment.py --datasets brats2021 brats2020_kaggle --epochs 10
```

## 디렉토리 구조 요구사항

```
<data_path>/
├── BraTS2021_Training_Data/          # BraTS2021 데이터셋
│   ├── BraTS2021_00000/
│   │   ├── *.nii.gz
│   ├── ...
│
├── MICCAI_BraTS2020_TrainingData/    # BraTS2020 Kaggle 데이터셋
│   └── BraTS2020_training_data/
│       └── content/data/
│           ├── *.h5
│
└── (다른 데이터셋들)
```

## 경로 변경 시 체크리스트

1. ✅ `--data_path` 인자로 데이터 루트 경로 지정
2. ✅ 하위 디렉토리 구조 확인 (BraTS2021_Training_Data 등)
3. ✅ 경로 구분자 확인 (Windows: `\`, Linux/Mac: `/`)
4. ✅ 절대 경로 사용 시 이식성 고려
5. ✅ 결과 저장 경로 확인 (`baseline_results/`)

## 주의사항

- Windows에서 경로에 한글이 있으면 문제 발생 가능
- 경로에 공백이 있으면 인용부호 사용: `--data_path "C:/My Projects/Data"`
- 상대 경로는 `python` 실행 위치 기준
- 절대 경로는 운영체제 독립적이지 않음 (Windows `\` vs Unix `/`)

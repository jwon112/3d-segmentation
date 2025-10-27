# 사용 예시

## 기본 사용법

### 1. 단일 데이터셋, 단일 시드
```bash
python integrated_experiment.py --epochs 10 --seeds 24
```

### 2. 다중 시드 실험
```bash
python integrated_experiment.py --epochs 10 --seeds 24 42 123
```

### 3. 특정 모델만 실험
```bash
python integrated_experiment.py --models unet3d unetr --epochs 10
```

### 4. 다중 데이터셋 실험
```bash
python integrated_experiment.py --datasets brats2021 brats2020_kaggle --epochs 10 --seeds 24 42
```

### 5. 자동 데이터셋 감지
```bash
python integrated_experiment.py --datasets auto --epochs 10
```

## 경로 설정 예시

### 로컬 환경
```bash
python integrated_experiment.py --data_path ./data --epochs 10
```

### 원격 서버
```bash
python integrated_experiment.py --data_path /data/brats --epochs 50
```

### 다른 드라이브 (Windows)
```bash
python integrated_experiment.py --data_path D:/Data/brats --epochs 10
```

## 고급 사용법

### 전체 옵션 사용
```bash
python integrated_experiment.py \
  --data_path ./data \
  --epochs 30 \
  --batch_size 1 \
  --seeds 24 42 123 \
  --models unet3d unetr swin_unetr \
  --datasets brats2021 brats2020_kaggle
```

### 빠른 테스트
```bash
python integrated_experiment.py --epochs 2 --seeds 24 --models unet3d
```

## 예상 결과

실행하면:
1. **데이터셋 로드**: 각 데이터셋별로 로딩 메시지 출력
2. **모델별 훈련**: 각 모델-데이터셋-시드 조합으로 훈련
3. **결과 저장**: CSV, PNG, HTML 파일 생성
4. **요약 출력**: 최종 성능 요약

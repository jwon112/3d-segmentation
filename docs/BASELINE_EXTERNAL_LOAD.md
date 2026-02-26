# Baseline 모델: 외부 로드 + Train from Scratch

Baseline은 우리가 직접 구현하지 않고 **외부 라이브러리(MONAI, Hugging Face 등)에서 모델만 로드**하고, **가중치는 로드하지 않고 train from scratch** 하는 방식을 권장할 수 있다.

- **장점**: 코드 양 감소, 유지보수·오류 가능성 감소.
- **유일한 고려**: 외부에서 로드한 모델이 **우리 학습 파이프라인과 제대로 맞물리는지** 확인하는 것.

아래는 그 “맞물림”을 점검할 때 쓰는 체크리스트다.

---

## 1. 입력(Input)

- **Shape**: `(B, C, D, H, W)` (3D) 또는 `(B, C, H, W)` (2D).  
  우리 데이터로더는 보통 `C=4` (BraTS), `B`는 배치 크기.
- **의미**: 채널 수(`n_channels`), `dim='2d'`/`'3d'`에 맞는 차원 수.

→ 외부 모델의 `in_channels`(및 2D/3D)를 우리 `n_channels`·`dim`과 맞추면 됨.

---

## 2. 출력(Output)

- **단일 텐서**: `(B, n_classes, D, H, W)` (또는 2D 대응).  
  로짓(softmax 전)이어야 하며, loss에서 `long` 타입 target과 함께 쓰인다.
- **Deep Supervision**: `list` 또는 `tuple` of tensors도 가능.  
  - 학습 시 `utils/runner/training.py`에서 `dummy_output = model(dummy_input)`으로 리스트 여부를 감지하고, 리스트면 `DeepSupervisionWrapper`로 loss 계산.  
  - 평가·sliding window에서는 **첫 번째 요소 `output[0]`만** 사용 (메인 출력).

→ 외부 모델이 단일 로짓을 반환하면 그대로 사용 가능. 리스트를 반환하면 우리 파이프라인이 이미 첫 번째만 쓰도록 되어 있음.

---

## 3. Loss

- **입력**: 예측 로짓(softmax 전) + target(long, 클래스 인덱스).  
  `combined_loss` / `combined_loss_nnunet_style` 및 `DeepSupervisionWrapper`가 이 형식을 기대함.
- **Deep Supervision**: 여러 스케일 로짓 리스트면, 우리 쪽에서 가중치를 붙여 합산하므로 **외부 모델이 리스트만 맞춰 주면 됨**.

→ 로짓 형태와 target dtype만 맞으면 됨. 외부 모델이 자체 loss를 쓰도록 강제하지 않으면 호환됨.

---

## 4. Sliding window / 평가

- `sliding_window_inference_3d(model, volume, patch_size=..., overlap=0.5, ...)`  
  - 내부에서 `logits = model(patch)` 호출 후, **리스트/튜플이면 `logits[0]`만 사용**해 블렌딩.  
- 평가·검증에서도 동일하게 `model(inputs)` 또는 sliding window 결과의 **단일 로짓**으로 Dice 등 계산.

→ 3D patch in → (단일 로짓 또는 리스트) out 형태만 맞으면 됨.

---

## 5. 기타

- **Device**: `model.to(device)` 로 이동.  
- **DDP**: `DistributedDataParallel` 래핑 시 `model.module`으로 실제 모델 접근하는 패턴 이미 사용 중.  
- **선택**: 우리 custom 모델용 `get_hybrid_stats()` 같은 건 baseline에 필수 아님. 없어도 학습/평가에는 문제 없음.

---

## 요약

| 항목 | 요구사항 |
|------|----------|
| 입력 | `(B, n_channels, ...)` 2D/3D 일치 |
| 출력 | 단일 로짓 `(B, n_classes, ...)` 또는 리스트(첫 요소가 메인 로짓) |
| Loss | 로짓(softmax 전) + long target → 우리 loss 함수와 호환 |
| Sliding window | patch in → 로짓(또는 리스트) out, 리스트면 [0] 사용 |

외부에서 모델만 로드하고 가중치는 로드하지 않은 뒤, 위 항목만 확인하면 train from scratch로 우리 파이프라인에 그대로 끼워 넣을 수 있다.

---

## 적용 현황

- **unet3d_***: `dynamic_network_architectures.PlainConvUNet` 로드 (기존과 동일).
- **unetr (3D)**: `monai.networks.nets.UNETR` 로드. 2D는 기존 `UNETR_Simplified` 유지.
- **swin_unetr (3D)**: `monai.networks.nets.SwinUNETR` 로드. 2D는 기존 `SwinUNETR_Simplified` 유지.
- **mobile_unetr_3d**, **segformer3d**: MONAI에 동일 3D 모델 없음 → 현재는 in-repo 구현 유지.

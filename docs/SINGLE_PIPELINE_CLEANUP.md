# 단일 파이프라인 정리 가이드

Cascade(ROI→Seg) 파이프라인이 제거된 뒤, 전체는 **단일 슬라이딩 윈도우 파이프라인**만 사용합니다.  
**적용 완료**: `custom/architecture/cascade` 폴더 전체 제거. ShuffleNetV2·Patch Conv Transformer seg_model 계열은 삭제했고, 재도입 시에는 **custom 직속** (예: `custom/shufflenet_v2.py`, `custom/shufflenet_v2_segnext.py`)으로 두면 됩니다.

---

## 1단계: 확실히 지울 수 있는 것 (선 삭제)

**✅ 적용 완료**: Cascade 베이스라인 3종 + `baseline_models.py` 삭제됨.

### 1.1 Cascade 베이스라인 래퍼 3종 + `baseline_models.py` (삭제됨)

| 삭제 대상 | 설명 |
|----------|------|
| `cascade_unet3d_*` | 단일 파이프라인에 이미 `unet3d_*` 존재. cascade 버전은 CoordConv 등 입력 래퍼일 뿐이라 중복. |
| `cascade_unetr_*` | 동일하게 `unetr` 존재. cascade 래퍼만 제거. |
| `cascade_swin_unetr_*` | 동일하게 `swin_unetr` 존재. cascade 래퍼만 제거. |
| `seg_model/baseline_models.py` | 위 3종의 CascadeUNet3D, CascadeUNETR, CascadeSwinUNETR 및 build_cascade_unet3d/unetr/swin_unetr 정의. |

**적용한 조치**
- `experiment_utils.py`: `cascade_unet3d_`, `cascade_unetr_`, `cascade_swin_unetr_` 분기 및 SUPPORTED_MODEL_PATTERNS에서 제거
- `experiment_config.py`: SIZE_SUFFIX_MODELS에서 cascade_unet3d_/unetr_/swin_unetr_ 제거
- `seg_model/__init__.py`: baseline_models import/__all__ 제거
- `baseline_models.py` 파일 삭제

### 1.2 cascade_patch_conv_transformer_ — 삭제 완료

- Patch Conv Transformer 계열 제거됨. 필요 시 custom 직속으로 재구현 가능.

---

## cascade 폴더 제거 완료

- **삭제**: `models/custom/architecture/` 전체 (그 안의 `cascade/`, `cascade/seg_model/` 포함).
- **patch_conv_transformer**: 이전에 삭제됨.
- **shufflenet_v2 / shufflenet_v2_segnext**: seg_model 쪽 파일은 이전에 삭제됨. 재도입 시 **custom 직속**에 두면 됨 (예: `custom/shufflenet_v2.py`, `custom/shufflenet_v2_segnext.py`).
- **유지**: `dualbranch_19_shufflenet_v2_stage3fused_*` 는 `custom/dualbranch_shufflenet_v2.py` + `modules/shufflenet_modules.py` 만 사용하므로 그대로 사용 가능.

---

## 추가 삭제 추천 (선택)

### A. 미구현 모델 prefix 제거 (권장)
- **dualbranch_10_unet_**, **dualbranch_11_unet_**: `create_model()` 분기가 없고 config/SUPPORTED_MODEL_PATTERNS에만 있음. 제거 시 혼란 감소.

### B. Cascade/ROI 결과 필드 제거 (권장)
- **utils/result_utils.py**: `create_result_dict()`의 `cascade_metrics`, `roi_model_name` 인자 및 내부 처리 제거. 호출처(experiment_orchestrator)에서 이미 넘기지 않음.  
- 단, 과거 실험 결과 JSON에 `cascade_*`, `roi_model_name` 키가 있다면 로딩 시 무시만 되므로 제거해도 무방.

### C. 문서/README 정리
- **README 프로젝트 구조**: `train_roi.py`, `utils/runner/roi_training.py` 항목 제거 (이미 파일 없음).

### D. ShuffleNetV2 / Patch Conv Transformer — **적용 완료**
- cascade_patch_conv_transformer, seg_model ShuffleNetV2 계열 제거함. create_model 분기·config·`seg_model/shufflenet_v2.py`, `shufflenet_v2_segnext.py`, `patch_conv_transformer.py` 삭제. `dualbranch_19_shufflenet_v2_stage3fused_`(custom/dualbranch_shufflenet_v2)는 유지.

### E. nnUNet 관련 — **적용 완료**
- 모델 이름 `nnunet` 및 `scripts/compare_with_nnunet.py` 제거함.

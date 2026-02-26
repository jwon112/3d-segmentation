# Dual-Branch 시리즈 모델 목록 (정리용)

`models/` 내 dualbranch 관련 파일별로, **어떤 실험 이름(prefix)**으로 쓰이고 **어떤 클래스(모델)**들이 있는지 정리했습니다.  
ablation 정리 시 하나씩 쳐낼 때 참고용입니다.

**삭제됨 (정리 완료):**  
- **dualbranch_basic.py** (01, 02, 03) — 공통 블록은 `models/modules/dualbranch_blocks.py`로 이전  
- **dualbranch_mobile.py** — 공통 블록은 `models/modules/dualbranch_blocks.py`로 이전  
- **dualbranch_16_unet.py** (Shuffle Hybrid)  
- **dualbranch_17_unet.py** (PAM-lite) — `modules/shufflenet_pamlite_modules.py` 함께 삭제  
- **dualbranch_shufflenet.py** (ShuffleNet V1, 18)

---

## 1. ~~`dualbranch_basic.py`~~ — **삭제됨** (이전: 01, 02, 03)

| 실험 이름(prefix) | 클래스 | 설명 |
|------------------|--------|------|
| `dualbranch_01_unet_` | `DualBranchUNet3D` | MaxPool 기반 Down3D (baseline) |
| `dualbranch_02_unet_` | `DualBranchUNet3D_Stride` | Stride Conv 다운샘플링 (MaxPool 대신) |
| `dualbranch_03_unet_` | `DualBranchUNet3D_StrideDilated` | FLAIR 브랜치에 dilated conv (넓은 receptive field) |

- **사이즈**: `_xs`, `_s`, `_m`, `_l` 지원  
- **비고**: `dualbranch_2modal_unet_s` 도 여기서 `DualBranchUNet3D` 사용 (고정 size='s')

---

## 2. `dualbranch_replk.py` — RepLK + MViT ablation (04, 05, 06, 07)

| 실험 이름(prefix) | 클래스 | 설명 |
|------------------|--------|------|
| `dualbranch_04_unet_` | `DualBranchUNet3D_StrideLK` | RepLK(13×13×13) FLAIR 브랜치 |
| `dualbranch_05_unet_` | `DualBranchUNet3D_StrideLK_FFN2` | RepLK + FFN2 (expansion_ratio=2) |
| `dualbranch_06_unet_` | `DualBranchUNet3D_StrideLK_FFN2_MViT` | RepLK + FFN2 + MViT Stage 4,5 |
| `dualbranch_07_unet_` | `DualBranchUNet3D_StrideLK_FFN2_MViT_Stage5` | RepLK + FFN2 + MViT Stage 5만 |

- **사이즈**: `_xs`, `_s`, `_m`, `_l` 지원

---

## 3. ~~`dualbranch_mobile.py`~~ — **삭제됨**

| 실험 이름(prefix) | 클래스 | 설명 |
|------------------|--------|------|
| `dualbranch_mobilenetv2_dilated_` | `DualBranchUNet3D_MobileNetV2` | Stage3-fused MobileNetV2 (표준 디코더) |
| `dualbranch_mobilenetv2_dilated_fixed_` | `DualBranchUNet3D_MobileNetV2` | 동일 + fixed decoder 채널 |

- **사이즈**: `_xs`, `_s`, `_m`, `_l`  
- **추가**: `DualBranchUNet3D_MobileNetV2_FixedDecoder` 등 크기별 서브클래스 존재

---

## 4. `dualbranch_mvit.py` — MobileViT Extended (13)

| 실험 이름(prefix) | 클래스 | 설명 |
|------------------|--------|------|
| `dualbranch_13_unet_` | `DualBranchUNet3D_MViT_Extended` | FLAIR에 MobileViT V3 Stage 3,4 + Stage5 MViT |

- **사이즈**: `_xs`, `_s`, `_m`, `_l`  
- **의존**: `dualbranch_mobile` (MobileNetV2 블록)

---

## 5. `dualbranch_backbone_unet.py` — Backbone 비교 (PAM/backbone ablation)

**실험 이름**: `dualbranch_backbone_<backbone>_<size>` (예: `dualbranch_backbone_shufflenetv2_s`)

| backbone 키 | 클래스 | 설명 |
|-------------|--------|------|
| `mobilenetv2_expand2` | `DualBranchUNet3D_MobileNetV2_Expand2` | MobileNetV2 expand_ratio=2 |
| `ghostnet` | `DualBranchUNet3D_GhostNet` | GhostNet |
| `dilated` | `DualBranchUNet3D_Dilated` | Dilated conv |
| `convnext` | `DualBranchUNet3D_ConvNeXt` | ConvNeXt |
| `shufflenetv2` | `DualBranchUNet3D_ShuffleNetV2` | ShuffleNetV2 |
| `shufflenetv2_crossattn` | `DualBranchUNet3D_ShuffleNetV2_CrossAttn` | ShuffleNetV2 + Cross Attention |
| `shufflenetv2_dilated` | `DualBranchUNet3D_ShuffleNetV2_Dilated` | ShuffleNetV2 + Dilated |
| `shufflenetv2_lk` | `DualBranchUNet3D_ShuffleNetV2_LK` | ShuffleNetV2 + Large Kernel |

- **사이즈**: `_xs`, `_s`, `_m`, `_l` (일부는 Small만 명시된 서브클래스만 있음)  
- **비고**: 파일이 크고(1000줄+) 여러 모듈 의존(ghostnet, convnext, cross_attention 등)

---

## 6. ~~`dualbranch_16_unet.py`~~ — **삭제됨**

| 실험 이름(prefix) | 클래스 | 설명 |
|------------------|--------|------|
| `dualbranch_16_shufflenet_hybrid_` | `DualBranchUNet3D_ShuffleHybrid` | ShuffleNetV2 (Stage2–3) + Hybrid Transformer (Stage4–5) |
| `dualbranch_16_shufflenet_hybrid_ln_` | `DualBranchUNet3D_ShuffleHybrid_AllLN` | 동일 + All LayerNorm 변형 |

- **사이즈**: `_xs`, `_s`, `_m`, `_l`  
- **의존**: `dualbranch_backbone_unet` (Stem3x3), `shufflenet_modules`, `shufflenet_hybrid_modules`

---

## 7. ~~`dualbranch_17_unet.py`~~ — **삭제됨** (PAM-lite)

**PAM이란?**  
이 코드베이스에서 **PAM-lite**는 “Position Attention Module”을 쓰지 않고 비슷한 효과를 내는 **경량 블록**을 말합니다.  
`shufflenet_pamlite_modules.py`에서는 **Dilated depthwise conv(rate 1/2/5) + GhostModule3D**로 전역에 가까운 receptive field를 늘리면서, 파라미터/연산을 크게 늘리지 않도록 했습니다.  
즉, **attention 없이 dilated + Ghost로 넓은 receptive field를 쓰는 ShuffleNetV2 변형**이라고 보면 됩니다.

| 실험 이름(prefix) | 클래스 | 설명 |
|------------------|--------|------|
| `dualbranch_17_shufflenet_pamlite_` | `DualBranchUNet3D_ShufflePamLite` | ShuffleNet PAM-lite 블록 (Stage1–4 dual, Stage5 single) |
| `dualbranch_17_shufflenet_pamlite_v3_` | `DualBranchUNet3D_ShufflePamLiteV3` | PAM-lite V3 (Hybrid V3 블록) |

- **사이즈**: `_xs`, `_s`, `_m`, `_l`  
- **의존**: `dualbranch_backbone_unet` (Stem3x3), `shufflenet_pamlite_modules`, `shufflenet_hybrid_modules`

---

## 8. ~~`dualbranch_shufflenet.py`~~ — **삭제됨** (ShuffleNet V1, 18)

| 실험 이름(prefix) | 클래스 | 설명 |
|------------------|--------|------|
| `dualbranch_18_shufflenet_v1_` | `DualBranchUNet3D_ShuffleNetV1` | ShuffleNet V1 + CBAM |
| `dualbranch_18_shufflenet_v1_stage3fused_` | `DualBranchUNet3D_ShuffleNetV1_Stage3Fused` | ShuffleNet V1, Stage3에서 fuse (4-stage 구조) |

- **사이즈**: `_xs`, `_s`, `_m`, `_l`  
- **비고**: config에 `_stage3fused_fixed_decoder_`, `_stage3fused_half_decoder_` 등 추가 variant 있음 (실제 create_model에서 처리 여부는 코드 확인 필요)

---

## 9. `dualbranch_shufflenet_v2.py` — ShuffleNet V2 Stage3 Fused (19)

| 실험 이름(prefix) | 클래스 | 설명 |
|------------------|--------|------|
| `dualbranch_19_shufflenet_v2_stage3fused_` | `DualBranchUNet3D_ShuffleNetV2_Stage3Fused` | ShuffleNet V2, Stage3에서 fuse (4-stage) |

- **사이즈**: `_xs`, `_s`, `_m`, `_l`  
- **비고**: config에 `_stage3fused_fixed_decoder_`, `_stage3fused_half_decoder_` 등 있음

---

## 10. 미사용/미구현 prefix

- **`dualbranch_10_unet_`**, **`dualbranch_11_unet_`**: `experiment_config.py` / `experiment_utils.py` prefix 목록에만 있고, `create_model()` 분기 없음 → **현재 미구현** (과거 번호만 남은 것으로 추정).

---

## 파일 ↔ 실험 이름 매핑 요약

| 파일 | 실험 번호/이름 |
|-----|----------------|
| `dualbranch_basic.py` | 01, 02, 03, dualbranch_2modal_unet_s |
| `dualbranch_replk.py` | 04, 05, 06, 07 |
| `dualbranch_mobile.py` | dualbranch_mobilenetv2_dilated_*, dualbranch_mobilenetv2_dilated_fixed_* |
| `dualbranch_mvit.py` | 13 |
| `dualbranch_backbone_unet.py` | dualbranch_backbone_* (backbone 8종) |
| `dualbranch_16_unet.py` | 16_shufflenet_hybrid, 16_shufflenet_hybrid_ln |
| `dualbranch_17_unet.py` | 17_shufflenet_pamlite, 17_shufflenet_pamlite_v3 |
| `dualbranch_shufflenet.py` | 18_shufflenet_v1, 18_shufflenet_v1_stage3fused |
| `dualbranch_shufflenet_v2.py` | 19_shufflenet_v2_stage3fused |

---

## 의존 관계 (쳐낼 때 참고)

- **dualbranch_backbone_unet.py**: `dualbranch_mobile`, `dualbranch_basic`, `channel_configs`, `modules`(mvit, ghostnet, shufflenet, convnext, cross_attention_3d)  
- **dualbranch_16_unet.py**: `dualbranch_backbone_unet`(Stem3x3), shufflenet_modules, shufflenet_hybrid_modules  
- **dualbranch_17_unet.py**: `dualbranch_backbone_unet`(Stem3x3), shufflenet_pamlite_modules, shufflenet_hybrid_modules  
- **dualbranch_mvit.py**: `dualbranch_mobile`

모델을 하나씩 제거할 때는 `utils/experiment_utils.py`의 `create_model()`, `utils/experiment_config.py`의 prefix/사이즈 목록, 그리고 `evaluate_experiment.py` / `experiment_orchestrator.py`의 dualbranch_04~07 등 prefix 하드코딩도 함께 정리해야 합니다.

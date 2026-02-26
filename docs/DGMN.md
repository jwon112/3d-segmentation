SYSTEM SPECIFICATION: 3D Dynamic Gated Mamba Network (DGMN)

1. System Metadata

Task: 3D Volumetric Medical Image Segmentation

Input Data: 3D Tensor $X \in \mathbb{R}^{B \times C_{in} \times D \times H \times W}$

Output Data: Segmentation Map $Y \in \mathbb{R}^{B \times N_{class} \times D \times H \times W}$

Core Philosophy:

Efficient 3D Operation: Minimize dense 3D Convolutions; maximize Depth-wise Conv & Mamba.

Dynamic Gating: Replace static weights with input-dependent dynamic weights (GLU, Spatial Gate, Softmax Fusion).

Decoder-free: Use All-MLP style multi-scale fusion instead of symmetric decoder.

2. Global Hyperparameters

Normalization: LayerNorm (Block Input), InstanceNorm3d (Conv Internal)

Activation: GELU (Features), Sigmoid (Gating coefficients), Softmax (Fusion weights)

Upsampling Mode: Trilinear Interpolation

Embedding Dim ($C_{emb}$): 128 (Default for fusion projection)

3. Module Specifications (Micro-Architecture)

3.1. Gating Modules (Primitives)

Module A: GLUBlock (Channel Gating)

Input: $X \in \mathbb{R}^{B \times C \times D \times H \times W}$

Operation:

$X_{proj} = Conv3d_{1\times1}(X)$ where $C_{out} = 2C$

$A, B = Split(X_{proj}, dim=1)$ where $A, B \in \mathbb{R}^{B \times C \times \dots}$

$Gate = Sigmoid(B)$

$Output = A \odot Gate$ (Element-wise multiplication)

Output: $Y \in \mathbb{R}^{B \times C \times D \times H \times W}$

Module B: SpatialGatingBlock (Position Gating)

Input: $X \in \mathbb{R}^{B \times C \times D \times H \times W}$

Operation:

$S_{map} = Conv3d_{1\times1}(X)$ where $C_{out} = 1$

$Attention = Sigmoid(S_{map})$

$Output = X \odot Attention$ (Broadcasting)

Output: $Y \in \mathbb{R}^{B \times C \times D \times H \times W}$

3.2. Encoder Block: ParallelBranchBlock

Input: $X_{in}$

Parameters: in_channels ($C$)

Logic Flow:

Pre-norm: $X = LayerNorm(X_{in})$

Branching:

Path A (Local): $F_{local} = DepthwiseConv3d_{7\times7\times7}(X)$

Path B (Global): $F_{global} = MambaBlock3d(X)$ (or Dilated Conv)

Independent Gating (Sequential Application):

$F_{local}' = SpatialGatingBlock(GLUBlock(F_{local}))$

$F_{global}' = SpatialGatingBlock(GLUBlock(F_{global}))$

Fusion (Cross-Channel Mixing):

$F_{concat} = Concat([F_{local}', F_{global}'], dim=1)$ $\to$ Channels: $2C$

$F_{fused} = Conv3d_{1\times1}(F_{concat})$ $\to$ Channels: $C$ (Projection & Mixing)

Residual:

$Output = F_{fused} + X_{in}$

3.3. Decoder Fusion: MultiScaleSoftmaxFusion

Input: List of tensors $[S_1, S_2, S_3, S_4]$ from Encoder Stages.

Target Resolution: Size of $S_1$ $(D/4, H/4, W/4)$.

Logic Flow:

Projection & Upsampling:

For each $S_i$:

Project to $C_{emb}$ using $Conv3d_{1\times1}$.

Upsample to Target Resolution (Trilinear).

Result: $[P_1, P_2, P_3, P_4]$ where all shapes are $(B, C_{emb}, D_{target}, \dots)$.

Concatenation:

$U = Concat([P_1, P_2, P_3, P_4], dim=1)$ $\to$ Shape: $(B, 4C_{emb}, \dots)$

Weight Generation:

$W_{logits} = Conv3d_{1\times1}(U)$ where $C_{out}=4$.

$W_{softmax} = Softmax(W_{logits}, dim=1)$ $\to$ Shape: $(B, 4, D_{target}, \dots)$.

Dynamic Aggregation:

$Y = \sum_{i=1}^{4} (P_i \odot W_{softmax}[:, i:i+1])$

Output: Fused Feature Map ($B, C_{emb}, D_{target}, \dots$)

3.4. MambaBlock3D (using Mamba-2)

목표: 기존 3D 파이프라인(`B×C×D×H×W` 입력, 3D 슬라이딩 윈도우)과 자연스럽게 호환되는 형태로 **Mamba-2**를 래핑한다.

Interface (conceptual):

- 입력: $X \in \mathbb{R}^{B \times C \times D \times H \times W}$
- 출력: $Y \in \mathbb{R}^{B \times C \times D \times H \times W}$ (채널·공간 차원 동일)

Sequence 축 정의:

- Mamba-2는 1D sequence $(B, L, C)$ 입력을 가정하므로, **깊이 축 $D$만 sequence 축**으로 사용한다.
- 변환:
  - $X$를 $(B \cdot H \cdot W, D, C)$ 로 reshape → 각 $(H,W)$ 위치별로 길이 $D$인 sequence.
  - Mamba-2 블록을 적용한 뒤 다시 $(B, C, D, H, W)$ 로 reshape.

Hyper-parameters (예시 기본값):

- $d\_model = C$ (입력 채널 수)
- $d\_state \in \{64, 128\}$ (기본 64)
- $d\_conv \in \{3, 4\}$ (local conv window)
- expand factor: 2

구현 시에는 `mamba-ssm` 패키지의 **Mamba-2** 모듈을 사용하고,  
`MambaBlock3D` 내부에서 위와 같이 reshape 만 처리하는 **얇은 래퍼**로 두는 것을 기본으로 한다.


4. Pipeline Execution Flow

Stage 1: Encoding

Iterate through Stages $i \in [1, 2, 3, 4]$:

Apply Downsampling (Stride Conv) if $i > 1$.

Apply $N_i$ x ParallelBranchBlock.

Save output features $S_i$.

Stage 2: Decoding

Input: $[S_1, S_2, S_3, S_4]$

Apply MultiScaleSoftmaxFusion.

Output: $F_{final}$.

Stage 3: Prediction

Apply Conv3d_{1\times1} to $F_{final}$ to map $C_{emb} \to N_{class}$.

5. Experiment Protocols (Ablation Study)

Config 파일에서 아래 플래그/옵션으로 ablation을 제어한다.

- `use_glu: bool` — GLUBlock 사용 여부
- `use_spatial_gate: bool` — SpatialGatingBlock 사용 여부
- `fusion_type: {'concat_linear', 'softmax_attention'}` — Multi-scale fusion 방식

실험 설정 예시:

| Experiment ID | use_glu | use_spatial_gate | fusion_type        | 설명 |
|---------------|--------:|-----------------:|--------------------|------|
| Baseline      |  False  |          False   | `"concat_linear"`  | 단순 concat + 1×1 Conv, 게이팅 없음 |
| Exp 1         |  True   |          False   | `"concat_linear"`  | 채널 게이팅(GLU)만 적용 |
| Exp 2         |  False  |          True    | `"concat_linear"`  | 공간 게이팅(Spatial)만 적용 |
| Exp 3         |  False  |          False   | `"softmax_attention"` | 게이팅 없이 softmax 기반 스케일 가중합만 사용 |
| Proposed      |  True   |          True    | `"softmax_attention"` | 채널/공간 게이팅 + softmax fusion (제안 모델) |

6. Implementation Constraints

Tensor Shapes: Ensure explicit shape checks at fusion points.

Memory Optimization: Use checkpointing for Mamba blocks if GPU VRAM is limited.

Gating Overhead: The 1x1 Convs in Gating modules must be lightweight. Do not use large hidden dimensions.

Mamba Backend:

- 구현 초기 버전은 **Mamba-2 (mamba-ssm)** 를 기본 backend로 사용한다.
- `MambaBlock3D` 의 외부 인터페이스는 항상 `B×C×D×H×W → B×C×D×H×W` 를 유지하여,  
  기존 3D 파이프라인(`train_model`, `sliding_window_inference_3d`, `evaluate_model`)과 완전히 호환되도록 설계한다.
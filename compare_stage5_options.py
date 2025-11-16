"""
Stage 5 블록 변경 시 파라미터 및 FLOPs 비교 분석

비교 대상:
1. ShuffleNetV2 블록 (현재)
2. MobileViT 블록
3. Concat (현재)
4. Cross Attention
"""

# size='s' 기준
branch4 = 128  # Stage 4 출력 채널 (각 분기)
down5 = 256    # Stage 5 출력 채널 (각 분기)
resolution_in = 16 * 16 * 16  # 4096
resolution_out = 8 * 8 * 8    # 512

print("=" * 80)
print("Stage 5 블록 변경 시 파라미터 및 FLOPs 비교 (size='s' 기준)")
print("=" * 80)
print(f"입력: {branch4} channels, {resolution_in} spatial locations")
print(f"출력: {down5} channels, {resolution_out} spatial locations")
print()

# ============================================================================
# 1. ShuffleNetV2 블록 (현재)
# ============================================================================
print("1. ShuffleNetV2 블록 (Down3DShuffleNetV2)")
print("-" * 80)

# Unit1 (stride=2): Branch1 + Branch2
# Branch1: DWConv(128, stride=2) -> 1x1(64)
branch1_unit1_params = (3*3*128) + (128*64)  # DWConv + 1x1
branch1_unit1_flops = (3*3*128 * resolution_in // 4) + (128*64 * resolution_out)  # stride=2

# Branch2: 1x1(128) -> DWConv(128, stride=2) -> 1x1(64)
branch2_unit1_params = (128*128) + (3*3*128) + (128*64)
branch2_unit1_flops = (128*128 * resolution_in) + (3*3*128 * resolution_in // 4) + (128*64 * resolution_out)

unit1_params = branch1_unit1_params + branch2_unit1_params
unit1_flops = branch1_unit1_flops + branch2_unit1_flops

# Unit2 (stride=1): Split -> Branch1 (identity) + Branch2
# Branch1: Identity (no params)
# Branch2: DWConv(128) -> 1x1(128) -> DWConv(128) -> 1x1(128)
branch2_unit2_params = (3*3*128) + (128*128) + (3*3*128) + (128*128)
branch2_unit2_flops = (3*3*128 * resolution_out) + (128*128 * resolution_out) + (3*3*128 * resolution_out) + (128*128 * resolution_out)

unit2_params = branch2_unit2_params
unit2_flops = branch2_unit2_flops

shufflenet_params = unit1_params + unit2_params
shufflenet_flops = unit1_flops + unit2_flops

print(f"  파라미터: {shufflenet_params:,}")
print(f"  FLOPs: {shufflenet_flops:,}")
print()

# ============================================================================
# 2. MobileViT 블록
# ============================================================================
print("2. MobileViT 블록 (Down3DStrideMViT)")
print("-" * 80)

# Transition3D(128, 256)
# PW: 1x1(128->256)
# DW: 3x3(256, stride=2)
transition_params = (128*256) + (3*3*256)
transition_flops = (128*256 * resolution_in) + (3*3*256 * resolution_out)

# MobileViT3DBlock(256)
# Local conv: 3x3(256)
local_params = 3*3*256
local_flops = 3*3*256 * resolution_out

# Projections
proj_in_params = 256*256  # 1x1
proj_out_params = 256*256  # 1x1
proj_flops = (256*256 * resolution_out) * 2

# Multi-head attention (embed_dim=256, num_heads=4, head_dim=64)
# Q, K, V projections are included in proj_in
# Attention: Q@K^T (256*512) @ (512*256) -> (256*256)
# Then @V: (256*256) @ (256*256) -> (256*256)
attn_flops = (256 * 512 * 256) + (256 * 256 * 256)  # QK^T + AV

# MLP (mlp_ratio=2): 256 -> 512 -> 256
mlp_params = (256*512) + (512*256)
mlp_flops = (256*512 * resolution_out) + (512*256 * resolution_out)

mvit_params = transition_params + local_params + proj_in_params + proj_out_params + mlp_params
mvit_flops = transition_flops + local_flops + proj_flops + attn_flops + mlp_flops

print(f"  파라미터: {mvit_params:,}")
print(f"  FLOPs: {mvit_flops:,}")
print()

# ============================================================================
# 3. Concat (현재)
# ============================================================================
print("3. Concat (현재)")
print("-" * 80)
concat_params = 0
concat_flops = 0  # 메모리 복사만, 연산은 거의 없음
print(f"  파라미터: {concat_params:,}")
print(f"  FLOPs: {concat_flops:,}")
print()

# ============================================================================
# 4. Cross Attention (BidirectionalCrossAttention3D)
# ============================================================================
print("4. Cross Attention (BidirectionalCrossAttention3D)")
print("-" * 80)

# CrossAttention3D(256) x 2
# Q, K, V projections: 3 * (256*256)
proj_params = 3 * (256*256)

# Output projection: 256*256
out_proj_params = 256*256

# Single CrossAttention
single_cross_attn_params = proj_params + out_proj_params

# Attention computation: Q@K^T (256*512) @ (512*256) -> (256*256)
# Then @V: (256*256) @ (256*256) -> (256*256)
single_cross_attn_flops = (256 * 512 * 256) + (256 * 256 * 256)

# Bidirectional: 2 * CrossAttention + Fusion
# Fusion: 1x1(512->256)
fusion_params = 512*256
fusion_flops = 512*256 * resolution_out

cross_attn_params = 2 * single_cross_attn_params + fusion_params
cross_attn_flops = 2 * single_cross_attn_flops + fusion_flops

print(f"  파라미터: {cross_attn_params:,}")
print(f"  FLOPs: {cross_attn_flops:,}")
print()

# ============================================================================
# 비교 요약
# ============================================================================
print("=" * 80)
print("비교 요약")
print("=" * 80)

print("\n[블록 변경: ShuffleNetV2 -> MobileViT]")
print(f"  파라미터 증가: {mvit_params - shufflenet_params:,} ({((mvit_params / shufflenet_params - 1) * 100):.1f}%)")
print(f"  FLOPs 증가: {mvit_flops - shufflenet_flops:,} ({((mvit_flops / shufflenet_flops - 1) * 100):.1f}%)")

print("\n[융합 변경: Concat -> Cross Attention]")
print(f"  파라미터 증가: {cross_attn_params - concat_params:,} (무한대% - concat은 파라미터 없음)")
print(f"  FLOPs 증가: {cross_attn_flops - concat_flops:,} (무한대% - concat은 FLOPs 거의 없음)")

print("\n[절대값 비교]")
print(f"  MobileViT 블록 파라미터: {mvit_params:,}")
print(f"  Cross Attention 파라미터: {cross_attn_params:,}")
print(f"  MobileViT 블록 FLOPs: {mvit_flops:,}")
print(f"  Cross Attention FLOPs: {cross_attn_flops:,}")

print("\n[결론]")
if mvit_params > cross_attn_params:
    print(f"  파라미터: MobileViT 블록이 Cross Attention보다 {mvit_params - cross_attn_params:,}개 더 많음")
else:
    print(f"  파라미터: Cross Attention이 MobileViT 블록보다 {cross_attn_params - mvit_params:,}개 더 많음")

if mvit_flops > cross_attn_flops:
    print(f"  FLOPs: MobileViT 블록이 Cross Attention보다 {mvit_flops - cross_attn_flops:,}개 더 많음")
else:
    print(f"  FLOPs: Cross Attention이 MobileViT 블록보다 {cross_attn_flops - mvit_flops:,}개 더 많음")


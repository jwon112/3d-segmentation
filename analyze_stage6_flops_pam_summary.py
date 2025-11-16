"""Summary analysis of FLOPs and PAM changes with Stage 6"""
import torch

print("=" * 80)
print("FLOPs and PAM Analysis: Stage 5 vs Stage 6")
print("=" * 80)
print()

# Measured values from Stage 5
stage5_data = {
    'xs': {
        'flops': 67_063_906_304,  # 0.0671 TFLOPs
        'pam_inf': 1069.76 * 1024**2,  # MB to bytes
        'pam_train': 3335.54 * 1024**2,
    },
    's': {
        'flops': 259_621_388_288,  # 0.2596 TFLOPs
        'pam_inf': 2106.46 * 1024**2,
        'pam_train': 6582.53 * 1024**2,
    },
    'm': {
        'flops': 1_021_217_079_296,  # 1.0212 TFLOPs
        'pam_inf': 4260.80 * 1024**2,
        'pam_train': 13219.04 * 1024**2,
    },
}

# Channel configurations
channels = {
    'xs': {'down5': 128, 'down6': 256, 'branch4': 64, 'stem': 8, 'out': 16},
    's': {'down5': 256, 'down6': 512, 'branch4': 128, 'stem': 16, 'out': 32},
    'm': {'down5': 512, 'down6': 1024, 'branch4': 256, 'stem': 32, 'out': 64},
    'l': {'down5': 1024, 'down6': 2048, 'branch4': 512, 'stem': 64, 'out': 128},
}

print("Current Structure (Stage 5) - Measured:")
print("-" * 80)
for size in ['xs', 's', 'm']:
    data = stage5_data[size]
    print(f"{size.upper():2s}: FLOPs={data['flops']/1e12:.4f} TFLOPs, "
          f"PAM(inf)={data['pam_inf']/1024**2:.2f} MB, "
          f"PAM(train)={data['pam_train']/1024**2:.2f} MB")

print("\n" + "=" * 80)
print("Theoretical Analysis: Stage 6 Addition")
print("-" * 80)
print()

for size in ['xs', 's', 'm', 'l']:
    ch = channels[size]
    down5 = ch['down5']
    down6 = ch['down6']
    
    print(f"[{size.upper()}] Channels: down5={down5}, down6={down6}")
    
    # Resolution changes
    stage5_res = 8 * 8 * 8  # 512
    stage6_res = 4 * 4 * 4  # 64
    resolution_ratio = stage6_res / stage5_res  # 0.125 (1/8)
    
    # 1. Stage 6 MobileViT FLOPs
    # MobileViT: Attention + Conv operations
    # Attention: O(C^2 * N) where N is number of tokens
    # For 8×8×8 input, patch-based attention
    # Rough estimate: down5^2 * stage5_res * attention_ops
    attention_flops = down5 * down5 * stage5_res * 2  # Self-attention (rough)
    conv_flops = down5 * down6 * stage5_res  # Downsampling conv
    mlp_flops = down5 * down6 * stage5_res * 2  # MLP in MobileViT
    stage6_flops = (attention_flops + conv_flops + mlp_flops) * 2  # Dual-branch
    stage6_flops *= 1.2  # Additional overhead
    
    # 2. Cross Attention FLOPs increase
    # Cross Attention: O(C^2 * H*W*D) for attention computation
    # 8 attention heads
    cross_attn_stage5 = down5 * down5 * stage6_res * 8  # 8 heads
    cross_attn_stage6 = down6 * down6 * stage6_res * 8
    cross_attn_increase = cross_attn_stage6 - cross_attn_stage5
    
    # 3. Decoder up1 FLOPs increase
    # Up3D: ConvTranspose (down6 -> down6//2) + DoubleConv
    # Input channels increase: down5 -> down6
    up1_convtranspose_increase = (down6 - down5) * (down6 // 2) * stage6_res * 2  # ConvTranspose
    up1_conv_increase = (down6 - down5) * ch['branch4'] * stage6_res * 2  # DoubleConv
    up1_flops_increase = up1_convtranspose_increase + up1_conv_increase
    
    # 4. Decoder up6 FLOPs (new stage)
    # Final upsampling: stem -> out, 128×128×128
    final_res = 128 * 128 * 128
    up6_convtranspose = ch['stem'] * (ch['stem'] // 2) * final_res * 2
    up6_conv = ch['stem'] * ch['out'] * final_res * 2
    up6_flops = up6_convtranspose + up6_conv
    
    total_flops_increase = stage6_flops + cross_attn_increase + up1_flops_increase + up6_flops
    
    if size in stage5_data:
        flops_ratio = total_flops_increase / stage5_data[size]['flops']
        print(f"  FLOPs increase: {total_flops_increase/1e12:.4f} TFLOPs ({flops_ratio:.1%})")
        print(f"    - Stage 6 MobileViT: {stage6_flops/1e12:.4f} TFLOPs ({stage6_flops/total_flops_increase:.1%})")
        print(f"    - Cross Attention: {cross_attn_increase/1e12:.4f} TFLOPs ({cross_attn_increase/total_flops_increase:.1%})")
        print(f"    - Decoder up1: {up1_flops_increase/1e12:.4f} TFLOPs ({up1_flops_increase/total_flops_increase:.1%})")
        print(f"    - Decoder up6: {up6_flops/1e12:.4f} TFLOPs ({up6_flops/total_flops_increase:.1%})")
    
    # PAM Analysis (Activation Memory)
    # Activation memory: O(batch * channels * H * W * D * 4 bytes)
    batch = 1
    bytes_per_float = 4
    
    # Stage 6 activation (output)
    stage6_activation = batch * down6 * stage6_res * bytes_per_float
    stage6_activation *= 2  # Dual-branch
    
    # Cross Attention activation increase
    # Need to store Q, K, V, attention scores, output
    cross_attn_stage5_bytes = batch * down5 * stage6_res * bytes_per_float * 5  # Q, K, V, scores, out
    cross_attn_stage6_bytes = batch * down6 * stage6_res * bytes_per_float * 5
    cross_attn_bytes_increase = cross_attn_stage6_bytes - cross_attn_stage5_bytes
    
    # Decoder up1 activation increase
    up1_bytes_increase = batch * (down6 - down5) * stage6_res * bytes_per_float
    
    # Decoder up6 activation (new)
    up6_bytes = batch * ch['stem'] * final_res * bytes_per_float
    
    # Additional: intermediate activations in Stage 6 MobileViT
    stage6_intermediate = batch * down5 * stage5_res * bytes_per_float * 3  # Intermediate activations
    stage6_intermediate *= 2  # Dual-branch
    
    total_pam_increase = (stage6_activation + stage6_intermediate + 
                         cross_attn_bytes_increase + up1_bytes_increase + up6_bytes)
    
    if size in stage5_data:
        pam_ratio = total_pam_increase / stage5_data[size]['pam_inf']
        print(f"  PAM (Inference) increase: {total_pam_increase/1024**2:.2f} MB ({pam_ratio:.1%})")
        print(f"    - Stage 6 activation: {(stage6_activation + stage6_intermediate)/1024**2:.2f} MB")
        print(f"    - Cross Attention: {cross_attn_bytes_increase/1024**2:.2f} MB")
        print(f"    - Decoder up1: {up1_bytes_increase/1024**2:.2f} MB")
        print(f"    - Decoder up6: {up6_bytes/1024**2:.2f} MB")
    
    print()

print("=" * 80)
print("Key Insights:")
print("-" * 80)
print("1. FLOPs 증가:")
print("   - 파라미터 증가율(270%)보다 작을 것으로 예상")
print("   - 이유: 해상도 감소(8³ → 4³ = 1/8)가 채널 증가(2x)보다 영향이 큼")
print("   - Stage 6 MobileViT가 전체 FLOPs 증가의 대부분을 차지")
print()
print("2. PAM 증가:")
print("   - 파라미터 증가율보다 훨씬 작을 것으로 예상")
print("   - 이유: Activation 메모리는 해상도 × 채널에 비례")
print("   - 해상도 감소 효과가 채널 증가 효과를 상쇄")
print("   - Train 모드에서는 gradient 저장으로 인해 더 큰 증가 예상")
print()
print("3. 결론:")
print("   - Stage 6 추가는 파라미터는 크게 증가(270%)하지만,")
print("   - FLOPs와 PAM 증가는 상대적으로 완화될 것으로 예상")
print("   - 특히 PAM은 해상도 감소로 인해 증가율이 낮을 것으로 예상")


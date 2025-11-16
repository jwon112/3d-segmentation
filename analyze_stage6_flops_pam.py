"""Analyze FLOPs and PAM changes when adding Stage 6"""
import torch
import torch.nn as nn
from models.dualbranch_14_unet import DualBranchUNet3D_MobileNetV2_Expand2
from models.modules.mvit_modules import Down3DStrideMViT
from models.modules.cross_attention_3d import BidirectionalCrossAttention3D
from models.model_3d_unet import Up3D
from models.channel_configs import get_dualbranch_channels
from utils.experiment_utils import calculate_flops, calculate_pam

print("=" * 80)
print("FLOPs and PAM Analysis: Stage 5 vs Stage 6")
print("=" * 80)
print()

# Input size for 3D models
INPUT_SIZE_3D = (1, 2, 128, 128, 128)  # (batch, channels, D, H, W)

sizes = ['xs', 's', 'm', 'l']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Current structure (Stage 5):")
print("-" * 80)

stage5_flops = {}
stage5_pam_inference = {}
stage5_pam_train = {}

for size in sizes:
    print(f"\n[{size.upper()}]")
    model = DualBranchUNet3D_MobileNetV2_Expand2(size=size).to(device)
    model.eval()
    
    # FLOPs calculation
    try:
        flops = calculate_flops(model, input_size=INPUT_SIZE_3D)
        stage5_flops[size] = flops
        print(f"  FLOPs: {flops:,.0f} ({flops/1e12:.4f} TFLOPs)")
    except Exception as e:
        print(f"  FLOPs calculation failed: {e}")
        stage5_flops[size] = 0
    
    # PAM calculation
    try:
        pam_inf_list, _ = calculate_pam(model, input_size=INPUT_SIZE_3D, mode='inference', 
                                        stage_wise=False, device=device, num_runs=3)
        pam_train_list, _ = calculate_pam(model, input_size=INPUT_SIZE_3D, mode='train', 
                                         stage_wise=False, device=device, num_runs=3)
        
        if pam_inf_list:
            pam_inf_mean = sum(pam_inf_list) / len(pam_inf_list)
            stage5_pam_inference[size] = pam_inf_mean
            print(f"  PAM (Inference): {pam_inf_mean / 1024**2:.2f} MB")
        
        if pam_train_list:
            pam_train_mean = sum(pam_train_list) / len(pam_train_list)
            stage5_pam_train[size] = pam_train_mean
            print(f"  PAM (Train): {pam_train_mean / 1024**2:.2f} MB")
    except Exception as e:
        print(f"  PAM calculation failed: {e}")
        stage5_pam_inference[size] = 0
        stage5_pam_train[size] = 0
    
    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n" + "=" * 80)
print("Estimated changes with Stage 6:")
print("-" * 80)
print()
print("Stage 6 adds:")
print("  - Stage 6 MobileViT: 8×8×8 → 4×4×4, channels down5 → down6 (2x)")
print("  - Cross Attention: down5 → down6 channels (2x)")
print("  - Decoder up1: input channels down5 → down6 (2x)")
print("  - Decoder up6: new stage")
print()

# Theoretical analysis
print("Theoretical Analysis:")
print("-" * 80)

for size in sizes:
    ch = get_dualbranch_channels(size)
    down5 = ch['down5']
    down6 = down5 * 2
    
    print(f"\n[{size.upper()}]")
    print(f"  Channels: down5={down5}, down6={down6}")
    
    # Stage 6 MobileViT FLOPs estimate
    # MobileViT FLOPs ≈ O(C_in * C_out * H * W * D) for attention + conv operations
    # Resolution: 8×8×8 → 4×4×4
    stage6_resolution = 8 * 8 * 8  # Input resolution
    stage6_output_resolution = 4 * 4 * 4  # Output resolution
    
    # Rough estimate: MobileViT has attention (O(N^2 * C)) and conv operations
    # For simplicity, estimate based on channel and resolution
    # Attention: ~down5 * down5 * 8*8*8 (self-attention within patches)
    # Conv operations: ~down5 * down6 * 8*8*8 (downsampling)
    stage6_flops_estimate = (down5 * down5 * stage6_resolution * 2 +  # Attention (rough)
                            down5 * down6 * stage6_resolution) * 2  # Conv + dual-branch
    stage6_flops_estimate *= 1.5  # Additional MLP and other operations
    
    # Cross Attention FLOPs increase
    # Cross Attention: O(C^2 * H * W * D) for attention computation
    cross_attn_stage5_flops = down5 * down5 * stage6_output_resolution * 8  # 8 heads
    cross_attn_stage6_flops = down6 * down6 * stage6_output_resolution * 8
    cross_attn_flops_increase = cross_attn_stage6_flops - cross_attn_stage5_flops
    
    # Decoder up1 FLOPs increase
    # Up3D: ConvTranspose + DoubleConv
    # Input channels: down5 → down6
    up1_flops_increase = (down6 - down5) * ch['branch4'] * stage6_output_resolution * 2  # Rough estimate
    
    # Decoder up6 FLOPs (new)
    up6_flops = ch['stem'] * ch['out'] * 128 * 128 * 128  # Final upsampling
    
    total_flops_increase = stage6_flops_estimate + cross_attn_flops_increase + up1_flops_increase + up6_flops
    
    if stage5_flops[size] > 0:
        flops_increase_ratio = total_flops_increase / stage5_flops[size]
        print(f"  FLOPs increase: {total_flops_increase/1e12:.4f} TFLOPs ({flops_increase_ratio:.1%})")
        print(f"    - Stage 6 MobileViT: {stage6_flops_estimate/1e12:.4f} TFLOPs")
        print(f"    - Cross Attention: {cross_attn_flops_increase/1e12:.4f} TFLOPs")
        print(f"    - Decoder up1: {up1_flops_increase/1e12:.4f} TFLOPs")
        print(f"    - Decoder up6: {up6_flops/1e12:.4f} TFLOPs")
    
    # PAM estimate
    # PAM is mainly activation memory: O(batch * channels * H * W * D)
    # Stage 6 activation: batch * down6 * 4*4*4
    stage6_activation_bytes = 1 * down6 * 4 * 4 * 4 * 4  # batch=1, float32=4 bytes
    stage6_activation_bytes *= 2  # Dual-branch
    
    # Cross Attention activation increase
    cross_attn_stage5_bytes = 1 * down5 * 4 * 4 * 4 * 4
    cross_attn_stage6_bytes = 1 * down6 * 4 * 4 * 4 * 4
    cross_attn_bytes_increase = cross_attn_stage6_bytes - cross_attn_stage5_bytes
    
    # Decoder up1 activation increase
    up1_bytes_increase = 1 * (down6 - down5) * 4 * 4 * 4 * 4
    
    # Decoder up6 activation (new)
    up6_bytes = 1 * ch['stem'] * 128 * 128 * 128 * 4
    
    total_pam_increase = stage6_activation_bytes + cross_attn_bytes_increase + up1_bytes_increase + up6_bytes
    
    if stage5_pam_inference[size] > 0:
        pam_increase_ratio = total_pam_increase / stage5_pam_inference[size]
        print(f"  PAM increase: {total_pam_increase / 1024**2:.2f} MB ({pam_increase_ratio:.1%})")
        print(f"    - Stage 6 activation: {stage6_activation_bytes / 1024**2:.2f} MB")
        print(f"    - Cross Attention: {cross_attn_bytes_increase / 1024**2:.2f} MB")
        print(f"    - Decoder up1: {up1_bytes_increase / 1024**2:.2f} MB")
        print(f"    - Decoder up6: {up6_bytes / 1024**2:.2f} MB")

print("\n" + "=" * 80)
print("Summary:")
print("-" * 80)
print("Stage 6 추가 시:")
print("  - FLOPs: 파라미터 증가율(약 270%)보다 작을 것으로 예상")
print("    이유: 해상도가 8×8×8 → 4×4×4로 감소하여 연산량 증가가 완화됨")
print("  - PAM: 파라미터 증가율보다 훨씬 작을 것으로 예상")
print("    이유: Activation 메모리는 해상도와 채널의 곱에 비례")
print("          해상도 감소(8³ → 4³ = 1/8)가 채널 증가(2x)보다 영향이 큼")
print()
print("결론: Stage 6 추가는 파라미터는 크게 증가하지만,")
print("      FLOPs와 PAM 증가는 상대적으로 완화될 것으로 예상됩니다.")


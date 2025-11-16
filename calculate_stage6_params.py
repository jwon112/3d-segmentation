"""Calculate parameter counts with Stage 6 added"""
import torch
import torch.nn as nn
from models.dualbranch_14_unet import DualBranchUNet3D_MobileNetV2_Expand2
from models.modules.mvit_modules import Down3DStrideMViT
from models.channel_configs import get_dualbranch_channels

print("=" * 70)
print("Parameter Count Comparison: Current (Stage 5) vs With Stage 6")
print("=" * 70)
print()

# Current structure (Stage 5)
print("Current structure (Stage 5):")
print("-" * 70)
sizes = ['xs', 's', 'm', 'l']
current_params = {}
for size in sizes:
    model = DualBranchUNet3D_MobileNetV2_Expand2(size=size)
    params = sum(p.numel() for p in model.parameters())
    current_params[size] = params
    
    ch = get_dualbranch_channels(size)
    print(f"  {size:2s}: {params:>12,} params")
    print(f"        Channels: {ch['stem']}+{ch['stem']}, {ch['branch2']}+{ch['branch2']}, "
          f"{ch['branch3']}+{ch['branch3']}, {ch['branch4']}+{ch['branch4']}, {ch['down5']}+{ch['down5']}")
print()

# Calculate Stage 6 parameters for each size
print("With Stage 6 added (estimated):")
print("-" * 70)
print("Stage 6: Down3DStrideMViT(down5, down6) where down6 = down5 * 2")
print("Cross Attention: Stage 6 output (down6 channels)")
print("Decoder: 6 stages (up1, up2, up3, up4, up5, up6)")
print()

for size in sizes:
    ch = get_dualbranch_channels(size)
    down5_channels = ch['down5']
    down6_channels = down5_channels * 2  # Stage 6: 2x channel increase
    
    # Create Stage 6 block to count parameters
    stage6 = Down3DStrideMViT(down5_channels, down6_channels, norm='bn', num_heads=4, mlp_ratio=2)
    stage6_params = sum(p.numel() for p in stage6.parameters())
    
    # Cross Attention for Stage 6 (down6 channels instead of down5)
    from models.modules.cross_attention_3d import BidirectionalCrossAttention3D
    cross_attn_stage5 = BidirectionalCrossAttention3D(down5_channels, num_heads=8, norm='bn')
    cross_attn_stage6 = BidirectionalCrossAttention3D(down6_channels, num_heads=8, norm='bn')
    cross_attn_diff = sum(p.numel() for p in cross_attn_stage6.parameters()) - sum(p.numel() for p in cross_attn_stage5.parameters())
    
    # Decoder up1 change: down5 -> down6 input channels
    from models.model_3d_unet import Up3D
    up1_stage5 = Up3D(down5_channels, ch['branch4'], bilinear=False, norm='bn', skip_channels=ch['branch4'] * 2)
    up1_stage6 = Up3D(down6_channels, ch['branch4'], bilinear=False, norm='bn', skip_channels=ch['branch4'] * 2)
    up1_diff = sum(p.numel() for p in up1_stage6.parameters()) - sum(p.numel() for p in up1_stage5.parameters())
    
    # Additional up6 decoder stage
    up6 = Up3D(ch['stem'], ch['out'], bilinear=False, norm='bn', skip_channels=1 * 2)
    up6_params = sum(p.numel() for p in up6.parameters())
    
    # Total additional parameters
    additional_params = stage6_params * 2 + cross_attn_diff + up1_diff + up6_params  # *2 for dual-branch
    total_params = current_params[size] + additional_params
    
    increase_ratio = additional_params / current_params[size]
    
    print(f"  {size:2s}: {total_params:>12,} params (+{additional_params:>12,}, {increase_ratio:.1%} increase)")
    print(f"        Base: {current_params[size]:>12,} params")
    print(f"        Stage 6 ({down5_channels}->{down6_channels}): {stage6_params * 2:>12,} params (dual-branch)")
    print(f"        Cross Attention diff: {cross_attn_diff:>12,} params")
    print(f"        Decoder up1 diff: {up1_diff:>12,} params")
    print(f"        Decoder up6 (new): {up6_params:>12,} params")
    print(f"        Channels: {ch['stem']}+{ch['stem']}, {ch['branch2']}+{ch['branch2']}, "
          f"{ch['branch3']}+{ch['branch3']}, {ch['branch4']}+{ch['branch4']}, "
          f"{down5_channels}+{down5_channels}, {down6_channels}+{down6_channels}")
    print()

print("=" * 70)
print("Summary:")
print("-" * 70)
for size in sizes:
    ch = get_dualbranch_channels(size)
    down5_channels = ch['down5']
    down6_channels = down5_channels * 2
    
    stage6 = Down3DStrideMViT(down5_channels, down6_channels, norm='bn', num_heads=4, mlp_ratio=2)
    stage6_params = sum(p.numel() for p in stage6.parameters())
    
    from models.modules.cross_attention_3d import BidirectionalCrossAttention3D
    cross_attn_stage5 = BidirectionalCrossAttention3D(down5_channels, num_heads=8, norm='bn')
    cross_attn_stage6 = BidirectionalCrossAttention3D(down6_channels, num_heads=8, norm='bn')
    cross_attn_diff = sum(p.numel() for p in cross_attn_stage6.parameters()) - sum(p.numel() for p in cross_attn_stage5.parameters())
    
    from models.model_3d_unet import Up3D
    up1_stage5 = Up3D(down5_channels, ch['branch4'], bilinear=False, norm='bn', skip_channels=ch['branch4'] * 2)
    up1_stage6 = Up3D(down6_channels, ch['branch4'], bilinear=False, norm='bn', skip_channels=ch['branch4'] * 2)
    up1_diff = sum(p.numel() for p in up1_stage6.parameters()) - sum(p.numel() for p in up1_stage5.parameters())
    
    up6 = Up3D(ch['stem'], ch['out'], bilinear=False, norm='bn', skip_channels=1 * 2)
    up6_params = sum(p.numel() for p in up6.parameters())
    
    additional_params = stage6_params * 2 + cross_attn_diff + up1_diff + up6_params
    total = current_params[size] + additional_params
    increase_ratio = additional_params / current_params[size]
    
    print(f"{size:2s}: {current_params[size]:>12,} -> {total:>12,} params (+{increase_ratio:.1%})")

print()
print("Note: Stage 6 would reduce resolution from 8×8×8 to 4×4×4")
print("      Decoder would have 6 stages instead of 5")


"""
ShuffleNetV2 Hybrid Modules
ShuffleNetV2 기반 Conv-Transformer 하이브리드 블록

구조:
- ShuffleNetV2의 기본 구조 (split -> branch1 + branch2 -> concat -> shuffle)
- Branch 1: Conv 연산 (기존 ShuffleNetV2 스타일)
- Branch 2: Transformer 연산 (MobileViT 스타일 inverted bottleneck + Self-Attention)
"""

import torch
import torch.nn as nn

from ..model_3d_unet import _make_norm3d
from .shufflenet_modules import channel_shuffle_3d


class ShuffleNetV2HybridUnit3D(nn.Module):
    """3D ShuffleNetV2 Hybrid Unit (Conv + Transformer).
    
    ShuffleNetV2 구조에 Conv와 Transformer를 병렬로 융합:
    - Stride=1: Split -> Branch1 (Conv) + Branch2 (Transformer with inverted bottleneck) -> Concat -> Shuffle
    - Stride=2: No split -> Branch1 (Conv stride=2) + Branch2 (Transformer stride=2) -> Concat -> Shuffle
    
    Branch 2 (Transformer):
    - MobileViT 스타일: 3x3 DWConv + 1x1 Conv (expansion) -> Transformer -> 1x1 Conv (reduction)
    - Inverted bottleneck 구조로 채널 확장/축소
    - Transformer는 Self-Attention + MLP로 구성
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 norm: str = 'bn', expand_ratio: float = 4.0, 
                 num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        if stride == 1:
            # Stride=1: Split operation
            assert in_channels == out_channels, "For stride=1, in_channels must equal out_channels"
            mid_channels = out_channels // 2
            
            # Branch 1: Conv (기존 ShuffleNetV2 스타일)
            self.branch1 = nn.Sequential(
                # Depthwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, 
                         groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Pointwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
            )
            
            # Branch 2: Transformer with inverted bottleneck
            expanded_channels = int(mid_channels * expand_ratio)
            
            # MobileViT 스타일: 3x3 DWConv + 1x1 Conv (expansion)
            self.branch2_local = nn.Sequential(
                # 3x3 Depthwise Conv (local representation)
                nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, 
                         groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # 1x1 Conv (expansion)
                nn.Conv3d(mid_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, expanded_channels),
                nn.ReLU(inplace=True),
            )
            
            # Transformer (Self-Attention + MLP)
            self.branch2_attn_norm = nn.LayerNorm(expanded_channels)
            self.branch2_attn = nn.MultiheadAttention(expanded_channels, num_heads=num_heads, batch_first=True)
            self.branch2_ffn = nn.Sequential(
                nn.Linear(expanded_channels, expanded_channels * mlp_ratio),
                nn.GELU(),
                nn.Linear(expanded_channels * mlp_ratio, expanded_channels)
            )
            
            # 1x1 Conv (reduction)
            self.branch2_reduce = nn.Sequential(
                nn.Conv3d(expanded_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # Stride=2: No split, both branches process full input
            # Branch 1: Conv stride=2
            self.branch1 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels // 2),
                nn.ReLU(inplace=True),
            )
            
            # Branch 2: Transformer stride=2
            expanded_channels = int(in_channels * expand_ratio)
            
            # MobileViT 스타일: 3x3 DWConv stride=2 + 1x1 Conv (expansion)
            self.branch2_local = nn.Sequential(
                # 3x3 Depthwise Conv stride=2 (local representation + downsampling)
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.ReLU(inplace=True),
                # 1x1 Conv (expansion)
                nn.Conv3d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, expanded_channels),
                nn.ReLU(inplace=True),
            )
            
            # Transformer (Self-Attention + MLP)
            self.branch2_attn_norm = nn.LayerNorm(expanded_channels)
            self.branch2_attn = nn.MultiheadAttention(expanded_channels, num_heads=num_heads, batch_first=True)
            self.branch2_ffn = nn.Sequential(
                nn.Linear(expanded_channels, expanded_channels * mlp_ratio),
                nn.GELU(),
                nn.Linear(expanded_channels * mlp_ratio, expanded_channels)
            )
            
            # 1x1 Conv (reduction)
            self.branch2_reduce = nn.Sequential(
                nn.Conv3d(expanded_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels // 2),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            # Split channels
            x1, x2 = x.chunk(2, dim=1)
            
            # Branch 1: Conv
            out1 = self.branch1(x1)
            
            # Branch 2: Transformer with inverted bottleneck
            # Local representation (MobileViT 스타일)
            x2_local = self.branch2_local(x2)  # (B, expanded_channels, D, H, W)
            
            # To tokens: (B, N, C)
            b, c, d, h, w = x2_local.shape
            tokens = x2_local.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, c)
            
            # Transformer
            tokens_norm = self.branch2_attn_norm(tokens)
            attn_out, _ = self.branch2_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens = tokens + attn_out  # Residual connection
            tokens = tokens + self.branch2_ffn(self.branch2_attn_norm(tokens))  # FFN with residual
            
            # Back to 3D: (B, N, C) -> (B, C, D, H, W)
            tokens = tokens.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
            
            # Reduction
            out2 = self.branch2_reduce(tokens)
            
            # Concat
            out = torch.cat([out1, out2], dim=1)
        else:
            # Both branches process full input
            # Branch 1: Conv
            out1 = self.branch1(x)
            
            # Branch 2: Transformer with inverted bottleneck
            # Local representation (MobileViT 스타일)
            x2_local = self.branch2_local(x)  # (B, expanded_channels, D, H, W)
            
            # To tokens: (B, N, C)
            b, c, d, h, w = x2_local.shape
            tokens = x2_local.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, c)
            
            # Transformer
            tokens_norm = self.branch2_attn_norm(tokens)
            attn_out, _ = self.branch2_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens = tokens + attn_out  # Residual connection
            tokens = tokens + self.branch2_ffn(self.branch2_attn_norm(tokens))  # FFN with residual
            
            # Back to 3D: (B, N, C) -> (B, C, D, H, W)
            tokens = tokens.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
            
            # Reduction
            out2 = self.branch2_reduce(tokens)
            
            # Concat
            out = torch.cat([out1, out2], dim=1)
        
        # Channel Shuffle
        out = channel_shuffle_3d(out, groups=2)
        return out


class Down3DShuffleNetV2Hybrid(nn.Module):
    """Downsampling using ShuffleNetV2 Hybrid unit (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', 
                 expand_ratio: float = 4.0, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = ShuffleNetV2HybridUnit3D(in_channels, out_channels, stride=2, 
                                               norm=norm, expand_ratio=expand_ratio, 
                                               num_heads=num_heads, mlp_ratio=mlp_ratio)
        # Second unit: stride=1 for feature refinement
        self.unit2 = ShuffleNetV2HybridUnit3D(out_channels, out_channels, stride=1, 
                                               norm=norm, expand_ratio=expand_ratio, 
                                               num_heads=num_heads, mlp_ratio=mlp_ratio)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


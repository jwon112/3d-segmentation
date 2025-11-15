"""
3D Cross Attention Module for Dual-Branch Feature Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .replk_modules import _make_norm3d


class CrossAttention3D(nn.Module):
    """3D Cross Attention for fusing dual-branch features.
    
    Given two feature maps from different branches (e.g., FLAIR and T1CE),
    this module uses cross-attention to fuse them. One branch provides queries,
    the other provides keys and values.
    
    Args:
        channels: Number of channels in each branch (assumed to be the same)
        num_heads: Number of attention heads
        norm: Normalization type ('bn', 'gn', 'in')
    """
    def __init__(self, channels: int, num_heads: int = 8, norm: str = 'bn'):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections for query, key, value
        # Branch 1 (e.g., FLAIR) -> Query
        # Branch 2 (e.g., T1CE) -> Key, Value
        self.q_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer norm for input features
        self.norm1 = _make_norm3d(norm, channels)
        self.norm2 = _make_norm3d(norm, channels)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: (B, C, D, H, W) - First branch features (e.g., FLAIR) - used as Query
            x2: (B, C, D, H, W) - Second branch features (e.g., T1CE) - used as Key, Value
        
        Returns:
            fused: (B, C, D, H, W) - Fused features
        """
        B, C, D, H, W = x1.shape
        
        # Normalize inputs
        x1_norm = self.norm1(x1)
        x2_norm = self.norm2(x2)
        
        # Project to Q, K, V
        q = self.q_proj(x1_norm)  # (B, C, D, H, W)
        k = self.k_proj(x2_norm)  # (B, C, D, H, W)
        v = self.v_proj(x2_norm)  # (B, C, D, H, W)
        
        # Reshape for multi-head attention: (B, C, D, H, W) -> (B, num_heads, head_dim, D*H*W)
        q = q.view(B, self.num_heads, self.head_dim, D * H * W)  # (B, H, d, N)
        k = k.view(B, self.num_heads, self.head_dim, D * H * W)  # (B, H, d, N)
        v = v.view(B, self.num_heads, self.head_dim, D * H * W)  # (B, H, d, N)
        
        # Transpose for attention computation: (B, H, d, N) -> (B, H, N, d)
        q = q.transpose(-2, -1)  # (B, H, N, d)
        k = k.transpose(-2, -1)  # (B, H, N, d)
        v = v.transpose(-2, -1)  # (B, H, N, d)
        
        # Compute attention scores: (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)  # (B, H, N, N)
        
        # Apply attention to values: (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
        out = torch.matmul(attn, v)  # (B, H, N, d)
        
        # Reshape back: (B, H, N, d) -> (B, C, D, H, W)
        out = out.transpose(-2, -1).contiguous()  # (B, H, d, N)
        out = out.view(B, C, D, H, W)  # (B, C, D, H, W)
        
        # Residual connection and output projection
        out = out + x1  # Residual from query branch
        out = self.out_proj(out)
        
        return out


class BidirectionalCrossAttention3D(nn.Module):
    """Bidirectional Cross Attention for symmetric feature fusion.
    
    Applies cross-attention in both directions and combines the results.
    This allows both branches to attend to each other.
    
    Args:
        channels: Number of channels in each branch
        num_heads: Number of attention heads
        norm: Normalization type
    """
    def __init__(self, channels: int, num_heads: int = 8, norm: str = 'bn'):
        super().__init__()
        self.cross_attn_1to2 = CrossAttention3D(channels, num_heads, norm)
        self.cross_attn_2to1 = CrossAttention3D(channels, num_heads, norm)
        
        # Fusion layer to combine both directions
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: (B, C, D, H, W) - First branch features
            x2: (B, C, D, H, W) - Second branch features
        
        Returns:
            fused: (B, C, D, H, W) - Fused features
        """
        # Cross attention in both directions
        out1 = self.cross_attn_1to2(x1, x2)  # x1 attends to x2
        out2 = self.cross_attn_2to1(x2, x1)  # x2 attends to x1
        
        # Concatenate and fuse
        fused = torch.cat([out1, out2], dim=1)  # (B, 2*C, D, H, W)
        fused = self.fusion(fused)  # (B, C, D, H, W)
        
        return fused


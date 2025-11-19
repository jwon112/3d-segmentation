"""
3D Cross Attention Module for Dual-Branch Feature Fusion

Two implementations:
1. Direct 3D Cross Attention: Works directly on 3D feature maps (spatial attention)
2. Transformer-based Cross Attention: Tokenizes features, applies cross-attention, then restores (global attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_3d_unet import _make_norm3d


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


# ============================================================================
# Transformer-based Cross Attention (Token-based, MobileViT-style)
# ============================================================================

class CrossAttentionTransformerLayer3D(nn.Module):
    """Single Cross-Attention Transformer layer for token-based cross attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        
        # Cross Attention: Query from branch1, Key/Value from branch2
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # FFN
        hidden_dim = dim * mlp_ratio
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.ffn_dropout = nn.Dropout(ffn_dropout)
    
    def forward(self, tokens_q: torch.Tensor, tokens_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens_q: (B, N, C) - Query tokens from branch 1
            tokens_kv: (B, M, C) - Key/Value tokens from branch 2 (can have different sequence length)
        
        Returns:
            out: (B, N, C) - Output tokens (same shape as tokens_q)
        """
        # Cross Attention: Query from branch1, Key/Value from branch2
        attn_input_q = self.norm1_q(tokens_q)
        attn_input_kv = self.norm1_kv(tokens_kv)
        attn_out, _ = self.cross_attn(attn_input_q, attn_input_kv, attn_input_kv)
        tokens = tokens_q + self.attn_dropout(attn_out)
        
        # FFN
        ffn_input = self.norm2(tokens)
        tokens = tokens + self.ffn_dropout(self.ffn(ffn_input))
        return tokens


class CrossAttentionTransformer3D(nn.Module):
    """Multi-layer Cross-Attention Transformer stack."""
    
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionTransformerLayer3D(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, tokens_q: torch.Tensor, tokens_kv: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens_q = layer(tokens_q, tokens_kv)
        return tokens_q


class CrossAttentionTransformerBlock3D(nn.Module):
    """Complete Cross-Attention Transformer Block with tokenization and restoration.
    
    Similar to MobileViT but uses cross-attention between two branches instead of self-attention.
    
    Flow:
    1. Input feature maps (x1, x2) -> Conv projection
    2. Unfold to tokens
    3. Cross-Attention Transformer (x1 tokens attend to x2 tokens)
    4. Fold back to feature maps
    5. Conv projection + residual
    """
    
    def __init__(
        self,
        channels: int,
        hidden_dim: int = None,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        norm: str = 'bn',
        patch_size: int = 2,
        num_transformer_layers: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim or channels
        self.patch_size = patch_size
        
        # Projection to hidden dimension
        self.proj_q = nn.Sequential(
            nn.Conv3d(channels, self.hidden_dim, kernel_size=1, bias=False),
            _make_norm3d(norm, self.hidden_dim),
        )
        self.proj_kv = nn.Sequential(
            nn.Conv3d(channels, self.hidden_dim, kernel_size=1, bias=False),
            _make_norm3d(norm, self.hidden_dim),
        )
        
        # Cross-Attention Transformer
        self.transformer = CrossAttentionTransformer3D(
            dim=self.hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )
        
        # Projection back to channels
        self.proj_out = nn.Sequential(
            nn.Conv3d(self.hidden_dim, channels, kernel_size=1, bias=False),
            _make_norm3d(norm, channels),
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: (B, C, D, H, W) - Query branch features
            x2: (B, C, D, H, W) - Key/Value branch features
        
        Returns:
            out: (B, C, D, H, W) - Fused features (same shape as x1)
        """
        residual = x1
        
        # Project to hidden dimension
        q_features = self.proj_q(x1)
        kv_features = self.proj_kv(x2)
        
        # Unfold to tokens
        q_tokens, q_info = self._unfold(q_features)
        kv_tokens, kv_info = self._unfold(kv_features)
        
        # Cross-Attention Transformer
        out_tokens = self.transformer(q_tokens, kv_tokens)
        
        # Fold back to feature maps
        out_features = self._fold(out_tokens, q_info)
        
        # Project back to channels
        out = self.proj_out(out_features)
        
        # Residual connection
        return out + residual
    
    def _unfold(self, features: torch.Tensor):
        """Convert (B, C, D, H, W) to tokens (B*patch_area, num_patches, C)."""
        B, C, D, H, W = features.shape
        p = self.patch_size
        
        pad_d = (p - D % p) % p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_d or pad_h or pad_w:
            features = F.pad(features, (0, pad_w, 0, pad_h, 0, pad_d))
            D += pad_d
            H += pad_h
            W += pad_w
        
        num_patch_d = D // p
        num_patch_h = H // p
        num_patch_w = W // p
        num_patches = num_patch_d * num_patch_h * num_patch_w
        patch_area = p ** 3
        
        patches = features.view(B, C, num_patch_d, p, num_patch_h, p, num_patch_w, p)
        patches = patches.permute(0, 3, 5, 7, 2, 4, 6, 1).contiguous()
        patches = patches.view(B * patch_area, num_patches, C)
        
        info = {
            "batch_size": B,
            "channels": C,
            "orig_size": (D - pad_d, H - pad_h, W - pad_w),
            "padded_size": (D, H, W),
            "num_patch_d": num_patch_d,
            "num_patch_h": num_patch_h,
            "num_patch_w": num_patch_w,
            "patch_size": p,
            "patch_area": patch_area,
        }
        return patches, info
    
    def _fold(self, patches: torch.Tensor, info: dict) -> torch.Tensor:
        """Inverse of _unfold."""
        B = info["batch_size"]
        C = info["channels"]
        num_patch_d = info["num_patch_d"]
        num_patch_h = info["num_patch_h"]
        num_patch_w = info["num_patch_w"]
        p = info["patch_size"]
        patch_area = info["patch_area"]
        D, H, W = info["padded_size"]
        orig_D, orig_H, orig_W = info["orig_size"]
        
        num_patches = num_patch_d * num_patch_h * num_patch_w
        patches = patches.view(B, patch_area, num_patches, C)
        patches = patches.permute(0, 3, 2, 1).contiguous()
        patches = patches.view(B, C, num_patch_d, num_patch_h, num_patch_w, p, p, p)
        patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
        features = patches.view(B, C, D, H, W)
        
        if (D, H, W) != (orig_D, orig_H, orig_W):
            features = features[:, :, :orig_D, :orig_H, :orig_W]
        return features


class BidirectionalCrossAttentionTransformer3D(nn.Module):
    """Bidirectional Cross-Attention Transformer Block.
    
    Applies cross-attention in both directions (x1->x2 and x2->x1) and fuses the results.
    This allows both branches to attend to each other with global context.
    """
    
    def __init__(
        self,
        channels: int,
        hidden_dim: int = None,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        norm: str = 'bn',
        patch_size: int = 2,
        num_transformer_layers: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn_1to2 = CrossAttentionTransformerBlock3D(
            channels=channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm=norm,
            patch_size=patch_size,
            num_transformer_layers=num_transformer_layers,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )
        self.cross_attn_2to1 = CrossAttentionTransformerBlock3D(
            channels=channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm=norm,
            patch_size=patch_size,
            num_transformer_layers=num_transformer_layers,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )
        
        # Fusion layer to combine both directions
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
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


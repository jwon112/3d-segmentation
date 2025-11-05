import torch
import torch.nn as nn
import torch.nn.functional as F

from .dualbranch_04_unet import Transition3D, RepLKBlock3D, ConvFFN3D
from .dualbranch_05_unet import Down3DStride
from .model_3d_unet import Up3D, OutConv3D, DoubleConv3D, _make_norm3d


class MobileViT3DBlock(nn.Module):
    """A lightweight 3D MobileViT-like block.

    conv3d (local) -> tokens (global) with MHSA -> MLP -> fuse back -> residual.
    """
    def __init__(self, channels: int, embed_dim: int = None, num_heads: int = 4, mlp_ratio: int = 2, norm: str = 'bn'):
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim or channels
        self.local = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm or 'bn', channels),
            nn.ReLU(inplace=True),
        )
        # projection to transformer dim
        self.proj_in = nn.Conv3d(channels, self.embed_dim, kernel_size=1, bias=True)
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * mlp_ratio), nn.GELU(), nn.Linear(self.embed_dim * mlp_ratio, self.embed_dim)
        )
        self.proj_out = nn.Conv3d(self.embed_dim, channels, kernel_size=1, bias=True)
        self.out_bn = _make_norm3d(norm or 'bn', channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # local conv
        y = self.local(x)
        b, c, d, h, w = y.shape
        # to tokens: (B, N, C)
        t = self.proj_in(y)  # (B, E, D, H, W)
        e = t.size(1)
        t = t.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, e)
        # transformer
        t = self.attn_norm(t)
        attn_out, _ = self.attn(t, t, t)
        t = t + attn_out
        t = t + self.ffn(self.attn_norm(t))
        # back to 3D
        t = t.view(b, d, h, w, e).permute(0, 4, 1, 2, 3).contiguous()
        t = self.proj_out(t)
        out = self.out_bn(t + x)
        return out


class Down3DStrideMViT(nn.Module):
    """Downsample then MobileViT3D block."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        # Use RepLKNet-style Transition for downsample for stability
        self.trans = Transition3D(in_channels, out_channels, norm=norm)
        self.mvit = MobileViT3DBlock(out_channels, embed_dim=out_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trans(x)
        x = self.mvit(x)
        return x


class DualBranchUNet3D_StrideLK_FFN2_MViT_Small(nn.Module):
    """Based on dualbranch_05 (FFN ratio=2), but use MobileViT3D at stages 4 and 5 for FLAIR branch.
    t1ce branch keeps stride conv downsamples.
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear

        # Stage 1
        self.stem_flair = DoubleConv3D(1, 16, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 16, norm=self.norm)

        # Stage 2 (RepLK + FFN2)
        self.branch_flair = nn.Sequential(
            Transition3D(16, 32, norm=self.norm),
            RepLKBlock3D(32, norm=self.norm, dw_ratio=1.5),
            ConvFFN3D(32, expansion_ratio=2, norm=self.norm),
        )
        self.branch_t1ce = Down3DStride(16, 32, norm=self.norm)

        # Stage 3 (RepLK + FFN2)
        self.branch_flair3 = nn.Sequential(
            Transition3D(32, 64, norm=self.norm),
            RepLKBlock3D(64, norm=self.norm, dw_ratio=1.5),
            ConvFFN3D(64, expansion_ratio=2, norm=self.norm),
        )
        self.branch_t1ce3 = Down3DStride(32, 64, norm=self.norm)

        # Stage 4 and 5: Single fused branch with MobileViT (no re-splitting)
        factor = 2 if self.bilinear else 1
        self.down3 = Down3DStrideMViT(64 + 64, 256, norm=self.norm, num_heads=4, mlp_ratio=2)
        self.down4 = Down3DStrideMViT(256, 512 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)

        # Decoder (standard)
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1 stems
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)

        # Stage 2
        b_flair = self.branch_flair(x1_flair)
        b_t1ce  = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)

        # Stage 3
        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce  = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)

        # Stage 4 and 5 on fused features (single branch)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        # RepLK reparam in earlier stages
        for m in [self.branch_flair[1], self.branch_flair3[1]]:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


class DualBranchUNet3D_StrideLK_FFN2_MViT_Medium(nn.Module):
    """Medium channels variant with MobileViT3D at stages 4 and 5 (based on dualbranch_05)."""
    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear

        # Stage 1
        self.stem_flair = DoubleConv3D(1, 32, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 32, norm=self.norm)

        # Stage 2/3
        self.branch_flair = nn.Sequential(
            Transition3D(32, 64, norm=self.norm),
            RepLKBlock3D(64, norm=self.norm, dw_ratio=1.5),
            ConvFFN3D(64, expansion_ratio=2, norm=self.norm),
        )
        self.branch_t1ce = Down3DStride(32, 64, norm=self.norm)
        self.branch_flair3 = nn.Sequential(
            Transition3D(64, 128, norm=self.norm),
            RepLKBlock3D(128, norm=self.norm, dw_ratio=1.5),
            ConvFFN3D(128, expansion_ratio=2, norm=self.norm),
        )
        self.branch_t1ce3 = Down3DStride(64, 128, norm=self.norm)

        # Stage 4/5 MobileViT (single fused branch)
        factor = 2 if self.bilinear else 1
        self.down3 = Down3DStrideMViT(128 + 128, 512, norm=self.norm, num_heads=4, mlp_ratio=2)
        self.down4 = Down3DStrideMViT(512, 1024 // factor, norm=self.norm, num_heads=4, mlp_ratio=2)

        # Decoder
        self.up1 = Up3D(1024, 512 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(128, 64, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)

        b_flair = self.branch_flair(x1_flair)
        b_t1ce  = self.branch_t1ce(x1_t1ce)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)

        b2_flair = self.branch_flair3(b_flair)
        b2_t1ce  = self.branch_t1ce3(b_t1ce)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)

        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        for m in [self.branch_flair[1], self.branch_flair3[1]]:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()



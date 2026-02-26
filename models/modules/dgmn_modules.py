import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm3D(nn.Module):
    """LayerNorm applied over channel dimension for 5D tensors (B, C, D, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) -> (B, D, H, W, C)
        x_perm = x.permute(0, 2, 3, 4, 1)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 4, 1, 2, 3)


class GLUBlock3D(nn.Module):
    """3D GLU-style channel gating using 1x1x1 Conv.

    Input/Output: (B, C, D, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv3d(channels, 2 * channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        a, b = torch.chunk(x_proj, 2, dim=1)
        gate = torch.sigmoid(b)
        return a * gate


class SpatialGatingBlock3D(nn.Module):
    """3D spatial gating: produces a single-channel attention map and gates the input.

    Input/Output: (B, C, D, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.conv(x))  # (B, 1, D, H, W)
        return x * attn


class MambaBlock3D(nn.Module):
    """Wrapper around Mamba-2 for 3D tensors.

    - Uses depth axis D as the sequence dimension.
    - Requires mamba-ssm to be installed when use_mamba=True.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.use_mamba = use_mamba

        if use_mamba:
            # Mamba-2 implementation from mamba-ssm
            try:
                from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "MambaBlock3D requires the 'mamba-ssm' package to be installed "
                    "when use_mamba=True. Install it with `pip install mamba-ssm`."
                ) from e

            self.mamba = Mamba2(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.fallback = None
        else:
            self.mamba = None
            # Simple depthwise-dilated conv branch for ablations (not an import fallback)
            self.fallback = nn.Sequential(
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    groups=channels,
                    bias=False,
                ),
                nn.InstanceNorm3d(channels),
                nn.GELU(),
                nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        if self.use_mamba:
            assert self.mamba is not None
            b, c, d, h, w = x.shape
            # Treat depth as sequence dimension, flatten spatial dims
            x_perm = x.permute(0, 3, 4, 2, 1)  # (B, H, W, D, C)
            x_seq = x_perm.reshape(b * h * w, d, c)  # (B*H*W, D, C)

            y_seq = self.mamba(x_seq)  # (B*H*W, D, C)

            y_perm = y_seq.reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)
            # back to (B, C, D, H, W)
            return y_perm

        assert self.fallback is not None
        return self.fallback(x)


class ParallelBranchBlock3D(nn.Module):
    """Encoder block with parallel local/global branches and dynamic gating.

    - Local branch: depthwise 3D conv
    - Global branch: MambaBlock3D (or dilated conv fallback)
    - Both branches use GLU + Spatial gating, then are fused via 1x1 Conv with residual.
    """

    def __init__(
        self,
        channels: int,
        use_mamba: bool = True,
        use_glu: bool = True,
        use_spatial_gate: bool = True,
    ):
        super().__init__()
        self.use_glu = use_glu
        self.use_spatial_gate = use_spatial_gate

        self.norm = ChannelLayerNorm3D(channels)

        # Local path
        self.local_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            stride=1,
            groups=channels,
            bias=False,
        )
        self.local_in = nn.InstanceNorm3d(channels, affine=True)

        # Global path (Mamba-2 or conv fallback)
        self.global_block = MambaBlock3D(channels, use_mamba=use_mamba)

        # Gating modules
        if use_glu:
            self.local_glu = GLUBlock3D(channels)
            self.global_glu = GLUBlock3D(channels)
        if use_spatial_gate:
            self.local_spatial = SpatialGatingBlock3D(channels)
            self.global_spatial = SpatialGatingBlock3D(channels)

        # Fusion
        self.proj = nn.Conv3d(2 * channels, channels, kernel_size=1, bias=True)

    def _apply_gates(self, x: torch.Tensor, glu: nn.Module | None, spatial: nn.Module | None) -> torch.Tensor:
        out = x
        if glu is not None:
            out = glu(out)
        if spatial is not None:
            out = spatial(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        residual = x
        x_norm = self.norm(x)

        # Local
        f_local = self.local_dw(x_norm)
        f_local = self.local_in(f_local)
        f_local = F.gelu(f_local)

        # Global
        f_global = self.global_block(x_norm)

        # Gating
        local_glu = getattr(self, "local_glu", None)
        global_glu = getattr(self, "global_glu", None)
        local_spatial = getattr(self, "local_spatial", None)
        global_spatial = getattr(self, "global_spatial", None)

        f_local = self._apply_gates(f_local, local_glu, local_spatial)
        f_global = self._apply_gates(f_global, global_glu, global_spatial)

        # Fuse
        f_concat = torch.cat([f_local, f_global], dim=1)
        f_fused = self.proj(f_concat)
        return f_fused + residual


class MultiScaleSoftmaxFusion3D(nn.Module):
    """Multi-scale feature fusion with optional softmax-based scale gating.

    - fusion_type == 'softmax_attention': softmax-generated per-scale weights (DGMN full).
    - fusion_type == 'concat_linear': simple concat + 1x1 Conv (no scale gating).

    Expects a list of features [S1..Sk] from different encoder stages.
    """

    def __init__(
        self,
        in_channels_list: list[int],
        embed_dim: int = 128,
        fusion_type: str = "softmax_attention",
    ):
        super().__init__()
        assert fusion_type in ("softmax_attention", "concat_linear")
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type

        self.proj = nn.ModuleList(
            [nn.Conv3d(c, embed_dim, kernel_size=1, bias=True) for c in in_channels_list]
        )

        k = len(in_channels_list)
        if fusion_type == "softmax_attention":
            # Produce k logits for softmax weights
            self.weight_conv = nn.Conv3d(
                embed_dim * k,
                k,
                kernel_size=1,
                bias=True,
            )
            self.linear_conv = None
        else:
            # Simple linear fusion: concat -> Conv(k*C_emb -> C_emb)
            self.weight_conv = None
            self.linear_conv = nn.Conv3d(
                embed_dim * k,
                embed_dim,
                kernel_size=1,
                bias=True,
            )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of tensors [S1..Sk], each (B, C_i, D_i, H_i, W_i)

        Returns:
            Fused feature: (B, C_emb, D_t, H_t, W_t)
        """
        assert len(features) == len(self.proj), "Mismatch between features and projections"
        # Use first feature as target resolution
        target = features[0]
        b, _, d_t, h_t, w_t = target.shape

        projected = []
        for x, conv in zip(features, self.proj):
            p = conv(x)
            if p.shape[2:] != (d_t, h_t, w_t):
                p = F.interpolate(p, size=(d_t, h_t, w_t), mode="trilinear", align_corners=False)
            projected.append(p)

        # (B, k*C_emb, D_t, H_t, W_t)
        U = torch.cat(projected, dim=1)

        if self.fusion_type == "softmax_attention":
            assert self.weight_conv is not None
            logits = self.weight_conv(U)  # (B, k, D_t, H_t, W_t)
            weights = F.softmax(logits, dim=1)

            fused = 0.0
            for i, p in enumerate(projected):
                w = weights[:, i : i + 1]  # (B, 1, D_t, H_t, W_t)
                fused = fused + p * w
            return fused

        # concat_linear: simple linear projection after concatenation
        assert self.linear_conv is not None
        fused = self.linear_conv(U)  # (B, C_emb, D_t, H_t, W_t)
        return fused


__all__ = [
    "ChannelLayerNorm3D",
    "GLUBlock3D",
    "SpatialGatingBlock3D",
    "MambaBlock3D",
    "ParallelBranchBlock3D",
    "MultiScaleSoftmaxFusion3D",
]


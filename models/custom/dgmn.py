import torch
import torch.nn as nn

from models.modules.dgmn_modules import (
    ParallelBranchBlock3D,
    MultiScaleSoftmaxFusion3D,
)


class DGMN3D(nn.Module):
    """3D Dynamic Gated Mamba Network (DGMN).

    - Encoder: 4 stages with ParallelBranchBlock3D.
    - Decoder: multi-scale softmax fusion (no UNet-style decoder).
    - Output: segmentation logits (B, n_classes, D, H, W).
    """

    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 4,
        base_channels: int = 32,
        use_mamba: bool = True,
        use_glu: bool = True,
        use_spatial: bool = True,
        fusion_type: str = "softmax_attention",
        embed_dim: int = 128,
    ):
        super().__init__()
        assert n_channels > 0
        assert n_classes > 0

        self.n_channels = n_channels
        self.n_classes = n_classes

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(n_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c1),
            nn.GELU(),
        )

        # Downsampling convs (stride=2) between stages
        self.down12 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.down23 = nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)
        self.down34 = nn.Conv3d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        # Encoder blocks per stage
        self.stage1_blocks = nn.Sequential(
            ParallelBranchBlock3D(c1, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
            ParallelBranchBlock3D(c1, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
        )
        self.stage2_blocks = nn.Sequential(
            ParallelBranchBlock3D(c2, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
            ParallelBranchBlock3D(c2, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
        )
        self.stage3_blocks = nn.Sequential(
            ParallelBranchBlock3D(c3, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
            ParallelBranchBlock3D(c3, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
            ParallelBranchBlock3D(c3, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
        )
        self.stage4_blocks = nn.Sequential(
            ParallelBranchBlock3D(c4, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
            ParallelBranchBlock3D(c4, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
            ParallelBranchBlock3D(c4, use_mamba=use_mamba, use_glu=use_glu, use_spatial_gate=use_spatial),
        )

        # Multi-scale fusion (uses S1..S4)
        self.fusion = MultiScaleSoftmaxFusion3D(
            in_channels_list=[c1, c2, c3, c4],
            embed_dim=embed_dim,
            fusion_type=fusion_type,
        )

        # Final prediction head
        self.head = nn.Conv3d(embed_dim, n_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)

        Returns:
            logits: (B, n_classes, D, H, W)
        """
        # Encoder
        x1 = self.stem(x)  # (B, c1, D, H, W)
        x1 = self.stage1_blocks(x1)  # S1

        x2 = self.down12(x1)
        x2 = self.stage2_blocks(x2)  # S2

        x3 = self.down23(x2)
        x3 = self.stage3_blocks(x3)  # S3

        x4 = self.down34(x3)
        x4 = self.stage4_blocks(x4)  # S4

        # Multi-scale fusion
        fused = self.fusion([x1, x2, x3, x4])

        # Prediction head
        logits = self.head(fused)
        return logits


__all__ = ["DGMN3D"]


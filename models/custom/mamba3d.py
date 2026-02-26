import torch
import torch.nn as nn

from models.modules.dgmn_modules import MambaBlock3D


class Mamba3D(nn.Module):
    """Simple 3D segmentation baseline using only Mamba-2 blocks.

    - No gating or multi-scale fusion.
    - Keeps spatial resolution; applies several MambaBlock3D layers and a 1x1 Conv head.
    """

    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 4,
        base_channels: int = 64,
        num_layers: int = 4,
        use_mamba: bool = True,
    ):
        super().__init__()
        assert n_channels > 0
        assert n_classes > 0

        self.n_channels = n_channels
        self.n_classes = n_classes

        c = base_channels

        # Stem projection
        self.stem = nn.Sequential(
            nn.Conv3d(n_channels, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c),
            nn.GELU(),
        )

        # Stack of MambaBlock3D layers
        self.blocks = nn.Sequential(
            *[MambaBlock3D(c, use_mamba=use_mamba) for _ in range(num_layers)]
        )

        # Prediction head
        self.head = nn.Conv3d(c, n_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)

        Returns:
            logits: (B, n_classes, D, H, W)
        """
        x = self.stem(x)
        x = self.blocks(x)
        logits = self.head(x)
        return logits


__all__ = ["Mamba3D"]


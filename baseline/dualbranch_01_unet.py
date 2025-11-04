import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse common UNet building blocks
from .model_3d_unet import DoubleConv3D, Down3D, Up3D, OutConv3D, _make_norm3d


class DualBranchUNet3D(nn.Module):
    """Dual-branch 3D U-Net (v0.1)

    - Input (B, 2, D, H, W): two modalities (FLAIR, t1ce)
    - Stage1 (stem): grouped double conv (groups=2) to keep modalities independent (64 ch)
    - Stage2: split channels into two 32-ch branches, apply symmetrical Down3D per branch to 64 ch each, then concat (128 ch)
    - Stages 3+: standard UNet encoder/decoder with skip connections
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2, "DualBranchUNet3D v0.1 expects exactly 2 input channels (FLAIR, t1ce)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear

        # Stage 1: modality-specific stems (split channels, each 1->32), then concat -> 64ch
        self.stem_flair = DoubleConv3D(1, 32, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 32, norm=self.norm)

        # Stage 2: modality-specific symmetrical branches
        # Split 64 -> (32, 32) and process separately to 64 each, concat => 128
        self.branch_flair = Down3D(32, 64, norm=self.norm)
        self.branch_t1ce = Down3D(32, 64, norm=self.norm)

        # Stages 3+: standard UNet encoder
        self.down2 = Down3D(128, 256, norm=self.norm)
        self.down3 = Down3D(256, 512, norm=self.norm)
        factor = 2 if self.bilinear else 1
        self.down4 = Down3D(512, 1024 // factor, norm=self.norm)

        # Decoder
        self.up1 = Up3D(1024, 512 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(128, 64, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1 (modality-specific stems)
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)  # (B,64,...)

        # Stage 2 branches: process each modality branch independently then fuse
        b_flair = self.branch_flair(x1_flair)  # (B,64,.../2)
        b_t1ce = self.branch_t1ce(x1_t1ce)    # (B,64,.../2)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)  # (B,128,.../2)

        # Stages 3+ encoder
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualBranchUNet3D(n_channels=2, n_classes=4).to(device)
    inp = torch.randn(1, 2, 64, 64, 64).to(device)
    with torch.no_grad():
        out = model(inp)
    print(f"Input: {tuple(inp.shape)} -> Output: {tuple(out.shape)}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse common UNet building blocks
from .model_3d_unet import DoubleConv3D, Up3D, OutConv3D, _make_norm3d


def _fold_bn_to_conv(conv: nn.Conv3d, bn: nn.Module):
    """Fold BatchNorm3d into Conv3d weights/bias and return (weight, bias).
    Assumes bn is BatchNorm3d-like with running_mean/var, weight(gamma), bias(beta).
    conv.bias can be None.
    """
    W = conv.weight  # (out_c, in_c/groups, kD, kH, kW)
    b = conv.bias if conv.bias is not None else torch.zeros(W.size(0), device=W.device)
    gamma = getattr(bn, 'weight') if hasattr(bn, 'weight') and bn.weight is not None else torch.ones(W.size(0), device=W.device)
    beta = getattr(bn, 'bias') if hasattr(bn, 'bias') and bn.bias is not None else torch.zeros(W.size(0), device=W.device)
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    W_fold = W * (gamma / std).reshape(-1, 1, 1, 1, 1)
    b_fold = (b - mean) * (gamma / std) + beta
    return W_fold, b_fold


class RepLKBlock3D(nn.Module):
    """RepLKNet-style 3D block with depthwise large kernel branch + depthwise 3x3 + depthwise 1x1 (pointwise-per-channel).

    Train mode: parallel branches with BN+ReLU, residual add, then ReLU.
    Deploy mode: fused into a single depthwise 7x7x7 Conv3d with bias.
    All branches are depthwise (groups=channels) to enable kernel summation reparam.
    """
    def __init__(self, channels: int, norm: str = 'bn'):
        super().__init__()
        self.channels = channels
        self.norm = norm or 'bn'
        self.deploy = False

        # depthwise 7x7x7
        self.dw7 = nn.Conv3d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.bn7 = _make_norm3d(self.norm, channels)

        # depthwise 3x3x3
        self.dw3 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn3 = _make_norm3d(self.norm, channels)

        # depthwise 1x1x1 (per-channel scale/shift)
        self.dw1 = nn.Conv3d(channels, channels, kernel_size=1, padding=0, groups=channels, bias=False)
        self.bn1 = _make_norm3d(self.norm, channels)

        # activation
        self.act = nn.ReLU(inplace=True)

        # placeholder for reparam conv in deploy
        self.reparam_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy and self.reparam_conv is not None:
            out = self.reparam_conv(x)
            out = self.act(out)
            return out

        y = 0
        y = y + self.act(self.bn7(self.dw7(x)))
        y = y + self.act(self.bn3(self.dw3(x)))
        y = y + self.act(self.bn1(self.dw1(x)))
        # residual
        out = self.act(x + y)
        return out

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        # Fold BN into each conv
        W7, b7 = _fold_bn_to_conv(self.dw7, self.bn7)
        W3, b3 = _fold_bn_to_conv(self.dw3, self.bn3)
        W1, b1 = _fold_bn_to_conv(self.dw1, self.bn1)

        # Pad 3x3 to 7x7 (pad last 3 dims: W, H, D)
        # F.pad format: (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
        W3_p = F.pad(W3, (2, 2, 2, 2, 2, 2))

        # Pad 1x1 to 7x7
        W1_p = F.pad(W1, (3, 3, 3, 3, 3, 3))

        # Create identity kernel for residual (7x7x7 depthwise with center=1)
        # Shape: (channels, 1, 7, 7, 7), center at (3,3,3) is 1
        identity_kernel = torch.zeros(self.channels, 1, 7, 7, 7, device=W7.device, dtype=W7.dtype)
        identity_kernel[:, :, 3, 3, 3] = 1.0

        # Sum weights/biases (all depthwise, same grouping)
        # Include residual identity in weight sum: conv(x) + x = (W + I) * x
        W_sum = W7 + W3_p + W1_p + identity_kernel
        b_sum = b7 + b3 + b1

        # Create reparam depthwise 7x7 conv with bias
        rep = nn.Conv3d(self.channels, self.channels, kernel_size=7, padding=3, groups=self.channels, bias=True)
        rep.weight.data.copy_(W_sum)
        rep.bias.data.copy_(b_sum)

        # Drop training branches
        self.reparam_conv = rep
        self.deploy = True
        del self.dw7, self.bn7, self.dw3, self.bn3, self.dw1, self.bn1
        self.dw7 = self.bn7 = self.dw3 = self.bn3 = self.dw1 = self.bn1 = None


class Down3DStride(nn.Module):
    """Downsampling with stride-2 Conv instead of MaxPool.
    Pattern: Conv(stride=2) -> Norm -> ReLU -> Conv -> Norm -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3DStrideRepLK(nn.Module):
    """Downsampling (stride-2) followed by RepLKBlock3D (DW7 + DW3 + DW1).
    Keeps channels at out_channels.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm3d(norm, out_channels),
            nn.ReLU(inplace=True),
        )
        self.replk = RepLKBlock3D(out_channels, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = self.replk(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        """Switch RepLKBlock3D to deploy mode (fuse branches)."""
        self.replk.switch_to_deploy()


class DualBranchUNet3D_StrideLK_Small(nn.Module):
    """Dual-branch 3D U-Net (v0.4 - LK conv for FLAIR) - Small channel version

    - FLAIR branch uses two depthwise 7x7x7 convs (RepLKNet-style large kernel) after stride downsampling
    - t1ce branch uses standard 3x3 convs (Down3DStride)
    - Applied at Stage 2 and Stage 3 (branching extended through Stage 3)
    - Width matches UNet3D_Small for fair comparison.
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2, "DualBranchUNet3D_StrideLK_Small expects exactly 2 input channels (FLAIR, t1ce)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear

        # Stage 1: modality-specific stems (1->16 each), then concat -> 32ch
        self.stem_flair = DoubleConv3D(1, 16, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 16, norm=self.norm)

        # Stage 2 branches: FLAIR uses RepLK, t1ce standard
        self.branch_flair = Down3DStrideRepLK(16, 32, norm=self.norm)
        self.branch_t1ce = Down3DStride(16, 32, norm=self.norm)

        # Stage 3 branches: extend same pattern (32->64 each), concat -> 128
        self.branch_flair3 = Down3DStrideRepLK(32, 64, norm=self.norm)
        self.branch_t1ce3 = Down3DStride(32, 64, norm=self.norm)

        # Stages 4+: encoder (128->256->512)
        self.down3 = Down3DStride(128, 256, norm=self.norm)
        factor = 2 if self.bilinear else 1
        self.down4 = Down3DStride(256, 512 // factor, norm=self.norm)

        # Decoder
        self.up1 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(128, 64 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(64, 32, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)  # (B,32,...)

        # Stage 2 branches
        b_flair = self.branch_flair(x1_flair)  # (B,32,.../2)
        b_t1ce  = self.branch_t1ce(x1_t1ce)   # (B,32,.../2)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)  # (B,64,.../2)

        # Stage 3 branches
        b2_flair = self.branch_flair3(b_flair)  # (B,64,.../4)
        b2_t1ce  = self.branch_t1ce3(b_t1ce)   # (B,64,.../4)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)  # (B,128,.../4)

        # Encoder
        x4 = self.down3(x3)   # 256
        x5 = self.down4(x4)   # 512

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def switch_to_deploy(self):
        """Switch all RepLK blocks in FLAIR branches to deploy mode (fuse branches)."""
        self.branch_flair.switch_to_deploy()
        self.branch_flair3.switch_to_deploy()


class DualBranchUNet3D_StrideLK_Medium(nn.Module):
    """Dual-branch 3D U-Net (v0.4 - LK conv for FLAIR) - Medium channel version

    - FLAIR branch uses two depthwise 7x7x7 convs after stride downsampling
    - t1ce branch uses standard 3x3 convs
    - Applied at Stage 2 and 3
    - Width matches UNet3D_Medium for fair comparison.
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False):
        super().__init__()
        assert n_channels == 2, "DualBranchUNet3D_StrideLK_Medium expects exactly 2 input channels (FLAIR, t1ce)."
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = (norm or 'bn')
        self.bilinear = bilinear

        # Stage 1: modality-specific stems (1->32 each), then concat -> 64ch
        self.stem_flair = DoubleConv3D(1, 32, norm=self.norm)
        self.stem_t1ce = DoubleConv3D(1, 32, norm=self.norm)

        # Stage 2 branches (32->64 each), FLAIR uses RepLK
        self.branch_flair = Down3DStrideRepLK(32, 64, norm=self.norm)
        self.branch_t1ce = Down3DStride(32, 64, norm=self.norm)

        # Stage 3 branches (64->128 each), FLAIR uses RepLK
        self.branch_flair3 = Down3DStrideRepLK(64, 128, norm=self.norm)
        self.branch_t1ce3 = Down3DStride(64, 128, norm=self.norm)

        # Stages 4+: encoder (256->512->1024)
        self.down3 = Down3DStride(256, 512, norm=self.norm)
        factor = 2 if self.bilinear else 1
        self.down4 = Down3DStride(512, 1024 // factor, norm=self.norm)

        # Decoder
        self.up1 = Up3D(1024, 512 // factor, self.bilinear, norm=self.norm)
        self.up2 = Up3D(512, 256 // factor, self.bilinear, norm=self.norm)
        self.up3 = Up3D(256, 128 // factor, self.bilinear, norm=self.norm)
        self.up4 = Up3D(128, 64, self.bilinear, norm=self.norm)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x1_flair = self.stem_flair(x[:, :1])
        x1_t1ce  = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([x1_flair, x1_t1ce], dim=1)  # (B,64,...)

        # Stage 2 branches
        b_flair = self.branch_flair(x1_flair)  # (B,64,.../2)
        b_t1ce  = self.branch_t1ce(x1_t1ce)   # (B,64,.../2)
        x2 = torch.cat([b_flair, b_t1ce], dim=1)  # (B,128,.../2)

        # Stage 3 branches
        b2_flair = self.branch_flair3(b_flair)  # (B,128,.../4)
        b2_t1ce  = self.branch_t1ce3(b_t1ce)   # (B,128,.../4)
        x3 = torch.cat([b2_flair, b2_t1ce], dim=1)  # (B,256,.../4)

        # Encoder
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def switch_to_deploy(self):
        """Switch all RepLK blocks in FLAIR branches to deploy mode (fuse branches)."""
        self.branch_flair.switch_to_deploy()
        self.branch_flair3.switch_to_deploy()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualBranchUNet3D_StrideLK_Small(n_channels=2, n_classes=4).to(device)
    inp = torch.randn(1, 2, 64, 64, 64).to(device)
    with torch.no_grad():
        out = model(inp)
    print(f"Input: {tuple(inp.shape)} -> Output: {tuple(out.shape)}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



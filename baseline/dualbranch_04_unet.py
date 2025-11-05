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


def _conv_bn_3d(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                groups: int = 1, norm: str = 'bn') -> nn.Sequential:
    seq = nn.Sequential()
    seq.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups, bias=False))
    seq.add_module('bn', _make_norm3d(norm, out_channels))
    return seq


def _conv_bn_relu_3d(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                     groups: int = 1, norm: str = 'bn') -> nn.Sequential:
    seq = _conv_bn_3d(in_channels, out_channels, kernel_size, stride, padding, groups, norm)
    seq.add_module('nonlinear', nn.ReLU(inplace=True))
    return seq


class ReparamLargeKernelConv3D(nn.Module):
    """3D depthwise large-kernel conv + optional small-kernel branch with BN, re-parameterizable to single conv.

    Follows RepLKNet's ReparamLargeKernelConv but for 3D depthwise conv.
    """
    def __init__(self, channels: int, kernel_size: int = 7, small_kernel: int = 3, norm: str = 'bn'):
        super().__init__()
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.norm = norm or 'bn'
        # Large kernel depthwise conv + BN
        self.lkb = _conv_bn_3d(channels, channels, kernel_size, 1, padding, groups=channels, norm=self.norm)
        # Optional small kernel depthwise conv + BN
        if small_kernel is not None and small_kernel > 0:
            sp = small_kernel // 2
            self.skb = _conv_bn_3d(channels, channels, small_kernel, 1, sp, groups=channels, norm=self.norm)
        else:
            self.skb = None
        # After merge
        self.lkb_reparam: nn.Conv3d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lkb_reparam is not None:
            return self.lkb_reparam(x)
        out = self.lkb(x)
        if self.skb is not None:
            out = out + self.skb(x)
        return out

    @torch.no_grad()
    def get_equivalent_kernel_bias(self):
        # fold BN for lkb
        k_l, b_l = _fold_bn_to_conv(self.lkb.conv, self.lkb.bn)
        k_eq, b_eq = k_l, b_l
        if self.skb is not None:
            k_s, b_s = _fold_bn_to_conv(self.skb.conv, self.skb.bn)
            pad = (self.kernel_size - self.small_kernel) // 2
            k_s_padded = F.pad(k_s, (pad, pad, pad, pad, pad, pad))
            k_eq = k_eq + k_s_padded
            b_eq = b_eq + b_s
        return k_eq, b_eq

    @torch.no_grad()
    def merge_kernel(self):
        if self.lkb_reparam is not None:
            return
        k_eq, b_eq = self.get_equivalent_kernel_bias()
        ch = k_eq.size(0)
        self.lkb_reparam = nn.Conv3d(ch, ch, kernel_size=self.kernel_size, stride=1,
                                      padding=self.kernel_size // 2, groups=ch, bias=True)
        self.lkb_reparam.weight.data.copy_(k_eq)
        self.lkb_reparam.bias.data.copy_(b_eq)
        # remove train branches
        del self.lkb
        if self.skb is not None:
            del self.skb
        self.lkb = None
        self.skb = None


class RepLKBlock3D(nn.Module):
    """RepLK block for 3D with Pre-BN -> PW(1x1, expand) -> DW Large-Kernel -> ReLU -> PW(1x1, project) -> Residual.

    dw_ratio follows paper defaults (~1.5). ConvFFN uses ratio 4 separately.
    """
    def __init__(self, channels: int, norm: str = 'bn', dw_ratio: float = 1.5):
        super().__init__()
        self.norm = norm or 'bn'
        self.pre_bn = _make_norm3d(self.norm, channels)
        expanded = max(1, int(round(channels * dw_ratio)))
        # pointwise expand then contract (inverted bottleneck)
        self.pw1 = _conv_bn_relu_3d(channels, expanded, kernel_size=1, stride=1, padding=0, groups=1, norm=self.norm)
        self.lk = ReparamLargeKernelConv3D(expanded, kernel_size=7, small_kernel=3, norm=self.norm)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = _conv_bn_3d(expanded, channels, kernel_size=1, stride=1, padding=0, groups=1, norm=self.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_bn(x)
        out = self.pw1(out)
        out = self.lk(out)
        out = self.act(out)
        out = self.pw2(out)
        return x + out

    @torch.no_grad()
    def switch_to_deploy(self):
        # merge only large-kernel branch; PW convs stay with fused BN handled by framework
        if hasattr(self.lk, 'merge_kernel'):
            self.lk.merge_kernel()


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


class Transition3D(nn.Module):
    """RepLKNet-style Transition for 3D: 1x1x1 PW (BN+ReLU) -> 3x3x3 DW stride=2 (BN+ReLU).
    Adjusts channels then downsamples.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.pw = _conv_bn_relu_3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, norm=norm)
        self.dw = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels, bias=False),
            _make_norm3d(norm or 'bn', out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw(x)
        x = self.dw(x)
        return x


class Down3DStrideRepLK(nn.Module):
    """Transition (PW 1x1 -> DW 3x3 stride=2) -> RepLKBlock3D -> ConvFFN3D.
    Keeps channels at out_channels.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.conv_down = Transition3D(in_channels, out_channels, norm=norm)
        self.replk = RepLKBlock3D(out_channels, norm=norm, dw_ratio=1.5)
        self.ffn = ConvFFN3D(out_channels, expansion_ratio=4, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = self.replk(x)
        x = self.ffn(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        """Switch RepLK parts to deploy mode (fuse large/small DW branches)."""
        self.replk.switch_to_deploy()


class ConvFFN3D(nn.Module):
    """Conv Feed-Forward Network for 3D: Pre-BN -> 1x1 expand -> GELU -> 1x1 project -> Residual."""
    def __init__(self, channels: int, expansion_ratio: int = 4, norm: str = 'bn'):
        super().__init__()
        self.pre_bn = _make_norm3d(norm or 'bn', channels)
        hidden = channels * expansion_ratio
        self.pw1 = _conv_bn_3d(channels, hidden, kernel_size=1, stride=1, padding=0, groups=1, norm=norm)
        self.act = nn.GELU()
        self.pw2 = _conv_bn_3d(hidden, channels, kernel_size=1, stride=1, padding=0, groups=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_bn(x)
        out = self.pw1(out)
        out = self.act(out)
        out = self.pw2(out)
        return x + out


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



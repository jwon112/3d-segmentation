"""
RepLK (Reparameterizable Large Kernel) Modules
RepLKNet 스타일의 3D Large Kernel Convolution 모듈들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_3d_unet import _make_norm3d


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
        # ensure new conv is created on the same device as folded weights
        device = k_eq.device
        self.lkb_reparam = nn.Conv3d(ch, ch, kernel_size=self.kernel_size, stride=1,
                                      padding=self.kernel_size // 2, groups=ch, bias=True).to(device)
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
        self.lk = ReparamLargeKernelConv3D(expanded, kernel_size=13, small_kernel=3, norm=self.norm)
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


class Down3DStrideRepLK_FFN2(nn.Module):
    """Transition (PW 1x1 -> DW 3x3 stride=2) -> RepLKBlock3D(dw_ratio=1.5) -> ConvFFN3D(expansion_ratio=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        self.conv_down = Transition3D(in_channels, out_channels, norm=norm)
        self.replk = RepLKBlock3D(out_channels, norm=norm, dw_ratio=1.5)
        self.ffn = ConvFFN3D(out_channels, expansion_ratio=2, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = self.replk(x)
        x = self.ffn(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        self.replk.switch_to_deploy()


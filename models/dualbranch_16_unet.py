import torch
import torch.nn as nn

from .model_3d_unet import Up3D, OutConv3D
from .dualbranch_14_unet import Stem3x3
from .modules.shufflenet_modules import Down3DShuffleNetV2
from .modules.shufflenet_hybrid_modules import Down3DShuffleNetV2Hybrid
from .channel_configs import get_dualbranch_channels_stage4_fused


class DualBranchUNet3D_ShuffleHybrid(nn.Module):
    """
    Dual-branch UNet with ShuffleNetV2 (Stage2-3) + Hybrid Transformer blocks (Stage4-5).

    - Stage 1: Stem3x3 (각 branch 독립)
    - Stage 2-3: Down3DShuffleNetV2 (dual-branch 유지)
    - Stage 4: Down3DShuffleNetV2Hybrid (dual-branch 유지)
    - Stage 5: Down3DShuffleNetV2Hybrid (single branch, Stage4 concat 이후)
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, norm: str = 'bn', bilinear: bool = False,
                 size: str = 's', hybrid_expand_ratio: float = 4.0, hybrid_num_heads: int = 4, hybrid_mlp_ratio: int = 2,
                 hybrid_patch_size: int = 4):
        super().__init__()
        assert n_channels == 2
        self.norm = norm or 'bn'
        self.bilinear = bilinear
        self.size = size

        channels = get_dualbranch_channels_stage4_fused(size)

        # Stage 1
        self.stem_flair = Stem3x3(1, channels['stem'], norm=self.norm)
        self.stem_t1ce = Stem3x3(1, channels['stem'], norm=self.norm)

        # Stage 2
        self.branch_flair2 = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)
        self.branch_t1ce2 = Down3DShuffleNetV2(channels['stem'], channels['branch2'], norm=self.norm)

        # Stage 3
        self.branch_flair3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)
        self.branch_t1ce3 = Down3DShuffleNetV2(channels['branch2'], channels['branch3'], norm=self.norm)

        # Stage 4
        self.branch_flair4 = Down3DShuffleNetV2Hybrid(
            channels['branch3'], channels['branch4'], norm=self.norm,
            expand_ratio=hybrid_expand_ratio, num_heads=hybrid_num_heads,
            mlp_ratio=hybrid_mlp_ratio, patch_size=hybrid_patch_size
        )
        self.branch_t1ce4 = Down3DShuffleNetV2Hybrid(
            channels['branch3'], channels['branch4'], norm=self.norm,
            expand_ratio=hybrid_expand_ratio, num_heads=hybrid_num_heads,
            mlp_ratio=hybrid_mlp_ratio, patch_size=hybrid_patch_size
        )

        # Stage 5 (single branch)
        fused_stage4_channels = channels['branch4'] * 2
        self.down5 = Down3DShuffleNetV2Hybrid(
            fused_stage4_channels, channels['down5'], norm=self.norm,
            expand_ratio=hybrid_expand_ratio, num_heads=hybrid_num_heads,
            mlp_ratio=hybrid_mlp_ratio, patch_size=hybrid_patch_size
        )

        # Decoder
        factor = 2 if self.bilinear else 1
        if self.bilinear:
            self.up1 = Up3D(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm,
                            skip_channels=fused_stage4_channels)
            self.up2 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm,
                            skip_channels=channels['branch3'] * 2)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm,
                            skip_channels=channels['branch2'] * 2)
            self.up4 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm,
                            skip_channels=channels['stem'] * 2)
            self.up5 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=2)
        else:
            self.up1 = Up3D(channels['down5'], channels['branch4'] // factor, self.bilinear, norm=self.norm,
                            skip_channels=fused_stage4_channels)
            self.up2 = Up3D(channels['branch4'], channels['branch3'] // factor, self.bilinear, norm=self.norm,
                            skip_channels=channels['branch3'] * 2)
            self.up3 = Up3D(channels['branch3'], channels['branch2'] // factor, self.bilinear, norm=self.norm,
                            skip_channels=channels['branch2'] * 2)
            self.up4 = Up3D(channels['branch2'], channels['stem'] // factor, self.bilinear, norm=self.norm,
                            skip_channels=channels['stem'] * 2)
            self.up5 = Up3D(channels['stem'], channels['out'], self.bilinear, norm=self.norm, skip_channels=2)
        self.outc = OutConv3D(channels['out'], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x

        s1_flair = self.stem_flair(x[:, :1])
        s1_t1ce = self.stem_t1ce(x[:, 1:2])
        x1 = torch.cat([s1_flair, s1_t1ce], dim=1)

        b2_flair = self.branch_flair2(s1_flair)
        b2_t1ce = self.branch_t1ce2(s1_t1ce)
        x2 = torch.cat([b2_flair, b2_t1ce], dim=1)

        b3_flair = self.branch_flair3(b2_flair)
        b3_t1ce = self.branch_t1ce3(b2_t1ce)
        x3 = torch.cat([b3_flair, b3_t1ce], dim=1)

        b4_flair = self.branch_flair4(b3_flair)
        b4_t1ce = self.branch_t1ce4(b3_t1ce)
        x4 = torch.cat([b4_flair, b4_t1ce], dim=1)

        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x_input[:, :2])
        return self.outc(x)


class DualBranchUNet3D_ShuffleHybrid_XS(DualBranchUNet3D_ShuffleHybrid):
    def __init__(self, **kwargs):
        super().__init__(size='xs', **kwargs)


class DualBranchUNet3D_ShuffleHybrid_Small(DualBranchUNet3D_ShuffleHybrid):
    def __init__(self, **kwargs):
        super().__init__(size='s', **kwargs)


class DualBranchUNet3D_ShuffleHybrid_Medium(DualBranchUNet3D_ShuffleHybrid):
    def __init__(self, **kwargs):
        super().__init__(size='m', **kwargs)


class DualBranchUNet3D_ShuffleHybrid_Large(DualBranchUNet3D_ShuffleHybrid):
    def __init__(self, **kwargs):
        super().__init__(size='l', **kwargs)


__all__ = [
    'DualBranchUNet3D_ShuffleHybrid',
    'DualBranchUNet3D_ShuffleHybrid_XS',
    'DualBranchUNet3D_ShuffleHybrid_Small',
    'DualBranchUNet3D_ShuffleHybrid_Medium',
    'DualBranchUNet3D_ShuffleHybrid_Large',
]


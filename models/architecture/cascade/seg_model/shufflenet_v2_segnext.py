"""
Cascade ShuffleNet V2 encoder + SegNeXt-style decoder (3D, LKA-hybrid encoder).

- Encoder:
    Stem3x3 (96^3) ->
    Down3DShuffleNetV2 (48^3) ->
    Down3DShuffleNetV2_LKAHybrid + extra LKA blocks (24^3) ->
    Stage4 Hybrid LKA (12^3)  [reused from CascadeShuffleNetV2UNet3D_LKAHybrid]

- Decoder (SegNeXt-style):
    - Trilinear upsample (x2)
    - 1x1x1 projections for high-level and skip feature
    - Add fusion (no concat)
    - Depthwise 3x3x3 + Pointwise 1x1x1 (lightweight refinement)
    - Hierarchical multi-scale fusion:
        x4(12^3) -> 24^3 (fuse x3) -> 48^3 (fuse x2) -> 96^3 (fuse x1) -> 1x1x1 seg head
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.channel_configs import get_singlebranch_channels_2step_decoder, get_activation_type
from models.model_3d_unet import _make_norm3d, _make_activation
from models.modules.shufflenet_modules import Down3DShuffleNetV2, ShuffleNetV2Unit3D_LKAHybrid, channel_shuffle_3d
from models.modules.lka_hybrid_modules import LKAHybridCBAM3D, drop_path
from models.modules.cbam_modules import ChannelAttention3D
from .shufflenet_v2 import (
    Stem3x3,
    Down3DShuffleNetV2_LKAHybrid,
    _build_shufflenet_v2_lka_extra_blocks,
    P3DStem3x3,
    P3DDown3DShuffleNetV2,
    P3DConv3d,
)


def _get_depth_config(size: str) -> dict:
    """Get depth configuration (number of blocks per stage) based on model size.
    
    Args:
        size: Model size ('xs', 's', 'm', 'l')
    
    Returns:
        Dictionary with depth config:
        - stage1_blocks: Number of blocks in Stage 1 (Stem)
        - stage2_blocks: Number of blocks in Stage 2
        - stage3_blocks: Total number of blocks in Stage 3 (including base 2 blocks)
        - stage4_blocks: Number of blocks in Stage 4
    """
    if size in ['xs', 's']:
        return {
            'stage1_blocks': 2,  # Stem: 2 blocks (base)
            'stage2_blocks': 2,  # Down3DShuffleNetV2: 2 blocks (base)
            'stage3_blocks': 4,  # Down3DShuffleNetV2_LKAHybrid (2) + extra (2)
            'stage4_blocks': 2,  # down4_lka1 (1) + down4_lka2 (1)
        }
    elif size in ['m', 'l']:
        return {
            'stage1_blocks': 3,  # Stem: 2 blocks (base) + 1 extra
            'stage2_blocks': 3,  # Down3DShuffleNetV2: 2 blocks (base) + 1 extra
            'stage3_blocks': 6,  # Down3DShuffleNetV2_LKAHybrid (2) + extra (4)
            'stage4_blocks': 3,  # down4_lka1 (1) + down4_lka2 (1) + 1 extra
        }
    else:
        raise ValueError(f"Unknown size: {size}. Must be one of ['xs', 's', 'm', 'l']")


def _build_stem_extra_blocks(
    channels: int,
    num_extra_blocks: int,
    norm: str,
    activation: str,
    size: str = "s",
) -> nn.Module:
    """Build extra blocks for Stem (Stage 1).
    
    Args:
        channels: Number of channels
        num_extra_blocks: Number of extra blocks to add (beyond base 2 blocks)
        norm: Normalization type
        activation: Activation type
        size: Model size ('xs', 's', 'm', 'l') - used to determine if drop path should be applied
    """
    if num_extra_blocks <= 0:
        return nn.Identity()
    
    # Only apply drop path for m/l models (extra blocks beyond xs/s)
    use_drop_path = size in ['m', 'l']
    
    layers = []
    for idx in range(num_extra_blocks):
        # For m/l models, apply drop path only to extra blocks (beyond xs/s baseline)
        # xs/s has 2 blocks, m/l has 3 blocks, so only the extra 1 block gets drop path
        if use_drop_path:
            # Wrap with drop path wrapper (Stem blocks don't have built-in drop path)
            # Since Stem blocks are simple conv blocks, we'll add drop path wrapper
            from models.modules.lka_hybrid_modules import drop_path as drop_path_fn
            
            class StemBlockWithDropPath(nn.Module):
                def __init__(self, block, drop_rate):
                    super().__init__()
                    self.block = block
                    self.drop_rate = drop_rate
                
                def forward(self, x):
                    if self.training and self.drop_rate > 0:
                        x = drop_path_fn(x, self.drop_rate, self.training)
                    return self.block(x)
            
            block = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
                _make_norm3d(norm, channels),
                _make_activation(activation, inplace=True),
            )
            # Apply drop path: 0.1 for the extra block
            drop_rate = 0.1
            layers.append(StemBlockWithDropPath(block, drop_rate))
        else:
            layers.append(
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
                    _make_norm3d(norm, channels),
                    _make_activation(activation, inplace=True),
                )
            )
    return nn.Sequential(*layers)


def _build_stage2_extra_blocks(
    channels: int,
    num_extra_blocks: int,
    norm: str,
    reduction: int,
    activation: str,
    size: str = "s",
) -> nn.Module:
    """Build extra blocks for Stage 2.
    
    Args:
        channels: Number of channels
        num_extra_blocks: Number of extra blocks to add (beyond base 2 blocks)
        norm: Normalization type
        reduction: Channel attention reduction ratio
        activation: Activation type
        size: Model size ('xs', 's', 'm', 'l') - used to determine if drop path should be applied
    """
    if num_extra_blocks <= 0:
        return nn.Identity()
    
    from models.modules.shufflenet_modules import ShuffleNetV2Unit3D
    from models.modules.lka_hybrid_modules import drop_path as drop_path_fn
    
    # Only apply drop path for m/l models (extra blocks beyond xs/s)
    use_drop_path = size in ['m', 'l']
    
    layers = []
    for idx in range(num_extra_blocks):
        # For m/l models, apply drop path only to extra blocks (beyond xs/s baseline)
        # xs/s has 2 blocks, m/l has 3 blocks, so only the extra 1 block gets drop path
        # ShuffleNetV2Unit3D doesn't support drop_path, so we wrap it
        block = ShuffleNetV2Unit3D(
            channels,
            channels,
            stride=1,
            norm=norm,
            use_channel_attention=True,
            reduction=reduction,
            activation=activation,
        )
        
        if use_drop_path:
            # Wrap with drop path
            class BlockWithDropPath(nn.Module):
                def __init__(self, block, drop_rate):
                    super().__init__()
                    self.block = block
                    self.drop_rate = drop_rate
                
                def forward(self, x):
                    if self.training and self.drop_rate > 0:
                        x = drop_path_fn(x, self.drop_rate, self.training)
                    return self.block(x)
            
            drop_rate = 0.1
            layers.append(BlockWithDropPath(block, drop_rate))
        else:
            layers.append(block)
    return nn.Sequential(*layers)


def _build_stage4_extra_blocks(
    channels: int,
    num_extra_blocks: int,
    norm: str,
    activation: str,
    size: str = "s",
) -> nn.Module:
    """Build extra blocks for Stage 4.
    
    Args:
        channels: Number of channels
        num_extra_blocks: Number of extra blocks to add (beyond base 2 blocks)
        norm: Normalization type
        activation: Activation type
        size: Model size ('xs', 's', 'm', 'l') - used to determine if drop path should be applied
    """
    if num_extra_blocks <= 0:
        return nn.Identity()
    
    # Only apply drop path for m/l models (extra blocks beyond xs/s)
    use_drop_path = size in ['m', 'l']
    
    layers = []
    for idx in range(num_extra_blocks):
        # For m/l models, apply drop path only to extra blocks (beyond xs/s baseline)
        # xs/s has 2 blocks, m/l has 3 blocks, so only the extra 1 block gets drop path
        drop_path_rate = 0.15 if use_drop_path else 0.0
        layers.append(
            LKAHybridCBAM3D(
                channels=channels,
                reduction=4,
                norm=norm,
                activation=activation,
                use_residual=True,
                stride=1,
                drop_path_rate=drop_path_rate,
                drop_channel_rate=0.05 if use_drop_path else 0.0,
            )
        )
    return nn.Sequential(*layers)


def _build_p3d_stem_extra_blocks(
    channels: int,
    num_extra_blocks: int,
    norm: str,
    activation: str,
    size: str = "s",
) -> nn.Module:
    """Build extra blocks for P3D Stem (Stage 1).
    
    Args:
        channels: Number of channels
        num_extra_blocks: Number of extra blocks to add (beyond base 2 blocks)
        norm: Normalization type
        activation: Activation type
        size: Model size ('xs', 's', 'm', 'l') - used to determine if drop path should be applied
    """
    if num_extra_blocks <= 0:
        return nn.Identity()
    
    # Only apply drop path for m/l models (extra blocks beyond xs/s)
    use_drop_path = size in ['m', 'l']
    
    layers = []
    for idx in range(num_extra_blocks):
        block_layers = [
            P3DConv3d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _make_norm3d(norm, channels),
            _make_activation(activation, inplace=True),
        ]
        
        if use_drop_path:
            from models.modules.lka_hybrid_modules import drop_path as drop_path_fn
            
            class P3DStemBlockWithDropPath(nn.Module):
                def __init__(self, block, drop_rate):
                    super().__init__()
                    self.block = nn.Sequential(*block)
                    self.drop_rate = drop_rate
                
                def forward(self, x):
                    if self.training and self.drop_rate > 0:
                        x = drop_path_fn(x, self.drop_rate, self.training)
                    return self.block(x)
            
            # Apply drop path: 0.1 for the extra block
            drop_rate = 0.1
            layers.append(P3DStemBlockWithDropPath(block_layers, drop_rate))
        else:
            layers.append(nn.Sequential(*block_layers))
    return nn.Sequential(*layers)


def _build_p3d_stage2_extra_blocks(
    channels: int,
    num_extra_blocks: int,
    norm: str,
    reduction: int,
    activation: str,
    size: str = "s",
) -> nn.Module:
    """Build extra blocks for P3D Stage 2.
    
    Args:
        channels: Number of channels
        num_extra_blocks: Number of extra blocks to add (beyond base 2 blocks)
        norm: Normalization type
        reduction: Channel attention reduction ratio
        activation: Activation type
        size: Model size ('xs', 's', 'm', 'l') - used to determine if drop path should be applied
    """
    if num_extra_blocks <= 0:
        return nn.Identity()
    
    from models.modules.shufflenet_modules import P3DShuffleNetV2Unit3D
    from models.modules.lka_hybrid_modules import drop_path as drop_path_fn
    
    # Only apply drop path for m/l models (extra blocks beyond xs/s)
    use_drop_path = size in ['m', 'l']
    
    layers = []
    for idx in range(num_extra_blocks):
        # For m/l models, apply drop path only to extra blocks (beyond xs/s baseline)
        # xs/s has 2 blocks, m/l has 3 blocks, so only the extra 1 block gets drop path
        # P3DShuffleNetV2Unit3D doesn't support drop_path, so we wrap it
        block = P3DShuffleNetV2Unit3D(
            channels,
            channels,
            stride=1,
            norm=norm,
            use_channel_attention=True,
            reduction=reduction,
            activation=activation,
        )
        
        if use_drop_path:
            # Wrap with drop path
            class P3DBlockWithDropPath(nn.Module):
                def __init__(self, block, drop_rate):
                    super().__init__()
                    self.block = block
                    self.drop_rate = drop_rate
                
                def forward(self, x):
                    if self.training and self.drop_rate > 0:
                        x = drop_path_fn(x, self.drop_rate, self.training)
                    return self.block(x)
            
            drop_rate = 0.1
            layers.append(P3DBlockWithDropPath(block, drop_rate))
        else:
            layers.append(block)
    return nn.Sequential(*layers)


def _build_p3d_stage4_extra_blocks(
    channels: int,
    num_extra_blocks: int,
    norm: str,
    activation: str,
    size: str = "s",
) -> nn.Module:
    """Build extra blocks for P3D Stage 4.
    
    Args:
        channels: Number of channels
        num_extra_blocks: Number of extra blocks to add (beyond base 2 blocks)
        norm: Normalization type
        activation: Activation type
        size: Model size ('xs', 's', 'm', 'l') - used to determine if drop path should be applied
    """
    if num_extra_blocks <= 0:
        return nn.Identity()
    
    # Only apply drop path for m/l models (extra blocks beyond xs/s)
    use_drop_path = size in ['m', 'l']
    
    # P3DLKAHybridCBAM3D is defined in this file, no need to import
    layers = []
    for idx in range(num_extra_blocks):
        # For m/l models, apply drop path only to extra blocks (beyond xs/s baseline)
        # xs/s has 2 blocks, m/l has 3 blocks, so only the extra 1 block gets drop path
        drop_path_rate = 0.15 if use_drop_path else 0.0
        layers.append(
            P3DLKAHybridCBAM3D(
                channels=channels,
                reduction=4,
                norm=norm,
                activation=activation,
                use_residual=True,
                stride=1,
                drop_path_rate=drop_path_rate,
                drop_channel_rate=0.05 if use_drop_path else 0.0,
            )
        )
    return nn.Sequential(*layers)


class SegNeXtDecoderBlock3D(nn.Module):
    """
    3D SegNeXt-style decoder block:

    - Upsample high-level feature by 2x (trilinear)
    - Project high-level and skip feature to the same channels with 1x1x1 conv
    - Fuse via addition (no concat)
    - Lightweight Depthwise 3x3x3 + Pointwise 1x1x1 refinement
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.norm = norm or "bn"

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        # Project high-level feature to out_channels
        self.proj_high = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, out_channels),
        )

        # Project skip feature to out_channels
        self.proj_skip = nn.Sequential(
            nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, out_channels),
        )

        # Depthwise separable refinement: DW 3x3x3 + PW 1x1x1
        self.refine = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,  # depthwise
                bias=False,
            ),
            _make_norm3d(self.norm, out_channels),
            _make_activation(activation, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_norm3d(self.norm, out_channels),
            _make_activation(activation, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample high-level feature
        x_up = self.up(x)

        # Spatial alignment with skip (padding if needed)
        diffZ = skip.size(2) - x_up.size(2)
        diffY = skip.size(3) - x_up.size(3)
        diffX = skip.size(4) - x_up.size(4)
        if diffZ != 0 or diffY != 0 or diffX != 0:
            x_up = F.pad(
                x_up,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )

        # Project to same channel dimension
        x_high = self.proj_high(x_up)
        x_skip = self.proj_skip(skip)

        # Fuse via addition
        x_fused = x_high + x_skip

        # Lightweight refinement
        out = self.refine(x_fused)
        return out


class CascadeShuffleNetV2SegNeXt3D_LKA(nn.Module):
    """
    Cascade ShuffleNet V2 encoder with LKA-hybrid (same as CascadeShuffleNetV2UNet3D_LKAHybrid)
    + SegNeXt-style lightweight decoder.

    - Encoder: 96^3 -> 48^3 -> 24^3 -> 12^3
      * Stage2: Down3DShuffleNetV2
      * Stage3: Down3DShuffleNetV2_LKAHybrid + extra LKA units
      * Stage4: LKAHybridCBAM3D (stride=2) + LKAHybridCBAM3D (stride=1)

    - Decoder: hierarchical upsample + add-fusion + DW/PW conv refinement
      * 12^3 -> 24^3 (fuse Stage3)
      * 24^3 -> 48^3 (fuse Stage2)
      * 48^3 -> 96^3 (fuse Stem)
    """

    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        norm: str = "bn",
        size: str = "s",
        include_coords: bool = True,
    ) -> None:
        super().__init__()
        self.n_image_channels = n_image_channels
        self.n_coord_channels = n_coord_channels
        self.include_coords = include_coords and n_coord_channels > 0
        self.norm = norm or "bn"
        self.size = size

        activation = get_activation_type(size)
        channels = get_singlebranch_channels_2step_decoder(size)
        
        # Get depth configuration based on model size
        depth_config = _get_depth_config(size)

        stem_in = n_image_channels + (n_coord_channels if self.include_coords else 0)
        self.stem = Stem3x3(stem_in, channels["stem"], norm=self.norm, activation=activation)
        # Stage 1 extra blocks (if needed)
        stage1_extra = depth_config['stage1_blocks'] - 2  # Base stem has 2 blocks
        self.stem_extra = _build_stem_extra_blocks(
            channels["stem"],
            stage1_extra,
            self.norm,
            activation,
            size=size,  # Pass size for drop path control
        )

        # Stage 2: ShuffleNetV2 Down block (48^3)
        self.down2 = Down3DShuffleNetV2(
            channels["stem"],
            channels["branch2"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        # Stage 2 extra blocks (if needed)
        stage2_extra = depth_config['stage2_blocks'] - 2  # Base down2 has 2 blocks
        self.down2_extra = _build_stage2_extra_blocks(
            channels["branch2"],
            stage2_extra,
            self.norm,
            reduction=8,
            activation=activation,
            size=size,  # Pass size for drop path control
        )

        # Stage 3: Hybrid LKA Down block (24^3) + extra LKA units
        self.down3 = Down3DShuffleNetV2_LKAHybrid(
            channels["branch2"],
            channels["branch3"],
            norm=self.norm,
            reduction=4,
            activation=activation,
        )
        # down3에 이미 2개 블록이 있으므로 extra의 첫 블록부터 drop path 적용
        stage3_extra = depth_config['stage3_blocks'] - 2  # Base down3 has 2 blocks
        # Generate drop path rates for extra blocks
        # xs/s: 2 extra blocks (total 4), m/l: 4 extra blocks (total 6)
        # Only apply drop path to blocks beyond xs/s baseline (m/l의 마지막 2개만)
        if size in ['xs', 's']:
            # xs/s: no drop path for extra blocks
            drop_path_rates = [0.0] * stage3_extra if stage3_extra > 0 else []
            drop_channel_rates = [0.0] * stage3_extra if stage3_extra > 0 else []
        else:  # m/l
            # m/l: only last 2 extra blocks get drop path (beyond xs/s baseline of 4 total blocks)
            # xs/s has 4 total (2 base + 2 extra), m/l has 6 total (2 base + 4 extra)
            # So only the last 2 extra blocks in m/l get drop path
            drop_path_rates = [0.0] * (stage3_extra - 2) + [0.1, 0.15] if stage3_extra >= 2 else [0.0] * stage3_extra
            drop_channel_rates = [0.0] * (stage3_extra - 2) + [0.05, 0.05] if stage3_extra >= 2 else [0.0] * stage3_extra
        self.down3_extra = _build_shufflenet_v2_lka_extra_blocks(
            channels["branch3"],
            stage3_extra,
            self.norm,
            reduction=4,
            activation=activation,
            # down3의 2개 블록 이후이므로 첫 블록부터 drop path 적용
            drop_path_rates=drop_path_rates,
            # Spatial dropout (width 앙상블) - Stage 3에 적용
            drop_channel_rates=drop_channel_rates,
        )

        fused_channels = channels["branch3"]
        expanded_channels = channels["down4"]

        # Stage 4: 24^3 -> 12^3 via LKA stride=2, then one more LKA (stride=1)
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # lka1은 항상 실행되어 Stage 4가 완전히 사라지지 않도록 보장
        self.down4_lka1 = LKAHybridCBAM3D(
            channels=expanded_channels,
            reduction=4,
            norm=self.norm,
            activation=activation,
            use_residual=True,
            stride=2,
            drop_path_rate=0.0,  # 첫 번째 블록은 항상 실행
            drop_channel_rate=0.0,  # 첫 번째 블록은 spatial dropout 미적용
        )
        # lka2만 drop path 및 spatial dropout 적용하여 regularization 효과
        self.down4_lka2 = LKAHybridCBAM3D(
            channels=expanded_channels,
            reduction=4,
            norm=self.norm,
            activation=activation,
            use_residual=True,
            stride=1,
            drop_path_rate=0.15,  # 두 번째 블록에만 drop path 적용
            drop_channel_rate=0.05,  # Bottleneck이므로 낮은 spatial dropout
        )
        # Stage 4 extra blocks (if needed)
        stage4_extra = depth_config['stage4_blocks'] - 2  # Base stage4 has 2 blocks (lka1 + lka2)
        self.down4_extra = _build_stage4_extra_blocks(
            expanded_channels,
            stage4_extra,
            self.norm,
            activation,
            size=size,  # Pass size for drop path control
        )

        # SegNeXt-style decoder
        # Level 3: 12^3 -> 24^3, fuse Stage3 (branch3)
        self.dec3 = SegNeXtDecoderBlock3D(
            in_channels=channels["down4"],
            skip_channels=channels["branch3"],
            out_channels=channels["up1"],
            norm=self.norm,
            activation=activation,
        )
        # Level 2: 24^3 -> 48^3, fuse Stage2 (branch2)
        self.dec2 = SegNeXtDecoderBlock3D(
            in_channels=channels["up1"],
            skip_channels=channels["branch2"],
            out_channels=channels["up2"],
            norm=self.norm,
            activation=activation,
        )
        # Level 1: 48^3 -> 96^3, fuse Stem
        self.dec1 = SegNeXtDecoderBlock3D(
            in_channels=channels["up2"],
            skip_channels=channels["stem"],
            out_channels=channels["out"],
            norm=self.norm,
            activation=activation,
        )

        # Final segmentation head
        self.outc = nn.Conv3d(channels["out"], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input split: image + coords
        if self.include_coords and self.n_coord_channels > 0:
            img = x[:, : self.n_image_channels]
            coord = x[:, self.n_image_channels : self.n_image_channels + self.n_coord_channels]
            x_in = torch.cat([img, coord], dim=1)
        else:
            x_in = x[:, : self.n_image_channels]

        # Encoder
        x1 = self.stem(x_in)        # 96^3
        x1 = self.stem_extra(x1)    # 96^3 (extra blocks if any)
        x2 = self.down2(x1)         # 48^3
        x2 = self.down2_extra(x2)   # 48^3 (extra blocks if any)
        x3 = self.down3(x2)         # 24^3
        x3 = self.down3_extra(x3)   # 24^3 (refined)

        x4 = self.down4_expand(x3)  # 24^3
        x4 = self.down4_lka1(x4)    # 12^3
        x4 = self.down4_lka2(x4)    # 12^3
        x4 = self.down4_extra(x4)   # 12^3 (extra blocks if any)

        # SegNeXt-style decoder
        d3 = self.dec3(x4, x3)      # 12^3 -> 24^3
        d2 = self.dec2(d3, x2)      # 24^3 -> 48^3
        d1 = self.dec1(d2, x1)      # 48^3 -> 96^3

        out = self.outc(d1)
        return out


def build_cascade_shufflenet_v2_segnext_lka(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2SegNeXt3D_LKA:
    return CascadeShuffleNetV2SegNeXt3D_LKA(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
    )


# ============================================================================
# P3D (Pseudo-3D) Variants
# ============================================================================


class P3DSegNeXtDecoderBlock3D(nn.Module):
    """
    3D SegNeXt-style decoder block with P3D:

    - Upsample high-level feature by 2x (trilinear)
    - Project high-level and skip feature to the same channels with 1x1x1 conv
    - Fuse via addition (no concat)
    - P3D Depthwise 3x3x3 + Pointwise 1x1x1 refinement
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.norm = norm or "bn"

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        # Project high-level feature to out_channels
        self.proj_high = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, out_channels),
        )

        # Project skip feature to out_channels
        self.proj_skip = nn.Sequential(
            nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, out_channels),
        )

        # P3D Depthwise separable refinement: P3D DW 3x3x3 + PW 1x1x1
        self.refine = nn.Sequential(
            P3DConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,  # depthwise
                bias=False,
            ),
            _make_norm3d(self.norm, out_channels),
            _make_activation(activation, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_norm3d(self.norm, out_channels),
            _make_activation(activation, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample high-level feature
        x_up = self.up(x)

        # Spatial alignment with skip (padding if needed)
        diffZ = skip.size(2) - x_up.size(2)
        diffY = skip.size(3) - x_up.size(3)
        diffX = skip.size(4) - x_up.size(4)
        if diffZ != 0 or diffY != 0 or diffX != 0:
            x_up = F.pad(
                x_up,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )

        # Project to same channel dimension
        x_high = self.proj_high(x_up)
        x_skip = self.proj_skip(skip)

        # Fuse via addition
        x_fused = x_high + x_skip

        # Lightweight refinement
        out = self.refine(x_fused)
        return out


class P3DLKAKernel3D(nn.Module):
    """
    P3D LKA-style kernel (dense + sparse + mixer), attention 없이 순수 커널만 구현.

    - Dense:  P3D 3x3x3 depthwise conv (2D spatial + 1D depth)
    - Sparse: P3D 3x3x3 depthwise conv with dilation=3 (2D spatial dilated + 1D depth dilated)
    - Mixer:  1x1x1 pointwise conv (채널 혼합)

    P3D로 분해하여 연산량을 절감하면서도 ERF는 7x7x7을 유지합니다.
    Depth 방향은 dense하게 커버하여 더 많은 정보를 활용합니다.
    """

    def __init__(
        self,
        channels: int,
        norm: str = "bn",
        activation: str = "relu",
        stride_dense: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.norm = norm or "bn"

        # Dense branch: P3D 3x3x3 depthwise conv
        # 2D spatial: 3x3 conv
        self.conv_dense_2d = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride_dense,
            padding=1,
            groups=channels,  # depthwise
            bias=False,
        )
        # 1D depth: 3 conv
        self.conv_dense_1d = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=stride_dense,
            padding=1,
            groups=channels,  # depthwise
            bias=False,
        )
        self.dense_bn = _make_norm3d(self.norm, channels)
        self.dense_act = _make_activation(activation, inplace=True)

        # Sparse branch: P3D 3x3x3 depthwise conv with dilation=3
        # 2D spatial: 3x3 conv with dilation=3
        self.conv_sparse_2d = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            groups=channels,  # depthwise
            bias=False,
        )
        # 1D depth: 3 conv with dilation=3
        self.conv_sparse_1d = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            groups=channels,  # depthwise
            bias=False,
        )
        self.sparse_bn = _make_norm3d(self.norm, channels)
        self.sparse_act = _make_activation(activation, inplace=True)

        # Mixer: 1x1x1 pointwise conv (채널 혼합)
        self.mixer = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            _make_norm3d(self.norm, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)

        Returns:
            P3D LKA-style 커널이 적용된 출력 텐서 (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        # Dense branch: P3D
        # 2D spatial conv
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        x_2d = self.conv_dense_2d(x_2d)  # (B*D, C, H', W')
        _, _, H_out, W_out = x_2d.shape
        x_3d = x_2d.view(B, D, C, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()
        # 1D depth conv
        x_1d = x_3d.permute(0, 3, 4, 1, 2).contiguous().view(B * H_out * W_out, C, D)
        x_1d = self.conv_dense_1d(x_1d)  # (B*H*W, C, D')
        _, _, D_out = x_1d.shape
        out = x_1d.view(B, H_out, W_out, C, D_out).permute(0, 3, 4, 1, 2).contiguous()
        out = self.dense_bn(out)
        out = self.dense_act(out)

        # Sparse branch: P3D with dilation
        # 2D spatial conv with dilation
        x_2d = out.permute(0, 2, 1, 3, 4).contiguous().view(B * D_out, C, H_out, W_out)
        x_2d = self.conv_sparse_2d(x_2d)  # (B*D, C, H'', W'')
        _, _, H_out2, W_out2 = x_2d.shape
        x_3d = x_2d.view(B, D_out, C, H_out2, W_out2).permute(0, 2, 1, 3, 4).contiguous()
        # 1D depth conv with dilation
        x_1d = x_3d.permute(0, 3, 4, 1, 2).contiguous().view(B * H_out2 * W_out2, C, D_out)
        x_1d = self.conv_sparse_1d(x_1d)  # (B*H*W, C, D'')
        _, _, D_out2 = x_1d.shape
        out = x_1d.view(B, H_out2, W_out2, C, D_out2).permute(0, 3, 4, 1, 2).contiguous()
        out = self.sparse_bn(out)
        out = self.sparse_act(out)

        # Mixer
        out = self.mixer(out)
        return out


class P3DLKAHybridCBAM3D(nn.Module):
    """
    P3D LKA Hybrid Block with CBAM Channel Attention.

    - P3D LKA 커널 (dense + sparse + mixer)을 통해 7x7x7 ERF를 갖는 특징을 추출
    - 그 결과에 대해 CBAM의 ChannelAttention3D를 적용하여 채널별 중요도를 재조정
    - 선택적으로 residual connection(x + F(x))을 적용 가능

    Args:
        channels: 입력/출력 채널 수
        reduction: CBAM 채널 축소 비율
        norm: 정규화 타입 ('bn', 'in', 'gn')
        activation: 활성화 함수 타입 ('relu', 'hardswish', 'gelu')
        use_residual: 입력과 출력 사이에 residual connection을 사용할지 여부
        drop_path_rate: Stochastic depth (depth 앙상블) 비율
        drop_channel_rate: Spatial dropout (width 앙상블) 비율
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        norm: str = "bn",
        activation: str = "relu",
        use_residual: bool = True,
        stride: int = 1,
        drop_path_rate: float = 0.0,
        drop_channel_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.stride = stride

        self.lka_kernel = P3DLKAKernel3D(
            channels=channels,
            norm=norm,
            activation=activation,
            stride_dense=stride,
        )
        self.channel_attention = ChannelAttention3D(
            channels=channels,
            reduction=reduction,
        )

        # Stochastic depth 비율 (depth 앙상블, 0이면 사용 안 함)
        self.drop_path_rate = float(drop_path_rate)

        # Spatial dropout (width 앙상블, CNN에 적합한 채널 단위 dropout)
        if drop_channel_rate > 0:
            self.drop_channel = nn.Dropout3d(drop_channel_rate)
        else:
            self.drop_channel = nn.Identity()

        # stride > 1인 경우 residual connection을 유지하기 위한 projection 경로
        if self.use_residual and self.stride > 1:
            self.proj = nn.Sequential(
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                _make_norm3d(norm, channels),
            )
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)

        Returns:
            P3D LKA 커널 + CBAM Channel Attention (+ optional residual)이 적용된 출력
        """
        out = self.lka_kernel(x)  # P3D 커널 구조로 특징 추출 (7x7x7 ERF + stride에 따른 downsample)
        out = self.channel_attention(out)  # 채널 어텐션 적용

        # Spatial dropout (width 앙상블) - CBAM 이후 적용
        out = self.drop_channel(out)

        # Projection 있는 정석 residual:
        # - stride == 1: out + x
        # - stride > 1: out + proj(x)  (spatial 크기를 맞춘 identity)
        if self.use_residual:
            identity = x
            if self.proj is not None:
                identity = self.proj(identity)
            # Stochastic depth (DropPath)를 residual branch에만 적용 (depth 앙상블)
            out = identity + drop_path(out, self.drop_path_rate, self.training)
        return out


class P3DShuffleNetV2Unit3D_LKAHybrid(nn.Module):
    """3D ShuffleNetV2 Unit with Hybrid LKA branch (P3D version).

    전체적으로 P3D를 일관되게 적용하여 효율성과 일관성을 확보합니다.

    - Stride=1:
        Split -> Branch1 (identity) +
        Branch2 (1x1 -> P3DLKAHybridCBAM3D -> 출력) -> Concat -> Shuffle
    - Stride=2:
        No split ->
        Branch1 (P3D DWConv stride=2 -> 1x1) +
        Branch2 (1x1 -> P3DLKAHybridCBAM3D stride=2 -> 출력 채널) -> Concat -> Shuffle

    모든 conv 연산에 P3D를 적용하여 메모리와 연산량을 절감합니다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "bn",
        reduction: int = 16,
        activation: str = "relu",
        drop_path: float = 0.0,
        drop_channel: float = 0.0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.drop_path = float(drop_path)
        self.drop_channel = float(drop_channel)

        activation_fn = _make_activation(activation, inplace=True)

        if stride == 1:
            # Stride=1: Split operation
            assert in_channels == out_channels, "For stride=1, in_channels must equal out_channels"
            mid_channels = out_channels // 2

            # Branch 1: Identity (no operation)
            self.branch1 = nn.Identity()

            # Branch 2: 1x1 -> Hybrid LKA
            self.branch2_conv1 = nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.branch2_bn1 = _make_norm3d(norm, mid_channels)
            self.branch2_activation = activation_fn

            # Hybrid LKA block: P3D 적용
            self.branch2_lka = P3DLKAHybridCBAM3D(
                channels=mid_channels,
                reduction=reduction,
                norm=norm,
                activation=activation,
                use_residual=True,
                stride=1,
                drop_path_rate=self.drop_path,
                drop_channel_rate=self.drop_channel,
            )
        else:
            # Stride=2: No split, both branches process full input
            mid_out = out_channels // 2

            # Branch 1: P3D DWConv stride=2 -> 1x1
            self.branch1_conv1 = P3DConv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            )
            self.branch1_bn1 = _make_norm3d(norm, in_channels)
            self.branch1_conv2 = nn.Conv3d(
                in_channels,
                mid_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.branch1_bn2 = _make_norm3d(norm, mid_out)
            self.branch1_activation = activation_fn

            # Branch 2: 1x1 -> Hybrid LKA (stride=2) for downsampling
            self.branch2_conv1 = nn.Conv3d(
                in_channels,
                mid_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.branch2_bn1 = _make_norm3d(norm, mid_out)
            self.branch2_activation = activation_fn

            # Hybrid LKA block (stride=2): P3D 적용
            self.branch2_lka = P3DLKAHybridCBAM3D(
                channels=mid_out,
                reduction=reduction,
                norm=norm,
                activation=activation,
                use_residual=True,
                stride=2,
                drop_path_rate=self.drop_path,
                drop_channel_rate=self.drop_channel,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            # Split channels
            x1, x2 = x.chunk(2, dim=1)
            # Branch 1: Identity
            out1 = self.branch1(x1)
            # Branch 2: Processing
            out2 = self.branch2_conv1(x2)
            out2 = self.branch2_bn1(out2)
            out2 = self.branch2_activation(out2)
            out2 = self.branch2_lka(out2)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        else:
            # Both branches process full input
            # Branch 1
            out1 = self.branch1_conv1(x)
            out1 = self.branch1_bn1(out1)
            out1 = self.branch1_conv2(out1)
            out1 = self.branch1_bn2(out1)
            out1 = self.branch1_activation(out1)
            # Branch 2
            out2 = self.branch2_conv1(x)
            out2 = self.branch2_bn1(out2)
            out2 = self.branch2_activation(out2)
            out2 = self.branch2_lka(out2)
            # Concat
            out = torch.cat([out1, out2], dim=1)

        # Channel Shuffle
        out = channel_shuffle_3d(out, groups=2)
        return out


class P3DDown3DShuffleNetV2_LKAHybrid(nn.Module):
    """Downsampling using P3DShuffleNetV2Unit3D_LKAHybrid (stride=2 then stride=1).

    - unit1: stride=2 (해상도 절반으로 감소)
    - unit2: stride=1 (해상도 유지, 특징 정제)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "bn",
        reduction: int = 16,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = P3DShuffleNetV2Unit3D_LKAHybrid(
            in_channels,
            out_channels,
            stride=2,
            norm=norm,
            reduction=reduction,
            activation=activation,
        )
        # Second unit: stride=1 for feature refinement
        self.unit2 = P3DShuffleNetV2Unit3D_LKAHybrid(
            out_channels,
            out_channels,
            stride=1,
            norm=norm,
            reduction=reduction,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


def _build_p3d_shufflenet_v2_lka_extra_blocks(
    channels: int,
    num_blocks: int,
    norm: str,
    reduction: int,
    activation: str,
    drop_path_rates: list[float] | None = None,
    drop_channel_rates: list[float] | None = None,
) -> nn.Module:
    """P3D ShuffleNetV2 + Hybrid LKA extra blocks (stride=1).

    Stage3/Stage4에서 해상도는 유지하고, 채널만 유지한 채로
    P3DShuffleNetV2Unit3D_LKAHybrid 블록을 반복적으로 적용합니다.
    """
    if num_blocks <= 0:
        return nn.Identity()

    layers: list[nn.Module] = []
    for idx in range(num_blocks):
        drop_rate = 0.0
        if drop_path_rates is not None and idx < len(drop_path_rates):
            drop_rate = float(drop_path_rates[idx])

        drop_channel_rate = 0.0
        if drop_channel_rates is not None and idx < len(drop_channel_rates):
            drop_channel_rate = float(drop_channel_rates[idx])

        layers.append(
            P3DShuffleNetV2Unit3D_LKAHybrid(
                channels,
                channels,
                stride=1,
                norm=norm,
                reduction=reduction,
                activation=activation,
                drop_path=drop_rate,
                drop_channel=drop_channel_rate,
            )
        )
    return nn.Sequential(*layers)


class CascadeShuffleNetV2SegNeXt3D_P3D_LKA(nn.Module):
    """
    Cascade ShuffleNet V2 encoder with P3D + LKA-hybrid
    + SegNeXt-style lightweight decoder with P3D.

    - Encoder: 96^3 -> 48^3 -> 24^3 -> 12^3
      * Stem: P3DStem3x3
      * Stage2: P3DDown3DShuffleNetV2
      * Stage3: P3DDown3DShuffleNetV2_LKAHybrid + extra P3D LKA units
      * Stage4: P3DLKAHybridCBAM3D (stride=2) + P3DLKAHybridCBAM3D (stride=1) [전체 P3D 적용]

    - Decoder: hierarchical upsample + add-fusion + P3D DW/PW conv refinement
      * 12^3 -> 24^3 (fuse Stage3)
      * 24^3 -> 48^3 (fuse Stage2)
      * 48^3 -> 96^3 (fuse Stem)
    """

    def __init__(
        self,
        n_image_channels: int = 4,
        n_coord_channels: int = 3,
        n_classes: int = 4,
        norm: str = "bn",
        size: str = "s",
        include_coords: bool = True,
    ) -> None:
        super().__init__()
        self.n_image_channels = n_image_channels
        self.n_coord_channels = n_coord_channels
        self.include_coords = include_coords and n_coord_channels > 0
        self.norm = norm or "bn"
        self.size = size

        activation = get_activation_type(size)
        channels = get_singlebranch_channels_2step_decoder(size)
        
        # Get depth configuration based on model size
        depth_config = _get_depth_config(size)

        stem_in = n_image_channels + (n_coord_channels if self.include_coords else 0)
        self.stem = P3DStem3x3(stem_in, channels["stem"], norm=self.norm, activation=activation)
        # Stage 1 extra blocks (if needed)
        stage1_extra = depth_config['stage1_blocks'] - 2  # Base stem has 2 blocks
        self.stem_extra = _build_p3d_stem_extra_blocks(
            channels["stem"],
            stage1_extra,
            self.norm,
            activation,
            size=size,  # Pass size for drop path control
        )

        # Stage 2: P3D ShuffleNetV2 Down block (48^3)
        self.down2 = P3DDown3DShuffleNetV2(
            channels["stem"],
            channels["branch2"],
            norm=self.norm,
            use_channel_attention=True,
            reduction=8,
            activation=activation,
        )
        # Stage 2 extra blocks (if needed)
        stage2_extra = depth_config['stage2_blocks'] - 2  # Base down2 has 2 blocks
        self.down2_extra = _build_p3d_stage2_extra_blocks(
            channels["branch2"],
            stage2_extra,
            self.norm,
            reduction=8,
            activation=activation,
            size=size,  # Pass size for drop path control
        )

        # Stage 3: P3D Hybrid LKA Down block (24^3) + extra P3D LKA units
        self.down3 = P3DDown3DShuffleNetV2_LKAHybrid(
            channels["branch2"],
            channels["branch3"],
            norm=self.norm,
            reduction=4,
            activation=activation,
        )
        # down3에 이미 2개 블록이 있으므로 extra의 첫 블록부터 drop path 적용
        stage3_extra = depth_config['stage3_blocks'] - 2  # Base down3 has 2 blocks
        # Generate drop path rates for extra blocks
        # xs/s: 2 extra blocks (total 4), m/l: 4 extra blocks (total 6)
        # Only apply drop path to blocks beyond xs/s baseline (m/l의 마지막 2개만)
        if size in ['xs', 's']:
            # xs/s: no drop path for extra blocks
            drop_path_rates = [0.0] * stage3_extra if stage3_extra > 0 else []
            drop_channel_rates = [0.0] * stage3_extra if stage3_extra > 0 else []
        else:  # m/l
            # m/l: only last 2 extra blocks get drop path (beyond xs/s baseline of 4 total blocks)
            # xs/s has 4 total (2 base + 2 extra), m/l has 6 total (2 base + 4 extra)
            # So only the last 2 extra blocks in m/l get drop path
            drop_path_rates = [0.0] * (stage3_extra - 2) + [0.1, 0.15] if stage3_extra >= 2 else [0.0] * stage3_extra
            drop_channel_rates = [0.0] * (stage3_extra - 2) + [0.05, 0.05] if stage3_extra >= 2 else [0.0] * stage3_extra
        self.down3_extra = _build_p3d_shufflenet_v2_lka_extra_blocks(
            channels["branch3"],
            stage3_extra,
            self.norm,
            reduction=4,
            activation=activation,
            # down3의 2개 블록 이후이므로 첫 블록부터 drop path 적용
            drop_path_rates=drop_path_rates,
            # Spatial dropout (width 앙상블) - Stage 3에 적용
            drop_channel_rates=drop_channel_rates,
        )

        fused_channels = channels["branch3"]
        expanded_channels = channels["down4"]

        # Stage 4: 24^3 -> 12^3 via P3D LKA stride=2, then one more P3D LKA (stride=1)
        # 전체 P3D 적용으로 일관성 확보
        self.down4_expand = nn.Sequential(
            nn.Conv3d(fused_channels, expanded_channels, kernel_size=1, bias=False),
            _make_norm3d(self.norm, expanded_channels),
            _make_activation(activation, inplace=True),
        )
        # lka1은 항상 실행되어 Stage 4가 완전히 사라지지 않도록 보장
        self.down4_lka1 = P3DLKAHybridCBAM3D(
            channels=expanded_channels,
            reduction=4,
            norm=self.norm,
            activation=activation,
            use_residual=True,
            stride=2,
            drop_path_rate=0.0,  # 첫 번째 블록은 항상 실행
            drop_channel_rate=0.0,  # 첫 번째 블록은 spatial dropout 미적용
        )
        # lka2만 drop path 및 spatial dropout 적용하여 regularization 효과
        self.down4_lka2 = P3DLKAHybridCBAM3D(
            channels=expanded_channels,
            reduction=4,
            norm=self.norm,
            activation=activation,
            use_residual=True,
            stride=1,
            drop_path_rate=0.15,  # 두 번째 블록에만 drop path 적용
            drop_channel_rate=0.05,  # Bottleneck이므로 낮은 spatial dropout
        )
        # Stage 4 extra blocks (if needed)
        stage4_extra = depth_config['stage4_blocks'] - 2  # Base stage4 has 2 blocks (lka1 + lka2)
        self.down4_extra = _build_p3d_stage4_extra_blocks(
            expanded_channels,
            stage4_extra,
            self.norm,
            activation,
            size=size,  # Pass size for drop path control
        )

        # P3D SegNeXt-style decoder
        # Level 3: 12^3 -> 24^3, fuse Stage3 (branch3)
        self.dec3 = P3DSegNeXtDecoderBlock3D(
            in_channels=channels["down4"],
            skip_channels=channels["branch3"],
            out_channels=channels["up1"],
            norm=self.norm,
            activation=activation,
        )
        # Level 2: 24^3 -> 48^3, fuse Stage2 (branch2)
        self.dec2 = P3DSegNeXtDecoderBlock3D(
            in_channels=channels["up1"],
            skip_channels=channels["branch2"],
            out_channels=channels["up2"],
            norm=self.norm,
            activation=activation,
        )
        # Level 1: 48^3 -> 96^3, fuse Stem
        self.dec1 = P3DSegNeXtDecoderBlock3D(
            in_channels=channels["up2"],
            skip_channels=channels["stem"],
            out_channels=channels["out"],
            norm=self.norm,
            activation=activation,
        )

        # Final segmentation head
        self.outc = nn.Conv3d(channels["out"], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input split: image + coords
        if self.include_coords and self.n_coord_channels > 0:
            img = x[:, : self.n_image_channels]
            coord = x[:, self.n_image_channels : self.n_image_channels + self.n_coord_channels]
            x_in = torch.cat([img, coord], dim=1)
        else:
            x_in = x[:, : self.n_image_channels]

        # Encoder
        x1 = self.stem(x_in)        # 96^3
        x1 = self.stem_extra(x1)    # 96^3 (extra blocks if any)
        x2 = self.down2(x1)         # 48^3
        x2 = self.down2_extra(x2)   # 48^3 (extra blocks if any)
        x3 = self.down3(x2)         # 24^3
        x3 = self.down3_extra(x3)   # 24^3 (refined)

        x4 = self.down4_expand(x3)  # 24^3
        x4 = self.down4_lka1(x4)    # 12^3
        x4 = self.down4_lka2(x4)    # 12^3
        x4 = self.down4_extra(x4)   # 12^3 (extra blocks if any)

        # SegNeXt-style decoder
        d3 = self.dec3(x4, x3)      # 12^3 -> 24^3
        d2 = self.dec2(d3, x2)      # 24^3 -> 48^3
        d1 = self.dec1(d2, x1)      # 48^3 -> 96^3

        out = self.outc(d1)
        return out


def build_cascade_shufflenet_v2_segnext_p3d_lka(
    *,
    n_image_channels: int = 4,
    n_coord_channels: int = 3,
    n_classes: int = 4,
    norm: str = "bn",
    size: str = "s",
    include_coords: bool = True,
) -> CascadeShuffleNetV2SegNeXt3D_P3D_LKA:
    return CascadeShuffleNetV2SegNeXt3D_P3D_LKA(
        n_image_channels=n_image_channels,
        n_coord_channels=n_coord_channels,
        n_classes=n_classes,
        norm=norm,
        size=size,
        include_coords=include_coords,
    )


__all__ = [
    "CascadeShuffleNetV2SegNeXt3D_LKA",
    "build_cascade_shufflenet_v2_segnext_lka",
    "CascadeShuffleNetV2SegNeXt3D_P3D_LKA",
    "build_cascade_shufflenet_v2_segnext_p3d_lka",
]



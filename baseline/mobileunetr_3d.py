# MobileUNETR 3D Architecture
# 3D version of MobileUNETR for medical volume segmentation
# Architecture: 3D MobileViT-based encoder with 3D decoder

import math
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


##############################################################################
# 3D Conv and Basic Components
##############################################################################
def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU(),
    )


def conv_3x3x3_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU(),
    )


##############################################################################
# 3D Transformer Components
##############################################################################
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """3D Self-Attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        """x: (B, num_patches, dim)"""
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


##############################################################################
# 3D MobileViT Block
##############################################################################
class MobileViTv3Block(nn.Module):
    """3D version of MobileViT block"""
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, conv_stide=1):
        super().__init__()
        self.ph, self.pw, self.pd = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, conv_stide)
        self.conv2 = conv_1x1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim)

        self.conv3 = conv_1x1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, 1)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten spatial dimensions to sequence for transformer: (B, C=dim, D, H, W) -> (B, N=DHW, dim)
        b, c, d, h, w = x.shape  # here c == dim
        x = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
        # Global representations through transformer
        x = self.transformer(x)
        # Back to (B, C=dim, D, H, W)
        x = x.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)

        # Fusion of local and global features
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)

        return x


def conv_nxn_bn(inp, oup, stride=1):
    return nn.Sequential(
        # Use standard 3D convolution (no depthwise grouping) to satisfy out_channels % groups == 0
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU(),
    )


##############################################################################
# 3D MobileViT Encoder
##############################################################################
class MV2Block(nn.Module):
    """3D MobileNetV2 block"""
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


class MobileViTv3Block_NoStem(nn.Module):
    """3D MobileViT block without stem"""
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim=128):
        super().__init__()
        self.ph, self.pw, self.pd = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, 1)
        self.conv2 = conv_1x1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim)

        self.conv3 = conv_1x1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, 1)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten spatial dims to sequence for transformer: (B, C=dim, D, H, W) -> (B, N=DHW, dim)
        b, c, d, h, w = x.shape  # here c == dim
        x = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
        # Global representations through transformer
        x = self.transformer(x)
        # Back to (B, C=dim, D, H, W)
        x = x.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)

        # Fusion of local and global features
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)

        return x


class MobileViTv3Encoder(nn.Module):
    """3D MobileViT encoder without pretrained weights (scratch)"""
    def __init__(self, image_size=(240, 240, 155), in_channels=2, patch_size=(2, 2, 2)):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        
        # Stem layers
        self.conv_stem = nn.Conv3d(in_channels, 16, 3, 2, 1, bias=False)
        self.bn_stem = nn.BatchNorm3d(16)
        self.activation_stem = nn.SiLU()
        
        # Stage 1
        self.stage1 = nn.Sequential(
            MV2Block(16, 16, 1, 1),
            MV2Block(16, 24, 2, 4),
            MV2Block(24, 24, 1, 3),
            MV2Block(24, 24, 1, 3),
            MV2Block(24, 48, 2, 4),
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            MV2Block(48, 64, 2, 4),
            MV2Block(64, 64, 1, 3),
            MV2Block(64, 96, 2, 4),
        )
        
        # Stage 3 with MobileViT blocks
        self.stage3 = nn.Sequential(
            MV2Block(96, 160, 2, 4),
            MobileViTv3Block_NoStem(dim=144, depth=2, channel=160, kernel_size=3, 
                                  patch_size=patch_size, mlp_dim=384),
            MobileViTv3Block_NoStem(dim=192, depth=4, channel=160, kernel_size=3,
                                  patch_size=patch_size, mlp_dim=576),
            MV2Block(160, 320, 1, 4),
            MobileViTv3Block_NoStem(dim=240, depth=3, channel=320, kernel_size=3,
                                  patch_size=patch_size, mlp_dim=960),
        )
        
    def forward(self, x):
        """x: (B, C, D, H, W)"""
        # Stem
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = self.activation_stem(x)
        
        # Stage 1
        x = self.stage1(x)
        h1 = x
        
        # Stage 2
        x = self.stage2(x)
        h2 = x
        
        # Stage 3
        x = self.stage3(x)
        h3 = x
        
        return {
            "raw_input": x,
            "hidden_states": [h1, h2, h3]
        }


##############################################################################
# 3D Bottleneck and Decoder
##############################################################################
class MV3DBottleneck(nn.Module):
    """3D Bottleneck layer"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
    
    def forward(self, x):
        """x: (B, C, D, H, W) -> (B, n_patches, dim)"""
        B, C, D, H, W = x.shape
        
        # Flatten spatial dimensions
        x = x.reshape(B, C, -1)  # (B, C, D*H*W)
        x = x.permute(0, 2, 1)  # (B, D*H*W, C)
        
        x = self.transformer(x)
        
        return x


class MV3DDecoders(nn.Module):
    """3D Decoder with skip connections

    Build a chain of Conv3d blocks that progressively reduce channels. The first
    block takes `base_channels` (e.g., encoder's last feature channels), and the
    rest follow the provided `channels` list.
    """
    def __init__(self, base_channels: int, channels, num_classes):
        super().__init__()
        self.decoders = nn.ModuleList()

        in_c = base_channels
        for i, out_c in enumerate(channels):
            self.decoders.append(
                nn.Sequential(
                    nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(out_c),
                    nn.SiLU(),
                )
            )
            in_c = out_c
        # final projection to num_classes
        self.final = nn.Conv3d(in_c, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x, skip_connections):
        """x: (B, D*H*W, dim), skip_connections: list of (B, C, D, H, W)"""
        B = x.shape[0]
        
        # Reshape transformer output to 3D
        # Assuming we need to match the last skip connection
        last_skip = skip_connections[-1]
        B, C, D, H, W = last_skip.shape
        x = x.reshape(B, C, D, H, W)
        
        # Process through decoders
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            # Upsample if needed (simple progressive upsampling)
            if i < len(self.decoders) - 1:
                x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        # Final logits
        x = self.final(x)
        return x


##############################################################################
# Complete 3D MobileUNETR Model
##############################################################################
class MobileUNETR_3D(nn.Module):
    """3D MobileUNETR for volume segmentation"""
    def __init__(
        self,
        image_size=(240, 240, 155),
        patch_size=(2, 2, 2),
        in_channels=2,
        out_channels=4,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder = MobileViTv3Encoder(
            image_size=image_size,
            in_channels=in_channels,
            patch_size=patch_size
        )
        
        # Bottleneck
        self.bottleneck = MV3DBottleneck(
            dim=320,
            depth=3,
            heads=4,
            dim_head=64,
            mlp_dim=1280,
            dropout=0.1
        )
        
        # Decoder
        # Use encoder's last feature channels (320) as base, then reduce
        self.decoder = MV3DDecoders(
            base_channels=320,
            channels=[196, 160, 128, 96, 64, 32, 16],
            num_classes=self.out_channels
        )
    
    def forward(self, x):
        # Encoder
        enc_dict = self.encoder(x)
        
        # Bottleneck
        bneck = self.bottleneck(enc_dict["hidden_states"][-1])
        
        # Decoder
        dec_out = self.decoder(bneck, enc_dict["hidden_states"])
        
        return dec_out


def MobileUNETR_3D_Wrapper(img_size=(240, 240, 155), patch_size=(2, 2, 2), in_channels=2, out_channels=4):
    """Wrapper function for 3D MobileUNETR"""
    model = MobileUNETR_3D(
        image_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return model

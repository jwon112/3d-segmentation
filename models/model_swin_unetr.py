import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union
import math

class PatchEmbedding3D(nn.Module):
    """3D Patch Embedding for Swin Transformer"""
    def __init__(self, img_size: Union[int, Tuple[int, int, int]], 
                 patch_size: Union[int, Tuple[int, int, int]], 
                 in_channels: int = 1, 
                 embed_dim: int = 96):
        super().__init__()
        
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
            
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        self.projection = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.projection(x)  # (B, embed_dim, D', H', W')
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, D'*H'*W', embed_dim)
        return x

class WindowAttention3D(nn.Module):
    """3D Window-based Multi-head Self Attention"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        # Convert window_size to tuple if it's an int
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size, window_size)
        else:
            self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # Use indexing='ij' for compatibility, fallback to default if not available
        try:
            coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))  # 3, Wh, Ww
        except TypeError:
            # Fallback for older PyTorch versions
            coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 3
        
        # In-place 연산 전에 clone()하여 완전히 독립적인 텐서로 만들기
        relative_coords = relative_coords.clone()
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        
        # 최종 계산 전에 다시 clone()하여 완전히 독립적으로 만들기
        relative_position_index = relative_coords.sum(-1).clone()  # Wh*Ww, Wh*Ww
        # register_buffer 전에 한 번 더 clone()하여 완전한 독립성 보장
        relative_position_index = relative_position_index.clone()
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block for 3D"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        # Convert window_size to tuple if it's an int
        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= min(self.window_size):
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            min_res = min(self.input_resolution)
            self.window_size = (min_res, min_res, min_res)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, L, C = x.shape
        # Compute actual resolution from input size (for dynamic size support)
        H, W, D = self.input_resolution
        # If L doesn't match, compute from L (approximate cubic root)
        if L != H * W * D:
            # Infer resolution from L (for dynamic sizes)
            side = int(round((L ** (1/3))))
            H = W = D = side
            # Adjust to make H*W*D == L
            while H * W * D < L:
                if H <= W and H <= D:
                    H += 1
                elif W <= D:
                    W += 1
                else:
                    D += 1
            # If still too large, reduce
            while H * W * D > L and H > 1 and W > 1 and D > 1:
                if H >= W and H >= D:
                    H -= 1
                elif W >= D:
                    W -= 1
                else:
                    D -= 1
        assert L == H * W * D, f"input feature has wrong size: L={L}, H*W*D={H*W*D} (H={H}, W={W}, D={D})"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, D, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)  # nW*B, window_size*window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W, D)  # B H' W' D' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = x.view(B, H * W * D, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, D, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, window_size, C)
        """
        B, H, W, D, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W, D):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image
            D (int): Depth of image

        Returns:
            x: (B, H, W, D, C)
        """
        B = int(windows.shape[0] / (H * W * D / window_size / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
        return x

class PatchMerging3D(nn.Module):
    """3D Patch Merging Layer"""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        # Compute actual resolution from input size
        H, W, D = self.input_resolution
        # If L doesn't match, compute from L (approximate cubic root)
        if L != H * W * D:
            # Infer resolution from L (for dynamic sizes)
            side = int(round((L ** (1/3))))
            H = W = D = side
            # Adjust to make H*W*D == L
            while H * W * D < L:
                if H <= W and H <= D:
                    H += 1
                elif W <= D:
                    W += 1
                else:
                    D += 1
            # If still too large, reduce
            while H * W * D > L and H > 1 and W > 1 and D > 1:
                if H >= W and H >= D:
                    H -= 1
                elif W >= D:
                    W -= 1
                else:
                    D -= 1
        assert L == H * W * D, f"input feature has wrong size: L={L}, H*W*D={H*W*D} (H={H}, W={W}, D={D})"
        assert H % 2 == 0 and W % 2 == 0 and D % 2 == 0, f"x size ({H}*{W}*{D}) are not even."

        x = x.view(B, H, W, D, C)

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 D/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*D/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class SwinUNETR(nn.Module):
    """Swin UNETR: Swin Transformer for 3D Medical Image Segmentation"""
    def __init__(self, 
                 img_size: Union[int, Tuple[int, int, int]] = (128, 128, 128),
                 patch_size: Union[int, Tuple[int, int, int]] = (4, 4, 4),
                 in_channels: int = 4,
                 out_channels: int = 4,
                 embed_dim: int = 96,
                 depths: Sequence[int] = (2, 2, 6, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24),
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(self.img_size, self.patch_size, in_channels, embed_dim)
        
        # Calculate patch resolution
        self.patch_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2]
        )
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            dim = embed_dim * 2 ** i_layer
            for i_block in range(depths[i_layer]):
                layer.append(SwinTransformerBlock3D(
                    dim=dim,
                    input_resolution=(
                        self.patch_resolution[0] // (2 ** i_layer),
                        self.patch_resolution[1] // (2 ** i_layer),
                        self.patch_resolution[2] // (2 ** i_layer)
                    ),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                ))
            if i_layer < self.num_layers - 1:
                layer.append(PatchMerging3D(
                    input_resolution=(
                        self.patch_resolution[0] // (2 ** i_layer),
                        self.patch_resolution[1] // (2 ** i_layer),
                        self.patch_resolution[2] // (2 ** i_layer)
                    ),
                    dim=dim
                ))
            self.layers.append(layer)
        
        # Decoder
        self.decoder_conv1 = nn.ConvTranspose3d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2)
        self.decoder_conv2 = nn.ConvTranspose3d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.decoder_conv3 = nn.ConvTranspose3d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        
        # Final output layer
        self.final_conv = nn.Conv3d(embed_dim, out_channels, kernel_size=1)
        
        # Skip connection convolutions
        self.skip_conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.skip_conv2 = nn.Conv3d(embed_dim, embed_dim * 2, kernel_size=3, padding=1)
        self.skip_conv3 = nn.Conv3d(embed_dim * 2, embed_dim * 4, kernel_size=3, padding=1)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Encoder
        skip_features = []
        
        # Create skip features at different scales
        skip_features.append(self.skip_conv1(x))  # Original scale
        skip_features.append(self.skip_conv2(F.max_pool3d(skip_features[0], 2)))  # 1/2 scale
        skip_features.append(self.skip_conv3(F.max_pool3d(skip_features[1], 2)))  # 1/4 scale
        
        # Process through Swin Transformer layers
        for layer in self.layers:
            for block in layer[:-1]:  # All blocks except the last one (PatchMerging)
                x_patches = block(x_patches)
            if len(layer) > 1:  # If there's a PatchMerging layer
                x_patches = layer[-1](x_patches)
        
        # Reshape back to spatial dimensions for the deepest feature
        patch_d = D // self.patch_size[0] // (2 ** (self.num_layers - 1))
        patch_h = H // self.patch_size[1] // (2 ** (self.num_layers - 1))
        patch_w = W // self.patch_size[2] // (2 ** (self.num_layers - 1))
        
        x_encoded = x_patches.transpose(1, 2).view(B, self.embed_dim * (2 ** (self.num_layers - 1)), patch_d, patch_h, patch_w)
        
        # Decoder with skip connections
        x_dec = self.decoder_conv1(x_encoded)
        x_dec = x_dec + skip_features[2]
        
        x_dec = self.decoder_conv2(x_dec)
        x_dec = x_dec + skip_features[1]
        
        x_dec = self.decoder_conv3(x_dec)
        x_dec = x_dec + skip_features[0]
        
        # Final output
        output = self.final_conv(x_dec)
        
        return output

class SwinUNETR_Simplified(nn.Module):
    """Simplified Swin UNETR for memory efficiency"""
    def __init__(self, 
                 img_size: Union[int, Tuple[int, int, int]] = (64, 64, 64),
                 patch_size: Union[int, Tuple[int, int, int]] = (4, 4, 4),
                 in_channels: int = 4,
                 out_channels: int = 4,
                 embed_dim: int = 48,
                 depths: Sequence[int] = (2, 2, 2),
                 num_heads: Sequence[int] = (3, 6, 12),
                 window_size: int = 4):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(self.img_size, self.patch_size, in_channels, embed_dim)
        
        # Calculate patch resolution
        self.patch_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2]
        )
        
        # Simplified Swin Transformer layers
        self.swin_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_blocks = nn.ModuleList()
            dim = embed_dim * 2 ** i_layer
            for i_block in range(depths[i_layer]):
                layer_blocks.append(SwinTransformerBlock3D(
                    dim=dim,
                    input_resolution=(
                        self.patch_resolution[0] // (2 ** i_layer),
                        self.patch_resolution[1] // (2 ** i_layer),
                        self.patch_resolution[2] // (2 ** i_layer)
                    ),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                    mlp_ratio=2.0
                ))
            self.swin_layers.append(layer_blocks)
            
            # Add PatchMerging except for the last layer
            if i_layer < self.num_layers - 1:
                self.swin_layers.append(PatchMerging3D(
                    input_resolution=(
                        self.patch_resolution[0] // (2 ** i_layer),
                        self.patch_resolution[1] // (2 ** i_layer),
                        self.patch_resolution[2] // (2 ** i_layer)
                    ),
                    dim=dim
                ))
        
        # Simplified decoder
        self.decoder_conv1 = nn.ConvTranspose3d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.decoder_conv2 = nn.ConvTranspose3d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        
        # Final output layer
        self.final_conv = nn.Conv3d(embed_dim, out_channels, kernel_size=1)
        
        # Skip connection convolutions
        self.skip_conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.skip_conv2 = nn.Conv3d(embed_dim, embed_dim * 2, kernel_size=3, padding=1)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Create skip features
        skip1 = self.skip_conv1(x)  # Original scale
        skip2 = self.skip_conv2(F.max_pool3d(skip1, 2))  # 1/2 scale
        
        # Process through Swin Transformer layers
        layer_idx = 0
        for i, layer in enumerate(self.swin_layers):
            if isinstance(layer, nn.ModuleList):  # SwinTransformerBlock layers
                for block in layer:
                    x_patches = block(x_patches)
            else:  # PatchMerging layer
                x_patches = layer(x_patches)
        
        # Reshape back to spatial dimensions
        patch_d = D // self.patch_size[0] // (2 ** (self.num_layers - 1))
        patch_h = H // self.patch_size[1] // (2 ** (self.num_layers - 1))
        patch_w = W // self.patch_size[2] // (2 ** (self.num_layers - 1))
        
        x_encoded = x_patches.transpose(1, 2).view(B, self.embed_dim * (2 ** (self.num_layers - 1)), patch_d, patch_h, patch_w)
        
        # Decoder with skip connections
        x_dec = self.decoder_conv1(x_encoded)
        x_dec = x_dec + skip2
        
        x_dec = self.decoder_conv2(x_dec)
        x_dec = x_dec + skip1
        
        # Final output
        output = self.final_conv(x_dec)
        
        return output

if __name__ == "__main__":
    # 모델 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 간소화된 Swin UNETR 모델 사용
    model = SwinUNETR_Simplified(img_size=(64, 64, 64), patch_size=(4, 4, 4), 
                                in_channels=4, out_channels=4).to(device)
    
    # 더미 입력 생성
    batch_size = 1
    input_tensor = torch.randn(batch_size, 4, 64, 64, 64).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

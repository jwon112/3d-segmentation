import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union

class PatchEmbedding3D(nn.Module):
    """3D Patch Embedding for Vision Transformer"""
    def __init__(self, img_size: Union[int, Tuple[int, int, int]], 
                 patch_size: Union[int, Tuple[int, int, int]], 
                 in_channels: int = 1, 
                 embed_dim: int = 768):
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

class PositionalEncoding3D(nn.Module):
    """3D Positional Encoding"""
    def __init__(self, embed_dim: int, max_len: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (seq_len, batch_size, embed_dim)
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x

class UNETR(nn.Module):
    """UNETR: Transformers for 3D Medical Image Segmentation"""
    def __init__(self, 
                 img_size: Union[int, Tuple[int, int, int]] = (128, 128, 128),
                 patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 16),
                 in_channels: int = 4,
                 out_channels: int = 4,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(self.img_size, self.patch_size, in_channels, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(embed_dim)
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder heads
        self.decoder_heads = nn.ModuleList([
            nn.ConvTranspose3d(embed_dim, 512, kernel_size=2, stride=2),
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
        ])
        
        # Final output layer
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        
        # Skip connection convolutions
        self.skip_convs = nn.ModuleList([
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
        ])
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x_patches = x_patches.transpose(0, 1)  # (num_patches, B, embed_dim)
        x_patches = self.pos_encoding(x_patches)
        x_patches = x_patches.transpose(0, 1)  # (B, num_patches, embed_dim)
        
        # Transformer encoding
        for transformer_block in self.transformer_blocks:
            x_patches = transformer_block(x_patches)
        
        # Reshape back to spatial dimensions
        patch_d = D // self.patch_size[0]
        patch_h = H // self.patch_size[1]
        patch_w = W // self.patch_size[2]
        
        x_encoded = x_patches.transpose(1, 2).view(B, self.embed_dim, patch_d, patch_h, patch_w)
        
        # Decoder with skip connections
        skip_features = []
        
        # Create skip features at different scales
        skip_features.append(self.skip_convs[0](x))  # Original scale
        skip_features.append(self.skip_convs[1](F.max_pool3d(skip_features[0], 2)))  # 1/2 scale
        skip_features.append(self.skip_convs[2](F.max_pool3d(skip_features[1], 2)))  # 1/4 scale
        skip_features.append(self.skip_convs[3](F.max_pool3d(skip_features[2], 2)))  # 1/8 scale
        
        # Decoder
        x_dec = x_encoded
        for i, decoder_head in enumerate(self.decoder_blocks):
            x_dec = decoder_head(x_dec)
            if i < len(skip_features):
                # Add skip connection
                x_dec = x_dec + skip_features[-(i+1)]
        
        # Final output
        output = self.final_conv(x_dec)
        
        return output

class UNETR_Simplified(nn.Module):
    """Simplified UNETR for memory efficiency"""
    def __init__(self, 
                 img_size: Union[int, Tuple[int, int, int]] = (64, 64, 64),
                 patch_size: Union[int, Tuple[int, int, int]] = (8, 8, 8),
                 in_channels: int = 4,
                 out_channels: int = 4,
                 embed_dim: int = 384,
                 num_heads: int = 6,
                 num_layers: int = 6):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(self.img_size, self.patch_size, in_channels, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(embed_dim)
        
        # Simplified transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2.0, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Simplified decoder
        self.decoder_conv1 = nn.ConvTranspose3d(embed_dim, 256, kernel_size=2, stride=2)
        self.decoder_conv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        
        # Final output layer
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        
        # Skip connection convolutions
        self.skip_conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.skip_conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.skip_conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x_patches = x_patches.transpose(0, 1)  # (num_patches, B, embed_dim)
        x_patches = self.pos_encoding(x_patches)
        x_patches = x_patches.transpose(0, 1)  # (B, num_patches, embed_dim)
        
        # Transformer encoding
        for transformer_block in self.transformer_blocks:
            x_patches = transformer_block(x_patches)
        
        # Reshape back to spatial dimensions
        patch_d = D // self.patch_size[0]
        patch_h = H // self.patch_size[1]
        patch_w = W // self.patch_size[2]
        
        x_encoded = x_patches.transpose(1, 2).view(B, self.embed_dim, patch_d, patch_h, patch_w)
        
        # Create skip features
        skip1 = self.skip_conv1(x)  # Original scale
        skip2 = self.skip_conv2(F.max_pool3d(skip1, 2))  # 1/2 scale
        skip3 = self.skip_conv3(F.max_pool3d(skip2, 2))  # 1/4 scale
        
        # Decoder with skip connections
        x_dec = self.decoder_conv1(x_encoded)
        x_dec = x_dec + skip3
        
        x_dec = self.decoder_conv2(x_dec)
        x_dec = x_dec + skip2
        
        x_dec = self.decoder_conv3(x_dec)
        x_dec = x_dec + skip1
        
        # Final output
        output = self.final_conv(x_dec)
        
        return output

if __name__ == "__main__":
    # 모델 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 간소화된 UNETR 모델 사용
    model = UNETR_Simplified(img_size=(64, 64, 64), patch_size=(8, 8, 8), 
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

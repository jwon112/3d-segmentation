"""
ShuffleNet Modules
ShuffleNetV1 및 ShuffleNetV2 스타일의 3D 모듈들
"""

import torch
import torch.nn as nn

from ..model_3d_unet import _make_norm3d


def channel_shuffle_3d(x: torch.Tensor, groups: int) -> torch.Tensor:
    """3D Channel Shuffle operation.
    
    Args:
        x: Input tensor of shape (B, C, D, H, W)
        groups: Number of groups to shuffle
    
    Returns:
        Shuffled tensor
    """
    B, C, D, H, W = x.size()
    channels_per_group = C // groups
    
    # Reshape: (B, groups, channels_per_group, D, H, W)
    x = x.view(B, groups, channels_per_group, D, H, W)
    
    # Transpose: (B, channels_per_group, groups, D, H, W)
    x = x.transpose(1, 2).contiguous()
    
    # Flatten: (B, C, D, H, W)
    x = x.view(B, C, D, H, W)
    
    return x


# ============================================================================
# ShuffleNetV1 Modules
# ============================================================================

class ShuffleNetV1Unit3D(nn.Module):
    """3D ShuffleNetV1 Unit.
    
    ShuffleNetV1의 핵심 블록:
    - Stride=1: Pointwise Group Conv -> Channel Shuffle -> Depthwise Conv -> Pointwise Group Conv + Residual
    - Stride=2: Pointwise Group Conv -> Channel Shuffle -> Depthwise Conv (stride=2) -> Pointwise Group Conv
    
    Args:
        dilation: Dilation rate for depthwise convolution (기본값: 1)
                  stride=1일 때만 사용 가능, 전역 문맥 포착을 위해 사용
    
    Reference:
        ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices (Zhang et al., CVPR 2018)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, norm: str = 'bn', dilation: int = 1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dilation = dilation
        
        # Dilation은 stride=1일 때만 사용 가능
        if dilation > 1:
            assert stride == 1, "Dilation can only be used with stride=1"
        
        # Pointwise Group Convolution
        mid_channels = out_channels // 4  # ShuffleNetV1에서 일반적으로 사용하는 비율
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, 
                               groups=groups, bias=False)
        self.bn1 = _make_norm3d(norm, mid_channels)
        
        # Depthwise Convolution (dilation 지원)
        padding = dilation * (3 - 1) // 2  # kernel_size=3에 대한 padding 계산
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=padding,
                               dilation=dilation, groups=mid_channels, bias=False)
        self.bn2 = _make_norm3d(norm, mid_channels)
        
        # Pointwise Group Convolution
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0,
                               groups=groups, bias=False)
        self.bn3 = _make_norm3d(norm, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection for stride=1 only
        if stride == 1:
            assert in_channels == out_channels, "For stride=1, in_channels must equal out_channels"
            self.use_residual = True
        else:
            # For stride=2, no residual connection (channels change)
            self.use_residual = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pointwise Group Conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Channel Shuffle
        out = channel_shuffle_3d(out, groups=self.groups)
        
        # Depthwise Conv
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Pointwise Group Conv
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Residual connection (only for stride=1)
        if self.use_residual:
            out = out + x
        
        out = self.relu(out)
        return out


class MultiScaleDilatedDepthwise3D(nn.Module):
    """Multi-Scale Dilated Depthwise Convolution for 3D
    
    여러 dilation rate를 병렬로 적용하여 다양한 수용 영역의 문맥을 포착합니다.
    ASPP (Atrous Spatial Pyramid Pooling) 스타일의 접근법.
    
    Args:
        channels: 입력/출력 채널 수
        dilation_rates: Dilation rate 리스트 (기본값: [1, 2, 5])
        norm: Normalization 타입 ('bn', 'gn', 'in')
    """
    def __init__(self, channels: int, dilation_rates: list = [1, 2, 5], norm: str = 'bn'):
        super().__init__()
        self.dilation_rates = dilation_rates
        
        # 각 dilation rate에 대한 depth-wise convolution
        self.dilated_convs = nn.ModuleList()
        for dilation in dilation_rates:
            padding = dilation * (3 - 1) // 2  # kernel_size=3에 대한 padding
            conv = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=padding,
                         dilation=dilation, groups=channels, bias=False),
                _make_norm3d(norm, channels),
                nn.ReLU(inplace=True),
            )
            self.dilated_convs.append(conv)
        
        # 병렬 결과를 합치는 1x1 conv (선택적, 채널 수가 같으므로 더하기만 해도 됨)
        # 하지만 normalization과 activation을 위해 1x1 conv 추가
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            _make_norm3d(norm, channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)
        
        Returns:
            Multi-scale dilated convolution이 적용된 출력 텐서 (B, C, D, H, W)
        """
        # 각 dilation rate에 대해 병렬로 적용
        outputs = []
        for dilated_conv in self.dilated_convs:
            outputs.append(dilated_conv(x))
        
        # 결과를 더하기 (element-wise sum)
        out = sum(outputs)
        
        # Fusion (1x1 conv로 정규화)
        out = self.fusion(out)
        
        # Residual connection
        out = out + x
        
        return out


class Down3DShuffleNetV1(nn.Module):
    """Downsampling using ShuffleNetV1 unit (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, groups: int = 1, norm: str = 'bn'):
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = ShuffleNetV1Unit3D(in_channels, out_channels, stride=2, groups=groups, norm=norm)
        # Second unit: stride=1 for feature refinement
        self.unit2 = ShuffleNetV1Unit3D(out_channels, out_channels, stride=1, groups=groups, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


# ============================================================================
# ShuffleNetV2 Modules
# ============================================================================

class ShuffleNetV2Unit3D(nn.Module):
    """3D ShuffleNetV2 Unit.
    
    ShuffleNetV2의 핵심 블록:
    - Stride=1: Split -> Branch1 (identity) + Branch2 (DWConv -> 1x1 -> DWConv -> 1x1) -> Concat -> Shuffle
    - Stride=2: No split -> Branch1 (DWConv stride=2 -> 1x1) + Branch2 (1x1 -> DWConv stride=2 -> 1x1) -> Concat -> Shuffle
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = 'bn'):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if stride == 1:
            # Stride=1: Split operation
            assert in_channels == out_channels, "For stride=1, in_channels must equal out_channels"
            mid_channels = out_channels // 2
            
            # Branch 1: Identity (no operation)
            self.branch1 = nn.Identity()
            
            # Branch 2: DWConv -> 1x1 -> DWConv -> 1x1
            self.branch2 = nn.Sequential(
                # Depthwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, 
                         groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Pointwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Depthwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, 
                         groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                # Pointwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # Stride=2: No split, both branches process full input
            # Branch 1: DWConv stride=2 -> 1x1
            self.branch1 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels // 2),
                nn.ReLU(inplace=True),
            )
            
            # Branch 2: 1x1 -> DWConv stride=2 -> 1x1
            self.branch2 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels // 2),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            # Split channels
            x1, x2 = x.chunk(2, dim=1)
            # Branch 1: Identity
            out1 = self.branch1(x1)
            # Branch 2: Processing
            out2 = self.branch2(x2)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        else:
            # Both branches process full input
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        
        # Channel Shuffle
        out = channel_shuffle_3d(out, groups=2)
        return out


class ShuffleNetV2Unit3D_Dilated(nn.Module):
    """3D ShuffleNetV2 Unit with Dilated Convolution (rate [1,2,5]).
    
    ShuffleNetV2의 DWConv에 dilated convolution 적용:
    - Stride=1: Split -> Branch1 (identity) + Branch2 (DWDilatedConv[1,2,5] -> 1x1 -> DWDilatedConv[1,2,5] -> 1x1) -> Concat -> Shuffle
    - Stride=2: No split -> Branch1 (DWDilatedConv stride=2 -> 1x1) + Branch2 (1x1 -> DWDilatedConv stride=2 -> 1x1) -> Concat -> Shuffle
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = 'bn', dilation_rates: list = [1, 2, 5]):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rates = dilation_rates
        
        if stride == 1:
            # Stride=1: Split operation
            assert in_channels == out_channels, "For stride=1, in_channels must equal out_channels"
            mid_channels = out_channels // 2
            
            # Branch 1: Identity (no operation)
            self.branch1 = nn.Identity()
            
            # Branch 2: DWDilatedConv[rate1] -> 1x1 -> DWDilatedConv[rate2] -> 1x1 -> DWDilatedConv[rate5] -> 1x1
            # rate [1,2,5]를 모두 사용하여 더 넓은 receptive field 확보
            # Padding for dilated conv: padding = dilation * (kernel_size - 1) / 2
            # For kernel=5, dilation=1: padding = 1 * (5-1) / 2 = 2
            # For kernel=5, dilation=2: padding = 2 * (5-1) / 2 = 4
            # For kernel=5, dilation=5: padding = 5 * (5-1) / 2 = 10
            self.branch2 = nn.Sequential(
                # Depthwise Dilated Conv (rate 1)
                nn.Conv3d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, 
                         dilation=dilation_rates[0], groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Pointwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Depthwise Dilated Conv (rate 2)
                nn.Conv3d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=4, 
                         dilation=dilation_rates[1], groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Pointwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Depthwise Dilated Conv (rate 5)
                nn.Conv3d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=10, 
                         dilation=dilation_rates[2], groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                # Pointwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # Stride=2: No split, both branches process full input
            # Branch 1: DWDilatedConv stride=2 -> 1x1
            self.branch1 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels // 2),
                nn.ReLU(inplace=True),
            )
            
            # Branch 2: 1x1 -> DWDilatedConv stride=2 -> 1x1
            self.branch2 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels // 2),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            # Split channels
            x1, x2 = x.chunk(2, dim=1)
            # Branch 1: Identity
            out1 = self.branch1(x1)
            # Branch 2: Processing
            out2 = self.branch2(x2)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        else:
            # Both branches process full input
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        
        # Channel Shuffle
        out = channel_shuffle_3d(out, groups=2)
        return out


class ShuffleNetV2Unit3D_LK(nn.Module):
    """3D ShuffleNetV2 Unit with Large Kernel (7x7x7).
    
    ShuffleNetV2의 DWConv kernel_size를 7x7x7로 변경:
    - Stride=1: Split -> Branch1 (identity) + Branch2 (DWConv7x7 -> 1x1 -> DWConv7x7 -> 1x1) -> Concat -> Shuffle
    - Stride=2: No split -> Branch1 (DWConv7x7 stride=2 -> 1x1) + Branch2 (1x1 -> DWConv7x7 stride=2 -> 1x1) -> Concat -> Shuffle
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = 'bn'):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if stride == 1:
            # Stride=1: Split operation
            assert in_channels == out_channels, "For stride=1, in_channels must equal out_channels"
            mid_channels = out_channels // 2
            
            # Branch 1: Identity (no operation)
            self.branch1 = nn.Identity()
            
            # Branch 2: DWConv7x7 -> 1x1 -> DWConv7x7 -> 1x1
            self.branch2 = nn.Sequential(
                # Depthwise Conv (7x7x7)
                nn.Conv3d(mid_channels, mid_channels, kernel_size=7, stride=1, padding=3, 
                         groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Pointwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
                # Depthwise Conv (7x7x7)
                nn.Conv3d(mid_channels, mid_channels, kernel_size=7, stride=1, padding=3, 
                         groups=mid_channels, bias=False),
                _make_norm3d(norm, mid_channels),
                # Pointwise Conv
                nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # Stride=2: No split, both branches process full input
            # Branch 1: DWConv7x7 stride=2 -> 1x1
            self.branch1 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=7, stride=2, padding=3, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels // 2),
                nn.ReLU(inplace=True),
            )
            
            # Branch 2: 1x1 -> DWConv7x7 stride=2 -> 1x1
            self.branch2 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=7, stride=2, padding=3, 
                         groups=in_channels, bias=False),
                _make_norm3d(norm, in_channels),
                nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
                _make_norm3d(norm, out_channels // 2),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            # Split channels
            x1, x2 = x.chunk(2, dim=1)
            # Branch 1: Identity
            out1 = self.branch1(x1)
            # Branch 2: Processing
            out2 = self.branch2(x2)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        else:
            # Both branches process full input
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            # Concat
            out = torch.cat([out1, out2], dim=1)
        
        # Channel Shuffle
        out = channel_shuffle_3d(out, groups=2)
        return out


class Down3DShuffleNetV2(nn.Module):
    """Downsampling using ShuffleNetV2 unit (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = ShuffleNetV2Unit3D(in_channels, out_channels, stride=2, norm=norm)
        # Second unit: stride=1 for feature refinement
        self.unit2 = ShuffleNetV2Unit3D(out_channels, out_channels, stride=1, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


class Down3DShuffleNetV2_Dilated(nn.Module):
    """Downsampling using ShuffleNetV2 Dilated unit (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn', dilation_rates: list = [1, 2, 5]):
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = ShuffleNetV2Unit3D_Dilated(in_channels, out_channels, stride=2, norm=norm, dilation_rates=dilation_rates)
        # Second unit: stride=1 for feature refinement
        self.unit2 = ShuffleNetV2Unit3D_Dilated(out_channels, out_channels, stride=1, norm=norm, dilation_rates=dilation_rates)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


class Down3DShuffleNetV2_LK(nn.Module):
    """Downsampling using ShuffleNetV2 Large Kernel unit (stride=2)."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'bn'):
        super().__init__()
        # First unit: stride=2 for downsampling
        self.unit1 = ShuffleNetV2Unit3D_LK(in_channels, out_channels, stride=2, norm=norm)
        # Second unit: stride=1 for feature refinement
        self.unit2 = ShuffleNetV2Unit3D_LK(out_channels, out_channels, stride=1, norm=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        x = self.unit2(x)
        return x


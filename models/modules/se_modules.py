"""
Squeeze-and-Excitation (SE) Modules for 3D
채널 어텐션 메커니즘을 통한 3D SE 블록 구현
"""

import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation Block
    
    채널 어텐션 메커니즘을 통해 중요한 채널에 집중합니다.
    
    Args:
        channels: 입력 채널 수
        reduction: 채널 축소 비율 (기본값: 16)
    
    Reference:
        Squeeze-and-Excitation Networks (Hu et al., CVPR 2018)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (B, C, D, H, W)
        
        Returns:
            SE 블록이 적용된 출력 텐서 (B, C, D, H, W)
        """
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c)
        # Store latest excitation weights for logging/analysis
        self.last_excitation = y.detach().cpu()
        y = y.view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


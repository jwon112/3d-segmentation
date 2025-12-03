"""
3D patch-based training datasets (nnU-Net style sampling).
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .brats_base import BratsDataset3D


class BratsPatchDataset3D(Dataset):
    """3D BraTS 패치 데이터셋 (학습용) - nnU-Net 스타일 샘플링"""

    def __init__(self, base_dataset: BratsDataset3D, patch_size=(128, 128, 128), samples_per_volume: int = 16, augment: bool = False):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.augment = augment

        self.index_map = []
        for vidx in range(len(self.base_dataset)):
            for s in range(self.samples_per_volume):
                sampling_type = 0 if (s % 3 == 0) else 1
                self.index_map.append((vidx, s, sampling_type))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx):
        vidx, _, sampling_type = self.index_map[idx]
        image, mask = self.base_dataset[vidx]

        C, H, W, D = image.shape
        ph, pw, pd = self.patch_size

        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        pad_d = max(0, pd - D)
        if pad_h or pad_w or pad_d:
            image = F.pad(image, (0, pad_d, 0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_d, 0, pad_w, 0, pad_h))
            C, H, W, D = image.shape

        if sampling_type == 0:
            fg_classes = []
            for cls in [1, 2, 3]:
                if (mask == cls).any():
                    fg_classes.append(cls)
            if len(fg_classes) > 0:
                selected_class = fg_classes[torch.randint(0, len(fg_classes), (1,)).item()]
                fg_coords = (mask == selected_class).nonzero(as_tuple=False)
                if fg_coords.numel() > 0:
                    ci = fg_coords[torch.randint(0, fg_coords.shape[0], (1,)).item()]
                    cy, cx, cz = ci.tolist()
                else:
                    cy = torch.randint(0, H, (1,)).item()
                    cx = torch.randint(0, W, (1,)).item()
                    cz = torch.randint(0, D, (1,)).item()
            else:
                cy = torch.randint(0, H, (1,)).item()
                cx = torch.randint(0, W, (1,)).item()
                cz = torch.randint(0, D, (1,)).item()
        else:
            cy = torch.randint(0, H, (1,)).item()
            cx = torch.randint(0, W, (1,)).item()
            cz = torch.randint(0, D, (1,)).item()

        sy = max(0, min(cy - ph // 2, H - ph))
        sx = max(0, min(cx - pw // 2, W - pw))
        sz = max(0, min(cz - pd // 2, D - pd))

        img_patch = image[:, sy:sy+ph, sx:sx+pw, sz:sz+pd]
        msk_patch = mask[sy:sy+ph, sx:sx+pw, sz:sz+pd]

        img_patch, msk_patch = self._maybe_augment(img_patch, msk_patch)
        return img_patch, msk_patch

    def _maybe_augment(self, img_patch: torch.Tensor, msk_patch: torch.Tensor):
        if not self.augment:
            return img_patch, msk_patch
        
        # 1. Multi-axis flipping (각 축에 대해 독립적으로)
        if torch.rand(1).item() < 0.5:
            img_patch = torch.flip(img_patch, dims=(2,))
            msk_patch = torch.flip(msk_patch, dims=(1,))
        if torch.rand(1).item() < 0.5:
            img_patch = torch.flip(img_patch, dims=(3,))
            msk_patch = torch.flip(msk_patch, dims=(2,))
        if torch.rand(1).item() < 0.5:
            img_patch = torch.flip(img_patch, dims=(4,))
            msk_patch = torch.flip(msk_patch, dims=(3,))
        
        # 2. 90-degree rotations (의료 영상에 적합)
        if torch.rand(1).item() < 0.5:
            # XY plane rotation (90, 180, 270도)
            k = torch.randint(1, 4, (1,)).item()
            img_patch = torch.rot90(img_patch, k=k, dims=(3, 4))
            msk_patch = torch.rot90(msk_patch, k=k, dims=(2, 3))
        
        # 3. Intensity augmentation: Scale + Shift (강화)
        if torch.rand(1).item() < 0.5:
            scale = 1.0 + 0.15 * torch.randn(1).item()  # 0.1 -> 0.15
            shift = 0.08 * torch.randn(1).item()  # 0.05 -> 0.08
            img_patch = img_patch * scale + shift
        
        # 4. Gamma correction (강화)
        if torch.rand(1).item() < 0.4:  # 0.3 -> 0.4
            gamma = 0.6 + 0.8 * torch.rand(1).item()  # 0.7~1.3 -> 0.6~1.4
            sign = torch.sign(img_patch)
            img_patch = sign * (torch.abs(img_patch) ** gamma)
        
        # 5. Contrast adjustment
        if torch.rand(1).item() < 0.3:
            # Contrast: (x - mean) * factor + mean
            mean = img_patch.mean()
            factor = 0.7 + 0.6 * torch.rand(1).item()  # 0.7~1.3
            img_patch = (img_patch - mean) * factor + mean
        
        # 6. Brightness adjustment
        if torch.rand(1).item() < 0.3:
            brightness = 0.1 * torch.randn(1).item()
            img_patch = img_patch + brightness
        
        # 7. Gaussian noise (강화)
        if torch.rand(1).item() < 0.4:  # 0.3 -> 0.4
            noise_std = 0.02 + 0.03 * torch.rand(1).item()  # 0.02~0.05 (기존 0.03 고정)
            noise = torch.randn_like(img_patch) * noise_std
            img_patch = img_patch + noise
        
        return img_patch, msk_patch


__all__ = ["BratsPatchDataset3D"]




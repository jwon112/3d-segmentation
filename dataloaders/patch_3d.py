"""
3D patch-based training datasets (nnU-Net style sampling).
"""

import math
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .brats_base import BratsDataset3D


def _apply_anisotropy_resize_patch(
    img_patch: torch.Tensor,
    msk_patch: torch.Tensor,
    target_size: Tuple[int, int, int],
    prob: float = 0.4,
    scale_range: Tuple[float, float] = (0.7, 1.3),
):
    """
    Depth 축만 이방성 스케일 후 다시 target_size로 리샘플.
    img_patch: (C, H, W, D), msk_patch: (H, W, D)
    """
    if torch.rand(1).item() >= prob:
        return img_patch, msk_patch

    C, H, W, D = img_patch.shape
    tH, tW, tD = target_size
    log_r = torch.empty(1, device=img_patch.device).uniform_(math.log(scale_range[0]), math.log(scale_range[1])).exp().item()
    new_D = max(2, int(round(D * log_r)))

    img_cd = img_patch.permute(0, 3, 1, 2)  # (C, D, H, W)
    msk_dhw = msk_patch.permute(2, 0, 1)    # (D, H, W)

    def _interp(tensor, size, mode):
        kwargs = {"size": size, "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        out = F.interpolate(tensor.unsqueeze(0), **kwargs)
        return out.squeeze(0)

    img_scaled = _interp(img_cd, (new_D, H, W), "trilinear")
    msk_scaled = _interp(msk_dhw.unsqueeze(0).float(), (new_D, H, W), "nearest").squeeze(0).long()

    img_resized = _interp(img_scaled, (tD, tH, tW), "trilinear")
    msk_resized = _interp(msk_scaled.unsqueeze(0).float(), (tD, tH, tW), "nearest").squeeze(0).long()

    img_out = img_resized.permute(0, 2, 3, 1)  # (C, H, W, D)
    msk_out = msk_resized.permute(1, 2, 0)     # (H, W, D)
    return img_out, msk_out


class BratsPatchDataset3D(Dataset):
    """3D BraTS 패치 데이터셋 (학습용) - nnU-Net 스타일 샘플링"""

    def __init__(
        self,
        base_dataset: BratsDataset3D,
        patch_size=(128, 128, 128),
        samples_per_volume: int = 16,
        augment: bool = False,
        anisotropy_augment: bool = False,
        max_cache_size: int = 50,
    ):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.augment = augment
        self.anisotropy_augment = anisotropy_augment
        self.max_cache_size = max_cache_size

        self.index_map = []
        for vidx in range(len(self.base_dataset)):
            for s in range(self.samples_per_volume):
                sampling_type = 0 if (s % 3 == 0) else 1
                self.index_map.append((vidx, s, sampling_type))
        
        # LRU 캐시 (worker별로 독립적으로 유지됨)
        self._volume_cache = OrderedDict()

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx):
        vidx, _, sampling_type = self.index_map[idx]
        
        # Subset 인덱스 처리: Subset인 경우 실제 인덱스로 변환
        if hasattr(self.base_dataset, 'indices'):
            # Subset인 경우: 원본 데이터셋의 실제 인덱스 사용
            actual_vidx = self.base_dataset.indices[vidx]
            base_dataset = self.base_dataset.dataset
        else:
            actual_vidx = vidx
            base_dataset = self.base_dataset
        
        # LRU 캐시 확인 및 업데이트
        cache_key = (id(base_dataset), actual_vidx)  # 데이터셋 인스턴스와 인덱스 조합
        if cache_key in self._volume_cache:
            # 캐시 히트: 해당 항목을 맨 뒤로 이동 (가장 최근 사용)
            cached_data = self._volume_cache.pop(cache_key)
            self._volume_cache[cache_key] = cached_data
            if len(cached_data) == 3:
                image, mask, fg_coords_dict = cached_data
            else:
                image, mask = cached_data
                fg_coords_dict = None
        else:
            # 캐시 미스: 새로 로드
            loaded_data = base_dataset[actual_vidx]
            if len(loaded_data) == 3:
                image, mask, fg_coords_dict = loaded_data
            else:
                image, mask = loaded_data
                fg_coords_dict = None
            
            # 캐시 크기 제한 확인
            if len(self._volume_cache) >= self.max_cache_size:
                # 가장 오래된 항목 제거 (LRU)
                self._volume_cache.popitem(last=False)
            
            # 새 항목 추가
            if fg_coords_dict:
                self._volume_cache[cache_key] = (image, mask, fg_coords_dict)
            else:
                self._volume_cache[cache_key] = (image, mask)

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
            # 포그라운드 오버샘플링: 저장된 좌표 사용 (있으면), 없으면 기존 방식
            if fg_coords_dict is not None:
                # 저장된 포그라운드 좌표 사용 (매우 빠름)
                fg_classes = [cls for cls in [1, 2, 3] if cls in fg_coords_dict and len(fg_coords_dict[cls]) > 0]
                if len(fg_classes) > 0:
                    selected_class = fg_classes[torch.randint(0, len(fg_classes), (1,)).item()]
                    coords = fg_coords_dict[selected_class]
                    ci = coords[torch.randint(0, len(coords), (1,)).item()]
                    cy, cx, cz = ci.tolist()
                else:
                    cy = torch.randint(0, H, (1,)).item()
                    cx = torch.randint(0, W, (1,)).item()
                    cz = torch.randint(0, D, (1,)).item()
            else:
                # 기존 방식: 매번 전체 마스크 스캔 (느림, 하위 호환성)
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
        if not self.augment and not self.anisotropy_augment:
            return img_patch, msk_patch

        if self.anisotropy_augment:
            img_patch, msk_patch = _apply_anisotropy_resize_patch(
                img_patch, msk_patch, target_size=self.patch_size, prob=0.4, scale_range=(0.7, 1.3)
            )
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




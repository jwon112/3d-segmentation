"""
CoordConv / 볼륨 조작 관련 유틸리티 함수 모음.

좌표맵 생성 및 볼륨 crop/paste 유틸 함수들을 제공합니다.
"""

from typing import Sequence, Tuple, Optional, Dict

import torch
import torch.nn.functional as F

_COORD_MAP_CACHE: Dict[Tuple[int, int, int], torch.Tensor] = {}


def _to_3tuple(size: Sequence[int]) -> Tuple[int, int, int]:
    if len(size) != 3:
        raise ValueError(f"Expected 3 spatial dims, got {size}")
    return int(size[0]), int(size[1]), int(size[2])


def get_normalized_coord_map(spatial_shape: Sequence[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """Simple 3-channel normalized coordinate map (0-1 range)."""
    shape = _to_3tuple(spatial_shape)
    if shape not in _COORD_MAP_CACHE:
        h, w, d = shape
        ys = torch.linspace(0.0, 1.0, steps=h, dtype=torch.float32)
        xs = torch.linspace(0.0, 1.0, steps=w, dtype=torch.float32)
        zs = torch.linspace(0.0, 1.0, steps=d, dtype=torch.float32)
        grid = torch.meshgrid(ys, xs, zs, indexing='ij')
        coord = torch.stack(grid, dim=0)
        _COORD_MAP_CACHE[shape] = coord.contiguous()
    coord_map = _COORD_MAP_CACHE[shape]
    if device is not None:
        coord_map = coord_map.to(device)
    return coord_map


def get_hybrid_coord_map(spatial_shape: Sequence[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """Hybrid coordinate map: 3 linear channels + 6 Fourier feature channels (9 channels total)."""
    shape = _to_3tuple(spatial_shape)

    cache_key = ("hybrid", shape)
    if cache_key not in _COORD_MAP_CACHE:
        h, w, d = shape

        # Linear coordinates (normalized 0-1)
        ys = torch.linspace(0.0, 1.0, steps=h, dtype=torch.float32)
        xs = torch.linspace(0.0, 1.0, steps=w, dtype=torch.float32)
        zs = torch.linspace(0.0, 1.0, steps=d, dtype=torch.float32)
        grid_y, grid_x, grid_z = torch.meshgrid(ys, xs, zs, indexing="ij")

        # Linear coordinates (3 channels)
        linear_coord = torch.stack([grid_y, grid_x, grid_z], dim=0)

        # Fourier features: 6 channels (sin/cos for 3 axes at 1 frequency)
        freq = 1.0
        fourier_list = [
            torch.sin(2 * torch.pi * freq * grid_y),
            torch.cos(2 * torch.pi * freq * grid_y),
            torch.sin(2 * torch.pi * freq * grid_x),
            torch.cos(2 * torch.pi * freq * grid_x),
            torch.sin(2 * torch.pi * freq * grid_z),
            torch.cos(2 * torch.pi * freq * grid_z),
        ]
        fourier_coord = torch.stack(fourier_list, dim=0)  # (6, H, W, D)

        # Combine: 3 linear + 6 Fourier = 9 channels
        hybrid_coord = torch.cat([linear_coord, fourier_coord], dim=0)  # (9, H, W, D)
        _COORD_MAP_CACHE[cache_key] = hybrid_coord.contiguous()

    hybrid_coord = _COORD_MAP_CACHE[cache_key]
    if device is not None:
        hybrid_coord = hybrid_coord.to(device)
    return hybrid_coord


def get_coord_map(
    spatial_shape: Sequence[int],
    device: Optional[torch.device] = None,
    encoding_type: str = "simple",
) -> torch.Tensor:
    """Get coordinate map based on encoding type ('simple' or 'hybrid')."""
    if encoding_type == "simple":
        return get_normalized_coord_map(spatial_shape, device)
    if encoding_type == "hybrid":
        return get_hybrid_coord_map(spatial_shape, device)
    raise ValueError(f"Unknown encoding_type: {encoding_type}. Must be 'simple' or 'hybrid'")


def resize_volume(volume: torch.Tensor, target_size: Sequence[int], mode: str = "trilinear") -> torch.Tensor:
    """Resize 3D volume with shape (C, H, W, D) to target_size (H, W, D)."""
    if volume.ndim != 4:
        raise ValueError(f"Expected tensor with shape (C, H, W, D), got {volume.shape}")
    target = _to_3tuple(target_size)
    if list(volume.shape[1:]) == list(target):
        return volume

    vol = volume.unsqueeze(0)  # (1, C, H, W, D)
    # F.interpolate expects (N, C, D, H, W)
    vol = vol.permute(0, 1, 4, 2, 3)
    interp_kwargs = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interp_kwargs["align_corners"] = False
    target_dhw = (target[2], target[0], target[1])
    vol = F.interpolate(vol, size=target_dhw, mode=mode, **interp_kwargs)
    vol = vol.permute(0, 1, 3, 4, 2)
    return vol.squeeze(0)


def crop_volume_with_center(
    tensor: torch.Tensor,
    center: Sequence[float],
    crop_size: Sequence[int],
    return_origin: bool = False,
    debug_sample_idx: int = -1,
):
    """Crop a sub-volume centered at `center` from tensor with shape (C, H, W, D) or (H, W, D)."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dims, got {tensor.shape}")

    c, h, w, d = tensor.shape
    size = _to_3tuple(crop_size)
    cy, cx, cz = [float(c_val) for c_val in center]
    half_h, half_w, half_d = [s / 2.0 for s in size]

    start_h = int(round(cy - half_h))
    start_w = int(round(cx - half_w))
    start_d = int(round(cz - half_d))
    starts = [start_h, start_w, start_d]

    src_ranges = []
    dst_ranges = []
    origins = []
    for dim, start, sz in zip((h, w, d), starts, size):
        end = start + sz
        src_start = max(0, start)
        src_end = min(end, dim)
        dst_start = max(0, -start)
        copy_len = max(0, src_end - src_start)
        dst_end = dst_start + copy_len
        origins.append(src_start)
        src_ranges.append((src_start, src_end))
        dst_ranges.append((dst_start, dst_end))

    if debug_sample_idx == 0:
        import json
        import time
        import os

        log_dir = os.path.join(os.getcwd(), ".cursor")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "debug.log")
        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "coord-check",
                            "hypothesisId": "H5",
                            "location": "coord_utils.py:crop_volume_with_center",
                            "message": "Crop volume with center - coordinate mapping",
                            "data": {
                                "input_center": [float(cy), float(cx), float(cz)],
                                "tensor_shape": [int(c), int(h), int(w), int(d)],
                                "crop_size": list(size),
                                "half_sizes": [float(half_h), float(half_w), float(half_d)],
                                "starts": [int(start_h), int(start_w), int(start_d)],
                                "origins": list(origins),
                                "src_ranges": [[int(r[0]), int(r[1])] for r in src_ranges],
                                "dst_ranges": [[int(r[0]), int(r[1])] for r in dst_ranges],
                            },
                            "timestamp": int(time.time() * 1000),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass

    patch = tensor.new_zeros((c, size[0], size[1], size[2]))
    if all(r[1] - r[0] > 0 for r in src_ranges):
        patch[
            :,
            dst_ranges[0][0] : dst_ranges[0][1],
            dst_ranges[1][0] : dst_ranges[1][1],
            dst_ranges[2][0] : dst_ranges[2][1],
        ] = tensor[
            :,
            src_ranges[0][0] : src_ranges[0][1],
            src_ranges[1][0] : src_ranges[1][1],
            src_ranges[2][0] : src_ranges[2][1],
        ]

    if squeeze:
        patch = patch.squeeze(0)
    if return_origin:
        return patch, tuple(origins)
    return patch


def paste_patch_to_volume(
    tensor: torch.Tensor,
    origin: Sequence[int],
    full_shape: Sequence[int],
    debug_sample_idx: int = -1,
):
    """Paste a cropped patch back into a zero volume of shape (C, H, W, D) or (H, W, D)."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dims, got {tensor.shape}")

    full_shape = _to_3tuple(full_shape)
    full = tensor.new_zeros((tensor.shape[0],) + full_shape)
    y0, x0, z0 = [int(max(0, o)) for o in origin]
    y1 = min(y0 + tensor.shape[1], full_shape[0])
    x1 = min(x0 + tensor.shape[2], full_shape[1])
    z1 = min(z0 + tensor.shape[3], full_shape[2])

    if debug_sample_idx == 0:
        import json
        import time
        import os

        log_dir = os.path.join(os.getcwd(), ".cursor")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "debug.log")
        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "coord-check",
                            "hypothesisId": "H5",
                            "location": "coord_utils.py:paste_patch_to_volume",
                            "message": "Paste patch to volume - coordinate mapping",
                            "data": {
                                "input_origin": list(origin),
                                "tensor_shape": list(tensor.shape),
                                "full_shape": list(full_shape),
                                "paste_coords": {
                                    "y0": int(y0),
                                    "y1": int(y1),
                                    "x0": int(x0),
                                    "x1": int(x1),
                                    "z0": int(z0),
                                    "z1": int(z1),
                                },
                                "paste_size": [int(y1 - y0), int(x1 - x0), int(z1 - z0)],
                            },
                            "timestamp": int(time.time() * 1000),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass

    if y1 <= y0 or x1 <= x0 or z1 <= z0:
        return full.squeeze(0) if squeeze else full
    fy = y1 - y0
    fx = x1 - x0
    fz = z1 - z0
    full[:, y0:y1, x0:x1, z0:z1] = tensor[:, :fy, :fx, :fz]
    return full.squeeze(0) if squeeze else full


def compute_tumor_center(mask: torch.Tensor) -> Tuple[float, float, float]:
    """Return foreground center of mass; if empty, return volume center."""
    fg = (mask > 0).nonzero(as_tuple=False)
    if fg.numel() == 0:
        h, w, d = mask.shape
        return (h / 2.0, w / 2.0, d / 2.0)
    center = fg.float().mean(dim=0)
    cy, cx, cz = center.tolist()
    return float(cy), float(cx), float(cz)


__all__ = [
    "get_normalized_coord_map",
    "get_hybrid_coord_map",
    "get_coord_map",
    "resize_volume",
    "crop_volume_with_center",
    "paste_patch_to_volume",
    "compute_tumor_center",
]


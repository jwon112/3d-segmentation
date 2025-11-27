"""
MobileUNETR 기반 ROI detector builders.
"""

from typing import Sequence, Tuple

from models.mobileunetr_3d import MobileUNETR_3D


def build_roi_mobileunetr3d(
    *,
    img_size: Sequence[int] = (64, 64, 64),
    patch_size: Sequence[int] = (2, 2, 2),
    in_channels: int = 7,
    out_channels: int = 2,
) -> MobileUNETR_3D:
    """
    Create a MobileUNETR_3D instance configured for ROI detection.
    """

    def _to_tuple(value: Sequence[int]) -> Tuple[int, int, int]:
        if len(value) != 3:
            raise ValueError(f"img_size/patch_size must be length 3, got {value}")
        return int(value[0]), int(value[1]), int(value[2])

    image_size = _to_tuple(img_size)
    patch = _to_tuple(patch_size)
    return MobileUNETR_3D(
        image_size=image_size,
        patch_size=patch,
        in_channels=in_channels,
        out_channels=out_channels,
    )


__all__ = ["build_roi_mobileunetr3d"]



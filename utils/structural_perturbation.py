import random
from typing import Sequence

import torch
import torch.nn.functional as F


def _normalize_roi_size(roi_size: Sequence[int] | int) -> tuple[int, int, int]:
    if isinstance(roi_size, int):
        return (roi_size, roi_size, roi_size)
    if len(roi_size) != 3:
        raise ValueError("roi_size must be an int or a sequence of length 3.")
    return tuple(int(v) for v in roi_size)


def _sample_centers(mask: torch.Tensor, num_regions: int) -> torch.Tensor:
    coords = torch.nonzero(mask > 0, as_tuple=False)
    if coords.numel() == 0:
        return coords

    num_regions = min(num_regions, coords.shape[0])
    select_ids = torch.randperm(coords.shape[0], device=coords.device)[:num_regions]
    return coords[select_ids]


def build_pseudo_break_view(
    image: torch.Tensor,
    vessel_mask: torch.Tensor,
    num_regions: int = 6,
    roi_size: Sequence[int] | int = (16, 16, 16),
    break_strength: float = 0.85,
) -> torch.Tensor:
    """
    Build a complementary pseudo-break view by replacing vessel-centered cuboids
    with a smoothed version of the input, encouraging the peer network to
    preserve structure under partial vessel disruptions.
    """
    if image.ndim != 5:
        raise ValueError("image must be a 5D tensor shaped [B, C, D, H, W].")

    roi_d, roi_h, roi_w = _normalize_roi_size(roi_size)
    blurred = F.avg_pool3d(
        image,
        kernel_size=(max(3, roi_d // 2 * 2 + 1), max(3, roi_h // 2 * 2 + 1), max(3, roi_w // 2 * 2 + 1)),
        stride=1,
        padding=(max(1, roi_d // 2), max(1, roi_h // 2), max(1, roi_w // 2)),
    )
    perturbed = image.clone()

    for batch_idx in range(image.shape[0]):
        centers = _sample_centers(vessel_mask[batch_idx, 0], num_regions)
        if centers.numel() == 0:
            continue

        for center in centers:
            d, h, w = [int(v.item()) for v in center]
            d0 = max(0, d - roi_d // 2)
            d1 = min(image.shape[2], d0 + roi_d)
            h0 = max(0, h - roi_h // 2)
            h1 = min(image.shape[3], h0 + roi_h)
            w0 = max(0, w - roi_w // 2)
            w1 = min(image.shape[4], w0 + roi_w)

            source_patch = image[batch_idx : batch_idx + 1, :, d0:d1, h0:h1, w0:w1]
            blur_patch = blurred[batch_idx : batch_idx + 1, :, d0:d1, h0:h1, w0:w1]

            local_mask = vessel_mask[batch_idx : batch_idx + 1, :, d0:d1, h0:h1, w0:w1].float()
            region_gate = (local_mask > 0).float()
            mixed_patch = source_patch * (1.0 - break_strength * region_gate) + blur_patch * (break_strength * region_gate)
            perturbed[batch_idx : batch_idx + 1, :, d0:d1, h0:h1, w0:w1] = mixed_patch

    return perturbed


def maybe_apply_random_flip(image: torch.Tensor, probability: float = 0.5) -> torch.Tensor:
    flipped = image
    for dim in (-1, -2, -3):
        if random.random() < probability:
            flipped = torch.flip(flipped, dims=[dim])
    return flipped

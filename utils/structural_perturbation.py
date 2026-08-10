from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    result = mask.float()
    for _ in range(iterations):
        result = F.max_pool3d(result, kernel_size=3, stride=1, padding=1)
    return result


def _inpaint_volume(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply Telea inpainting independently to the three orthogonal planes."""
    image_np = image.detach().float().cpu().numpy()
    mask_np = (mask.detach().cpu().numpy() > 0).astype(np.uint8) * 255
    estimates = []

    for axis in range(3):
        volume = np.moveaxis(image_np, axis, 0)
        volume_mask = np.moveaxis(mask_np, axis, 0)
        restored = np.empty_like(volume, dtype=np.float32)
        for index, (plane, plane_mask) in enumerate(zip(volume, volume_mask)):
            if plane_mask.any():
                restored[index] = cv2.inpaint(
                    plane.astype(np.float32), plane_mask, 3, cv2.INPAINT_TELEA
                )
            else:
                restored[index] = plane
        estimates.append(np.moveaxis(restored, 0, axis))

    inpainted = np.mean(estimates, axis=0)
    return torch.as_tensor(inpainted, dtype=image.dtype, device=image.device)


@torch.no_grad()
def build_inpainted_view(
    images: torch.Tensor,
    source_vessel_mask: torch.Tensor,
    dilation_iterations: int = 3,
) -> torch.Tensor:
    if images.ndim != 5 or source_vessel_mask.ndim != 5:
        raise ValueError("images and source_vessel_mask must be [B, C, D, H, W]")

    dilated = _dilate(source_vessel_mask, dilation_iterations)
    result = images.clone()
    for batch_index in range(images.shape[0]):
        for channel_index in range(images.shape[1]):
            result[batch_index, channel_index] = _inpaint_volume(
                images[batch_index, channel_index], dilated[batch_index, 0]
            )
    return result


def _gaussian_patch(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coordinate = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))


@torch.no_grad()
def build_pseudo_break_view(
    images: torch.Tensor,
    inpainted: torch.Tensor,
    teacher_vessel_mask: torch.Tensor,
    num_centers: int = 6,
    patch_size: int = 15,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Blend inpainted content at vessel-centred 2-D patches (paper Eqs. 5-7)."""
    if patch_size % 2 != 1:
        raise ValueError("patch_size must be odd")
    blend_mask = torch.zeros_like(teacher_vessel_mask)
    kernel = _gaussian_patch(patch_size, sigma, images.device, images.dtype)
    radius = patch_size // 2

    for batch_index in range(images.shape[0]):
        coordinates = torch.nonzero(teacher_vessel_mask[batch_index, 0] > 0, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        count = min(num_centers, coordinates.shape[0])
        coordinates = coordinates[
            torch.randperm(coordinates.shape[0], device=coordinates.device)[:count]
        ]
        for depth, height, width in coordinates.tolist():
            h0, h1 = max(0, height - radius), min(images.shape[3], height + radius + 1)
            w0, w1 = max(0, width - radius), min(images.shape[4], width + radius + 1)
            kh0, kw0 = h0 - (height - radius), w0 - (width - radius)
            local = kernel[kh0 : kh0 + h1 - h0, kw0 : kw0 + w1 - w0]
            blend_mask[batch_index, 0, depth, h0:h1, w0:w1] = torch.maximum(
                blend_mask[batch_index, 0, depth, h0:h1, w0:w1], local
            )

    return images + blend_mask * (inpainted - images)

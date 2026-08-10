from __future__ import annotations

import warnings

import numpy as np
import torch

try:
    import cripser as cr
    from gudhi.wasserstein import wasserstein_distance

    TOPOLOGY_BACKEND_AVAILABLE = True
except Exception:
    cr = None
    wasserstein_distance = None
    TOPOLOGY_BACKEND_AVAILABLE = False


def _critical_points(likelihood: np.ndarray, persistence_threshold: float):
    inverted = 1.0 - likelihood
    pd = cr.computePH(inverted, maxdim=1, location="birth")
    zero_dim = pd[pd[:, 0] == 0]
    if zero_dim.size == 0:
        empty_points = np.zeros((0, 2))
        return empty_points, empty_points, empty_points, np.array([], dtype=int), np.array([], dtype=int)

    diagram = zero_dim[:, 1:3].copy()
    diagram[:, 1] = np.minimum(diagram[:, 1], 1.0)
    births = zero_dim[:, 3:5].astype(int)
    deaths = zero_dim[:, 6:8].astype(int)
    persistence = np.abs(diagram[:, 1] - diagram[:, 0])
    valid_idx = np.where(persistence > persistence_threshold)[0]
    noise_idx = np.where(persistence <= persistence_threshold)[0]
    return diagram, births, deaths, valid_idx, noise_idx


def _matching(student_diagram: np.ndarray, teacher_diagram: np.ndarray):
    if student_diagram.shape[0] == 0:
        return np.array([], dtype=int), np.zeros((0, 2), dtype=int)
    if teacher_diagram.shape[0] == 0:
        return np.arange(student_diagram.shape[0]), np.zeros((0, 2), dtype=int)

    _, raw_match = wasserstein_distance(student_diagram, teacher_diagram, matching=True)
    raw_match = np.asarray(raw_match, dtype=int)
    drop_idx = raw_match[raw_match[:, 1] == -1, 0]
    paired = raw_match[(raw_match[:, 0] != -1) & (raw_match[:, 1] != -1)]
    return drop_idx, paired


def _slice_topology_loss(
    student_slice: torch.Tensor,
    teacher_slice: torch.Tensor,
    patch_size: int,
    persistence_threshold: float,
) -> torch.Tensor:
    student_np = student_slice.detach().cpu().numpy()
    teacher_np = teacher_slice.detach().cpu().numpy()

    weight_map = np.zeros_like(student_np, dtype=np.float32)
    ref_map = np.zeros_like(student_np, dtype=np.float32)

    height, width = student_np.shape
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            stu_patch = student_np[y : min(y + patch_size, height), x : min(x + patch_size, width)]
            tea_patch = teacher_np[y : min(y + patch_size, height), x : min(x + patch_size, width)]

            if stu_patch.min() >= 1.0 or stu_patch.max() <= 0.0:
                continue
            if tea_patch.min() >= 1.0 or tea_patch.max() <= 0.0:
                continue

            stu_dgm, stu_birth, stu_death, stu_valid, stu_noise = _critical_points(stu_patch, persistence_threshold)
            tea_dgm, _, _, tea_valid, _ = _critical_points(tea_patch, persistence_threshold)
            if stu_dgm.shape[0] == 0:
                continue

            student_match_space = stu_dgm[stu_valid]
            teacher_match_space = tea_dgm[tea_valid]
            to_remove, paired = _matching(student_match_space, teacher_match_space)

            resolved_remove = [int(index) for index in stu_noise]
            for idx in to_remove:
                global_idx = np.where(np.all(stu_dgm == student_match_space[idx], axis=1))[0]
                if global_idx.size > 0:
                    resolved_remove.append(int(global_idx[0]))

            resolved_pairs = []
            for stu_idx, tea_idx in paired:
                s_idx = np.where(np.all(stu_dgm == student_match_space[stu_idx], axis=1))[0]
                t_idx = np.where(np.all(tea_dgm == teacher_match_space[tea_idx], axis=1))[0]
                if s_idx.size > 0 and t_idx.size > 0:
                    resolved_pairs.append((int(s_idx[0]), int(t_idx[0])))

            for stu_idx, tea_idx in resolved_pairs:
                by, bx = stu_birth[stu_idx]
                dy, dx = stu_death[stu_idx]
                if 0 <= by < stu_patch.shape[0] and 0 <= bx < stu_patch.shape[1]:
                    weight_map[y + by, x + bx] = 1.0
                    ref_map[y + by, x + bx] = tea_dgm[tea_idx][0]
                if 0 <= dy < stu_patch.shape[0] and 0 <= dx < stu_patch.shape[1]:
                    weight_map[y + dy, x + dx] = 1.0
                    ref_map[y + dy, x + dx] = tea_dgm[tea_idx][1]

            for stu_idx in resolved_remove:
                by, bx = stu_birth[stu_idx]
                dy, dx = stu_death[stu_idx]
                if 0 <= by < stu_patch.shape[0] and 0 <= bx < stu_patch.shape[1]:
                    weight_map[y + by, x + bx] = 1.0
                    ref_map[y + by, x + bx] = stu_patch[dy, dx] if 0 <= dy < stu_patch.shape[0] and 0 <= dx < stu_patch.shape[1] else 1.0
                if 0 <= dy < stu_patch.shape[0] and 0 <= dx < stu_patch.shape[1]:
                    weight_map[y + dy, x + dx] = 1.0
                    ref_map[y + dy, x + dx] = stu_patch[by, bx] if 0 <= by < stu_patch.shape[0] and 0 <= bx < stu_patch.shape[1] else 0.0

    weight_tensor = torch.as_tensor(weight_map, dtype=student_slice.dtype, device=student_slice.device)
    ref_tensor = torch.as_tensor(ref_map, dtype=student_slice.dtype, device=student_slice.device)
    return (((student_slice * weight_tensor) - ref_tensor) ** 2).sum()


def topology_consistency_loss(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    patch_size: int = 96,
    persistence_threshold: float = 0.1,
    max_slices: int = 4,
) -> torch.Tensor:
    """
    Slice-wise topological consistency for 3D vessel segmentation.
    The input tensors are expected to be shaped [B, D, H, W].
    """
    if not TOPOLOGY_BACKEND_AVAILABLE:
        warnings.warn(
            "Topology backend is unavailable. Install `cripser` and `gudhi` to enable topological regularization.",
            stacklevel=2,
        )
        return student_prob.new_zeros(())

    if student_prob.ndim != 4 or teacher_prob.ndim != 4:
        raise ValueError("student_prob and teacher_prob must be shaped [B, D, H, W].")

    total_loss = student_prob.new_zeros(())
    slice_counter = 0

    for batch_idx in range(student_prob.shape[0]):
        depth = student_prob.shape[1]
        height = student_prob.shape[2]
        width = student_prob.shape[3]

        depth_ids = np.linspace(0, max(depth - 1, 0), num=min(max_slices, depth), dtype=int)
        height_ids = np.linspace(0, max(height - 1, 0), num=min(max_slices, height), dtype=int)
        width_ids = np.linspace(0, max(width - 1, 0), num=min(max_slices, width), dtype=int)

        for d_idx in depth_ids:
            total_loss = total_loss + _slice_topology_loss(
                student_prob[batch_idx, d_idx],
                teacher_prob[batch_idx, d_idx],
                patch_size,
                persistence_threshold,
            )
            slice_counter += 1

        for h_idx in height_ids:
            total_loss = total_loss + _slice_topology_loss(
                student_prob[batch_idx, :, h_idx, :],
                teacher_prob[batch_idx, :, h_idx, :],
                patch_size,
                persistence_threshold,
            )
            slice_counter += 1

        for w_idx in width_ids:
            total_loss = total_loss + _slice_topology_loss(
                student_prob[batch_idx, :, :, w_idx],
                teacher_prob[batch_idx, :, :, w_idx],
                patch_size,
                persistence_threshold,
            )
            slice_counter += 1

    if slice_counter == 0:
        return total_loss
    return total_loss / slice_counter

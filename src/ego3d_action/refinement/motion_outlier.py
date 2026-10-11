"""Robust frame-level motion outlier screening, stage P1.

The detector works on displacement per frame so missing observations do not
inflate apparent speed. It combines bounded out-and-back segment rejection
with a robust Hampel test for isolated one-frame spikes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError

Array = np.ndarray


@dataclass(frozen=True)
class MotionOutlierResult:
    """Validity and diagnostic masks for one 3D trajectory."""

    valid: Array
    outlier: Array
    block_outlier: Array
    hampel_outlier: Array


def rotation_matrix_valid(rotations: Array) -> Array:
    """Return a conservative validity mask for rotation matrices."""
    r = np.asarray(rotations, dtype=np.float64)
    finite = np.isfinite(r).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], r, np.eye(3))
    gram_error = np.max(
        np.abs(np.swapaxes(safe, -1, -2) @ safe - np.eye(3)), axis=(-2, -1)
    )
    det = np.linalg.det(safe)
    return finite & (gram_error < 0.1) & (det > 0.5) & (det < 1.5)


def _mad_threshold(values: Array, *, k: float, floor: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(floor)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return max(float(floor), median + float(k) * 1.4826 * mad) + 1e-12


def detect_motion_outliers(
    position: Array,
    valid: Array,
    *,
    rotation: Array | None = None,
    position_step_threshold: float = 0.15,
    rotation_step_threshold_deg: float = 45.0,
    position_recovery: float = 0.15,
    rotation_recovery_deg: float = 30.0,
    max_block: int = 15,
    hampel_k: float = 3.0,
    position_floor: float = 0.10,
    rotation_floor_deg: float = 30.0,
) -> MotionOutlierResult:
    """Screen one position track shaped [T,3] and optional [T,3,3] rotations.

    A pair of fixed-threshold step spikes is rejected as a block only when the
    motion returns near its entry pose within max_block frames. Isolated spikes
    use a robust median/MAD threshold. Smooth, consistently fast motion remains.
    """
    p = np.asarray(position, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 3:
        raise StageIOError(f"position must be [T, 3], got {p.shape}")
    if mask.shape != (p.shape[0],):
        raise StageIOError(f"valid must be [{p.shape[0]}], got {mask.shape}")
    if max_block < 1:
        raise StageIOError(f"max_block must be >= 1, got {max_block}")
    params = (
        position_step_threshold, position_recovery, position_floor,
        rotation_step_threshold_deg, rotation_recovery_deg, rotation_floor_deg,
        hampel_k,
    )
    if not np.isfinite(params).all() or min(params) < 0.0:
        raise StageIOError("outlier thresholds must be finite and non-negative")
    if rotation is not None:
        rot = np.asarray(rotation, dtype=np.float64)
        if rot.shape != (p.shape[0], 3, 3):
            raise StageIOError(f"rotation must be [{p.shape[0]}, 3, 3], got {rot.shape}")
        rot_valid = rotation_matrix_valid(rot)
    else:
        rot = None
        rot_valid = np.ones(p.shape[0], dtype=bool)

    effective = mask & np.isfinite(p).all(axis=1) & rot_valid
    indices = np.flatnonzero(effective)
    block_mask = np.zeros(p.shape[0], dtype=bool)
    hampel_mask = np.zeros(p.shape[0], dtype=bool)
    malformed_mask = mask & ~effective
    if indices.size < 3:
        return MotionOutlierResult(
            valid=effective,
            outlier=malformed_mask,
            block_outlier=block_mask,
            hampel_outlier=hampel_mask,
        )

    gaps = np.diff(indices).astype(np.float64)
    pos_steps = np.linalg.norm(np.diff(p[indices], axis=0), axis=1) / gaps
    if rot is not None:
        relative = np.swapaxes(rot[indices[:-1]], -1, -2) @ rot[indices[1:]]
        traces = np.trace(relative, axis1=-2, axis2=-1)
        rot_steps = np.degrees(
            np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
        ) / gaps
    else:
        rot_steps = np.zeros_like(pos_steps)

    pos_fixed = pos_steps > position_step_threshold
    rot_fixed = rot_steps > rotation_step_threshold_deg
    fixed = pos_fixed | rot_fixed
    events = np.flatnonzero(fixed)
    cursor = 0
    while cursor + 1 < events.size:
        first = int(events[cursor])
        second = int(events[cursor + 1])
        frame_span = int(indices[second + 1] - indices[first])
        rejected_count = second - first
        if frame_span <= max_block + 1 and rejected_count > 0:
            checks: list[bool] = []
            if np.any(pos_fixed[first : second + 1]):
                net_position = float(
                    np.linalg.norm(p[indices[second + 1]] - p[indices[first]])
                )
                checks.append(net_position < position_recovery)
            if np.any(rot_fixed[first : second + 1]) and rot is not None:
                net_rotation = rot[indices[first]].T @ rot[indices[second + 1]]
                angle = float(
                    np.degrees(
                        np.arccos(
                            np.clip((np.trace(net_rotation) - 1.0) / 2.0, -1.0, 1.0)
                        )
                    )
                )
                checks.append(angle < rotation_recovery_deg)
            if checks and all(checks):
                start_frame = int(indices[first + 1])
                stop_frame = int(indices[second + 1])
                block_mask[start_frame:stop_frame] = True
                cursor += 2
                continue
        cursor += 1

    pos_threshold = _mad_threshold(pos_steps, k=hampel_k, floor=position_floor)
    hampel_step = pos_steps > pos_threshold
    if rot is not None:
        rot_threshold = _mad_threshold(rot_steps, k=hampel_k, floor=rotation_floor_deg)
        hampel_step |= rot_steps > rot_threshold

    for middle in range(1, indices.size - 1):
        if indices[middle] != indices[middle - 1] + 1:
            continue
        if indices[middle + 1] != indices[middle] + 1:
            continue
        if hampel_step[middle - 1] and hampel_step[middle]:
            hampel_mask[indices[middle]] = True

    outlier = malformed_mask | block_mask | hampel_mask
    return MotionOutlierResult(
        valid=effective & ~outlier,
        outlier=outlier,
        block_outlier=block_mask,
        hampel_outlier=hampel_mask,
    )

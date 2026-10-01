"""Phase 6: short-gap pose interpolation (the reference pipeline's P2).

The reference system interpolates missing poses with SLERP; that formula
applies to *rotation* parameters. A prediction in this project's contract
carries 21 joint positions (its MANO fields are placeholders), so a gap is
filled by per-joint linear interpolation between the surrounding valid
frames - the same convention the window-seam blender already uses
(``hand/temporal_blend.py`` blends positions linearly and reserves SLERP for
rotations).

Only runs of at most ``max_gap`` missing hand-frames bounded by valid frames
on **both** sides are filled. Leading/trailing missing frames and longer gaps
stay missing: without two anchors the fill would be an extrapolation, and a
long gap describes a hand whose motion is unknown, not a hand at rest. Every
filled frame is reported through the separate ``interpolated`` mask (and the
``hand_interpolated`` trajectory field) so evaluation can always separate real
predictions from interpolated ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray

#: Longest missing run that is interpolated (0.4 s at 30 fps).
DEFAULT_MAX_GAP = 12


@dataclass(frozen=True)
class GapFillResult:
    """Short-gap interpolation of the hand trajectory."""

    joints_camera: Array  # [T, 2, 21, 3] with filled frames
    confidence: Array  # [T, 2] interpolated across filled frames
    valid: Array  # [T, 2] bool - anchors | interpolated
    interpolated: Array  # [T, 2] bool - True exactly on filled frames
    gaps_filled: int
    frames_filled: int
    frames_left_missing: int


def interpolate_hand_gaps(
    joints_camera: Array,
    valid: Array,
    confidence: Array,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
) -> GapFillResult:
    """Fill short missing runs in ``[T, 2, 21, 3]`` camera-space joints.

    A hand-frame is an anchor when ``valid`` is set *and* every joint is
    finite; a ``valid`` frame with non-finite joints is treated as missing,
    never as an interpolation endpoint. Each gap of ``1 .. max_gap`` frames
    between two anchors is filled per joint and per coordinate with the linear
    blend of its anchors; the confidence follows the same blend so downstream
    confidence-weighted stages treat a filled frame like its neighbours.

    Args:
        joints_camera: ``[T, 2, 21, 3]`` camera-space joints (NaN = missing).
        valid: ``[T, 2]`` validity mask from detection/tracking.
        confidence: ``[T, 2]`` detection confidences, interpolated as well.
        max_gap: longest missing run to fill; ``>= 1``.

    Returns:
        :class:`GapFillResult`; the input arrays are never modified.

    Raises:
        StageIOError: on shape mismatches or a non-positive ``max_gap``.
    """
    joints = np.asarray(joints_camera, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise StageIOError(f"joints_camera must be [T, 2, 21, 3], got {joints.shape}")
    if mask.shape != joints.shape[:2]:
        raise StageIOError(f"valid must be {joints.shape[:2]}, got {mask.shape}")
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.shape != joints.shape[:2]:
        raise StageIOError(f"confidence must be {joints.shape[:2]}, got {conf.shape}")
    if max_gap < 1:
        raise StageIOError(f"max_gap must be >= 1, got {max_gap}")

    out = joints.copy()
    out_conf = conf.copy()
    interpolated = np.zeros(joints.shape[:2], dtype=bool)
    gaps_filled = 0
    frames_filled = 0

    for hand in range(2):
        anchor = mask[:, hand] & np.isfinite(joints[:, hand]).all(axis=(1, 2))
        for start, end in _fillable_gaps(anchor):
            if end - start > max_gap:
                continue
            left, right = start - 1, end
            weights = (np.arange(start, end, dtype=np.float64) - left) / float(right - left)
            blend = (1.0 - weights)[:, None, None] * joints[left, hand][None] + weights[
                :, None, None
            ] * joints[right, hand][None]
            out[start:end, hand] = blend
            out_conf[start:end, hand] = (1.0 - weights) * conf[left, hand] + weights * conf[
                right, hand
            ]
            interpolated[start:end, hand] = True
            gaps_filled += 1
            frames_filled += end - start

    valid_out = mask | interpolated
    left_missing = int(np.count_nonzero(~valid_out))
    if frames_filled:
        logger.info(
            "gap fill (max_gap=%d): %d gaps / %d hand-frames interpolated, "
            "%d hand-frames stay missing",
            max_gap,
            gaps_filled,
            frames_filled,
            left_missing,
        )
    else:
        logger.info("gap fill (max_gap=%d): no fillable gaps", max_gap)

    return GapFillResult(
        joints_camera=out,
        confidence=out_conf,
        valid=valid_out,
        interpolated=interpolated,
        gaps_filled=gaps_filled,
        frames_filled=frames_filled,
        frames_left_missing=left_missing,
    )


def _fillable_gaps(anchor: Array) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` missing runs strictly between two anchors."""
    flags = np.asarray(anchor, dtype=bool).reshape(-1)
    if flags.size < 2 or not flags.any():
        return []
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2], strict=False)]
    return [(prev_end, next_start) for (_, prev_end), (next_start, _) in zip(runs, runs[1:], strict=False)]

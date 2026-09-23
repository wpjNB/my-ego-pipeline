"""Phase 6.2: bone-scale correction.

HaWoR's MANO shape drifts slightly from frame to frame (a finger measuring
91.2 mm, then 94.0 mm, then 90.8 mm) although a real hand keeps its bone
lengths. The reference system therefore computes a clip-level mean bone length
and limits the per-frame correction to **3.5 %**.

Correction is applied as a forward-kinematic rescale: walking the kinematic
tree from the wrist, every bone is scaled by a clamped ratio, so the hand shape
stays consistent while the root stays where the reconstruction put it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError
from ..hand.mano import JOINT_PARENTS, NUM_JOINTS, bone_pairs

logger = logging.getLogger(__name__)

Array = np.ndarray

DEFAULT_MAX_CORRECTION = 0.035


@dataclass(frozen=True)
class BoneScaleResult:
    """Result of :func:`correct_bone_scale`."""

    joints: Array  # [T, 2, 21, 3]
    ratios: Array  # [T, 2, 20] clamp applied per bone
    clip_mean_lengths: Array  # [2, 20] metres
    max_deviation_before: Array  # [2, 20]
    max_deviation_after: Array  # [2, 20]


def _bone_index_pairs() -> tuple[Array, Array]:
    pairs = bone_pairs()
    parents = np.array([p for p, _ in pairs], dtype=np.int64)
    children = np.array([c for _, c in pairs], dtype=np.int64)
    return parents, children


def clip_mean_bone_lengths(joints: Array, valid: Array, *, reference: str = "median") -> Array:
    """Clip-level reference bone lengths ``[2, 20]``.

    ``reference='median'`` is the default because a few mis-reconstructed
    frames should not drag the reference, while the spec's "clip-level mean" is
    available with ``reference='mean'``.
    """
    arr = np.asarray(joints, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if arr.ndim != 4 or arr.shape[1:] != (2, NUM_JOINTS, 3):
        raise StageIOError(f"joints must be [T, 2, 21, 3], got {arr.shape}")
    if mask.shape != arr.shape[:2]:
        raise StageIOError(f"valid must be {arr.shape[:2]}, got {mask.shape}")
    if reference not in {"median", "mean"}:
        raise StageIOError(f"reference must be 'median' or 'mean', got '{reference}'")

    parents, children = _bone_index_pairs()
    out = np.full((2, parents.shape[0]), np.nan, dtype=np.float64)
    for hand in range(2):
        frames = mask[:, hand]
        if not frames.any():
            logger.warning("hand %d has no valid frames; bone-length reference stays NaN", hand)
            continue
        vectors = arr[frames, hand][:, children, :] - arr[frames, hand][:, parents, :]
        lengths = np.linalg.norm(vectors, axis=-1)
        reducer = np.nanmedian if reference == "median" else np.nanmean
        out[hand] = reducer(lengths, axis=0)
    return out


def correct_bone_scale(
    joints: Array,
    valid: Array,
    *,
    max_correction: float = DEFAULT_MAX_CORRECTION,
    reference: str = "median",
) -> BoneScaleResult:
    """Rescale per-frame bones toward the clip-level reference lengths.

    Args:
        joints: ``[T, 2, 21, 3]`` camera- or world-space joints.
        valid: ``[T, 2]`` validity mask; invalid frames are copied through.
        max_correction: per-frame bound (spec: ``0.035``).
        reference: ``'median'`` (default) or ``'mean'`` clip reference.

    Returns:
        :class:`BoneScaleResult` with corrected joints and diagnostics.

    Raises:
        StageIOError: on shape mismatches or a negative bound.
    """
    arr = np.asarray(joints, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if arr.ndim != 4 or arr.shape[1:] != (2, NUM_JOINTS, 3):
        raise StageIOError(f"joints must be [T, 2, 21, 3], got {arr.shape}")
    if mask.shape != arr.shape[:2]:
        raise StageIOError(f"valid must be {arr.shape[:2]}, got {mask.shape}")
    if max_correction < 0.0:
        raise StageIOError(f"max_correction must be >= 0, got {max_correction}")

    clip_lengths = clip_mean_bone_lengths(arr, mask, reference=reference)
    parents, children = _bone_index_pairs()
    num_frames = arr.shape[0]

    out = arr.copy()
    ratios = np.ones((num_frames, 2, parents.shape[0]), dtype=np.float64)
    before = np.zeros((2, parents.shape[0]), dtype=np.float64)
    after = np.zeros((2, parents.shape[0]), dtype=np.float64)

    if not 0.0 <= max_correction <= 1.0:
        raise StageIOError(f"max_correction must lie in [0, 1], got {max_correction}")

    for hand in range(2):
        if not np.isfinite(clip_lengths[hand]).all():
            logger.warning(
                "hand %d: no usable bone reference; leaving its trajectory unchanged", hand
            )
            continue
        for frame in range(num_frames):
            if not mask[frame, hand]:
                continue
            points = arr[frame, hand]
            new_points = points.copy()
            for bone, (parent, child) in enumerate(zip(parents, children, strict=False)):
                target = clip_lengths[hand, bone]
                if not np.isfinite(target) or target <= 0.0:
                    continue
                vector = points[child] - points[parent]
                length = float(np.linalg.norm(vector))
                if length <= 1e-9:
                    continue
                ratio = float(np.clip(target / length, 1.0 - max_correction, 1.0 + max_correction))
                ratios[frame, hand, bone] = ratio
                new_points[child] = new_points[parent] + ratio * (
                    points[child] - points[parent]
                )
            out[frame, hand] = new_points

        # Diagnostics: worst relative deviation from the clip reference.
        frames = mask[:, hand]
        if frames.any():
            corrected = out[frames, hand]
            raw = arr[frames, hand]
            ref = clip_lengths[hand]
            before[hand] = np.max(
                np.abs(np.linalg.norm(raw[:, children] - raw[:, parents], axis=-1) / ref - 1.0),
                axis=0,
            )
            after[hand] = np.max(
                np.abs(np.linalg.norm(corrected[:, children] - corrected[:, parents], axis=-1) / ref - 1.0),
                axis=0,
            )

    logger.info(
        "bone-scale correction (bound %.1f%%): worst deviation %.2f%% -> %.2f%%",
        100.0 * max_correction,
        100.0 * float(np.nanmax(before)) if np.isfinite(before).any() else 0.0,
        100.0 * float(np.nanmax(after)) if np.isfinite(after).any() else 0.0,
    )
    return BoneScaleResult(
        joints=out,
        ratios=ratios,
        clip_mean_lengths=clip_lengths,
        max_deviation_before=before,
        max_deviation_after=after,
    )

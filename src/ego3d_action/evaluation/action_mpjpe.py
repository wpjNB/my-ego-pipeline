"""Action-MPJPE, the metric used by the reference benchmark.

The trajectory is cut into **1-second action chunks**. For a chunk starting at
frame ``t`` every subsequent frame inside the chunk is expressed in the camera
frame at ``t`` - for the prediction with the predicted pose, for the ground
truth with the ground-truth pose - and the mean joint distance is taken:

``E = mean_(t,h) [ mean_(i,j) ||p_hat[t,h,i,j] - p[t,h,i,j]||_2 ]``

Neither prediction nor ground truth is re-translated, re-rotated or re-scaled
before comparing: doing so would cancel exactly the camera-motion and
metric-scale errors this benchmark exists to measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray

NUM_HANDS = 2
NUM_JOINTS = 21
WRIST_JOINT = 0


@dataclass(frozen=True)
class ActionMPJPEResult:
    """Action-MPJPE and its breakdown."""

    action_mpjpe_mm: float
    num_chunks: int
    num_terms: int
    per_hand_mm: Array  # [2]
    wrist_mm: float
    depth_mm: float
    per_joint_mm: Array  # [21]
    chunk_indices: Array
    per_chunk_mm: Array
    joint_coverage: float  # fraction of (frame, hand, joint) terms that were comparable

    def as_dict(self) -> dict[str, float]:
        return {
            "action_mpjpe_mm": self.action_mpjpe_mm,
            "wrist_mm": self.wrist_mm,
            "depth_mm": self.depth_mm,
            "left_mm": float(self.per_hand_mm[0]),
            "right_mm": float(self.per_hand_mm[1]),
        }


def to_camera_frame(points_world: Array, rotation_c2w: Array, translation_c2w: Array) -> Array:
    """Express world points in the camera frame given by ``(R_c2w, t_c2w)``."""
    rel = np.asarray(points_world, dtype=np.float64) - np.asarray(translation_c2w, dtype=np.float64)
    return np.einsum("ij,...j->...i", np.asarray(rotation_c2w, dtype=np.float64).T, rel)


def safe_nanmean(values: Array, axis: int) -> Array:
    """``np.nanmean`` that returns ``NaN`` (quietly) for all-NaN slices.

    A reference trajectory may legitimately contain joints that are never
    observed (see the wrist-only HOT3D sample), and ``np.nanmean`` would emit a
    "Mean of empty slice" warning for every one of them.
    """
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, data, 0.0), axis=axis)
    return np.where(count > 0, total / np.where(count > 0, count, 1), np.nan)


def action_mpjpe(
    prediction_world: Array,
    ground_truth_world: Array,
    *,
    prediction_rotation_c2w: Array,
    prediction_translation_c2w: Array,
    ground_truth_rotation_c2w: Array,
    ground_truth_translation_c2w: Array,
    fps: float,
    chunk_seconds: float = 1.0,
    prediction_valid: Array | None = None,
    ground_truth_valid: Array | None = None,
) -> ActionMPJPEResult:
    """Compute Action-MPJPE (millimetres) between two world-frame trajectories.

    Args:
        prediction_world: ``[T, 2, 21, 3]`` predicted joints, metres.
        ground_truth_world: ``[T, 2, 21, 3]`` reference joints, metres.
        prediction_rotation_c2w / prediction_translation_c2w: predicted camera.
        ground_truth_rotation_c2w / ground_truth_translation_c2w: reference camera.
        fps: frame rate that defines the 1-second chunk length.
        chunk_seconds: chunk duration (spec: ``1.0``).
        prediction_valid / ground_truth_valid: optional ``[T, 2]`` masks;
            frames missing in either sequence are excluded from the mean, but
            still counted by the coverage metric.

    Raises:
        StageIOError: on shape mismatches, a non-positive fps, a chunk longer
            than the clip, or when no overlapping valid term exists.
    """
    pred = np.asarray(prediction_world, dtype=np.float64)
    gt = np.asarray(ground_truth_world, dtype=np.float64)
    if pred.shape != gt.shape:
        raise StageIOError(f"prediction shape {pred.shape} != ground truth shape {gt.shape}")
    if pred.ndim != 4 or pred.shape[1:] != (NUM_HANDS, NUM_JOINTS, 3):
        raise StageIOError(f"trajectories must be [T, 2, 21, 3], got {pred.shape}")
    total = pred.shape[0]
    if total == 0:
        raise StageIOError("cannot evaluate an empty trajectory")
    if fps <= 0.0:
        raise StageIOError(f"fps must be positive, got {fps}")

    pr = np.asarray(prediction_rotation_c2w, dtype=np.float64)
    pt = np.asarray(prediction_translation_c2w, dtype=np.float64)
    gr = np.asarray(ground_truth_rotation_c2w, dtype=np.float64)
    gtt = np.asarray(ground_truth_translation_c2w, dtype=np.float64)
    for name, arr, shape in (
        ("prediction_rotation_c2w", pr, (total, 3, 3)),
        ("prediction_translation_c2w", pt, (total, 3)),
        ("ground_truth_rotation_c2w", gr, (total, 3, 3)),
        ("ground_truth_translation_c2w", gtt, (total, 3)),
    ):
        if arr.shape != shape:
            raise StageIOError(f"{name} must be {shape}, got {arr.shape}")

    chunk = max(1, int(round(fps * chunk_seconds)))
    if chunk > total:
        raise StageIOError(f"chunk length {chunk} exceeds the clip length {total}")

    # Finiteness is evaluated per *joint*, not per hand-frame: a reference that
    # only carries the wrist (the bundled HOT3D sample, which has no MANO mesh
    # model to derive finger positions) must still produce a meaningful number
    # instead of being discarded wholesale.
    finite_pred = np.isfinite(pred).all(axis=-1)  # [T, 2, 21] - per joint, not per coordinate
    finite_gt = np.isfinite(gt).all(axis=-1)
    if prediction_valid is not None:
        hand_mask = np.asarray(prediction_valid, dtype=bool)
        if hand_mask.shape != (total, NUM_HANDS):
            raise StageIOError(f"prediction_valid must be [{total}, 2], got {hand_mask.shape}")
        finite_pred = finite_pred & hand_mask[:, :, None]
    if ground_truth_valid is not None:
        hand_mask = np.asarray(ground_truth_valid, dtype=bool)
        if hand_mask.shape != (total, NUM_HANDS):
            raise StageIOError(f"ground_truth_valid must be [{total}, 2], got {hand_mask.shape}")
        finite_gt = finite_gt & hand_mask[:, :, None]
    usable = finite_pred & finite_gt  # [T, 2, 21]

    starts = np.arange(0, total - chunk + 1, dtype=np.int64)
    per_frame_error = np.full(
        (starts.size, chunk, NUM_HANDS, NUM_JOINTS), np.nan, dtype=np.float64
    )

    for slot, start in enumerate(starts):
        frames = np.arange(start, start + chunk, dtype=np.int64)
        pred_cam = to_camera_frame(pred[frames], pr[start], pt[start])
        gt_cam = to_camera_frame(gt[frames], gr[start], gtt[start])
        diff = np.linalg.norm(pred_cam - gt_cam, axis=-1)
        mask = usable[frames]
        per_frame_error[slot] = np.where(mask, diff, np.nan)

    if not np.isfinite(per_frame_error).any():
        raise StageIOError(
            "no overlapping valid (frame, hand, joint) terms between prediction and ground truth"
        )

    per_hand = np.array(
        [safe_nanmean(per_frame_error[..., hand, :], axis=None) for hand in range(NUM_HANDS)],
        dtype=np.float64,
    )
    per_joint = np.array(
        [safe_nanmean(per_frame_error[..., joint], axis=None) for joint in range(NUM_JOINTS)],
        dtype=np.float64,
    )
    per_chunk = np.array(
        [safe_nanmean(per_frame_error[slot], axis=None) for slot in range(starts.size)],
        dtype=np.float64,
    )

    depth_error = np.abs(pred[..., 2] - gt[..., 2])
    depth_mm = float(1000.0 * np.mean(depth_error[usable])) if usable.any() else float("nan")

    result = ActionMPJPEResult(
        action_mpjpe_mm=float(1000.0 * safe_nanmean(per_frame_error, axis=None)),
        num_chunks=int(starts.size),
        num_terms=int(np.count_nonzero(np.isfinite(per_frame_error))),
        per_hand_mm=1000.0 * per_hand,
        wrist_mm=float(1000.0 * per_joint[WRIST_JOINT]),
        depth_mm=depth_mm,
        per_joint_mm=1000.0 * per_joint,
        chunk_indices=starts,
        per_chunk_mm=1000.0 * per_chunk,
        joint_coverage=float(np.mean(usable)),
    )
    logger.info(
        "Action-MPJPE %.4f mm over %d chunks (%d terms); wrist %.3f mm",
        result.action_mpjpe_mm,
        result.num_chunks,
        result.num_terms,
        result.wrist_mm,
    )
    return result

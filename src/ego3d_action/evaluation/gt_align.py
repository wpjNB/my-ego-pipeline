"""Per-clip GT translation alignment (the mirror's hand-eye offset).

The HOT3D-mirror hand GT is rigidly misplaced relative to its own images: the
PRED-GT wrist gap is dominated by a *constant* offset in the camera frame
(measured 2026-10-01 on ep000/ep003: 1.7-6.9 cm constant vs 2.4-3.7 cm
per-frame residual - on ep000's right hand 97 % of the gap is constant). A
constant camera-frame offset is the signature of a hand-eye/rig calibration
error, not of estimation error, which should vary over time.

:func:`camera_frame_offset` estimates that offset per hand (median over all
valid (frame, joint) terms of prediction-minus-GT in the *camera frame*), and
:func:`apply_offset` bakes it into a GT world trajectory so downstream
metrics/renders measure time-varying agreement. The alignment never changes
GT's internal shape (distances between its joints are preserved) and leaves
the camera trajectory untouched, so camera error is unaffected.

Caveat, stated where the flag is offered: with the alignment applied, the
constant part of the placement error is removed *by construction* - the
resulting numbers measure time-varying agreement, not absolute placement.
"""

from __future__ import annotations

import numpy as np

from ..errors import StageIOError

Array = np.ndarray


def _to_camera(points_world: Array, rotation_c2w: Array, translation_c2w: Array) -> Array:
    t = np.asarray(translation_c2w, dtype=np.float64)
    if t.ndim == 1:
        t = t[None, None, :]
    elif t.ndim == 2:
        t = t[:, None, None, :]
    rel = np.asarray(points_world, dtype=np.float64) - t
    return np.einsum("njk,n...j->n...k", np.asarray(rotation_c2w, dtype=np.float64), rel)


def camera_frame_offset(
    pred_world: Array,
    gt_world: Array,
    *,
    pred_rotation_c2w: Array,
    pred_translation_c2w: Array,
    gt_rotation_c2w: Array,
    gt_translation_c2w: Array,
    pred_valid: Array | None = None,
    gt_valid: Array | None = None,
) -> Array:
    """Per-hand constant offset ``[2, 3]`` (metres) between pred and GT hands.

    Both trajectories are expressed in the **reference camera frame** (the GT
    poses): estimating the offset in each trajectory's own camera frame would
    absorb the camera-trajectory error, which varies over time and would
    poison a constant-offset model. The offset is the median over valid frames
    of the **wrist** (joint 0) difference - the one landmark whose identity is
    unambiguous. An all-joint median would instead centre the whole hand and
    silently absorb hand-shape differences (bone lengths, articulation) that
    the metric is supposed to measure: on ep000 that inflated the estimated
    offset to 12-14 cm along z while the wrist-only offset is ~2-7 cm.
    Returns zeros for a hand with no overlapping valid frames.
    """
    del pred_rotation_c2w, pred_translation_c2w  # both use the GT camera
    pred = np.asarray(pred_world, dtype=np.float64)
    gt = np.asarray(gt_world, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[1] != 2:
        raise StageIOError(f"trajectories must share [T, 2, J, 3] shapes, got {pred.shape} vs {gt.shape}")
    gr = np.asarray(gt_rotation_c2w, dtype=np.float64)
    gtt = np.asarray(gt_translation_c2w, dtype=np.float64)
    pred_wrist = _to_camera(pred[:, :, 0:1], gr, gtt)
    gt_wrist = _to_camera(gt[:, :, 0:1], gr, gtt)
    usable = np.isfinite(pred_wrist).all(axis=-1) & np.isfinite(gt_wrist).all(axis=-1)
    if pred_valid is not None:
        usable &= np.asarray(pred_valid, dtype=bool)[:, :, None]
    if gt_valid is not None:
        usable &= np.asarray(gt_valid, dtype=bool)[:, :, None]

    offsets = np.zeros((2, 3), dtype=np.float64)
    for hand in range(2):
        terms = (pred_wrist[:, hand] - gt_wrist[:, hand])[usable[:, hand]]
        if terms.shape[0] == 0:
            continue
        offsets[hand] = np.median(terms, axis=0)
    return offsets


def apply_offset(
    gt_world: Array,
    gt_rotation_c2w: Array,
    gt_translation_c2w: Array,
    offsets: Array,
) -> Array:
    """Bake per-hand camera-frame ``offsets`` into a GT world trajectory.

    ``G'_world(t) = R(t) @ (G_cam(t) + offset) + t(t)`` - the offset rides the
    camera (hand-eye style), so it stays constant in the camera frame while the
    camera moves. Joint-to-joint distances are untouched.
    """
    gt = np.asarray(gt_world, dtype=np.float64)
    rot = np.asarray(gt_rotation_c2w, dtype=np.float64)
    trans = np.asarray(gt_translation_c2w, dtype=np.float64)
    off = np.asarray(offsets, dtype=np.float64)
    if gt.shape[:2] != (len(rot), 2) or off.shape != (2, 3):
        raise StageIOError(f"shapes mismatch: gt {gt.shape}, R {rot.shape}, offsets {off.shape}")
    gt_cam = _to_camera(gt, rot, trans)
    gt_cam = gt_cam + off[None, :, None, :]
    world = np.einsum("nik,n...k->n...i", rot, gt_cam)
    return world + trans[:, None, None, :]

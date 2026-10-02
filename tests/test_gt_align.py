"""GT translation alignment: offset estimation and application."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.evaluation.gt_align import apply_offset, camera_frame_offset
from ego3d_action.errors import StageIOError


def _camera_poses(seed: int, total: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """A rotating, translating ego camera (c2w)."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.normal(scale=0.01, size=(total, 3)), axis=0)
    angles = np.cumsum(rng.normal(scale=0.05, size=total))
    rot = []
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        rot.append(np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]))
    return np.stack(rot), t


def _hands(rot: np.ndarray, t: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Hands that sit at a fixed camera-frame offset from the camera path."""
    total = rot.shape[0]
    cam_local = np.array([0.1, -0.2, 0.4]) + offset
    world = np.einsum("njk,k->nj", rot, cam_local) + t
    return np.broadcast_to(world[:, None, None, :], (total, 2, 21, 3)).copy()


def test_offset_recovery_under_camera_motion() -> None:
    """A rigid camera-frame offset must be recovered exactly despite motion."""
    rot, t = _camera_poses(0)
    true_offset = np.array([0.03, -0.05, 0.02])
    gt = _hands(rot, t, np.zeros(3))
    pred = _hands(rot, t, true_offset)
    est = camera_frame_offset(
        pred, gt,
        pred_rotation_c2w=rot, pred_translation_c2w=t,
        gt_rotation_c2w=rot, gt_translation_c2w=t,
    )
    assert np.allclose(est, true_offset[None, :].repeat(2, axis=0), atol=1e-9)


def test_apply_offset_is_shape_preserving_and_rides_the_camera() -> None:
    rot, t = _camera_poses(1)
    offset = np.array([[0.02, 0.0, -0.04], [-0.01, 0.03, 0.0]])
    gt = _hands(rot, t, np.zeros(3))
    aligned = apply_offset(gt, rot, t, offset)
    # alignment moves every joint by exactly the camera-frame offset, so
    # joint-to-joint distances are untouched
    d_orig = np.linalg.norm(gt[:, 0, 5] - gt[:, 0, 20], axis=-1)
    d_al = np.linalg.norm(aligned[:, 0, 5] - aligned[:, 0, 20], axis=-1)
    assert np.allclose(d_orig, d_al, atol=1e-12)
    # ... and in the camera frame the aligned GT sits at GT+offset exactly
    from ego3d_action.evaluation.gt_align import _to_camera

    cam = _to_camera(aligned, rot, t)
    base = _to_camera(gt, rot, t)
    assert np.allclose(cam - base, offset[None, :, None, :], atol=1e-12)


def test_round_trip_alignment_cancels_the_gap() -> None:
    rot, t = _camera_poses(2)
    true_offset = np.array([[0.04, 0.01, -0.02], [0.0, -0.03, 0.05]])
    gt = _hands(rot, t, np.zeros(3))
    pred = _hands(rot, t, true_offset.sum(axis=0) * 0 + true_offset[0])  # same for both hands
    est = camera_frame_offset(
        pred, gt,
        pred_rotation_c2w=rot, pred_translation_c2w=t,
        gt_rotation_c2w=rot, gt_translation_c2w=t,
    )
    aligned = apply_offset(gt, rot, t, est)
    resid = np.linalg.norm(pred[:, 0] - aligned[:, 0], axis=-1)
    assert np.median(resid) < 1e-9


def test_offset_is_zero_without_overlap() -> None:
    rot, t = _camera_poses(3)
    gt = _hands(rot, t, np.zeros(3))
    pred = _hands(rot, t, np.array([0.05, 0.0, 0.0]))
    no_overlap = np.zeros((rot.shape[0], 2), dtype=bool)
    est = camera_frame_offset(
        pred, gt,
        pred_rotation_c2w=rot, pred_translation_c2w=t,
        gt_rotation_c2w=rot, gt_translation_c2w=t,
        pred_valid=no_overlap, gt_valid=no_overlap,
    )
    assert np.all(est == 0.0)


def test_shape_mismatch_raises() -> None:
    rot, t = _camera_poses(4)
    gt = _hands(rot, t, np.zeros(3))
    with pytest.raises(StageIOError):
        camera_frame_offset(
            gt[:, :, :10], gt,
            pred_rotation_c2w=rot, pred_translation_c2w=t,
            gt_rotation_c2w=rot, gt_translation_c2w=t,
        )

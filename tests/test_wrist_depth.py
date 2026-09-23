"""Phase 6.3: ray-constrained wrist-depth optimisation."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.geometry.transforms import project_points
from ego3d_action.refinement.wrist_depth import (
    CONFIDENCE_POWER,
    confidence_weights,
    optimize_wrist_depth,
)


def make_intrinsics(num_frames: int) -> np.ndarray:
    k = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    return np.broadcast_to(k, (num_frames, 3, 3)).copy()


def make_joints(wrist_track: np.ndarray) -> np.ndarray:
    """Build a synthetic hand whose wrist follows ``wrist_track`` [T, 3]."""
    offsets = np.zeros((21, 3))
    offsets[:, 0] = np.linspace(0.0, 0.08, 21)
    offsets[:, 1] = np.linspace(0.0, 0.02, 21)
    return wrist_track[:, None, :] + offsets[None, :, :]


def test_projection_is_preserved_exactly() -> None:
    total = 40
    t = np.arange(total, dtype=np.float64)
    wrist = np.stack([0.05 * np.sin(0.3 * t), 0.03 * np.cos(0.2 * t), 1.0 + 0.2 * np.sin(0.5 * t)], axis=-1)
    joints = make_joints(wrist)[:, None, :, :].repeat(2, axis=1)
    intrinsics = make_intrinsics(total)
    confidence = np.full((total, 2), 0.9)
    valid = np.ones((total, 2), dtype=bool)

    result = optimize_wrist_depth(joints, intrinsics, confidence, valid)
    before = project_points(intrinsics, joints[:, 0, 0, :])
    after = project_points(intrinsics, result.joints_camera[:, 0, 0, :])
    assert np.allclose(before, after, atol=1e-9)


def test_intra_hand_geometry_is_preserved() -> None:
    total = 25
    t = np.arange(total, dtype=np.float64)
    wrist = np.stack([0.02 * t, np.zeros(total), 1.2 + 0.3 * np.sin(0.4 * t)], axis=-1)
    joints = make_joints(wrist)[:, None, :, :].repeat(2, axis=1)
    intrinsics = make_intrinsics(total)
    confidence = np.full((total, 2), 0.8)
    valid = np.ones((total, 2), dtype=bool)

    result = optimize_wrist_depth(joints, intrinsics, confidence, valid)
    offsets_before = joints[:, 0] - joints[:, 0, 0:1, :]
    offsets_after = result.joints_camera[:, 0] - result.joints_camera[:, 0, 0:1, :]
    assert np.allclose(offsets_before, offsets_after, atol=1e-9)


def test_high_confidence_frames_move_less_than_low_confidence_frames() -> None:
    total = 60
    t = np.arange(total, dtype=np.float64)
    clean_depth = 1.5 + 0.1 * np.sin(0.4 * t)
    noisy_depth = clean_depth.copy()
    rng = np.random.default_rng(0)

    confidence = np.full((total, 2), 0.9)
    low = slice(20, 40)
    confidence[low] = 0.2
    noisy_depth[low] += rng.normal(0.0, 0.08, size=20)

    wrist = np.stack([np.zeros(total), np.zeros(total), noisy_depth], axis=-1)
    joints = make_joints(wrist)[:, None, :, :].repeat(2, axis=1)
    intrinsics = make_intrinsics(total)
    valid = np.ones((total, 2), dtype=bool)

    result = optimize_wrist_depth(joints, intrinsics, confidence, valid, lam=0.2)
    moved = np.abs(result.depth_after[:, 0] - result.depth_before[:, 0])
    assert np.mean(moved[low]) > np.mean(moved[:20]) * 3.0


def test_noisy_depth_becomes_smoother() -> None:
    total = 80
    t = np.arange(total, dtype=np.float64)
    clean = 1.5 + 0.1 * np.sin(0.4 * t)
    rng = np.random.default_rng(1)
    noisy = clean + rng.normal(0.0, 0.05, size=total)

    wrist = np.stack([np.zeros(total), np.zeros(total), noisy], axis=-1)
    joints = make_joints(wrist)[:, None, :, :].repeat(2, axis=1)
    intrinsics = make_intrinsics(total)
    confidence = np.full((total, 2), 0.9)
    valid = np.ones((total, 2), dtype=bool)

    result = optimize_wrist_depth(joints, intrinsics, confidence, valid, lam=0.2)

    def roughness(values: np.ndarray) -> float:
        return float(np.mean(np.abs(np.diff(values, n=2))))

    error_before = float(np.mean(np.abs(result.depth_before[:, 0] - clean)))
    error_after = float(np.mean(np.abs(result.depth_after[:, 0] - clean)))

    assert roughness(result.depth_after[:, 0]) < roughness(result.depth_before[:, 0])
    assert error_after < 0.8 * error_before, (
        f"depth optimisation did not reduce the error against the clean track "
        f"({error_before:.4f} -> {error_after:.4f})"
    )


def test_lambda_zero_keeps_original_depth() -> None:
    total = 30
    rng = np.random.default_rng(2)
    depth = 1.0 + rng.normal(0.0, 0.05, size=total)
    wrist = np.stack([np.zeros(total), np.zeros(total), depth], axis=-1)
    joints = make_joints(wrist)[:, None, :, :].repeat(2, axis=1)
    result = optimize_wrist_depth(
        joints, make_intrinsics(total), np.full((total, 2), 0.9), np.ones((total, 2), dtype=bool), lam=0.0
    )
    assert np.allclose(result.depth_after[:, 0], depth, atol=1e-8)


def test_invalid_frames_are_untouched() -> None:
    total = 20
    rng = np.random.default_rng(3)
    depth = 1.2 + rng.normal(0.0, 0.02, size=total)
    wrist = np.stack([np.zeros(total), np.zeros(total), depth], axis=-1)
    joints = make_joints(wrist)[:, None, :, :].repeat(2, axis=1)
    valid = np.ones((total, 2), dtype=bool)
    valid[5:8] = False

    result = optimize_wrist_depth(
        joints, make_intrinsics(total), np.full((total, 2), 0.9), valid
    )
    assert np.allclose(result.joints_camera[5:8], joints[5:8])
    assert not np.isfinite(result.depth_after[5:8, 0]).any()


def test_gaps_split_the_optimisation_into_segments() -> None:
    total = 20
    rng = np.random.default_rng(4)
    depth = 1.2 + rng.normal(0.0, 0.02, size=total)
    wrist = np.stack([np.zeros(total), np.zeros(total), depth], axis=-1)
    joints = make_joints(wrist)[:, None, :, :].repeat(2, axis=1)
    valid = np.ones((total, 2), dtype=bool)
    valid[10, 0] = False

    result = optimize_wrist_depth(
        joints, make_intrinsics(total), np.full((total, 2), 0.9), valid
    )
    # Hand 0 is split into two runs, hand 1 stays a single run.
    assert result.segments == 3


def test_confidence_weights_follow_documented_formula() -> None:
    confidence = np.array([[0.9, 0.3], [0.9, 0.9]])
    valid = np.ones((2, 2), dtype=bool)
    weights = confidence_weights(confidence, valid)
    median = 0.9
    assert weights[0, 0] == pytest.approx(1.0)
    expected_low = np.clip(0.3 / median, 0.5, 1.5) ** CONFIDENCE_POWER
    assert weights[0, 1] == pytest.approx(expected_low)
    assert weights[0, 1] == pytest.approx(0.5**CONFIDENCE_POWER)


def test_max_depth_change_is_respected() -> None:
    total = 40
    rng = np.random.default_rng(5)
    depth = 1.4 + rng.normal(0.0, 0.2, size=total)
    wrist = np.stack([np.zeros(total), np.zeros(total), depth], axis=-1)
    joints = make_joints(wrist)[:, None, :, :].repeat(2, axis=1)
    result = optimize_wrist_depth(
        joints,
        make_intrinsics(total),
        np.full((total, 2), 0.9),
        np.ones((total, 2), dtype=bool),
        max_depth_change=0.01,
    )
    delta = np.abs(result.depth_after[:, 0] - result.depth_before[:, 0])
    assert np.all(delta <= 0.01 + 1e-9)


def test_invalid_arguments_raise() -> None:
    joints = np.zeros((10, 2, 21, 3))
    intrinsics = make_intrinsics(10)
    with pytest.raises(StageIOError):
        optimize_wrist_depth(joints, intrinsics, np.ones((10, 2)), np.ones((10, 2), bool), lam=-1.0)
    with pytest.raises(StageIOError):
        optimize_wrist_depth(joints, make_intrinsics(5), np.ones((10, 2)), np.ones((10, 2), bool))
    with pytest.raises(StageIOError):
        optimize_wrist_depth(joints, intrinsics, np.ones((9, 2)), np.ones((10, 2), bool))
    with pytest.raises(StageIOError):
        optimize_wrist_depth(np.zeros((10, 2, 20, 3)), intrinsics, np.ones((10, 2)), np.ones((10, 2), bool))

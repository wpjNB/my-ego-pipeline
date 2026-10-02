"""Phase 6 P3: constant-velocity UKF + unscented RTS smoothing of hand joints."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.refinement.ukf_smooth import MIN_VALID, smooth_hand_joints


def make_trajectory(total: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """``(joints [T, 2, 21, 3], valid [T, 2])`` fully valid, wrist on a ramp."""
    joints = np.zeros((total, 2, 21, 3))
    joints[:, :, :, 0] = np.arange(total)[:, None, None] * 0.01
    joints[:, 1, :, :] += 1.0
    valid = np.ones((total, 2), dtype=bool)
    return joints, valid


def roughness(signal: np.ndarray) -> float:
    """Second-difference energy - the quantity the smoother minimises."""
    return float(np.sum(np.diff(signal, n=2, axis=0) ** 2))


def test_constant_trajectory_is_unchanged() -> None:
    joints, valid = make_trajectory()
    joints[:] = 0.5
    result = smooth_hand_joints(joints, valid)
    # A constant signal drives the robust observation scale to its floor, so
    # the filter trusts the observations completely.
    assert np.allclose(result.joints_camera, joints)


def test_clean_constant_velocity_is_preserved() -> None:
    joints, valid = make_trajectory()
    result = smooth_hand_joints(joints, valid)
    assert np.allclose(result.joints_camera, joints, atol=1e-4)


def test_noise_is_smoothed_without_collapsing() -> None:
    rng = np.random.default_rng(11)
    joints, valid = make_trajectory(total=120)
    noise = rng.standard_normal(joints.shape) * 0.02
    noisy = joints + noise

    result = smooth_hand_joints(noisy, valid)

    assert roughness(result.joints_camera[:, 0, 0, :]) < 0.2 * roughness(noisy[:, 0, 0, :])
    # The smoother follows the signal; it must not flatten it to a line.
    drift = np.abs(result.joints_camera - joints).mean()
    assert drift < 0.05


def test_missing_frames_are_never_written() -> None:
    joints, valid = make_trajectory()
    joints[10:14, 0] = np.nan
    valid[10:14, 0] = False
    snapshot = joints.copy()

    result = smooth_hand_joints(joints, valid)

    assert np.isnan(result.joints_camera[10:14, 0]).all()
    assert not result.frames_smoothed[10:14, 0].any()
    assert result.frames_smoothed[:, 0].sum() == 36
    assert result.frames_smoothed[:, 1].all()
    assert result.hands_smoothed == 2
    # The hidden input values survive untouched.
    assert np.isnan(snapshot[10:14, 0]).all()


def test_gap_does_not_bridge_two_levels() -> None:
    joints, valid = make_trajectory(total=30)
    joints[10:20, 0] = np.nan
    valid[10:20, 0] = False
    joints[:10, 0, 0, 1] = 0.0
    joints[20:, 0, 0, 1] = 1.0  # the right-hand side sits one unit higher

    result = smooth_hand_joints(joints, valid)

    # Clean observations drive the observation scale to its floor, yet the
    # initial velocity variance is taken over the whole valid sequence -
    # including the jump across the hole (reference behaviour) - so the RTS
    # pass leaks a small, bounded amount of the step across the gap. Both
    # sides must stay near their own level.
    assert np.allclose(result.joints_camera[:10, 0, 0, 1], 0.0, atol=0.05)
    assert np.allclose(result.joints_camera[20:, 0, 0, 1], 1.0, atol=0.05)


def test_short_hands_are_skipped() -> None:
    joints, valid = make_trajectory(total=10)
    valid[3:, 0] = False  # only 3 valid frames (< MIN_VALID)

    result = smooth_hand_joints(joints, valid)

    assert result.hands_smoothed == 1
    assert not result.frames_smoothed[:, 0].any()
    assert result.frames_smoothed[:, 1].all()
    assert MIN_VALID == 4


def test_hands_are_independent() -> None:
    joints, valid = make_trajectory()
    valid[5:9, 1] = False
    joints[5:9, 1] = np.nan

    result = smooth_hand_joints(joints, valid)

    assert np.isnan(result.joints_camera[5:9, 1]).all()
    assert np.allclose(result.joints_camera[:, 0], joints[:, 0], atol=1e-4)


def test_inputs_are_not_mutated() -> None:
    joints, valid = make_trajectory()
    joints[10:14, 0] = np.nan
    valid[10:14, 0] = False
    joints_snapshot = joints.copy()
    valid_snapshot = valid.copy()

    smooth_hand_joints(joints, valid)

    assert np.array_equal(joints, joints_snapshot, equal_nan=True)
    assert np.array_equal(valid, valid_snapshot)


def test_parameter_validation() -> None:
    joints, valid = make_trajectory()
    with pytest.raises(StageIOError):
        smooth_hand_joints(joints, valid, q=0.05)
    with pytest.raises(StageIOError):
        smooth_hand_joints(joints, valid, r=3.0)
    with pytest.raises(StageIOError):
        smooth_hand_joints(joints, valid, beta=-1.0)
    with pytest.raises(StageIOError):
        smooth_hand_joints(joints[:, :1], valid)
    with pytest.raises(StageIOError):
        smooth_hand_joints(joints, valid[:, :1])


def test_rts_off_still_smooths() -> None:
    rng = np.random.default_rng(3)
    joints, valid = make_trajectory(total=80)
    noisy = joints + rng.standard_normal(joints.shape) * 0.02

    filtered = smooth_hand_joints(noisy, valid, rts=False)

    assert roughness(filtered.joints_camera[:, 0, 0, :]) < roughness(noisy[:, 0, 0, :])
    assert filtered.frames_smoothed.all()

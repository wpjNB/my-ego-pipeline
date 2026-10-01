"""Phase 6 P2: short-gap interpolation of the hand trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.refinement.gap_fill import interpolate_hand_gaps


def make_trajectory(total: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(joints [T, 2, 21, 3], valid [T, 2], confidence [T, 2])`` fully valid."""
    rng = np.random.default_rng(7)
    joints = rng.standard_normal((total, 2, 21, 3))
    valid = np.ones((total, 2), dtype=bool)
    confidence = np.full((total, 2), 0.8)
    return joints, valid, confidence


def test_short_gap_is_filled_linearly() -> None:
    joints, valid, confidence = make_trajectory()
    joints[3:6, 1] = np.nan
    valid[3:6, 1] = False

    result = interpolate_hand_gaps(joints, valid, confidence, max_gap=12)

    left, right = joints[2, 1], joints[6, 1]
    assert np.allclose(result.joints_camera[3, 1], 0.75 * left + 0.25 * right)
    assert np.allclose(result.joints_camera[4, 1], 0.50 * left + 0.50 * right)
    assert np.allclose(result.joints_camera[5, 1], 0.25 * left + 0.75 * right)
    assert np.isfinite(result.joints_camera[:, 1]).all()


def test_masks_and_counters() -> None:
    joints, valid, confidence = make_trajectory()
    joints[3:6, 1] = np.nan
    valid[3:6, 1] = False

    result = interpolate_hand_gaps(joints, valid, confidence, max_gap=12)

    assert result.interpolated.dtype == bool
    assert result.interpolated[:, 0].sum() == 0
    assert result.interpolated[3:6, 1].all()
    assert result.interpolated.sum() == 3
    assert result.valid[3:6, 1].all()
    assert result.gaps_filled == 1
    assert result.frames_filled == 3
    assert result.frames_left_missing == 0
    # Anchors and the untouched hand keep their values exactly.
    assert np.array_equal(result.joints_camera[2, 1], joints[2, 1])
    assert np.array_equal(result.joints_camera[:, 0], joints[:, 0])


def test_gap_longer_than_max_gap_stays_missing() -> None:
    joints, valid, confidence = make_trajectory()
    joints[2:9, 1] = np.nan
    valid[2:9, 1] = False

    result = interpolate_hand_gaps(joints, valid, confidence, max_gap=4)

    assert not np.isfinite(result.joints_camera[2:9, 1]).any()
    assert result.interpolated.sum() == 0
    assert result.gaps_filled == 0
    assert result.frames_left_missing == 7


def test_leading_and_trailing_missing_stay_missing() -> None:
    joints, valid, confidence = make_trajectory()
    joints[:3, 0] = np.nan
    valid[:3, 0] = False
    joints[-2:, 0] = np.nan
    valid[-2:, 0] = False

    result = interpolate_hand_gaps(joints, valid, confidence, max_gap=12)

    assert not np.isfinite(result.joints_camera[:3, 0]).any()
    assert not np.isfinite(result.joints_camera[-2:, 0]).any()
    assert result.interpolated.sum() == 0


def test_confidence_follows_the_same_blend() -> None:
    joints, valid, confidence = make_trajectory()
    confidence[:, 1] = np.linspace(0.4, 1.2, joints.shape[0])
    joints[3:6, 1] = np.nan
    valid[3:6, 1] = False

    result = interpolate_hand_gaps(joints, valid, confidence, max_gap=12)

    left, right = confidence[2, 1], confidence[6, 1]
    assert result.confidence[3, 1] == pytest.approx(0.75 * left + 0.25 * right)
    assert result.confidence[5, 1] == pytest.approx(0.25 * left + 0.75 * right)
    assert np.array_equal(result.confidence[:, 0], confidence[:, 0])


def test_valid_but_nonfinite_frame_is_filled_not_anchored() -> None:
    joints, valid, confidence = make_trajectory()
    # A "valid" detection that produced NaNs is missing, never an endpoint.
    joints[2, 1] = np.nan
    joints[4:6, 1] = np.nan

    result = interpolate_hand_gaps(joints, valid, confidence, max_gap=12)

    # Anchors are frames 0, 1, 3, 6..9 (frame 3 stays a real prediction), so
    # frames 2 and 4-5 form two fillable gaps.
    assert result.interpolated[2, 1]
    assert not result.interpolated[3, 1]
    assert result.interpolated[4:6, 1].all()
    assert np.allclose(result.joints_camera[2, 1], 0.5 * joints[1, 1] + 0.5 * joints[3, 1])
    assert np.allclose(result.joints_camera[5, 1], joints[3, 1] / 3.0 + 2.0 * joints[6, 1] / 3.0)


def test_multiple_gaps_fill_independently() -> None:
    joints, valid, confidence = make_trajectory(total=14)
    joints[2, 1] = np.nan
    valid[2, 1] = False
    joints[8:11, 1] = np.nan
    valid[8:11, 1] = False

    result = interpolate_hand_gaps(joints, valid, confidence, max_gap=12)

    assert result.gaps_filled == 2
    assert result.frames_filled == 4
    assert result.interpolated[2, 1] and result.interpolated[8:11, 1].all()
    assert result.frames_left_missing == 0


def test_inputs_are_not_mutated() -> None:
    joints, valid, confidence = make_trajectory()
    joints[3:6, 1] = np.nan
    valid[3:6, 1] = False
    joints_snapshot = joints.copy()
    valid_snapshot = valid.copy()
    confidence_snapshot = confidence.copy()

    interpolate_hand_gaps(joints, valid, confidence, max_gap=12)

    assert np.array_equal(joints, joints_snapshot, equal_nan=True)
    assert np.array_equal(valid, valid_snapshot)
    assert np.array_equal(confidence, confidence_snapshot)


def test_input_validation() -> None:
    joints, valid, confidence = make_trajectory()
    with pytest.raises(StageIOError):
        interpolate_hand_gaps(joints[:, :1], valid, confidence)
    with pytest.raises(StageIOError):
        interpolate_hand_gaps(joints, valid[:, :1], confidence)
    with pytest.raises(StageIOError):
        interpolate_hand_gaps(joints, valid, confidence[:, :1])
    with pytest.raises(StageIOError):
        interpolate_hand_gaps(joints, valid, confidence, max_gap=0)

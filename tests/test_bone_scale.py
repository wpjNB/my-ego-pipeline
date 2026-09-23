"""Phase 6.2: bone-scale correction."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.hand.mano import JOINT_PARENTS, bone_pairs, joint_children
from ego3d_action.refinement.bone_scale import (
    clip_mean_bone_lengths,
    correct_bone_scale,
)


def build_hand(scale: float) -> np.ndarray:
    """A canonical 21-joint hand scaled uniformly by ``scale``."""
    rng = np.random.default_rng(0)
    points = np.zeros((21, 3))
    for child, parent in enumerate(JOINT_PARENTS):
        if parent < 0:
            continue
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        points[child] = points[parent] + 0.03 * direction
    return points * scale


def test_mano_topology_is_consistent() -> None:
    assert len(JOINT_PARENTS) == 21
    assert JOINT_PARENTS[0] == -1
    assert len(bone_pairs()) == 20
    children = joint_children()
    assert children[0] == [1, 5, 9, 13, 17]
    assert np.allclose(
        [c for _, c in bone_pairs()], sorted(c for _, c in bone_pairs())
    )


def test_stable_hand_is_left_almost_unchanged() -> None:
    joints = np.stack([build_hand(1.0) for _ in range(20)])  # [T, 21, 3]
    joints = np.repeat(joints[:, None], 2, axis=1)
    valid = np.ones((20, 2), dtype=bool)

    result = correct_bone_scale(joints, valid)
    assert np.allclose(result.joints, joints, atol=1e-9)
    assert np.allclose(result.ratios, 1.0)


def test_wobbling_bone_lengths_are_corrected_within_the_bound() -> None:
    total = 40
    scales = 1.0 + 0.15 * np.sin(np.linspace(0.0, 6.0, total))
    joints = np.stack([build_hand(s) for s in scales])[:, None].repeat(2, axis=1)
    valid = np.ones((total, 2), dtype=bool)

    result = correct_bone_scale(joints, valid, max_correction=0.035)

    assert np.all(result.ratios >= 1.0 - 0.035 - 1e-12)
    assert np.all(result.ratios <= 1.0 + 0.035 + 1e-12)
    # Deviation from the clip reference must shrink in both directions.
    assert np.nanmax(result.max_deviation_after) < np.nanmax(result.max_deviation_before)
    # The bound genuinely clamps: 15 % wobble cannot be fully removed.
    assert np.nanmax(result.max_deviation_after) > 0.035


def test_clamped_correction_respects_the_bound_for_small_drift() -> None:
    total = 30
    scales = np.linspace(1.0, 1.02, total)
    joints = np.stack([build_hand(s) for s in scales])[:, None].repeat(2, axis=1)
    valid = np.ones((total, 2), dtype=bool)

    result = correct_bone_scale(joints, valid, max_correction=0.035)
    assert np.allclose(result.ratios, 1.0, atol=0.035)
    # Every bone ends within 3.5 % of the clip reference.
    assert np.nanmax(result.max_deviation_after) <= 0.035 + 1e-6


def test_invalid_frames_are_copied_through() -> None:
    total = 10
    joints = np.stack([build_hand(1.0) for _ in range(total)])
    joints = np.repeat(joints[:, None], 2, axis=1)
    valid = np.ones((total, 2), dtype=bool)
    valid[4] = False
    joints[4] = 0.0

    result = correct_bone_scale(joints, valid)
    assert np.allclose(result.joints[4], 0.0)


def test_clip_reference_uses_valid_frames_only() -> None:
    total = 12
    clean = np.repeat(np.stack([build_hand(1.0) for _ in range(total)])[:, None], 2, axis=1)
    junked = clean.copy()
    junked[3] = 5.0 * build_hand(1.0)

    valid = np.ones((total, 2), dtype=bool)
    valid[3] = False

    lengths = clip_mean_bone_lengths(junked, valid)
    reference = clip_mean_bone_lengths(clean, np.ones_like(valid))
    assert np.allclose(lengths, reference)


def test_mean_reference_option() -> None:
    joints = np.stack([build_hand(1.0) for _ in range(6)])
    joints = np.repeat(joints[:, None], 2, axis=1)
    valid = np.ones((6, 2), dtype=bool)
    assert np.allclose(
        clip_mean_bone_lengths(joints, valid, reference="mean"),
        clip_mean_bone_lengths(joints, valid, reference="median"),
    )
    with pytest.raises(StageIOError):
        clip_mean_bone_lengths(joints, valid, reference="mode")


def test_invalid_arguments_raise() -> None:
    joints = np.zeros((5, 2, 21, 3))
    with pytest.raises(StageIOError):
        correct_bone_scale(joints, np.ones((5, 2), dtype=bool), max_correction=-0.1)
    with pytest.raises(StageIOError):
        correct_bone_scale(joints, np.ones((5, 3), dtype=bool))
    with pytest.raises(StageIOError):
        correct_bone_scale(np.zeros((5, 2, 20, 3)), np.ones((5, 2), dtype=bool))

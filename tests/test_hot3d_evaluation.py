"""Evaluation against a wrist-only reference (the bundled HOT3D sample)."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.evaluation.action_mpjpe import action_mpjpe, safe_nanmean


def make_case(total: int = 90, *, wrist_only: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    joints = np.full((total, 2, 21, 3), np.nan)
    joints[:, :, 0, :] = rng.normal(0.0, 0.1, size=(total, 2, 3))
    if not wrist_only:
        joints[:, :, 1:, :] = rng.normal(0.0, 0.1, size=(total, 2, 20, 3))
    valid = np.ones((total, 2), dtype=bool)
    valid[10:14, 1] = False
    joints[~valid] = np.nan
    rotation = np.broadcast_to(np.eye(3), (total, 3, 3)).copy()
    translation = np.zeros((total, 3))
    return {
        "joints": joints,
        "valid": valid,
        "rotation": rotation,
        "translation": translation,
    }


def evaluate(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]):
    return action_mpjpe(
        pred["joints"],
        gt["joints"],
        prediction_rotation_c2w=pred["rotation"],
        prediction_translation_c2w=pred["translation"],
        ground_truth_rotation_c2w=gt["rotation"],
        ground_truth_translation_c2w=gt["translation"],
        fps=30.0,
        prediction_valid=pred["valid"],
        ground_truth_valid=gt["valid"],
    )


def test_wrist_only_reference_is_still_evaluated() -> None:
    case = make_case()
    result = evaluate(case, case)
    assert result.action_mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    # Exactly one of 21 joints is comparable.
    assert result.joint_coverage == pytest.approx(1.0 / 21.0, rel=0.05)
    assert np.isnan(result.per_joint_mm[5])
    assert result.per_joint_mm[0] == pytest.approx(0.0, abs=1e-9)


def test_wrist_only_offset_shows_up_in_the_wrist_error() -> None:
    gt = make_case()
    pred = {key: value.copy() for key, value in gt.items()}
    pred["joints"][:, :, 0, :] += np.array([0.0, 0.03, 0.04])  # 50 mm
    result = evaluate(pred, gt)
    assert result.action_mpjpe_mm == pytest.approx(50.0, abs=1e-6)
    assert result.wrist_mm == pytest.approx(50.0, abs=1e-6)


def test_full_reference_still_averages_every_joint() -> None:
    case = make_case(wrist_only=False)
    result = evaluate(case, case)
    # 4 hand-frames of 180 are marked invalid by the case builder.
    assert result.joint_coverage == pytest.approx(1.0 - 4.0 / (90 * 2), rel=1e-6)
    assert np.isfinite(result.per_joint_mm).all()


def test_joint_level_masking_beats_hand_level_masking() -> None:
    """A hand-frame with a partially valid reference must not be discarded."""
    gt = make_case()
    pred = {key: value.copy() for key, value in gt.items()}
    pred["joints"][:, :, 1:, :] = 1.0  # garbage on joints the reference never saw
    result = evaluate(pred, gt)
    assert result.action_mpjpe_mm == pytest.approx(0.0, abs=1e-9)


def test_no_comparable_terms_raises() -> None:
    gt = make_case()
    pred = {key: value.copy() for key, value in gt.items()}
    pred["joints"][:] = np.nan
    with pytest.raises(StageIOError):
        evaluate(pred, gt)


def test_safe_nanmean_is_quiet_and_explicit() -> None:
    values = np.array([[1.0, np.nan], [3.0, np.nan]])
    assert safe_nanmean(values, axis=0)[0] == pytest.approx(2.0)
    assert np.isnan(safe_nanmean(values, axis=0)[1])
    assert safe_nanmean(values, axis=None) == pytest.approx(2.0)
    assert np.isnan(safe_nanmean(np.full((2, 2), np.nan), axis=None))

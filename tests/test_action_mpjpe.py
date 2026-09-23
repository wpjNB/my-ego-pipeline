"""Phase 7: Action-MPJPE protocol and coverage."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego3d_action.errors import StageIOError
from ego3d_action.evaluation.action_mpjpe import action_mpjpe, to_camera_frame
from ego3d_action.evaluation.benchmark import (
    EvaluationReport,
    camera_pose_error_mm,
    measure_fps,
)
from ego3d_action.evaluation.coverage import coverage_by_hand, coverage_ratio, missing_runs


def make_trajectory(total: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(total, dtype=np.float64)
    rot = np.stack([Rotation.from_euler("y", 3.0 * i, degrees=True).as_matrix() for i in t])
    trans = np.stack([0.02 * t, 0.01 * np.sin(0.2 * t), np.zeros(total)], axis=-1)
    joints = rng.normal(0.0, 0.05, size=(total, 2, 21, 3)) + np.array([0.0, 0.0, 1.5])
    return joints, rot, trans


def test_identical_trajectories_give_zero_error() -> None:
    joints, rot, trans = make_trajectory(60)
    result = action_mpjpe(
        joints,
        joints,
        prediction_rotation_c2w=rot,
        prediction_translation_c2w=trans,
        ground_truth_rotation_c2w=rot,
        ground_truth_translation_c2w=trans,
        fps=30.0,
    )
    assert result.action_mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    assert result.num_chunks == 31
    assert result.wrist_mm == pytest.approx(0.0, abs=1e-9)


def test_global_rigid_offset_cancels_in_the_camera_frame() -> None:
    joints, rot, trans = make_trajectory(60)
    extra_rot = Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True).as_matrix()
    extra_trans = np.array([0.5, -0.3, 0.2])
    moved_joints = joints @ extra_rot.T + extra_trans
    moved_rot = np.einsum("ij,tjk->tik", extra_rot, rot)
    moved_trans = (extra_rot @ trans.T).T + extra_trans

    result = action_mpjpe(
        moved_joints,
        joints,
        prediction_rotation_c2w=moved_rot,
        prediction_translation_c2w=moved_trans,
        ground_truth_rotation_c2w=rot,
        ground_truth_translation_c2w=trans,
        fps=30.0,
    )
    assert result.action_mpjpe_mm == pytest.approx(0.0, abs=1e-6)


def test_constant_world_offset_equals_the_offset_length() -> None:
    joints, rot, trans = make_trajectory(60)
    offset = np.array([0.0, 0.03, 0.04])  # 50 mm
    result = action_mpjpe(
        joints + offset,
        joints,
        prediction_rotation_c2w=rot,
        prediction_translation_c2w=trans,
        ground_truth_rotation_c2w=rot,
        ground_truth_translation_c2w=trans,
        fps=30.0,
    )
    assert result.action_mpjpe_mm == pytest.approx(50.0, abs=1e-6)


def test_chunk_length_follows_fps() -> None:
    joints, rot, trans = make_trajectory(40)
    result_30 = action_mpjpe(
        joints,
        joints,
        prediction_rotation_c2w=rot,
        prediction_translation_c2w=trans,
        ground_truth_rotation_c2w=rot,
        ground_truth_translation_c2w=trans,
        fps=30.0,
    )
    result_10 = action_mpjpe(
        joints,
        joints,
        prediction_rotation_c2w=rot,
        prediction_translation_c2w=trans,
        ground_truth_rotation_c2w=rot,
        ground_truth_translation_c2w=trans,
        fps=10.0,
    )
    assert result_30.num_chunks == 40 - 30 + 1
    assert result_10.num_chunks == 40 - 10 + 1


def test_missing_frames_are_excluded_from_the_mean() -> None:
    joints, rot, trans = make_trajectory(60)
    corrupted = joints.copy()
    corrupted[:10] += 10.0  # 10 metre error if it were counted
    valid = np.ones((60, 2), dtype=bool)
    valid[:10] = False

    result = action_mpjpe(
        corrupted,
        joints,
        prediction_rotation_c2w=rot,
        prediction_translation_c2w=trans,
        ground_truth_rotation_c2w=rot,
        ground_truth_translation_c2w=trans,
        fps=30.0,
        prediction_valid=valid,
        ground_truth_valid=valid,
    )
    assert result.action_mpjpe_mm < 1e-6
    assert result.num_terms > 0


def test_no_overlap_raises() -> None:
    joints, rot, trans = make_trajectory(50)
    valid = np.zeros((50, 2), dtype=bool)
    with pytest.raises(StageIOError):
        action_mpjpe(
            joints,
            joints,
            prediction_rotation_c2w=rot,
            prediction_translation_c2w=trans,
            ground_truth_rotation_c2w=rot,
            ground_truth_translation_c2w=trans,
            fps=30.0,
            prediction_valid=valid,
            ground_truth_valid=valid,
        )


def test_shape_validation() -> None:
    joints, rot, trans = make_trajectory(30)
    with pytest.raises(StageIOError):
        action_mpjpe(
            joints,
            joints[:10],
            prediction_rotation_c2w=rot,
            prediction_translation_c2w=trans,
            ground_truth_rotation_c2w=rot[:10],
            ground_truth_translation_c2w=trans[:10],
            fps=30.0,
        )
    with pytest.raises(StageIOError):
        action_mpjpe(
            joints,
            joints,
            prediction_rotation_c2w=rot,
            prediction_translation_c2w=trans,
            ground_truth_rotation_c2w=rot,
            ground_truth_translation_c2w=trans,
            fps=0.0,
        )
    with pytest.raises(StageIOError):
        action_mpjpe(
            joints,
            joints,
            prediction_rotation_c2w=rot,
            prediction_translation_c2w=trans,
            ground_truth_rotation_c2w=rot,
            ground_truth_translation_c2w=trans,
            fps=1.0,
            chunk_seconds=100.0,
        )


def test_to_camera_frame_matches_manual_computation() -> None:
    rot = Rotation.from_euler("z", 45.0, degrees=True).as_matrix()
    trans = np.array([1.0, 2.0, 3.0])
    points = np.array([[1.0, 2.0, 4.0]])
    expected = rot.T @ (points[0] - trans)
    assert np.allclose(to_camera_frame(points, rot, trans)[0], expected)


def test_coverage_helpers() -> None:
    valid = np.array([[True, False], [True, True], [False, False]])
    assert coverage_ratio(valid) == pytest.approx(0.5)
    assert np.allclose(coverage_by_hand(valid), [2 / 3, 1 / 3])
    assert missing_runs(np.array([True, False, False, True, False])) == [(1, 3), (4, 5)]
    with pytest.raises(StageIOError):
        coverage_ratio(np.zeros((0, 2), dtype=bool))


def test_report_format_matches_the_spec_block() -> None:
    joints, rot, trans = make_trajectory(40)
    result = action_mpjpe(
        joints,
        joints,
        prediction_rotation_c2w=rot,
        prediction_translation_c2w=trans,
        ground_truth_rotation_c2w=rot,
        ground_truth_translation_c2w=trans,
        fps=30.0,
    )
    report = EvaluationReport.from_result(
        result, coverage=0.8123, fps=15.53, num_frames=40, camera_error=0.0
    )
    text = report.format()
    assert "HOT3D Evaluation" in text
    assert "Action MPJPE :" in text
    assert "Coverage     : 81.23 %" in text
    assert "FPS          : 15.53" in text
    assert "Camera error :" in text
    assert "Wrist error  :" in text
    assert "Depth error  :" in text
    assert report.as_dict()["coverage"] == pytest.approx(0.8123)


def test_measure_fps_and_camera_error() -> None:
    assert measure_fps(300, 20.0) == pytest.approx(15.0)
    with pytest.raises(StageIOError):
        measure_fps(300, 0.0)
    with pytest.raises(StageIOError):
        measure_fps(-1, 1.0)

    pred = np.zeros((3, 3))
    gt = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.002, 0.0]])
    assert camera_pose_error_mm(pred, gt) == pytest.approx(1000.0 * np.mean([0.0, 0.001, 0.002]))

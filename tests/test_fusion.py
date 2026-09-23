"""Phase 5: world fusion and the trajectory artefact contract."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego3d_action.errors import StageIOError
from ego3d_action.fusion.trajectory import (
    build_trajectory,
    camera_joints_to_world,
    trajectory_metadata,
)
from ego3d_action.io.serialization import TRAJECTORY_FIELDS, validate_trajectory


def make_inputs(total: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    joints = rng.normal(0.0, 0.02, size=(total, 2, 21, 3)) + np.array([0.0, 0.0, 1.5])
    valid = np.ones((total, 2), dtype=bool)
    valid[3, 1] = False
    rotation = np.stack(
        [Rotation.from_euler("y", 5.0 * t, degrees=True).as_matrix() for t in range(total)]
    )
    translation = np.stack([np.linspace(0, 0.2, total), np.zeros(total), np.full(total, 0.01)], axis=-1)
    intrinsics = np.broadcast_to(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        (total, 3, 3),
    ).copy()
    return {
        "joints_camera": joints,
        "hand_valid": valid,
        "hand_confidence": np.where(valid, 0.9, 0.0),
        "camera_rotation_c2w": rotation,
        "camera_translation_c2w": translation,
        "camera_intrinsics": intrinsics,
        "bbox": np.zeros((total, 2, 4)),
        "track_id": np.zeros((total, 2), dtype=np.int64),
    }


def test_camera_joints_to_world_matches_manual_formula() -> None:
    inputs = make_inputs()
    joints = inputs["joints_camera"]
    rot = inputs["camera_rotation_c2w"]
    trans = inputs["camera_translation_c2w"]

    world = camera_joints_to_world(joints, rot, trans)
    expected = np.einsum("tij,thnj->thni", rot, joints) + trans[:, None, None, :]
    assert np.allclose(world, expected, atol=1e-12)


def test_invalid_hand_frames_become_nan_in_world_space() -> None:
    inputs = make_inputs()
    world = camera_joints_to_world(
        inputs["joints_camera"],
        inputs["camera_rotation_c2w"],
        inputs["camera_translation_c2w"],
        hand_valid=inputs["hand_valid"],
    )
    assert not np.isfinite(world[3, 1]).any()
    assert np.isfinite(world[3, 0]).all()


def test_build_trajectory_satisfies_the_section_7_contract() -> None:
    arrays, metadata = build_trajectory(**make_inputs(), fps=30.0)
    assert sorted(arrays) == sorted(TRAJECTORY_FIELDS)
    assert validate_trajectory(arrays) == []
    assert arrays["hand_xyz_world"].shape == (12, 2, 21, 3)
    assert arrays["mano_hand_pose"].shape == (12, 2, 15, 3, 3)
    assert arrays["mano_betas"].shape == (12, 2, 10)
    assert metadata["camera_convention"] == "c2w"
    assert metadata["units"] == "meter"
    assert metadata["hand_representation"] == "21_joints_metric_xyz"
    assert metadata["world_frame"] == 0
    assert np.allclose(arrays["timestamps"], np.arange(12) / 30.0)


def test_validate_trajectory_reports_missing_and_misshaped_fields() -> None:
    arrays, _ = build_trajectory(**make_inputs(), fps=30.0)
    broken = dict(arrays)
    del broken["bbox"]
    problems = validate_trajectory(broken)
    assert any("bbox" in problem for problem in problems)

    broken = dict(arrays)
    broken["camera_K"] = np.zeros((12, 4, 4))
    problems = validate_trajectory(broken)
    assert any("camera_K" in problem for problem in problems)

    broken = dict(arrays)
    broken["hand_valid"] = broken["hand_valid"][:5]
    problems = validate_trajectory(broken)
    assert any("leading dim" in problem for problem in problems)


def test_trajectory_metadata_merges_extra_fields() -> None:
    metadata = trajectory_metadata(fps=25.0, num_frames=100, extra={"clip": "demo01"})
    assert metadata["fps"] == 25.0
    assert metadata["num_frames"] == 100
    assert metadata["clip"] == "demo01"


def test_shape_validation_raises() -> None:
    inputs = make_inputs()
    with pytest.raises(StageIOError):
        camera_joints_to_world(
            inputs["joints_camera"][:, :1],
            inputs["camera_rotation_c2w"],
            inputs["camera_translation_c2w"],
        )
    with pytest.raises(StageIOError):
        camera_joints_to_world(
            inputs["joints_camera"],
            inputs["camera_rotation_c2w"][:5],
            inputs["camera_translation_c2w"],
        )
    with pytest.raises(StageIOError):
        camera_joints_to_world(
            inputs["joints_camera"],
            inputs["camera_rotation_c2w"],
            inputs["camera_translation_c2w"],
            hand_valid=np.ones((12, 3), dtype=bool),
        )

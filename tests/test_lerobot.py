"""LeRobot v3 reader and HOT3D ground-truth conversion (uses the bundled sample)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego3d_action.datasets.hot3d_gt import convert_episode, write_episode
from ego3d_action.datasets.lerobot import LeRobotDataset
from ego3d_action.errors import StageIOError
from ego3d_action.io.serialization import load_json, load_npz, validate_trajectory

SAMPLE_ROOT = Path("data/samples/lerobot_v3")

pytestmark = pytest.mark.skipif(
    not SAMPLE_ROOT.is_dir(), reason="the bundled LeRobot sample is not present"
)


@pytest.fixture(scope="module")
def dataset() -> LeRobotDataset:
    return LeRobotDataset(SAMPLE_ROOT)


def test_info_is_parsed(dataset: LeRobotDataset) -> None:
    info = dataset.info
    assert info.codebase_version.startswith("v3")
    assert info.fps == pytest.approx(30.0)
    assert info.total_episodes == 8
    assert info.total_frames == 3600
    assert dataset.video_key == "observation.images.ego"
    assert info.feature_shape("extrinsics_w2c") == (16,)
    assert info.feature_shape("left_hand_pose") == (135,)


def test_episode_metadata_and_tasks(dataset: LeRobotDataset) -> None:
    assert dataset.episode_indices() == list(range(8))
    meta = dataset.episodes_meta()[0]
    assert int(meta["length"]) == 450
    tasks = dataset.tasks()
    assert tasks[0].startswith("hot3d ego hand-object interaction")


def test_load_episode_shapes(dataset: LeRobotDataset) -> None:
    episode = dataset.load_episode(0)
    assert episode.length == 450
    assert episode.task is not None and "hot3d" in episode.task
    assert episode.video_path.is_file()
    assert episode.column("extrinsics_w2c").shape == (450, 16)
    assert episode.column("intrinsics").shape == (450, 9)
    assert episode.column("state_mask").shape == (450, 2)
    assert episode.column("left_hand_pose").shape == (450, 135)
    # Scalar columns collapse to one dimension.
    assert episode.column("frame_index").shape == (450,)
    assert episode.column("observation.state").shape == (450, 122)


def test_unknown_episode_and_missing_root_raise(dataset: LeRobotDataset, tmp_path: Path) -> None:
    with pytest.raises(StageIOError, match="not in this dataset"):
        dataset.load_episode(99)
    with pytest.raises(StageIOError, match="not a LeRobot v3 dataset"):
        LeRobotDataset(tmp_path)
    with pytest.raises(StageIOError):
        dataset.load_episode(0).column("nope")


def test_convert_episode_satisfies_the_trajectory_contract(dataset: LeRobotDataset) -> None:
    episode = convert_episode(dataset, 0)
    assert validate_trajectory(episode.arrays, strict=True) == []
    assert episode.num_frames == 450
    assert episode.metadata["hand_joints"] == "wrist_only"
    assert episode.metadata["units"] == "meter"
    assert episode.metadata["camera_convention"] == "c2w"
    assert 0.5 < float(episode.metadata["coverage_left"]) <= 1.0


def test_reference_is_re_anchored_to_world_zero(dataset: LeRobotDataset) -> None:
    episode = convert_episode(dataset, 0)
    arrays = episode.arrays
    assert np.allclose(arrays["camera_R_c2w"][0], np.eye(3), atol=1e-9)
    assert np.allclose(arrays["camera_t_c2w"][0], np.zeros(3), atol=1e-9)
    # The HOT3D world stays reachable through the recorded anchor.
    anchor_rotation = np.asarray(episode.metadata["hot3d_world_anchor_rotation"])
    anchor_translation = np.asarray(episode.metadata["hot3d_world_anchor_translation"])
    assert np.isclose(np.linalg.det(anchor_rotation), 1.0, atol=1e-6)
    assert anchor_translation.shape == (3,)


def test_only_the_wrist_carries_a_reference_position(dataset: LeRobotDataset) -> None:
    episode = convert_episode(dataset, 0)
    world = episode.arrays["hand_xyz_world"]
    finite = np.isfinite(world).all(axis=-1)
    per_frame = finite.sum(axis=-1)  # [T, 2]
    assert set(np.unique(per_frame[:, 0])) <= {0, 1}
    assert set(np.unique(per_frame[:, 1])) <= {0, 1}
    # The wrist must be present wherever the hand is marked valid.
    valid = episode.arrays["hand_valid"]
    assert np.isfinite(world[:, :, 0, :][valid]).all()
    # ... and the MANO parameterisation is preserved for any later FK.
    assert episode.arrays["mano_hand_pose"].shape == (450, 2, 15, 3, 3)
    assert episode.arrays["mano_betas"].shape == (450, 2, 10)


def test_wrist_is_consistent_with_the_reference_camera(dataset: LeRobotDataset) -> None:
    """Projecting the wrist with the reference camera must land inside the frame."""
    episode = convert_episode(dataset, 0)
    arrays = episode.arrays
    world = arrays["hand_xyz_world"][:, 0, 0, :]
    rotation = arrays["camera_R_c2w"]
    translation = arrays["camera_t_c2w"]
    intrinsics = arrays["camera_K"]
    camera = np.einsum("tji,tj->ti", rotation, world - translation)
    pixel = np.einsum("tij,tj->ti", intrinsics, camera)
    pixel = pixel[:, :2] / pixel[:, 2:3]
    valid = arrays["hand_valid"][:, 0]
    inside = (
        (pixel[:, 0] >= 0) & (pixel[:, 0] < 512) & (pixel[:, 1] >= 0) & (pixel[:, 1] < 512)
    )
    assert valid.sum() > 100
    # Most valid frames must have the wrist in view; the hand legitimately leaves
    # the frame at times in an egocentric clip.
    assert float(np.mean(inside[valid])) > 0.5


def test_write_episode_roundtrip(dataset: LeRobotDataset, tmp_path: Path) -> None:
    episode = convert_episode(dataset, 1)
    trajectory, metadata = write_episode(
        episode, tmp_path / "ground_truth.npz", tmp_path / "ground_truth.json"
    )
    assert trajectory.is_file() and metadata.is_file()
    loaded = load_npz(trajectory)
    assert validate_trajectory(loaded, strict=True) == []
    assert load_json(metadata)["hand_joints"] == "wrist_only"

"""HOT3D reference conversion must re-anchor every world-space field."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ego3d_action.camera.camera_pose import world_frame_alignment
from ego3d_action.datasets.hot3d_gt import convert_episode
from ego3d_action.hand.mano_model import forward_kinematics
from ego3d_action.testing.synthetic import make_synthetic_mano_model


class _Episode:
    def __init__(self, columns: dict[str, np.ndarray], video_path: Path) -> None:
        self._columns = columns
        self.length = int(columns["extrinsics_w2c"].shape[0])
        self.video_path = video_path
        self.task = "synthetic HOT3D episode"

    def column(self, name: str) -> np.ndarray:
        return self._columns[name]

    def has(self, name: str) -> bool:
        return name in self._columns


class _Dataset:
    def __init__(self, episode: _Episode, root: Path) -> None:
        from types import SimpleNamespace

        self._episode = episode
        self.root = root
        self.video_key = "observation.images.ego"
        self.info = SimpleNamespace(
            fps=30.0,
            features={self.video_key: {"shape": [512, 512, 3]}},
            raw={"_source_dataset": "hot3d"},
        )

    def load_episode(self, episode_index: int) -> _Episode:
        assert episode_index == 0
        return self._episode


def test_mano_root_rotation_is_reanchored_with_hot3d_world() -> None:
    total = 3
    c2w_rotation = Rotation.from_euler(
        "zyx", [[72.0, -16.0, 23.0], [76.0, -13.0, 20.0], [80.0, -9.0, 17.0]], degrees=True
    ).as_matrix()
    c2w_translation = np.array(
        [[0.18, -0.11, 0.31], [0.21, -0.09, 0.29], [0.25, -0.07, 0.26]],
        dtype=np.float64,
    )
    w2c_rotation = np.transpose(c2w_rotation, (0, 2, 1))
    w2c_translation = -np.einsum("tij,tj->ti", w2c_rotation, c2w_translation)
    extrinsics_w2c = np.broadcast_to(np.eye(4), (total, 4, 4)).copy()
    extrinsics_w2c[:, :3, :3] = w2c_rotation
    extrinsics_w2c[:, :3, 3] = w2c_translation

    roots = {
        "left": Rotation.from_euler("xyz", [[-20, 35, 90], [-18, 32, 94], [-16, 29, 98]], degrees=True).as_matrix(),
        "right": Rotation.from_euler("xyz", [[15, -25, 10], [17, -21, 14], [20, -18, 18]], degrees=True).as_matrix(),
    }
    wrists = {
        "left": np.array([[0.02, 0.31, -0.12], [0.03, 0.32, -0.10], [0.04, 0.33, -0.08]]),
        "right": np.array([[0.20, 0.18, -0.13], [0.21, 0.19, -0.11], [0.22, 0.20, -0.09]]),
    }
    local_pose = np.broadcast_to(np.eye(3), (total, 15, 3, 3)).copy()
    columns = {
        "extrinsics_w2c": extrinsics_w2c.reshape(total, 16),
        "intrinsics": np.broadcast_to(
            np.array([[221.0, 0.0, 255.5], [0.0, 221.0, 255.5], [0.0, 0.0, 1.0]]),
            (total, 3, 3),
        ).reshape(total, 9),
        "state_mask": np.ones((total, 2), dtype=bool),
        "left_kept": np.ones(total, dtype=bool),
        "right_kept": np.ones(total, dtype=bool),
        "observation.state": np.zeros((total, 122), dtype=np.float64),
    }
    for side in ("left", "right"):
        columns[f"{side}_transl_world"] = wrists[side]
        columns[f"{side}_orient_world"] = roots[side].reshape(total, 9)
        columns[f"{side}_hand_pose"] = local_pose.reshape(total, 135)

    episode = _Episode(columns, Path("synthetic.mp4"))
    dataset = _Dataset(episode, Path("synthetic_hot3d"))
    model = make_synthetic_mano_model()
    converted = convert_episode(dataset, 0, mano_models={"left": model, "right": model})
    arrays = converted.arrays
    anchor_rotation, anchor_translation = world_frame_alignment(c2w_rotation, c2w_translation)
    assert not np.allclose(anchor_rotation, np.eye(3))

    for side_index, side in enumerate(("left", "right")):
        root_rotation_world0 = np.einsum("ij,tjk->tik", anchor_rotation, roots[side])
        wrist_world0 = np.einsum(
            "ij,tj->ti", anchor_rotation, wrists[side] - anchor_translation
        )
        np.testing.assert_allclose(arrays["mano_root_rot"][:, side_index], root_rotation_world0)
        expected_world = forward_kinematics(
            model,
            arrays["mano_betas"][:, side_index],
            arrays["mano_hand_pose"][:, side_index],
            root_rotation=root_rotation_world0,
            root_translation=wrist_world0,
        )
        np.testing.assert_allclose(arrays["hand_xyz_world"][:, side_index], expected_world)

        # This fixture must fail if the input HOT3D root is used unchanged after
        # the wrist has already moved into World-0.
        unanchored_world = forward_kinematics(
            model,
            arrays["mano_betas"][:, side_index],
            arrays["mano_hand_pose"][:, side_index],
            root_rotation=roots[side],
            root_translation=wrist_world0,
        )
        assert not np.allclose(expected_world, unanchored_world)

    expected_camera = np.einsum(
        "tji,thkj->thki",
        arrays["camera_R_c2w"],
        arrays["hand_xyz_world"] - arrays["camera_t_c2w"][:, None, None, :],
    )
    np.testing.assert_allclose(arrays["hand_xyz_camera"], expected_camera)

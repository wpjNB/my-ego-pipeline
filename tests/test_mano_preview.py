from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ego3d_action.refinement.mano_preview import build_processed_mano_preview


def test_processed_mesh_follows_filtered_root_and_wrist() -> None:
    frames = 4
    vertices = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (frames, 2, 1, 3)).copy()
    raw_joints = np.zeros((frames, 2, 21, 3), dtype=np.float64)
    clean_joints = raw_joints.copy()
    clean_joints[:, :, 0, 1] = 1.0
    raw_rot = np.broadcast_to(np.eye(3), (frames, 2, 3, 3)).copy()
    clean_rot = raw_rot.copy()
    clean_rot[:, :, :, :] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    valid = np.ones((frames, 2), dtype=bool)

    result = build_processed_mano_preview(
        vertices,
        raw_joints,
        clean_joints,
        valid,
        valid,
        np.zeros_like(valid),
        raw_root_rot=raw_rot,
        processed_root_rot=clean_rot,
    )

    assert result.valid.all()
    assert np.allclose(result.vertices_camera[:, :, 0, :], [0.0, 2.0, 0.0], atol=1e-6)


def test_short_invalid_mesh_gap_is_interpolated() -> None:
    vertices = np.zeros((3, 2, 2, 3), dtype=np.float64)
    vertices[0, :, :, 0] = 0.0
    vertices[2, :, :, 0] = 2.0
    vertices[1] = np.nan
    raw_joints = np.zeros((3, 2, 21, 3), dtype=np.float64)
    raw_joints[1] = np.nan
    clean_joints = np.zeros_like(raw_joints)
    raw_valid = np.ones((3, 2), dtype=bool)
    raw_valid[1] = False
    processed_valid = np.ones((3, 2), dtype=bool)

    result = build_processed_mano_preview(
        vertices,
        raw_joints,
        clean_joints,
        raw_valid,
        processed_valid,
        np.zeros_like(raw_valid),
        max_gap=2,
    )

    assert result.valid.all()
    assert result.interpolated[1].all()
    assert np.allclose(result.vertices_camera[1, :, :, 0], 1.0)

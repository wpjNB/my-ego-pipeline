"""Transforms: rigid application, inversion, projection, SLERP."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego3d_action.errors import InsufficientDataError
from ego3d_action.geometry.transforms import (
    apply_rigid,
    as_rotation_matrix,
    invert_rigid,
    project_points,
    rotation_angle,
    rotation_slerp,
    unproject_pixels,
)


def test_apply_and_invert_rigid_are_inverse() -> None:
    rotation = Rotation.from_euler("xyz", [20.0, -35.0, 10.0], degrees=True).as_matrix()
    translation = np.array([0.3, -0.2, 0.7])
    points = np.random.default_rng(0).normal(size=(16, 3))

    moved = apply_rigid(rotation, translation, points)
    back = apply_rigid(*invert_rigid(rotation, translation), moved)
    assert np.allclose(back, points, atol=1e-12)


def test_as_rotation_matrix_rejects_improper_rotation() -> None:
    bad = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(InsufficientDataError):
        as_rotation_matrix(bad)


def test_project_unproject_roundtrip() -> None:
    intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.1, -0.2, 1.5], [-0.4, 0.3, 2.0]])
    pixels = project_points(intrinsics, points)
    depth = points[:, 2]
    recovered = unproject_pixels(intrinsics, pixels, depth)
    assert np.allclose(recovered, points, atol=1e-9)


def test_project_points_marks_points_behind_camera() -> None:
    intrinsics = np.eye(3)
    pixels = project_points(intrinsics, np.array([[0.1, 0.1, 0.0]]))
    assert not np.all(np.isfinite(pixels))


def test_rotation_slerp_endpoints_and_midpoint() -> None:
    rot_a = np.eye(3)
    rot_b = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()

    assert np.allclose(rotation_slerp(rot_a, rot_b, 0.0), rot_a, atol=1e-12)
    assert np.allclose(rotation_slerp(rot_a, rot_b, 1.0), rot_b, atol=1e-12)

    mid = rotation_slerp(rot_a, rot_b, 0.5)
    assert np.isclose(np.degrees(rotation_angle(mid)), 45.0, atol=1e-6)


def test_rotation_slerp_uses_shortest_path() -> None:
    rot_a = np.eye(3)
    rot_b = Rotation.from_euler("z", 350.0, degrees=True).as_matrix()
    mid = rotation_slerp(rot_a, rot_b, 0.5)
    angle = np.degrees(rotation_angle(mid))
    assert angle < 10.0, f"expected the -10 deg path, got {angle:.3f} deg"


def test_rotation_slerp_is_batched_and_alpha_one_per_sample() -> None:
    rot_a = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    rot_b = np.stack([Rotation.from_euler("x", 10.0 * i, degrees=True).as_matrix() for i in range(4)])
    alpha = np.array([0.0, 0.25, 0.75, 1.0])
    out = rotation_slerp(rot_a, rot_b, alpha)
    assert out.shape == (4, 3, 3)
    assert np.allclose(out[0], rot_a[0])
    assert np.allclose(out[-1], rot_b[-1])


def test_rotation_slerp_rejects_shape_mismatch() -> None:
    with pytest.raises(InsufficientDataError):
        rotation_slerp(np.eye(3), np.broadcast_to(np.eye(3), (2, 3, 3)), 0.5)


def test_rotation_slerp_handles_identical_rotations() -> None:
    rot = Rotation.from_euler("y", 30.0, degrees=True).as_matrix()
    out = rotation_slerp(rot, rot, 0.5)
    assert np.allclose(out, rot, atol=1e-9)


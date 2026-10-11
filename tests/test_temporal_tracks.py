from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego3d_action.errors import StageIOError
from ego3d_action.refinement.temporal_tracks import (
    interpolate_rotation_gaps,
    interpolate_vector_gaps,
    smooth_rotation_tracks,
)


def test_vector_short_gap_is_interpolated_and_counted() -> None:
    values = np.arange(18, dtype=np.float64).reshape(6, 3)
    valid = np.ones(6, dtype=bool)
    values[2:4] = np.nan
    valid[2:4] = False

    result = interpolate_vector_gaps(values, valid, max_gap=2)

    assert np.allclose(result.values[2], values[1] * 2.0 / 3.0 + values[4] / 3.0)
    assert np.allclose(result.values[3], values[1] / 3.0 + values[4] * 2.0 / 3.0)
    assert result.valid.all()
    assert result.interpolated.sum() == 2
    assert result.gaps_filled == 1


def test_vector_long_and_boundary_gaps_stay_missing() -> None:
    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    valid = np.ones(10, dtype=bool)
    values[:1] = np.nan
    valid[:1] = False
    values[3:9] = np.nan
    valid[3:9] = False

    result = interpolate_vector_gaps(values, valid, max_gap=3)

    assert not result.valid[0]
    assert not result.valid[3:9].any()
    assert not result.interpolated.any()


def test_rotation_gap_uses_shortest_path_slerp() -> None:
    values = Rotation.from_rotvec(np.column_stack((np.zeros(5), np.zeros(5), np.radians([0, 0, 0, 0, 120])))).as_matrix()
    valid = np.array([True, False, False, False, True])

    result = interpolate_rotation_gaps(values, valid, max_gap=3)
    angles = Rotation.from_matrix(result.values).as_euler("xyz", degrees=True)[:, 2]

    assert np.allclose(angles, [0, 30, 60, 90, 120], atol=1e-5)
    assert result.interpolated.tolist() == [False, True, True, True, False]


def test_rotation_multi_track_gaps_are_independent() -> None:
    values = np.broadcast_to(np.eye(3), (5, 2, 3, 3)).copy()
    values[3, 1] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    valid = np.ones((5, 2), dtype=bool)
    valid[1:3, 1] = False

    result = interpolate_rotation_gaps(values, valid, max_gap=2)

    assert result.valid[:, 0].all()
    assert result.interpolated[1:3, 1].all()
    assert np.allclose(result.values[2, 1], Rotation.from_euler("x", 60, degrees=True).as_matrix())


def test_rotation_smoothing_reduces_local_jitter_and_keeps_valid_rotations() -> None:
    angles = np.array([0.0, 0.0, 20.0, 0.0, 0.0])
    values = Rotation.from_rotvec(np.column_stack((np.zeros(5), np.zeros(5), np.radians(angles)))).as_matrix()
    result = smooth_rotation_tracks(values, np.ones(5, dtype=bool), strength=1.0)

    before = Rotation.from_matrix(values).as_rotvec()
    after = Rotation.from_matrix(result).as_rotvec()
    assert np.linalg.norm(np.diff(after, n=2, axis=0)) < np.linalg.norm(np.diff(before, n=2, axis=0))
    assert np.allclose(np.linalg.det(result), 1.0)
    assert np.allclose(result[[0, -1]], values[[0, -1]])


def test_shape_and_parameter_validation() -> None:
    with pytest.raises(StageIOError):
        interpolate_vector_gaps(np.zeros(3), np.ones(3, dtype=bool), max_gap=2)
    with pytest.raises(StageIOError):
        interpolate_vector_gaps(np.zeros((4, 3)), np.ones(3, dtype=bool), max_gap=2)
    with pytest.raises(StageIOError):
        interpolate_rotation_gaps(np.zeros((4, 3, 3)), np.ones(3, dtype=bool), max_gap=2)
    with pytest.raises(StageIOError):
        smooth_rotation_tracks(np.zeros((4, 3, 3)), np.ones(3, dtype=bool))

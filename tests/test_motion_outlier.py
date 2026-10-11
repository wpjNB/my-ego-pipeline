from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego3d_action.errors import StageIOError
from ego3d_action.refinement.motion_outlier import detect_motion_outliers


def test_rejects_short_out_and_back_segment() -> None:
    position = np.zeros((12, 3), dtype=np.float64)
    position[:, 0] = np.arange(12) * 0.01
    position[4, 0] += 0.8
    valid = np.ones(12, dtype=bool)

    result = detect_motion_outliers(position, valid)

    assert result.outlier[4]
    assert result.block_outlier[4]
    assert result.valid.sum() == 11


def test_hampel_catches_isolated_spike_below_block_threshold() -> None:
    position = np.zeros((12, 3), dtype=np.float64)
    position[5, 0] = 0.4
    valid = np.ones(12, dtype=bool)

    result = detect_motion_outliers(
        position, valid, position_step_threshold=2.0, position_floor=0.1
    )

    assert result.hampel_outlier[5]
    assert not result.block_outlier[5]


def test_sustained_fast_motion_is_not_rejected_as_a_recovery_spike() -> None:
    position = np.zeros((12, 3), dtype=np.float64)
    position[:, 0] = np.arange(12) * 0.2

    result = detect_motion_outliers(position, np.ones(12, dtype=bool))

    assert not result.outlier.any()


def test_rotation_out_and_back_is_rejected() -> None:
    position = np.zeros((8, 3), dtype=np.float64)
    rotations = np.broadcast_to(np.eye(3), (8, 3, 3)).copy()
    rotations[3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()

    result = detect_motion_outliers(
        position, np.ones(8, dtype=bool), rotation=rotations
    )

    assert result.outlier[3]
    assert result.block_outlier[3]


def test_invalid_frame_is_not_an_interpolation_anchor() -> None:
    position = np.zeros((5, 3), dtype=np.float64)
    position[2] = np.nan

    result = detect_motion_outliers(position, np.ones(5, dtype=bool))

    assert not result.valid[2]
    assert result.outlier[2]


def test_shape_and_parameter_validation() -> None:
    with pytest.raises(StageIOError):
        detect_motion_outliers(np.zeros((4, 2)), np.ones(4, dtype=bool))
    with pytest.raises(StageIOError):
        detect_motion_outliers(np.zeros((4, 3)), np.ones(3, dtype=bool))
    with pytest.raises(StageIOError):
        detect_motion_outliers(np.zeros((4, 3)), np.ones(4, dtype=bool), max_block=0)

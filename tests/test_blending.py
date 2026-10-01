"""Phase 2: linear overlap blending of HaWoR windows."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego3d_action.errors import StageIOError
from ego3d_action.hand.temporal_blend import (
    HandWindow,
    blend_hand_windows,
    overlap_alpha,
    overlap_alpha_ramp,
    smooth_hand_trajectory,
)


def make_window(
    start: int,
    num_frames: int,
    fill: float,
    *,
    rotation_deg: float = 0.0,
    valid: np.ndarray | None = None,
) -> HandWindow:
    joints = np.full((num_frames, 2, 21, 3), fill, dtype=np.float64)
    joints[..., 0] = fill
    root_rot = np.broadcast_to(
        Rotation.from_euler("z", rotation_deg, degrees=True).as_matrix(),
        (num_frames, 2, 3, 3),
    ).copy()
    mask = np.ones((num_frames, 2), dtype=bool) if valid is None else valid
    return HandWindow(start=start, joints_camera=joints, valid=mask, root_rot=root_rot)


def test_single_window_passes_through() -> None:
    window = make_window(0, 5, 0.25)
    result = blend_hand_windows([window])
    assert result.joints_camera.shape == (5, 2, 21, 3)
    assert np.allclose(result.joints_camera, 0.25)
    assert result.valid.all()


def test_ramp_is_linear_across_overlap() -> None:
    window_a = make_window(0, 6, 0.0)
    window_b = make_window(4, 6, 1.0)
    result = blend_hand_windows([window_a, window_b])
    # Frames 0-3 come from A only, 4-5 are shared, 6-9 come from B only.
    assert np.allclose(result.joints_camera[:4], 0.0)
    assert np.allclose(result.joints_camera[6:], 1.0)
    assert np.isclose(result.joints_camera[4, 0, 0, 0], 0.0)
    assert np.isclose(result.joints_camera[5, 0, 0, 0], 1.0)
    assert np.allclose(result.weight[:4, 0], 1.0)
    assert np.allclose(result.weight[4:6, 0], 2.0)


def test_alpha_ramp_matches_documented_formula() -> None:
    window_a = make_window(0, 200, 0.0)
    window_b = make_window(160, 200, 1.0)
    ramp = overlap_alpha_ramp(window_b, window_a)
    assert ramp[:40][0] == pytest.approx(0.0)
    assert ramp[:40][-1] == pytest.approx(1.0)
    assert ramp[40:].tolist() == pytest.approx([1.0] * 160)
    assert np.allclose(ramp[:40], np.linspace(0.0, 1.0, 40))


def test_missing_frames_stay_missing() -> None:
    mask = np.zeros((4, 2), dtype=bool)
    mask[:, 0] = True
    window = make_window(0, 4, 0.5, valid=mask)
    result = blend_hand_windows([window])
    assert result.valid[:, 0].all()
    assert not result.valid[:, 1].any()
    assert np.allclose(result.weight[:, 1], 0.0)


def test_gap_between_windows_stays_missing() -> None:
    window_a = make_window(0, 3, 0.0)
    window_b = make_window(6, 3, 1.0)
    result = blend_hand_windows([window_a, window_b])
    assert result.joints_camera.shape[0] == 9
    assert not result.valid[3:6].any()


def test_rotation_uses_slerp() -> None:
    from ego3d_action.geometry.transforms import rotation_angle

    window_a = make_window(0, 6, 0.0, rotation_deg=0.0)
    window_b = make_window(2, 6, 0.0, rotation_deg=90.0)
    result = blend_hand_windows([window_a, window_b])
    assert np.allclose(result.root_rot[0], np.eye(3))
    # Overlap frames 2..5 share a [0, 1/3, 2/3, 1] ramp.
    observed = [np.degrees(rotation_angle(result.root_rot[f])) for f in range(2, 6)]
    assert observed == pytest.approx([0.0, 30.0, 60.0, 90.0], abs=1e-6)
    assert np.allclose(result.root_rot[7], window_b.root_rot[5], atol=1e-9)


def test_confidence_is_aggregated_with_max() -> None:
    window_a = HandWindow(
        start=0,
        joints_camera=np.zeros((3, 2, 21, 3)),
        valid=np.ones((3, 2), dtype=bool),
        confidence=np.full((3, 2), 0.9),
    )
    window_b = HandWindow(
        start=2,
        joints_camera=np.ones((3, 2, 21, 3)),
        valid=np.ones((3, 2), dtype=bool),
        confidence=np.full((3, 2), 0.4),
    )
    result = blend_hand_windows([window_a, window_b])
    assert np.allclose(result.confidence[0], 0.9)
    assert np.allclose(result.confidence[4], 0.4)
    assert np.allclose(result.confidence[2], 0.9)


def test_empty_window_list_raises() -> None:
    with pytest.raises(StageIOError):
        blend_hand_windows([])


def test_overlap_alpha_validates_input() -> None:
    assert overlap_alpha(0, 1) == 1.0
    assert overlap_alpha(1, 4) == pytest.approx(1.0 / 3.0)
    with pytest.raises(StageIOError):
        overlap_alpha(4, 4)
    with pytest.raises(StageIOError):
        overlap_alpha(0, 0)


def test_window_shape_validation() -> None:
    with pytest.raises(StageIOError):
        HandWindow(start=0, joints_camera=np.zeros((3, 2, 20, 3)), valid=np.ones((3, 2), dtype=bool))


def test_vertices_blend_like_joints() -> None:
    """Mesh vertices follow the same first-take-then-lerp rule as the joints."""
    verts_a = np.zeros((4, 2, 3, 3))
    verts_a[..., 0] = 1.0
    verts_b = np.zeros((4, 2, 3, 3))
    verts_b[..., 0] = 3.0  # window b starts at frame 1, overlapping 3 frames
    valid = np.ones((4, 2), dtype=bool)
    window_a = HandWindow(
        start=0, joints_camera=np.zeros((4, 2, 21, 3)), valid=valid,
        confidence=np.full((4, 2), 0.9), vertices_camera=verts_a,
    )
    window_b = HandWindow(
        start=1, joints_camera=np.zeros((4, 2, 21, 3)), valid=valid,
        confidence=np.full((4, 2), 0.9), vertices_camera=verts_b,
    )
    result = blend_hand_windows([window_a, window_b])
    assert result.vertices_camera is not None
    assert result.vertices_camera.shape == (5, 2, 3, 3)
    assert np.allclose(result.vertices_camera[0, 0, :, 0], 1.0)  # only a covers it
    assert np.allclose(result.vertices_camera[2, 0, :, 0], 2.0)  # ramp midpoint (a=0.5)
    assert np.allclose(result.vertices_camera[3, 0, :, 0], 3.0)  # ramp end (a=1)
    assert np.allclose(result.vertices_camera[4, 0, :, 0], 3.0)  # only b covers it

    # A window without vertices just does not contribute mesh geometry.
    mixed = blend_hand_windows([
        HandWindow(start=0, joints_camera=np.zeros((4, 2, 21, 3)), valid=valid),
        window_b,
    ])
    assert mixed.vertices_camera is not None
    assert np.allclose(mixed.vertices_camera[0, 0], 0.0)  # no geometry before b starts
    assert np.allclose(mixed.vertices_camera[1:, 0, :, 0], 3.0)

    with pytest.raises(StageIOError):
        HandWindow(
            start=0, joints_camera=np.zeros((3, 2, 21, 3)), valid=np.ones((3, 2), dtype=bool),
            vertices_camera=np.zeros((3, 2, 3)),
        )


def test_smoothing_damps_high_frequency_noise_without_crossing_gaps() -> None:
    """One binomial pass halves a single-frame spike; gaps and edges stay put."""
    joints = np.zeros((7, 2, 21, 3))
    joints[:, 0, 0, 2] = [0, 1, 2, 30, 4, 5, 6]  # a 30 mm spike on the left wrist
    valid = np.ones((7, 2), dtype=bool)
    valid[5:, 0] = False  # a gap after frame 4 - the spike must not smear into it
    smoothed, verts = smooth_hand_trajectory(joints.copy(), valid, passes=1)
    assert verts is None
    z = smoothed[:, 0, 0, 2]
    assert np.isclose(z[3], 16.5)  # spike damped: 0.25*2 + 0.5*30 + 0.25*4
    assert np.isclose(z[2], 8.75)  # the neighbour legitimately picks up spike energy
    assert z[4] == 4.0  # last frame of the run keeps its value (edge)
    assert z[0] == 0.0 and z[1] == 1.0  # run head keeps its edge values

    # vertices ride along with the same taps
    verts_in = np.zeros((7, 2, 4, 3))
    verts_in[:, 0, :, 2] = joints[:, 0, 0, 2][:, None]
    sm_j, sm_v = smooth_hand_trajectory(joints.copy(), valid, vertices_camera=verts_in, passes=1)
    assert np.isclose(sm_v[3, 0, :, 2], sm_j[3, 0, 0, 2]).all()

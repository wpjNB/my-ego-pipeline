"""Phase 4: depth-derived Sim(3) window stitching."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.camera.stitch import align_window_pair, stitch_camera_windows
from ego3d_action.camera.window import CameraWindow, WindowRange
from ego3d_action.errors import InsufficientDataError, StageIOError
from ego3d_action.geometry.sim3 import Sim3
from ego3d_action.geometry.transforms import rotation_angle

from conftest import build_two_windows


@pytest.fixture(scope="module")
def scaled_windows(
    synthetic_scene: dict[str, object], known_sim3: Sim3
) -> tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray]:
    return build_two_windows(synthetic_scene, known_sim3)


@pytest.fixture(scope="module")
def aligned_windows(
    synthetic_scene: dict[str, object]
) -> tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray]:
    return build_two_windows(synthetic_scene, Sim3.identity())


def test_align_window_pair_recovers_the_relative_sim3(
    scaled_windows: tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray],
    known_sim3: Sim3,
) -> None:
    window_a, window_b, _, _ = scaled_windows

    sim3, diagnostics = align_window_pair(window_b, window_a, stride=8, random_state=0)
    inverse = known_sim3.inverse()

    assert diagnostics.num_correspondences > 100
    assert diagnostics.inlier_ratio > 0.9
    assert np.isclose(sim3.scale, inverse.scale, rtol=1e-3)
    assert np.allclose(sim3.rotation, inverse.rotation, atol=1e-3)
    assert np.allclose(sim3.translation, inverse.translation, atol=5e-3)

def test_stitched_trajectory_lands_in_world_zero(
    scaled_windows: tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray],
    known_sim3: Sim3,
) -> None:
    window_a, window_b, rot_a, trans_a = scaled_windows
    stitched = stitch_camera_windows(
        [window_a, window_b], num_frames=360, stride=8, random_state=0
    )

    assert stitched.valid.all()
    assert np.allclose(stitched.rotation_c2w[0], np.eye(3), atol=1e-6)
    assert np.allclose(stitched.translation_c2w[0], np.zeros(3), atol=1e-6)

    translation_error = np.linalg.norm(stitched.translation_c2w - trans_a, axis=-1)
    rotation_error = np.degrees(rotation_angle(np.einsum("tji,tjk->tik", stitched.rotation_c2w, rot_a)))
    assert translation_error.max() < 5e-3, f"max translation error {translation_error.max():.4f} m"
    assert rotation_error.max() < 0.1, f"max rotation error {rotation_error.max():.4f} deg"

    assert len(stitched.sim3) == 2
    assert np.isclose(stitched.sim3[0].scale, 1.0)
    assert np.allclose(stitched.sim3[1].rotation, known_sim3.inverse().rotation, atol=1e-3)


def test_blending_is_continuous_across_the_overlap(
    blended_trajectory: tuple[np.ndarray, np.ndarray]
) -> None:
    _, translation = blended_trajectory
    steps = np.linalg.norm(np.diff(translation, axis=0), axis=-1)
    # No step inside or across the blend region may jump by more than a few cm.
    assert steps.max() < 0.05
    assert np.isfinite(translation).all()


@pytest.fixture(scope="module")
def blended_trajectory(
    scaled_windows: tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    window_a, window_b, _, _ = scaled_windows
    stitched = stitch_camera_windows([window_a, window_b], num_frames=360, stride=8, random_state=0)
    return stitched.rotation_c2w, stitched.translation_c2w


def test_single_window_stitch_is_a_passthrough(
    aligned_windows: tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray]
) -> None:
    window_a, _, rot_a, trans_a = aligned_windows
    stitched = stitch_camera_windows([window_a], num_frames=200, stride=8)
    assert np.allclose(stitched.translation_c2w, trans_a[:200], atol=1e-12)
    assert np.allclose(stitched.rotation_c2w, rot_a[:200], atol=1e-12)


def test_non_overlapping_windows_raise(
    synthetic_scene: dict[str, object],
    aligned_windows: tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray],
) -> None:
    window_a = aligned_windows[0]
    other = CameraWindow(
        window=WindowRange(index=1, start=500, end=520),
        rotation_c2w=np.broadcast_to(np.eye(3), (20, 3, 3)).copy(),
        translation_c2w=np.zeros((20, 3)),
        intrinsics=np.broadcast_to(np.asarray(synthetic_scene["intrinsics"]), (20, 3, 3)).copy(),
        depth=np.full((20, int(synthetic_scene["height"]), int(synthetic_scene["width"])), 1.5),
    )
    with pytest.raises(StageIOError):
        align_window_pair(other, window_a, stride=8)


def test_empty_and_unordered_windows_raise(
    aligned_windows: tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray]
) -> None:
    window_a, window_b, _, _ = aligned_windows
    with pytest.raises(StageIOError):
        stitch_camera_windows([], num_frames=10)
    with pytest.raises(StageIOError):
        stitch_camera_windows([window_b, window_a], num_frames=360)


def test_num_frames_smaller_than_last_window_raises(
    aligned_windows: tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray]
) -> None:
    window_a, window_b, _, _ = aligned_windows
    with pytest.raises(StageIOError):
        stitch_camera_windows([window_a, window_b], num_frames=300)


def test_depth_correspondences_require_depth_variation(synthetic_scene: dict[str, object]) -> None:
    """A window with no usable depth cannot be aligned."""
    intrinsics = np.asarray(synthetic_scene["intrinsics"])
    height = int(synthetic_scene["height"])
    width = int(synthetic_scene["width"])
    empty = CameraWindow(
        window=WindowRange(index=0, start=0, end=20),
        rotation_c2w=np.broadcast_to(np.eye(3), (20, 3, 3)).copy(),
        translation_c2w=np.zeros((20, 3)),
        intrinsics=np.broadcast_to(intrinsics, (20, 3, 3)).copy(),
        depth=np.full((20, height, width), np.nan),
    )
    with pytest.raises(StageIOError):
        align_window_pair(empty, empty, stride=8)


def test_depth_correspondences_vectorize_common_valid_samples_and_confidence() -> None:
    from ego3d_action.camera.depth import build_depth_correspondences

    frames, height, width = 3, 3, 4
    intrinsics = np.broadcast_to(
        np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
        (frames, 3, 3),
    ).copy()
    rotations = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
    translations_src = np.zeros((frames, 3))
    translations_dst = np.zeros((frames, 3))
    translations_dst[:, 0] = 9.0
    depth_src = np.full((frames, height, width), 2.0)
    depth_dst = np.full((frames, height, width), 2.0)
    confidence_src = np.full_like(depth_src, 0.8)
    confidence_dst = np.ones_like(depth_dst)

    # The windows share global frames 1 and 2. Each invalid sample must be
    # excluded from both paired arrays, including confidence-filtered samples.
    depth_src[1, 0, 0] = np.nan
    depth_dst[0, 0, 1] = np.nan
    confidence_src[1, 0, 2] = 0.2
    src = CameraWindow(
        window=WindowRange(index=0, start=0, end=3),
        rotation_c2w=rotations,
        translation_c2w=translations_src,
        intrinsics=intrinsics,
        depth=depth_src,
        depth_confidence=confidence_src,
    )
    dst = CameraWindow(
        window=WindowRange(index=1, start=1, end=4),
        rotation_c2w=rotations,
        translation_c2w=translations_dst,
        intrinsics=intrinsics,
        depth=depth_dst,
        depth_confidence=confidence_dst,
    )

    correspondences = build_depth_correspondences(
        src, dst, stride=1, min_confidence=0.5, max_points=None
    )

    assert correspondences.count == 21
    frame_ids, frame_counts = np.unique(correspondences.frames, return_counts=True)
    assert np.array_equal(frame_counts, [9, 12])
    assert np.array_equal(frame_ids, [1, 2])
    assert np.allclose(
        correspondences.points_dst - correspondences.points_src, [9.0, 0.0, 0.0]
    )
    assert np.allclose(correspondences.weights, 0.8)

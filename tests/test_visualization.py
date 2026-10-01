"""Debug video overlays and trajectory plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.visualization import overlay

cv2 = pytest.importorskip("cv2")


def write_frames(directory: Path, count: int = 4) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        frame[10:20, 10:20] = 200
        path = directory / f"{index:06d}.jpg"
        assert cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def test_draw_detections_marks_the_frame() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    boxes = np.array([[5.0, 5.0, 20.0, 20.0], [30.0, 5.0, 45.0, 20.0]])
    confidence = np.array([0.9, 0.2])
    valid = np.array([True, False])
    drawn = overlay.draw_detections(frame, boxes, confidence, valid)
    assert drawn.shape == frame.shape
    assert drawn.sum() > 0
    assert np.array_equal(frame, np.zeros_like(frame))  # input untouched


def test_draw_detections_validates_shapes() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    with pytest.raises(StageIOError):
        overlay.draw_detections(frame, np.zeros((3, 4)), np.zeros(2), np.ones(2, dtype=bool))


def test_draw_hand_projection_draws_a_skeleton() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    intrinsics = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]])
    joints = np.zeros((2, 21, 3))
    joints[..., 2] = 0.5
    joints[..., 0] = np.linspace(-0.05, 0.05, 21)[None, :]
    drawn = overlay.draw_hand_projection(frame, joints, intrinsics, np.array([True, False]))
    assert drawn.sum() > 0


def test_draw_hand_projection_validates_shapes() -> None:
    with pytest.raises(StageIOError):
        overlay.draw_hand_projection(
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((2, 20, 3)),
            np.eye(3),
            np.ones(2, dtype=bool),
        )


def test_write_detection_video_roundtrip(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", count=6)
    boxes = np.tile(np.array([[[5.0, 5.0, 20.0, 20.0], [30.0, 5.0, 45.0, 20.0]]]), (6, 1, 1))
    confidence = np.full((6, 2), 0.9)
    valid = np.ones((6, 2), dtype=bool)
    out = overlay.write_detection_video(
        frames, boxes, confidence, valid, tmp_path / "01_detection.mp4", fps=6.0
    )
    assert out.is_file() and out.stat().st_size > 0

    capture = cv2.VideoCapture(str(out))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    finally:
        capture.release()


def test_write_detection_video_rejects_too_few_boxes(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", count=4)
    with pytest.raises(StageIOError):
        overlay.write_detection_video(
            frames,
            np.zeros((2, 2, 4)),
            np.zeros((4, 2)),
            np.ones((4, 2), dtype=bool),
            tmp_path / "bad.mp4",
        )


def test_write_hand_video_roundtrip(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", count=3)
    joints = np.zeros((3, 2, 21, 3))
    joints[..., 2] = 1.0
    intrinsics = np.broadcast_to(
        np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]]), (3, 3, 3)
    ).copy()
    valid = np.ones((3, 2), dtype=bool)
    out = overlay.write_hand_video(
        frames, joints, intrinsics, valid, tmp_path / "02_hawor.mp4", fps=3.0
    )
    assert out.is_file() and out.stat().st_size > 0


def test_plot_world_trajectory_writes_a_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    joints = np.zeros((10, 2, 21, 3))
    joints[..., 0] = np.linspace(0.0, 0.4, 10)[:, None, None]
    out = overlay.plot_world_trajectory(joints, np.zeros((10, 3)), tmp_path / "trajectory.png")
    assert out.is_file() and out.stat().st_size > 1000

    with pytest.raises(StageIOError):
        overlay.plot_world_trajectory(np.zeros((10, 2, 20, 3)), None, tmp_path / "bad.png")


def test_wrist_comparison_video_and_stills(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", count=5)
    total = len(frames)
    joints = np.full((total, 2, 21, 3), np.nan)
    joints[:, :, 0, :] = np.array([0.0, 0.0, 1.5])
    rotation = np.broadcast_to(np.eye(3), (total, 3, 3)).copy()
    translation = np.zeros((total, 3))
    intrinsics = np.broadcast_to(
        np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]]), (total, 3, 3)
    ).copy()
    valid = np.ones((total, 2), dtype=bool)

    out = overlay.write_wrist_comparison_video(
        frames,
        joints,
        rotation,
        translation,
        intrinsics,
        valid,
        tmp_path / "gt_vs_pred.mp4",
        prediction_world=joints,
        fps=5.0,
        still_indices=[0, 3],
    )
    assert out.is_file() and out.stat().st_size > 0
    stills = sorted((tmp_path / "gt_vs_pred_stills").glob("*.png"))
    assert [path.name for path in stills] == ["000000.png", "000003.png"]

    capture = cv2.VideoCapture(str(out))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == total
    finally:
        capture.release()


def test_wrist_comparison_draws_only_projectable_points(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", count=2)
    joints = np.full((2, 2, 21, 3), np.nan)
    joints[:, :, 0, :] = np.array([0.0, 0.0, -1.0])  # behind the camera
    rotation = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    intrinsics = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    out = overlay.write_wrist_comparison_video(
        frames,
        joints,
        rotation,
        np.zeros((2, 3)),
        intrinsics,
        np.ones((2, 2), dtype=bool),
        tmp_path / "empty.mp4",
        fps=2.0,
    )
    assert out.is_file()


def test_world_to_camera_is_the_inverse_of_the_stored_pose() -> None:
    """Every overlay must apply c2w before projecting - never the intrinsics alone."""
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([1.0, 2.0, 3.0])
    point_world = np.array([1.0, 2.0, 2.0])  # one metre in front of the camera
    camera = overlay.world_to_camera(point_world, rotation, translation)
    assert np.allclose(camera, [0.0, 0.0, -1.0])  # camera looks down -z

    # Projecting a world point without the transform is a different pixel entirely.
    intrinsics = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    with_camera = intrinsics @ camera
    without = intrinsics @ point_world
    assert not np.allclose(with_camera[:2] / with_camera[2], without[:2] / without[2])


def test_wrist_comparison_can_draw_the_reference_skeleton(tmp_path: Path) -> None:
    """The skeleton needs finite joints; a wrist-only reference still renders."""
    frames = write_frames(tmp_path / "frames", count=3)
    total = len(frames)
    intrinsics = np.broadcast_to(
        np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]]), (total, 3, 3)
    ).copy()
    rotation = np.broadcast_to(np.eye(3), (total, 3, 3)).copy()
    translation = np.zeros((total, 3))
    valid = np.ones((total, 2), dtype=bool)

    full = np.full((total, 2, 21, 3), 0.5)
    full[:, 0, 0, :] = 0.0
    skeleton = overlay.write_wrist_comparison_video(
        frames,
        full,
        rotation,
        translation,
        intrinsics,
        valid,
        tmp_path / "skeleton.mp4",
        draw_skeleton=True,
        fps=3.0,
    )
    assert skeleton.is_file() and skeleton.stat().st_size > 0

    wrist_only = np.full((total, 2, 21, 3), np.nan)
    wrist_only[:, :, 0, :] = 0.5
    plain = overlay.write_wrist_comparison_video(
        frames,
        wrist_only,
        rotation,
        translation,
        intrinsics,
        valid,
        tmp_path / "wrist_only.mp4",
        draw_skeleton=True,
        fps=3.0,
    )
    assert plain.is_file() and plain.stat().st_size > 0


def test_draw_hand_mesh_rasterises_with_occlusion_and_holes() -> None:
    """Mesh drawing: near faces win, invalid hands and behind-camera faces skip."""
    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    # A tetrahedron at z = 2 m: one big far face plus a small near face that
    # overlaps it - the near face must win where they cover the same pixels.
    verts = np.zeros((2, 4, 3))
    verts[0, 0] = [-0.2, -0.2, 2.0]
    verts[0, 1] = [0.2, -0.2, 2.0]
    verts[0, 2] = [0.0, 0.2, 2.0]
    verts[0, 3] = [0.0, 0.0, 1.5]  # the near vertex
    verts[1] = verts[0]
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    intrinsics = np.array([[32.0, 0, 24], [0, 32.0, 24], [0, 0, 1]])
    canvas = overlay.draw_hand_mesh(
        frame, verts, faces, intrinsics, np.array([True, False]),
        colours=[(0, 0, 255), (0, 255, 0)],
    )
    drawn = (canvas.sum(axis=2) > 0).sum()
    assert 0 < drawn < 48 * 48  # something appeared, image not flooded
    assert (canvas[24, 24] > 0).any()  # the near faces cover the centre

    # Both hands invalid: canvas untouched.
    untouched = overlay.draw_hand_mesh(
        frame, verts, faces, intrinsics, np.array([False, False])
    )
    assert not (untouched.sum(axis=2) > 0).any()

    # A face entirely behind the camera is dropped, not projected through.
    behind = verts.copy()
    behind[..., 2] = -1.0
    empty = overlay.draw_hand_mesh(
        frame, behind, faces, intrinsics, np.array([True, False])
    )
    assert not (empty.sum(axis=2) > 0).any()

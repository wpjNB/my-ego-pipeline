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

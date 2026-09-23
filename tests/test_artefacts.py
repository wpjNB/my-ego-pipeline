"""On-disk layout and per-stage artefact round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego3d_action.camera.stitch import (
    StitchedCamera,
    load_sim3_transforms,
    load_stitched_camera,
    save_sim3_transforms,
    save_stitched_camera,
)
from ego3d_action.camera.window import load_camera_window, save_camera_window
from ego3d_action.errors import StageIOError
from ego3d_action.geometry.sim3 import Sim3
from ego3d_action.io.artefacts import (
    ClipLayout,
    load_detection,
    load_hand,
    save_detection,
    save_hand,
)

from conftest import make_intrinsics, make_window


def test_layout_matches_the_documented_tree(tmp_path: Path) -> None:
    layout = ClipLayout(data_root=tmp_path, clip="demo01")
    assert layout.frames_dir == tmp_path / "demo01" / "frames"
    assert layout.detection_path == tmp_path / "demo01" / "detection" / "detection.npz"
    assert layout.hand_path == tmp_path / "demo01" / "hand" / "hand_camera.npz"
    assert layout.camera_windows_dir == tmp_path / "demo01" / "camera" / "windows"
    assert layout.stitched_camera_path == tmp_path / "demo01" / "camera" / "stitched_camera.npz"
    assert layout.sim3_path == tmp_path / "demo01" / "stitched" / "sim3_transforms.npz"
    assert layout.trajectory_path == tmp_path / "demo01" / "trajectory" / "trajectory.npz"
    assert layout.window_path(0, 200).name == "000000_000199.npz"
    assert layout.window_path(160, 360).name == "000160_000359.npz"
    with pytest.raises(StageIOError):
        layout.window_path(200, 200)


def test_ensure_dirs_creates_every_stage_folder(tmp_path: Path) -> None:
    layout = ClipLayout(data_root=tmp_path, clip="demo01")
    layout.ensure_dirs()
    assert layout.visualization_dir.is_dir()
    assert layout.trajectory_dir.is_dir()


def test_detection_roundtrip(tmp_path: Path) -> None:
    layout = ClipLayout(data_root=tmp_path, clip="demo01")
    arrays = {
        "boxes": np.zeros((5, 2, 4)),
        "confidence": np.full((5, 2), 0.9),
        "valid": np.ones((5, 2), dtype=bool),
        "track_id": np.zeros((5, 2), dtype=np.int64),
    }
    save_detection(layout, arrays, metadata={"stage": "phase1_detection"})
    loaded = load_detection(layout)
    assert np.allclose(loaded["confidence"], 0.9)
    assert (layout.detection_dir / "boxes.npy").is_file()


def test_detection_requires_the_contract_fields(tmp_path: Path) -> None:
    layout = ClipLayout(data_root=tmp_path, clip="demo01")
    with pytest.raises(StageIOError, match="missing fields"):
        save_detection(layout, {"boxes": np.zeros((5, 2, 4))})
    with pytest.raises(StageIOError, match=r"\[T, 2\]"):
        save_detection(
            layout,
            {
                "boxes": np.zeros((5, 2, 4)),
                "confidence": np.zeros((5, 2)),
                "valid": np.zeros((5, 3), dtype=bool),
                "track_id": np.zeros((5, 2), dtype=np.int64),
            },
        )


def test_hand_roundtrip(tmp_path: Path) -> None:
    layout = ClipLayout(data_root=tmp_path, clip="demo01")
    save_hand(
        layout,
        {
            "joints_camera": np.zeros((4, 2, 21, 3)),
            "valid": np.ones((4, 2), dtype=bool),
            "confidence": np.ones((4, 2)),
            "root_rot": np.broadcast_to(np.eye(3), (4, 2, 3, 3)).copy(),
            "betas": np.zeros((4, 2, 10)),
        },
    )
    loaded = load_hand(layout)
    assert loaded["joints_camera"].shape == (4, 2, 21, 3)


def test_camera_window_roundtrip(tmp_path: Path) -> None:
    window = make_window(
        index=0,
        start=0,
        num_frames=3,
        rotation=np.broadcast_to(np.eye(3), (3, 3, 3)).copy(),
        translation=np.zeros((3, 3)),
        depth=np.full((3, 8, 6), 1.5),
        intrinsics=np.broadcast_to(make_intrinsics(6, 8), (3, 3, 3)).copy(),
    )
    path = tmp_path / "000000_000002.npz"
    save_camera_window(window, path)
    loaded = load_camera_window(path)
    assert loaded.start == 0 and loaded.end == 3
    assert np.allclose(loaded.depth, 1.5)


def test_stitched_camera_roundtrip(tmp_path: Path) -> None:
    total = 5
    stitched = StitchedCamera(
        rotation_c2w=np.broadcast_to(np.eye(3), (total, 3, 3)).copy(),
        translation_c2w=np.arange(total * 3, dtype=np.float64).reshape(total, 3),
        valid=np.ones(total, dtype=bool),
        weight=np.ones(total),
        intrinsics=np.broadcast_to(np.eye(3), (total, 3, 3)).copy(),
        num_frames=total,
    )
    save_stitched_camera(tmp_path / "stitched_camera.npz", stitched, metadata={"coverage": 1.0})
    loaded = load_stitched_camera(tmp_path / "stitched_camera.npz")
    assert np.allclose(loaded.translation_c2w, stitched.translation_c2w)
    assert loaded.intrinsics is not None
    assert (tmp_path / "stitched_camera.json").is_file()


def test_sim3_transform_roundtrip(tmp_path: Path) -> None:
    transforms = [Sim3.identity(), Sim3(scale=0.8, rotation=np.eye(3), translation=np.ones(3))]
    path = save_sim3_transforms(tmp_path / "sim3_transforms.npz", transforms)
    loaded = load_sim3_transforms(path)
    assert len(loaded) == 2
    assert np.isclose(loaded[1].scale, 0.8)
    assert np.allclose(loaded[1].translation, np.ones(3))
    with pytest.raises(StageIOError):
        save_sim3_transforms(tmp_path / "empty.npz", [])
    with pytest.raises(StageIOError):
        load_sim3_transforms(tmp_path / "missing.npz")

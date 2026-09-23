"""Backend adapters: probes, runner wiring, conversion and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego3d_action.camera import vggt_omega
from ego3d_action.detection import wilor
from ego3d_action.errors import (
    BackendNotAvailableError,
    StageIOError,
)
from ego3d_action.hand import hawor
from ego3d_action.hand.temporal_blend import HandWindow
from ego3d_action.runtime.subprocess_backend import BackendInvocation

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS = REPO_ROOT / "backends"


def mock_invocation(seed: int = 0) -> BackendInvocation:
    return BackendInvocation(
        mode="mock", backends_dir=BACKENDS, python_commands={}, timeout_seconds=120, seed=seed
    )


def real_invocation() -> BackendInvocation:
    return BackendInvocation(
        mode="real",
        backends_dir=BACKENDS,
        python_commands={"wilor": ("python",), "hawor": ("python",), "vggt": ("python",)},
        timeout_seconds=120,
    )


def make_frames(directory: Path, count: int = 30) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:06d}.jpg").write_bytes(b"stub")
    return directory


# ---------------------------------------------------------------- probes


def test_probes_report_missing_checkout_and_weights(tmp_path: Path) -> None:
    status = wilor.probe(tmp_path / "third_party", tmp_path / "weights")
    assert not status.available
    assert len(status.missing) == 2
    with pytest.raises(BackendNotAvailableError) as excinfo:
        wilor.require(tmp_path / "third_party", tmp_path / "weights")
    assert "WiLoR" in str(excinfo.value)
    assert "third_party" in excinfo.value.hint


def test_probes_report_available_when_both_exist(tmp_path: Path) -> None:
    third_party = tmp_path / "third_party"
    weights = tmp_path / "weights"
    (third_party / "WiLoR").mkdir(parents=True)
    (weights / "wilor").mkdir(parents=True)
    assert wilor.probe(third_party, weights).available


# ------------------------------------------------------------- Phase 1


def test_wilor_converts_raw_detections_to_handedness() -> None:
    frames = [
        [
            wilor.RawDetection(bbox=np.array([0.0, 0.0, 10.0, 10.0]), confidence=0.9, right_score=0.8),
            wilor.RawDetection(bbox=np.array([20.0, 0.0, 30.0, 10.0]), confidence=0.9, left_score=0.7),
        ]
    ]
    detections = wilor.to_hand_detections(frames)
    assert detections[0][0].handedness == 1
    assert detections[0][1].handedness == 0


def test_raw_detections_from_arrays_respects_count() -> None:
    arrays = {
        "boxes": np.zeros((3, 2, 4)),
        "confidence": np.full((3, 2), 0.9),
        "count": np.array([2, 1, 0]),
    }
    frames = wilor.raw_detections_from_arrays(arrays)
    assert [len(frame) for frame in frames] == [2, 1, 0]
    with pytest.raises(StageIOError):
        wilor.raw_detections_from_arrays({"boxes": np.zeros((3, 2, 3)), "confidence": np.zeros((3, 2))})
    with pytest.raises(StageIOError):
        wilor.raw_detections_from_arrays(
            {"boxes": np.zeros((3, 2, 4)), "confidence": np.zeros((3, 2)), "count": np.zeros(5)}
        )


def test_wilor_tracking_path_is_model_free() -> None:
    frames = []
    for index in range(6):
        frames.append(
            [
                wilor.RawDetection(
                    bbox=np.array([10.0 * index, 0.0, 10.0 * index + 10.0, 10.0]),
                    confidence=0.9,
                    right_score=0.9,
                )
            ]
        )
    arrays = wilor.track_clip(frames)
    assert arrays["valid"].shape == (6, 2)
    assert arrays["valid"][:, 1].all()
    assert not arrays["valid"][:, 0].any()


def test_wilor_detect_clip_in_mock_mode(tmp_path: Path) -> None:
    frames_dir = make_frames(tmp_path / "frames", count=40)
    arrays = wilor.detect_clip(
        frames_dir,
        tmp_path / "raw.npz",
        invocation=mock_invocation(),
        third_party=tmp_path / "third_party",
        weights_root=tmp_path / "weights",
        num_frames=40,
        width=320,
        height=240,
    )
    assert arrays["boxes"].shape[0] == 40
    # The mock plants a recoverable gap and an unrecoverable one.
    tracked = wilor.track_clip(wilor.raw_detections_from_arrays(arrays))
    assert tracked["valid"][20, 1]  # recovered low-confidence detection
    assert not tracked["valid"][18:20, 1].any()
    assert not tracked["valid"][30:40, 1].any()  # 10-frame hole stays missing


def test_wilor_detect_clip_requires_the_backend_in_real_mode(tmp_path: Path) -> None:
    frames_dir = make_frames(tmp_path / "frames", count=4)
    with pytest.raises(BackendNotAvailableError):
        wilor.detect_clip(
            frames_dir,
            tmp_path / "raw.npz",
            invocation=real_invocation(),
            third_party=tmp_path / "third_party",
            weights_root=tmp_path / "weights",
            num_frames=4,
        )


# ------------------------------------------------------------- Phase 2


def test_hawor_clip_request_schedule() -> None:
    request = hawor.HaworClipRequest(num_frames=40, frames_dir=Path("frames"))
    assert request.ranges() == [(0, 16), (8, 24), (16, 32), (24, 40)]
    single = hawor.HaworClipRequest(num_frames=10, frames_dir=Path("frames"))
    assert single.ranges() == [(0, 10)]


def test_hawor_window_roundtrip(tmp_path: Path) -> None:
    window = HandWindow(
        start=8,
        joints_camera=np.zeros((3, 2, 21, 3)),
        valid=np.ones((3, 2), dtype=bool),
        confidence=np.full((3, 2), 0.8),
    )
    path = hawor.save_hand_window(window, tmp_path / "000008_000010.npz")
    loaded = hawor.load_hand_window(path)
    assert loaded.start == 8
    assert loaded.joints_camera.shape == (3, 2, 21, 3)
    assert np.allclose(loaded.confidence, 0.8)


def test_hawor_load_window_validates_fields(tmp_path: Path) -> None:
    from ego3d_action.io.serialization import save_npz

    save_npz(tmp_path / "bad.npz", start=np.array([0]))
    with pytest.raises(StageIOError, match="missing required fields"):
        hawor.load_hand_window(tmp_path / "bad.npz")


def test_hawor_run_windows_in_mock_mode(tmp_path: Path) -> None:
    frames_dir = make_frames(tmp_path / "frames", count=40)
    request = hawor.HaworClipRequest(num_frames=40, frames_dir=frames_dir)
    paths = hawor.run_windows(
        request,
        tmp_path / "hand" / "windows",
        invocation=mock_invocation(),
        third_party=tmp_path / "third_party",
        weights_root=tmp_path / "weights",
    )
    assert [path.name for path in paths] == [
        "000000_000015.npz",
        "000008_000023.npz",
        "000016_000031.npz",
        "000024_000039.npz",
    ]
    loaded = hawor.load_hand_window(paths[0])
    assert loaded.joints_camera.shape == (16, 2, 21, 3)
    # Without a detection artefact the built-in gaps apply (left hand 20-21).
    second = hawor.load_hand_window(paths[1])  # covers frames 8-23
    assert not np.isfinite(second.joints_camera[12:14, 0]).any()


def test_hawor_run_windows_requires_backend_in_real_mode(tmp_path: Path) -> None:
    request = hawor.HaworClipRequest(num_frames=16, frames_dir=tmp_path)
    with pytest.raises(BackendNotAvailableError):
        hawor.run_windows(
            request,
            tmp_path / "windows",
            invocation=real_invocation(),
            third_party=tmp_path / "third_party",
            weights_root=tmp_path / "weights",
        )


# ------------------------------------------------------------- Phase 3


def test_vggt_window_request_validation() -> None:
    with pytest.raises(StageIOError):
        vggt_omega.VggtWindowRequest(start=10, end=10, resolution=416, checkpoint="x")
    with pytest.raises(StageIOError):
        vggt_omega.VggtWindowRequest(start=0, end=10, resolution=0, checkpoint="x")
    request = vggt_omega.VggtWindowRequest(
        start=0, end=200, resolution=416, checkpoint=vggt_omega.SUPPORTED_CHECKPOINTS[0]
    )
    assert request.num_frames == 200
    assert "VGGT-Omega-1B-416-Reproduction" in vggt_omega.SUPPORTED_CHECKPOINTS


def test_vggt_run_window_in_mock_mode(tmp_path: Path) -> None:
    request = vggt_omega.VggtWindowRequest(
        start=0, end=240, resolution=416, checkpoint=vggt_omega.SUPPORTED_CHECKPOINTS[0]
    )
    paths = vggt_omega.run_window(
        request,
        invocation=mock_invocation(),
        third_party=tmp_path / "third_party",
        weights_root=tmp_path / "weights",
        out_dir=tmp_path / "camera" / "windows",
        num_frames=240,
    )
    assert [path.name for path in paths] == ["000000_000199.npz", "000160_000239.npz"]
    from ego3d_action.camera.window import load_camera_window

    window = load_camera_window(paths[1])
    assert window.window.num_frames == 80
    assert np.isfinite(window.depth).any()


def test_vggt_run_window_requires_backend_in_real_mode(tmp_path: Path) -> None:
    request = vggt_omega.VggtWindowRequest(
        start=0, end=200, resolution=416, checkpoint=vggt_omega.SUPPORTED_CHECKPOINTS[0]
    )
    with pytest.raises(BackendNotAvailableError):
        vggt_omega.run_window(
            request,
            invocation=real_invocation(),
            third_party=tmp_path / "third_party",
            weights_root=tmp_path / "weights",
            out_dir=tmp_path / "windows",
            num_frames=200,
        )


def test_vggt_empty_window_placeholder() -> None:
    window = vggt_omega.empty_window(0, 20, height=8, width=6, intrinsics=np.eye(3))
    assert window.window.num_frames == 20
    assert not np.isfinite(window.depth).any()

"""Backend adapters: probes, conversion, and explicit not-implemented paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego3d_action.camera import vggt_omega
from ego3d_action.detection import wilor
from ego3d_action.errors import (
    BackendInvocationNotImplemented,
    BackendNotAvailableError,
    StageIOError,
)
from ego3d_action.hand import hawor


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
    assert arrays["valid"][:, 1].all()  # right hand
    assert not arrays["valid"][:, 0].any()  # nothing detected as left


def test_wilor_detector_is_explicitly_not_wired(tmp_path: Path) -> None:
    third_party = tmp_path / "third_party"
    weights = tmp_path / "weights"
    (third_party / "WiLoR").mkdir(parents=True)
    (weights / "wilor").mkdir(parents=True)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "000000.jpg").write_bytes(b"not-a-real-jpeg")

    with pytest.raises(BackendInvocationNotImplemented) as excinfo:
        wilor.detect_clip(frames_dir, third_party=third_party, weights_root=weights)
    assert "ego3d_wilor" in excinfo.value.hint

    with pytest.raises(StageIOError):
        wilor.detect_clip(tmp_path / "empty", third_party=third_party, weights_root=weights)


def test_hawor_window_request_and_invocation(tmp_path: Path) -> None:
    third_party = tmp_path / "third_party"
    weights = tmp_path / "weights"
    (third_party / "HaWoR").mkdir(parents=True)
    (weights / "hawor").mkdir(parents=True)
    request = hawor.HaworWindowRequest(
        start=0,
        end=16,
        frames_dir=tmp_path,
        boxes=np.zeros((16, 2, 4)),
        valid=np.ones((16, 2), dtype=bool),
        confidence=np.ones((16, 2)),
    )
    assert request.num_frames == 16
    with pytest.raises(BackendInvocationNotImplemented) as excinfo:
        hawor.run_window(request, third_party=third_party, weights_root=weights)
    assert "ego3d_hawor" in excinfo.value.hint


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


def test_vggt_invocation_is_explicitly_not_wired(tmp_path: Path) -> None:
    third_party = tmp_path / "third_party"
    weights = tmp_path / "weights"
    (third_party / "VGGT-Omega").mkdir(parents=True)
    (weights / "vggt-omega").mkdir(parents=True)
    request = vggt_omega.VggtWindowRequest(
        start=0, end=200, resolution=416, checkpoint=vggt_omega.SUPPORTED_CHECKPOINTS[0]
    )
    with pytest.raises(BackendInvocationNotImplemented) as excinfo:
        vggt_omega.run_window(request, third_party=third_party, weights_root=weights)
    assert "ego3d_vggt" in excinfo.value.hint


def test_vggt_empty_window_placeholder() -> None:
    window = vggt_omega.empty_window(0, 20, height=8, width=6, intrinsics=np.eye(3))
    assert window.window.num_frames == 20
    assert not np.isfinite(window.depth).any()

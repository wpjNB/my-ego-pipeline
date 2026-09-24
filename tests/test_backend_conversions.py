"""The model-output -> pipeline-artefact conversions used by the real runners."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ego3d_action.camera.vggt_omega import camera_window_from_output
from ego3d_action.camera.vggt_omega import find_checkpoint as find_vggt_checkpoint
from ego3d_action.camera.vggt_omega import resolve_checkpoint as resolve_vggt_checkpoint
from ego3d_action.detection.wilor import (
    RawDetection,
    build_raw_detection_arrays,
    detections_from_predictions,
    raw_detections_from_arrays,
)
from ego3d_action.errors import StageIOError
from ego3d_action.hand.hawor import (
    find_weights_files,
    hawor_tracks_from_detection,
    hand_windows_from_joints,
    load_hand_window,
    save_hawor_tracks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str) -> object:
    """Import a ``backends/*.py`` runner as a module."""
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ WiLoR


def test_build_raw_detection_arrays_compacts_slots() -> None:
    frames: list[list[RawDetection]] = [
        [
            RawDetection(bbox=np.array([0.0, 0.0, 10.0, 10.0]), confidence=0.9, left_score=0.9),
            RawDetection(bbox=np.array([20.0, 0.0, 30.0, 10.0]), confidence=0.8, right_score=0.8),
        ],
        [],
        [RawDetection(bbox=np.array([5.0, 5.0, 15.0, 15.0]), confidence=0.4, right_score=0.7)],
    ]
    arrays = build_raw_detection_arrays(frames)
    assert arrays["boxes"].shape == (3, 2, 4)
    assert arrays["count"].tolist() == [2, 0, 1]
    assert arrays["confidence"][2, 0] == pytest.approx(0.4)
    assert arrays["right_score"][0, 1] == pytest.approx(0.8)
    assert arrays["count"][1] == 0  # "no detection" is explicit, not a fake box

    # ... and the artefact round-trips back into detections.
    restored = raw_detections_from_arrays(arrays)
    assert [len(frame) for frame in restored] == [2, 0, 1]


def test_build_raw_detection_arrays_validates_frames() -> None:
    with pytest.raises(StageIOError):
        build_raw_detection_arrays([])  # nothing to infer a length from
    with pytest.raises(StageIOError):
        build_raw_detection_arrays(
            [[RawDetection(bbox=np.zeros(4), confidence=0.5)]], frame_ids=[7], num_frames=3
        )
    with pytest.raises(StageIOError):
        build_raw_detection_arrays(
            [[RawDetection(bbox=np.zeros(4), confidence=0.5)]], frame_ids=[0, 1]
        )
    # An explicitly empty clip is legal: "no hands anywhere" is a real outcome.
    empty = build_raw_detection_arrays([], num_frames=5)
    assert empty["count"].tolist() == [0] * 5


def test_detections_from_predictions_reads_a_dict() -> None:
    predictions = {
        "boxes": np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]]),
        "scores": np.array([0.9, 0.7]),
        "left": np.array([0.8, 0.1]),
        "right": np.array([0.2, 0.9]),
    }
    detections = detections_from_predictions(predictions, frame=3)
    assert len(detections) == 2
    assert detections[0].left_score == pytest.approx(0.8)
    assert detections[1].right_score == pytest.approx(0.9)


def test_detections_from_predictions_refuses_to_guess() -> None:
    with pytest.raises(NotImplementedError, match="Available"):
        detections_from_predictions({"something_else": np.zeros(3)}, frame=0)


def test_wilor_runner_reports_a_missing_checkpoint(tmp_path: Path) -> None:
    runner = load_script("wilor_runner_mod", "backends/wilor_runner.py")
    args = runner.build_parser().parse_args(["--weights", str(tmp_path / "weights")])
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        runner.load_detector(args)


# ------------------------------------------------------------------ HaWoR


def test_hawor_weights_are_found_in_both_layouts(tmp_path: Path) -> None:
    """``weights/hawor/checkpoints/...`` (upstream) and ``weights/hawor/...`` (flat)."""
    flat = tmp_path / "flat"
    (flat / "hawor").mkdir(parents=True)
    (flat / "hawor" / "hawor.ckpt").write_bytes(b"x")
    (flat / "hawor" / "infiller.pt").write_bytes(b"x")
    (flat / "hawor" / "model_config.yaml").write_text("model: hawor\n")
    found = find_weights_files(flat, third_party=tmp_path / "third_party")
    assert found["checkpoint"].name == "hawor.ckpt"
    assert found["infiller"].name == "infiller.pt"
    assert found["model_config"].name == "model_config.yaml"

    nested = tmp_path / "nested"
    (nested / "hawor" / "checkpoints").mkdir(parents=True)
    (nested / "hawor" / "checkpoints" / "hawor.ckpt").write_bytes(b"x")
    (nested / "hawor" / "checkpoints" / "infiller.pt").write_bytes(b"x")
    nested_found = find_weights_files(nested, third_party=tmp_path / "third_party")
    assert nested_found["checkpoint"].parent.name == "checkpoints"

    # model_config.yaml also ships inside the checkout.
    checkout = tmp_path / "third_party" / "HaWoR"
    checkout.mkdir(parents=True)
    (checkout / "model_config.yaml").write_text("model: hawor\n")
    from_checkout = find_weights_files(tmp_path / "empty", third_party=tmp_path / "third_party")
    assert from_checkout["model_config"].parent == checkout
    assert "checkpoint" not in from_checkout


def test_vggt_checkpoint_is_found_as_a_directory_or_a_file(tmp_path: Path) -> None:
    name = "VGGT-Omega-1B-416-Reproduction"

    documented = tmp_path / "weights" / "vggt-omega" / name
    documented.mkdir(parents=True)
    (documented / "model.safetensors").write_bytes(b"x")
    assert find_vggt_checkpoint(tmp_path / "weights", name) == documented

    flat = tmp_path / "flat" / "vggt-omega"
    flat.mkdir(parents=True)
    single = flat / "vggt_omega_1b_416_reproduce.pt"
    single.write_bytes(b"x")
    # The exact file for the requested checkpoint beats the enclosing directory.
    assert find_vggt_checkpoint(tmp_path / "flat", name) == single

    deeper = tmp_path / "deeper" / "vggt"
    deeper.mkdir(parents=True)
    torch_file = deeper / f"{name}.pt"
    torch_file.write_bytes(b"x")
    # An exact-name file beats the enclosing directory: it is the specific match.
    assert find_vggt_checkpoint(tmp_path / "deeper", name) == torch_file

    empty = tmp_path / "nothing"
    empty.mkdir()
    assert find_vggt_checkpoint(empty, name) is None


def test_vggt_checkpoint_substitution_is_reported(tmp_path: Path) -> None:
    """Running the 512 checkpoint when 416 was requested must not be silent."""
    root = tmp_path / "weights" / "vggt-omega"
    root.mkdir(parents=True)
    wanted = root / "vggt_omega_1b_416_reproduce.pt"
    wanted.write_bytes(b"x")
    (root / "vggt_omega_1b_512.pt").write_bytes(b"x")

    path, substituted = resolve_vggt_checkpoint(tmp_path / "weights", "VGGT-Omega-1B-416-Reproduction")
    assert path == wanted and substituted is None

    wanted.unlink()
    path, substituted = resolve_vggt_checkpoint(tmp_path / "weights", "VGGT-Omega-1B-416-Reproduction")
    assert path is not None and path.name == "vggt_omega_1b_512.pt"
    assert substituted == "vggt_omega_1b_512.pt"

    # The documented directory layout resolves with no substitution.
    (root / "VGGT-Omega-1B-416-Reproduction").mkdir()
    (root / "VGGT-Omega-1B-416-Reproduction" / "model.safetensors").write_bytes(b"x")
    path, substituted = resolve_vggt_checkpoint(tmp_path / "weights", "VGGT-Omega-1B-416-Reproduction")
    assert path is not None and path.name == "VGGT-Omega-1B-416-Reproduction"
    assert substituted is None

    nothing, substituted = resolve_vggt_checkpoint(tmp_path / "elsewhere", "VGGT-Omega-1B-416-Reproduction")
    assert nothing is None and substituted is None


def make_tracking(total: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.zeros((total, 2, 4))
    confidence = np.full((total, 2), 0.9)
    valid = np.ones((total, 2), dtype=bool)
    valid[10:12, 0] = False
    valid[20:, 1] = False
    for frame in range(total):
        for hand in range(2):
            if valid[frame, hand]:
                boxes[frame, hand] = [10 * frame, hand * 50, 10 * frame + 40, hand * 50 + 40]
    return boxes, confidence, valid


def test_hawor_tracks_use_our_tracking_decision() -> None:
    boxes, confidence, valid = make_tracking()
    model_boxes, tracks = hawor_tracks_from_detection(
        boxes=boxes, confidence=confidence, valid=valid
    )
    assert set(tracks) == {0, 1}
    assert len(tracks[0]) == int(valid[:, 0].sum())
    assert len(tracks[1]) == int(valid[:, 1].sum())
    # HaWoR expects one entry per kept frame, with a [1, 5] box and handedness.
    entry = tracks[0][0]
    assert entry["frame"] == 0
    assert entry["det"] is True
    assert np.asarray(entry["det_box"]).shape == (1, 5)
    assert np.asarray(entry["det_handedness"]).tolist() == [0]
    # The frames our tracker dropped must not appear.
    assert 11 not in [item["frame"] for item in tracks[0]]
    assert 25 not in [item["frame"] for item in tracks[1]]
    assert model_boxes.shape == (40,)


def test_hawor_tracks_round_trip_through_npy(tmp_path: Path) -> None:
    boxes, confidence, valid = make_tracking(8)
    model_boxes, tracks = hawor_tracks_from_detection(
        boxes=boxes, confidence=confidence, valid=valid
    )
    written = save_hawor_tracks(tmp_path / "tracks_0_8", model_boxes, tracks)
    assert [path.name for path in written] == ["model_boxes.npy", "model_tracks.npy"]
    reloaded = np.load(tmp_path / "tracks_0_8" / "model_tracks.npy", allow_pickle=True).item()
    assert sorted(reloaded) == [0, 1]
    assert reloaded[0][0]["frame"] == 0


def test_hawor_tracks_validate_shapes() -> None:
    with pytest.raises(StageIOError):
        hawor_tracks_from_detection(
            boxes=np.zeros((5, 3, 4)), confidence=np.zeros((5, 3)), valid=np.ones((5, 3), bool)
        )
    with pytest.raises(StageIOError):
        hawor_tracks_from_detection(
            boxes=np.zeros((5, 2, 4)), confidence=np.zeros((4, 2)), valid=np.ones((5, 2), bool)
        )


def test_hand_windows_from_joints_writes_the_16_8_schedule(tmp_path: Path) -> None:
    joints = np.zeros((40, 2, 21, 3))
    valid = np.ones((40, 2), dtype=bool)
    valid[10:12, 0] = False
    confidence = np.where(valid, 0.9, 0.0)
    paths = hand_windows_from_joints(
        joints, valid, confidence, out_dir=tmp_path / "windows"
    )
    assert [path.name for path in paths] == [
        "000000_000015.npz",
        "000008_000023.npz",
        "000016_000031.npz",
        "000024_000039.npz",
    ]
    window = load_hand_window(paths[1])  # covers frames 8..23
    assert window.start == 8
    assert not np.isfinite(window.joints_camera[2:4, 0]).any()  # frames 10, 11
    assert np.isfinite(window.joints_camera[2:4, 1]).all()


def test_hand_windows_validate_shapes(tmp_path: Path) -> None:
    with pytest.raises(StageIOError):
        hand_windows_from_joints(
            np.zeros((5, 2, 20, 3)), np.ones((5, 2), bool), np.ones((5, 2)), out_dir=tmp_path
        )
    with pytest.raises(StageIOError):
        hand_windows_from_joints(
            np.zeros((5, 2, 21, 3)), np.ones((4, 2), bool), np.ones((5, 2)), out_dir=tmp_path
        )


def test_hawor_runner_check_reports_the_missing_checkout(tmp_path: Path) -> None:
    runner = load_script("hawor_runner_mod", "backends/hawor_runner.py")
    available, detail = runner.backend_available(tmp_path)
    assert not available
    assert "checkout" in detail


def test_hawor_runner_prepares_frames_for_hawor(tmp_path: Path) -> None:
    runner = load_script("hawor_runner_mod2", "backends/hawor_runner.py")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(3):
        (frames_dir / f"{index:06d}.jpg").write_bytes(b"stub")
    # The runner resolves its weights before doing anything else.
    weights = tmp_path / "weights" / "hawor"
    weights.mkdir(parents=True)
    (weights / "hawor.ckpt").write_bytes(b"stub")
    (weights / "infiller.pt").write_bytes(b"stub")
    args = runner.build_parser().parse_args(
        [
            "--frames",
            str(frames_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--weights",
            str(tmp_path / "weights"),
        ]
    )
    namespace = runner.build_hawor_args(args, tmp_path / "seq", frames_dir)
    images = sorted((tmp_path / "seq" / "extracted_images").glob("*.jpg"))
    assert [path.name for path in images] == ["0000.jpg", "0001.jpg", "0002.jpg"]
    assert namespace.video_path.endswith("seq.mp4")
    assert namespace.img_focal is None
    assert namespace.checkpoint.endswith("hawor.ckpt")


def test_hawor_runner_explains_missing_weights(tmp_path: Path) -> None:
    runner = load_script("hawor_runner_mod3", "backends/hawor_runner.py")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    args = runner.build_parser().parse_args(
        ["--frames", str(frames_dir), "--out-dir", str(tmp_path / "out"), "--weights", str(tmp_path)]
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        runner.build_hawor_args(args, tmp_path / "seq", frames_dir)
    message = str(excinfo.value)
    assert "hawor/checkpoints/hawor.ckpt" in message  # the paths it looked at
    assert "download_weights.sh" in message


# ------------------------------------------------------------------ VGGT


def test_camera_window_from_output_validates() -> None:
    intrinsics = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    window = camera_window_from_output(
        start=10,
        end=14,
        rotation_c2w=np.broadcast_to(np.eye(3), (4, 3, 3)).copy(),
        translation_c2w=np.zeros((4, 3)),
        intrinsics=intrinsics,
        depth=np.full((4, 6, 8), 1.5),
    )
    assert window.window.start == 10 and window.window.num_frames == 4
    assert np.isfinite(window.depth).all()

    with pytest.raises(StageIOError, match="backend returned"):
        camera_window_from_output(
            start=0,
            end=5,
            rotation_c2w=np.broadcast_to(np.eye(3), (4, 3, 3)).copy(),
            translation_c2w=np.zeros((4, 3)),
            intrinsics=intrinsics,
            depth=np.full((4, 6, 8), 1.5),
        )


def test_vggt_runner_refuses_undecodable_output() -> None:
    runner = load_script("vggt_runner_mod", "backends/vggt_runner.py")
    with pytest.raises(NotImplementedError, match="Available"):
        runner.decode_predictions({"nonsense": np.zeros(3)}, image_size=(4, 4), resolution=416)

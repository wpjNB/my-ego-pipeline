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


def test_wilor_runner_reports_a_missing_detector(tmp_path: Path) -> None:
    runner = load_script("wilor_runner_mod", "backends/wilor_runner.py")
    weights = tmp_path / "weights"
    assert runner.find_detector(weights) is None
    available, detail = runner.backend_available(tmp_path / "third_party", weights)
    assert not available
    assert "detector.pt" in detail and "download_weights.sh" in detail

    # HaWoR's copy of the same YOLO detector is accepted as a stand-in.
    external = weights / "external"
    external.mkdir(parents=True)
    (external / "detector.pt").write_bytes(b"stub")
    assert runner.find_detector(weights) == external / "detector.pt"


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
    # ``checkout`` is optional; the runner passes it to fail early on a wrong
    # --third-party, but the argument assembly must work without it too.
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


def test_hawor_runner_rejects_a_wrong_checkout(tmp_path: Path) -> None:
    """A 4th argument (the checkout) is checked when it is supplied."""
    runner = load_script("hawor_runner_mod4", "backends/hawor_runner.py")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    args = runner.build_parser().parse_args(
        ["--frames", str(frames_dir), "--out-dir", str(tmp_path / "out"), "--weights", str(tmp_path)]
    )
    with pytest.raises(NotADirectoryError) as excinfo:
        runner.build_hawor_args(args, tmp_path / "seq", frames_dir, tmp_path / "no-such-checkout")
    assert "HaWoR checkout not found" in str(excinfo.value)


def test_gpu_runners_expose_a_precision_flag() -> None:
    """fp16 is what makes the 1B/3 GB checkpoints fit on a shared 12 GB GPU."""
    hawor = load_script("hawor_runner_mod5", "backends/hawor_runner.py")
    vggt = load_script("vggt_runner_mod5", "backends/vggt_runner.py")
    for runner in (hawor, vggt):
        parser = runner.build_parser()
        assert parser.parse_args([]).precision == "auto"
        assert parser.parse_args(["--precision", "fp16"]).precision == "fp16"
        with pytest.raises(SystemExit):  # an unknown precision is rejected, not ignored
            parser.parse_args(["--precision", "bf16"])


def test_hawor_runner_exposes_a_crop_size_knob() -> None:
    """192 px is the shared-GPU escape hatch for HaWoR's activation memory."""
    hawor = load_script("hawor_runner_mod6", "backends/hawor_runner.py")
    parser = hawor.build_parser()
    assert parser.parse_args([]).crop_size == 256  # upstream default
    assert parser.parse_args(["--crop-size", "192"]).crop_size == 192


def test_hawor_loader_forces_cpu_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """HaWoR's load_from_checkpoint has no map_location -> 4.3 GB lands on the GPU.

    ``torch`` is not installed in this environment, so the test drives the loader
    with a stub module: what matters is that ``map_location="cpu"`` is injected
    for the duration of the load and that an explicit caller value is respected.
    """
    import sys
    import types

    runner = load_script("hawor_runner_mod7", "backends/hawor_runner.py")
    calls: list[dict[str, object]] = []

    class _Model:
        def __init__(self) -> None:
            self.inference = lambda *a, **k: "inference"

        def half(self) -> "_Model":
            calls.append({"half": True})
            return self

    def fake_load(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return {"loaded": True}

    fake_torch = types.SimpleNamespace(
        load=fake_load,
        float16="float16",
        autocast=lambda *a, **k: __import__("contextlib").nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def upstream_loader(path: str) -> tuple[object, object]:
        fake_torch.load(path, weights_only=True)  # what Lightning ends up doing
        return _Model(), {"cfg": True}

    patched = runner._patched_loader(upstream_loader, half=True)
    model, cfg = patched("weights/hawor.ckpt")
    assert calls[0]["map_location"] == "cpu"  # forced
    assert calls[0]["weights_only"] is True  # caller's other kwargs survive
    assert {"half": True} in calls  # fp16 requested
    assert cfg == {"cfg": True}
    assert model.inference("x") == "inference"
    # The global torch.load is restored: no leaking patch.
    assert fake_torch.load is fake_load

    # Lightning passes its own ``_default_map_location`` callable (which would
    # pick CUDA); that must be steered to CPU too.
    def lightning_loader(path: str) -> tuple[object, object]:
        fake_torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
        return _Model(), {}

    calls.clear()
    runner._patched_loader(lightning_loader, half=False)("weights/hawor.ckpt")
    assert calls[0]["map_location"] == "cpu"
    assert calls[0]["weights_only"] is False

    def explicit_loader(path: str) -> tuple[object, object]:
        fake_torch.load(path, map_location="cuda:1")
        return _Model(), {}

    calls.clear()
    runner._patched_loader(explicit_loader, half=False)("weights/hawor.ckpt")
    assert calls[0]["map_location"] == "cuda:1"  # explicit wins
    assert {"half": True} not in calls  # fp32 path must not halve the model


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


def test_synthesised_camera_trajectory_is_float32(tmp_path: Path) -> None:
    """HaWoR feeds this into einsum next to float32 model outputs.

    Writing float64 raised "expected scalar type Double but found Float" inside
    ``cam2world_convert``; DROID-SLAM's own npz is float32, which is why the
    synthesised replacement has to match.
    """
    import argparse

    import numpy as np

    runner = load_script("hawor_runner_mod8", "backends/hawor_runner.py")
    args = argparse.Namespace(camera_windows=None, focal=None)
    path = runner.write_camera_trajectory(args, tmp_path / "seq", 0, 4)
    data = np.load(path)
    assert data["traj"].dtype == np.float32
    assert data["img_center"].dtype == np.float32
    assert float(data["scale"]) == 1.0
    assert data["traj"].shape == (4, 7)


def test_hawor_to_camera_space_uses_the_contract_layout() -> None:
    """Frame-major (T, hand, joint, xyz), NaN wherever either validity source fails."""
    import numpy as np

    runner = load_script("hawor_runner_mod9", "backends/hawor_runner.py")
    total = 3
    r_w2c = np.broadcast_to(np.eye(3), (total, 3, 3)).copy()
    t_w2c = np.tile(np.array([0.0, 0.0, -1.0]), (total, 1))
    landmarks = np.zeros((total, 2, 21, 3))
    landmarks[..., 2] = 2.0  # everything sits 2 m down +z
    valid = np.ones((total, 2), dtype=bool)
    valid[1, 0] = False
    pred_valid = np.ones((2, total))  # (hand, frame), as the infiller returns it
    pred_valid[1, 2] = 0.0
    confidence = np.full((total, 2), 0.9)

    result = runner.to_camera_space(
        r_w2c, t_w2c, landmarks, valid=valid, pred_valid=pred_valid, confidence=confidence
    )
    joints = result["joints_camera"]
    assert joints.shape == (total, 2, 21, 3)
    assert np.allclose(joints[0, 0], [0.0, 0.0, 1.0])  # identity pose, z-1
    assert np.isnan(joints[1, 0]).all()  # tracker missed it
    assert np.isnan(joints[2, 1]).all()  # infiller did not trust it
    assert result["valid"].tolist() == [[True, True], [False, True], [True, False]]
    assert result["confidence"][1, 0] == 0.0
    assert result["confidence"][0, 0] == 0.9

    with pytest.raises(ValueError, match="landmarks must be"):
        runner.to_camera_space(
            r_w2c, t_w2c, landmarks.transpose(1, 0, 2, 3), valid=valid,
            pred_valid=pred_valid, confidence=confidence,
        )


def test_hawor_to_camera_space_transforms_vertices_like_joints() -> None:
    """Mesh vertices ride the same w2c transform and validity mask as joints."""
    import numpy as np

    runner = load_script("hawor_runner_mod12", "backends/hawor_runner.py")
    total = 2
    r_w2c = np.broadcast_to(np.eye(3), (total, 3, 3)).copy()
    t_w2c = np.zeros((total, 3))
    landmarks = np.zeros((total, 2, 21, 3))
    vertices = np.zeros((total, 2, 5, 3))
    vertices[..., 2] = 0.5
    valid = np.ones((total, 2), dtype=bool)
    valid[1, 1] = False
    pred_valid = np.ones((2, total))
    confidence = np.full((total, 2), 0.8)

    result = runner.to_camera_space(
        r_w2c, t_w2c, landmarks,
        valid=valid, pred_valid=pred_valid, confidence=confidence,
        vertices=vertices,
    )
    verts = result["vertices_camera"]
    assert verts.shape == (total, 2, 5, 3)
    assert verts.dtype == np.float32
    assert np.allclose(verts[0, 0, :, 2], 0.5)  # same transform as the joints
    assert np.isnan(verts[1, 1]).all()  # invalid hand -> no mesh either
    assert np.isfinite(verts[1, 0]).all()

    with pytest.raises(ValueError, match="vertices must be"):
        runner.to_camera_space(
            r_w2c, t_w2c, landmarks,
            valid=valid, pred_valid=pred_valid, confidence=confidence,
            vertices=vertices[:-1],  # one frame short of the joints
        )


def test_hawor_absolutizes_paths_before_chdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HaWoR chdirs into its checkout; relative paths must not follow it there.

    ``--camera-windows data/<clip>/camera/windows`` resolved to
    ``third_party/HaWoR/data/...`` after the chdir and reported "holds no *.npz"
    while the directory had 39 windows.
    """
    import argparse

    runner = load_script("hawor_runner_mod10", "backends/hawor_runner.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "clip" / "camera" / "windows").mkdir(parents=True)
    args = argparse.Namespace(
        frames="data/clip/frames",
        out_dir="data/clip/hand/windows",
        detection="data/clip/detection/detection.npz",
        camera_windows="data/clip/camera/windows",
    )
    runner.absolutize_paths(args)
    for value in (args.frames, args.out_dir, args.detection, args.camera_windows):
        assert Path(value).is_absolute()
        assert Path(value).is_relative_to(tmp_path)
    # Empty/None values are left alone rather than turned into the cwd.
    runner.absolutize_paths(argparse.Namespace(camera_windows=None, frames=""))
    assert True


def test_resolve_focal_prefers_real_intrinsics_over_the_600px_default(
    tmp_path: Path,
) -> None:
    """HaWoR's silent 600 px default is what put hands 2.2x too deep on HOT3D."""
    import numpy as np

    from ego3d_action.io.artefacts import ClipLayout

    runner = load_script("run_hand_mod_focal", "scripts/run_hand.py")
    layout = ClipLayout(data_root=tmp_path, clip="clip")
    layout.ensure_dirs()
    (layout.metadata_path).write_text('{"width": 512, "height": 512}', encoding="utf-8")

    class _Config:
        def __init__(self, values: dict[str, object]) -> None:
            self.values = values

        def get(self, key: str, default: object = None) -> object:
            return self.values.get(key, default)

    # 1. nothing available -> the caller must warn rather than guess
    focal, source = runner.resolve_focal(layout, _Config({}))
    assert focal is None and source == "unavailable"

    # 2. a reference trajectory answers when the camera stage has not run yet
    intrinsics = np.broadcast_to(
        np.array([[221.14, 0.0, 255.8], [0.0, 221.14, 255.8], [0.0, 0.0, 1.0]]), (3, 3, 3)
    ).copy()
    np.savez(layout.trajectory_dir / "ground_truth.npz", camera_K=intrinsics)
    focal, source = runner.resolve_focal(layout, _Config({}))
    assert abs(focal - 221.14) < 1e-6 and source == "reference camera_K"

    # 3. Phase 3's windows win, rescaled from the depth grid to the frame size
    np.savez(
        layout.window_path(0, 4),
        intrinsics=np.broadcast_to(
            np.array([[110.57, 0.0, 127.9], [0.0, 110.57, 127.9], [0.0, 0.0, 1.0]]), (4, 3, 3)
        ).copy(),
        depth=np.zeros((4, 256, 256), dtype=np.float32),
    )
    focal, source = runner.resolve_focal(layout, _Config({}))
    assert abs(focal - 221.14) < 0.05 and source == "Phase 3 camera windows"

    # 4. an explicit config value wins over everything
    focal, source = runner.resolve_focal(layout, _Config({"hand.focal": 300.0}))
    assert focal == 300.0 and source == "config hand.focal"


def test_hawor_drops_cached_tracks_when_the_focal_changes(tmp_path: Path) -> None:
    """A cached track computed at the default 600 px must not survive a real focal."""
    runner = load_script("hawor_runner_mod11", "backends/hawor_runner.py")
    seq = tmp_path / "seq"
    (seq / "tracks_0_450").mkdir(parents=True)
    (seq / "tracks_0_450" / "frame_chunks_all.npy").write_bytes(b"stale")

    runner.invalidate_stale_hand_cache(seq, 221.14)
    assert (seq / "est_focal.txt").read_text() == "221.14"
    assert (seq / "tracks_0_450").is_dir()  # nothing to compare against -> keep

    runner.invalidate_stale_hand_cache(seq, 600.0)
    assert not (seq / "tracks_0_450").exists()  # focal changed -> drop

    # Same focal again: no further deletion, and a missing focal changes nothing.
    (seq / "tracks_1_2").mkdir()
    runner.invalidate_stale_hand_cache(seq, 600.0)
    assert (seq / "tracks_1_2").is_dir()
    runner.invalidate_stale_hand_cache(seq, None)
    assert (seq / "tracks_1_2").is_dir()

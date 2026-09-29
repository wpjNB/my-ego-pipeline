"""HaWoR backend adapter (Phase 2).

HaWoR reconstructs 16-frame windows with an 8-frame overlap. The backend runs
in its own environment (``ego3d_hawor``: Python 3.10 / torch 1.13 / CUDA 11.7,
with DROID-SLAM and Metric3D available but unused here, because the whole point
of this project is to replace them with VGGT-Omega).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import StageIOError
from ..runtime.backend import BackendSpec, BackendStatus, probe_backend, require_backend
from ..runtime.sharding import WindowSelection
from ..runtime.subprocess_backend import BackendInvocation, run_runner
from .temporal_blend import HandWindow, blend_hand_windows

logger = logging.getLogger(__name__)

HAWOR_SPEC = BackendSpec(
    name="HaWoR",
    env="ego3d_hawor",
    install_hint=(
        "Clone HaWoR into third_party/HaWoR and place its MANO checkpoints in weights/hawor, "
        "then install environment-hawor.yml (Python 3.10 / torch 1.13 / CUDA 11.7)."
    ),
    repository="https://github.com/ThunderVVV/HaWoR",
)


@dataclass(frozen=True)
class HaworClipRequest:
    """A whole-clip HaWoR request: the orchestrator owns the 16/8 schedule.

    HaWoR loads its model once per process, so the backend is invoked once for
    the clip and writes every window; the orchestrator then blends them.
    """

    num_frames: int
    frames_dir: Path
    window: int = 16
    overlap: int = 8
    device: str = "auto"
    #: Focal length in *frame* pixels. HaWoR falls back to a hard-coded 600 px
    #: when this is None, which mis-scales every reconstructed depth.
    focal: float | None = None

    @property
    def starts(self) -> list[int]:
        stride = self.window - self.overlap
        starts = list(range(0, max(1, self.num_frames - self.overlap), stride))
        return starts or [0]

    def ranges(self) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for start in self.starts:
            end = min(start + self.window, self.num_frames)
            if spans and end <= spans[-1][1]:
                break
            spans.append((start, end))
        return spans


def probe(third_party: str | Path, weights_root: str | Path) -> BackendStatus:
    return probe_backend(HAWOR_SPEC, third_party=Path(third_party), weights_root=Path(weights_root))


#: Where ``hawor.ckpt`` / ``infiller.pt`` / ``model_config.yaml`` may live. Both
#: the upstream layout (``weights/hawor/checkpoints/...``) and the flat one that
#: ``scripts/download_weights.sh`` produces (``weights/hawor/...``) are accepted.
HAWOR_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "checkpoint": ("hawor/checkpoints/hawor.ckpt", "hawor/hawor.ckpt", "hawor.ckpt"),
    "infiller": (
        "hawor/checkpoints/infiller.pt",
        "hawor/infiller.pt",
        "infiller.pt",
    ),
    "model_config": (
        "hawor/checkpoints/model_config.yaml",
        "hawor/model_config.yaml",
        "model_config.yaml",
    ),
}


def find_weights_files(
    weights_root: str | Path, third_party: str | Path = "third_party"
) -> dict[str, Path]:
    """Resolve HaWoR's weight files across the layouts users actually end up with.

    Returns the subset that exists; ``model_config.yaml`` also ships inside the
    checkout, so it is looked up there as well.
    """
    root = Path(weights_root)
    found: dict[str, Path] = {}
    for name, relatives in HAWOR_FILE_CANDIDATES.items():
        for relative in relatives:
            candidate = root / relative
            if candidate.is_file():
                found[name] = candidate
                break
    if "model_config" not in found:
        for relative in HAWOR_FILE_CANDIDATES["model_config"]:
            candidate = Path(third_party) / "HaWoR" / relative.split("/")[-1]
            if candidate.is_file():
                found["model_config"] = candidate
                break
            candidate = Path(third_party) / "HaWoR" / "weights" / relative
            if candidate.is_file():
                found["model_config"] = candidate
                break
    return found


def require(third_party: str | Path, weights_root: str | Path) -> BackendStatus:
    return require_backend(HAWOR_SPEC, third_party=Path(third_party), weights_root=Path(weights_root))


def run_windows(
    request: HaworClipRequest,
    out_dir: str | Path,
    *,
    invocation: BackendInvocation,
    third_party: str | Path,
    weights_root: str | Path,
    detection_path: str | Path | None = None,
    camera_windows: str | Path | None = None,
    log_path: Path | None = None,
    selection: WindowSelection | None = None,
    skip_existing: bool = False,
    precision: str | None = None,
    crop_size: int | None = None,
) -> list[Path]:
    """Reconstruct every window of a clip and return the written files.

    In mock mode this executes ``backends/mock_backend.py hawor``; in real mode it
    requires the checkout/weights and runs ``backends/hawor_runner.py`` in the
    configured interpreter (normally ``ego3d_hawor``). Every written window is
    validated before it is trusted.

    ``camera_windows`` points at Phase 3's ``camera/windows`` directory. HaWoR's
    infiller needs a world->camera trajectory, which upstream obtains from
    DROID-SLAM; this project replaces DROID-SLAM with VGGT-Omega, so the runner
    synthesises that trajectory from these VGGT poses. Omit it and the camera
    stays constant - a degraded, non-metric result the runner reports as
    ``camera_source: constant``.

    ``selection`` slices the clip's 16/8 schedule into one shard's worth of
    windows (the schedule is always derived from the *global* window/overlap, so
    the union of shards is exactly the unsliced run). ``skip_existing`` lets a
    shard be reused when its windows and provenance marker are already on disk.

    ``precision`` is forwarded to the runner (``auto``/``fp32``/``fp16``); fp16
    halves the hand checkpoint's resident memory, which is what makes Phase 2
    fit on a 12 GB GPU that another job is sharing.

    ``crop_size`` overrides HaWoR's 256 px hand crop; a smaller crop cuts
    activation memory (192 px is 56 % of the area) at some accuracy cost.

    Raises:
        BackendNotAvailableError: the checkpoint/env is not present.
        BackendExecutionError: the runner failed or timed out.
        StageIOError: a window is missing or malformed.
    """
    if not invocation.is_mock:
        require(third_party, weights_root)

    active = selection or WindowSelection()
    spec, prefix = invocation.resolve("hawor", mock_subcommand="hawor")
    args = [
        *prefix,
        "--out-dir",
        str(out_dir),
        "--frames",
        str(request.frames_dir),
        "--num-frames",
        str(request.num_frames),
        "--window",
        str(request.window),
        "--overlap",
        str(request.overlap),
        "--device",
        request.device,
        "--third-party",
        str(third_party),
        "--weights",
        str(weights_root),
    ]
    if not active.is_whole:
        args += ["--shard", f"{active.shard.index}/{active.shard.count}"]
        if active.window_start is not None and active.window_end is not None:
            args += ["--window-range", f"{active.window_start}-{active.window_end}"]
    if skip_existing:
        args.append("--skip-existing")
    if precision is not None:
        args += ["--precision", str(precision)]
    if crop_size is not None:
        args += ["--crop-size", str(crop_size)]
    if request.focal is not None:
        # Without this HaWoR silently uses its 600 px default and every hand is
        # reconstructed at the wrong depth (2-3x too far on a 98-degree ego lens).
        args += ["--focal", repr(float(request.focal))]
    if detection_path is not None:
        args += ["--detection", str(detection_path)]
    if camera_windows is not None:
        args += ["--camera-windows", str(camera_windows)]
    payload = run_runner(spec, args, log_path=log_path)
    logger.info("HaWoR runner reported %s", json.dumps(payload, sort_keys=True))

    target = Path(out_dir)
    expected_ranges = active.select(request.ranges())
    if not expected_ranges:
        raise StageIOError(
            f"selection '{active.describe()}' matched no HaWoR window; "
            f"the schedule has {len(request.ranges())} window(s)"
        )
    paths: list[Path] = []
    for start, end in expected_ranges:
        path = target / f"{start:06d}_{end - 1:06d}.npz"
        if not path.is_file():
            raise StageIOError(
                f"the hand backend did not produce window {path.name}; it reported "
                f"{payload.get('windows', 'no window list')}"
            )
        load_hand_window(path)  # validates shapes before the file is trusted
        paths.append(path)
    return paths


def load_hand_window(path: str | Path) -> HandWindow:
    """Load a persisted HaWoR window, validating the stage contract."""
    from ..io.serialization import load_npz

    data = load_npz(
        path,
        required=("joints_camera", "valid", "confidence", "start"),
    )
    start = int(np.asarray(data["start"]).reshape(-1)[0])
    return HandWindow(
        start=start,
        joints_camera=data["joints_camera"],
        valid=np.asarray(data["valid"], dtype=bool),
        confidence=data["confidence"],
        root_rot=data.get("root_rot"),
        betas=data.get("betas"),
    )


def save_hand_window(window: HandWindow, path: str | Path) -> Path:
    """Persist a HaWoR window (used by the runners and by tests)."""
    from ..io.serialization import save_npz

    payload: dict[str, np.ndarray] = {
        "start": np.array([window.start], dtype=np.int64),
        "joints_camera": window.joints_camera,
        "valid": window.valid,
        "confidence": (
            window.confidence
            if window.confidence is not None
            else np.where(window.valid, 1.0, 0.0)
        ),
    }
    if window.root_rot is not None:
        payload["root_rot"] = window.root_rot
    if window.betas is not None:
        payload["betas"] = window.betas
    return save_npz(path, **payload)


def hawor_tracks_from_detection(
    *,
    boxes: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, dict[int, list[dict[str, object]]]]:
    """Build HaWoR's ``(model_boxes, model_tracks)`` structures from our tracking.

    HaWoR's demo starts from ``detect_track(imgfiles, thresh=0.2)``, which both
    detects *and* decides tracking. This project deliberately replaces that
    rule set with the conservative tracker of Phase 1, and this function is the
    seam: our ``detection.npz`` becomes the ``model_tracks`` file that
    ``hawor_motion_estimation`` consumes, so HaWoR reconstructs exactly the
    frames our tracker kept - never more.

    Args:
        boxes: ``[T, 2, 4]`` tracked boxes (NaN where invalid).
        confidence: ``[T, 2]`` detector confidences.
        valid: ``[T, 2]`` tracker decision.

    Raises:
        StageIOError: on shape mismatches.
    """
    box_array = np.asarray(boxes, dtype=np.float64)
    conf_array = np.asarray(confidence, dtype=np.float64)
    valid_array = np.asarray(valid, dtype=bool)
    if box_array.ndim != 3 or box_array.shape[1:] != (2, 4):
        raise StageIOError(f"boxes must be [T, 2, 4], got {box_array.shape}")
    if conf_array.shape != box_array.shape[:2] or valid_array.shape != box_array.shape[:2]:
        raise StageIOError(
            f"confidence {conf_array.shape} and valid {valid_array.shape} must match "
            f"{box_array.shape[:2]}"
        )

    tracks: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    per_frame_boxes: list[list[np.ndarray]] = [[] for _ in range(box_array.shape[0])]
    for frame in range(box_array.shape[0]):
        for hand in (0, 1):
            if not valid_array[frame, hand]:
                continue
            entry = np.array([*box_array[frame, hand], conf_array[frame, hand]], dtype=np.float64)
            tracks[hand].append(
                {
                    "frame": frame,
                    "det": True,
                    "det_box": entry[None, :],
                    "det_handedness": np.array([hand]),
                }
            )
            per_frame_boxes[frame].append(entry)
    # HaWoR's own detect_track returns an empty object array here; keep the same
    # shape for compatibility but fill it with what we know.
    model_boxes = np.array(per_frame_boxes, dtype=object)
    return model_boxes, tracks


def save_hawor_tracks(
    directory: str | Path,
    model_boxes: np.ndarray,
    tracks: dict[int, list[dict[str, object]]],
) -> list[Path]:
    """Write ``model_boxes.npy`` / ``model_tracks.npy`` where HaWoR expects them."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in (("model_boxes.npy", model_boxes), ("model_tracks.npy", tracks)):
        path = target / name
        np.save(path, payload)
        written.append(path)
    return written


def hand_windows_from_joints(
    joints_camera: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    *,
    out_dir: str | Path,
    window: int = 16,
    overlap: int = 8,
    root_rot: np.ndarray | None = None,
    betas: np.ndarray | None = None,
    selection: WindowSelection | None = None,
) -> list[Path]:
    """Split camera-space joints into the 16/8 HaWoR windows on disk.

    ``joints_camera`` is ``[T, 2, 21, 3]`` in metres; frames without a hand must
    be ``NaN`` and invalid. Nothing is interpolated.

    ``selection`` restricts the written windows to one shard of the clip's
    schedule. The schedule is still derived from the global ``window``/``overlap``,
    so the union of all shards is exactly the unsliced result.
    """
    joints = np.asarray(joints_camera, dtype=np.float64)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise StageIOError(f"joints_camera must be [T, 2, 21, 3], got {joints.shape}")
    valid_array = np.asarray(valid, dtype=bool)
    conf_array = np.asarray(confidence, dtype=np.float64)
    if valid_array.shape != joints.shape[:2] or conf_array.shape != joints.shape[:2]:
        raise StageIOError(
            f"valid {valid_array.shape} and confidence {conf_array.shape} must match "
            f"{joints.shape[:2]}"
        )
    request = HaworClipRequest(num_frames=joints.shape[0], frames_dir=Path("."), window=window, overlap=overlap)
    active = selection or WindowSelection()
    written: list[Path] = []
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for start, end in active.select(request.ranges()):
        window_obj = HandWindow(
            start=start,
            joints_camera=np.where(valid_array[start:end, :, None, None], joints[start:end], np.nan),
            valid=valid_array[start:end],
            confidence=conf_array[start:end],
            root_rot=None if root_rot is None else np.asarray(root_rot)[start:end],
            betas=None if betas is None else np.asarray(betas)[start:end],
        )
        written.append(save_hand_window(window_obj, target / f"{start:06d}_{end - 1:06d}.npz"))
    return written


def blend_windows(windows: list[HandWindow]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend overlapping windows (the model-free half of Phase 2)."""
    result = blend_hand_windows(windows)
    return result.joints_camera, result.valid, result.confidence, result.root_rot

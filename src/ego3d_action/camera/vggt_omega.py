"""VGGT-Omega backend adapter (Phase 3).

The reference configuration uses the 416 px / 200-frame / 40-overlap setup with
the official ``VGGT-Omega-1B-416-Reproduction`` checkpoint. That checkpoint is
published as a *reproduction* checkpoint: even with correct code there is no
guarantee of reproducing the blog's 52.0435 mm, so the pipeline never treats
that number as a target.
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
from .window import CameraWindow, WindowRange, load_camera_window

logger = logging.getLogger(__name__)

VGGT_SPEC = BackendSpec(
    name="VGGT-Omega",
    env="ego3d_vggt",
    install_hint=(
        "Clone VGGT-Omega into third_party/VGGT-Omega, download "
        "VGGT-Omega-1B-416-Reproduction into weights/vggt-omega, then install "
        "environment-vggt.yml."
    ),
    repository="https://github.com/facebookresearch/vggt",
)

SUPPORTED_CHECKPOINTS = (
    "VGGT-Omega-1B-416-Reproduction",  # default for this project
    "VGGT-Omega-1B-512",
    "VGGT-Omega-1B-256-Text-Alignment",
)

#: The file name each published checkpoint is distributed under. A checkpoint
#: may arrive as a directory of shards or as a single ``.pt``; both are accepted.
CHECKPOINT_FILENAMES: dict[str, tuple[str, ...]] = {
    "VGGT-Omega-1B-416-Reproduction": (
        "vggt_omega_1b_416_reproduce.pt",
        "vggt_omega_1b_416_reproduction.pt",
        "model.safetensors",
    ),
    "VGGT-Omega-1B-512": ("vggt_omega_1b_512.pt", "model.safetensors"),
    "VGGT-Omega-1B-256-Text-Alignment": ("vggt_omega_1b_256_text_alignment.pt", "model.safetensors"),
}

#: Any of these identifies a VGGT-Omega checkpoint, whatever it was requested as.
GENERIC_CHECKPOINT_FILENAMES = (
    "vggt_omega_1b_512.pt",
    "vggt_omega_1b_416_reproduce.pt",
    "vggt_omega_1b_416_reproduction.pt",
    "vggt_omega_1b_256_text_alignment.pt",
)

#: Payload of the most recent backend run, for provenance (which checkpoint was
#: actually used, how many windows were written, ...).
LAST_RUN: dict[str, object] = {}


@dataclass(frozen=True)
class VggtWindowRequest:
    """A window inference request expressed in video frame indices."""

    start: int
    end: int
    resolution: int
    checkpoint: str
    device: str = "auto"
    use_depth_confidence: bool = True

    @property
    def num_frames(self) -> int:
        return self.end - self.start

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise StageIOError(f"invalid window range [{self.start}, {self.end})")
        if self.resolution <= 0:
            raise StageIOError(f"resolution must be positive, got {self.resolution}")
        if self.checkpoint not in SUPPORTED_CHECKPOINTS:
            logger.warning(
                "checkpoint '%s' is not one of the documented checkpoints %s; "
                "proceeding anyway, but record it in the ablation table",
                self.checkpoint,
                list(SUPPORTED_CHECKPOINTS),
            )


def probe(third_party: str | Path, weights_root: str | Path) -> BackendStatus:
    return probe_backend(VGGT_SPEC, third_party=Path(third_party), weights_root=Path(weights_root))


def find_checkpoint(weights_root: str | Path, checkpoint: str) -> Path | None:
    """Locate a VGGT-Omega checkpoint as a directory *or* a single file.

    Accepts the documented layout (``weights/vggt-omega/<checkpoint>/``) and the
    flatter variants people end up with after a manual download
    (``weights/vggt-omega/<file>``, ``weights/vggt/<file>``, ``weights/<file>``).
    """
    root = Path(weights_root)
    stem = checkpoint.replace("/", "_")
    wanted = CHECKPOINT_FILENAMES.get(checkpoint, ())
    candidates: list[Path] = [
        root / checkpoint,
        root / "vggt-omega" / checkpoint,
        root / "vggt" / checkpoint,
    ]
    for directory in (root / "vggt-omega", root / "vggt", root):
        candidates.extend(directory / name for name in wanted)
    candidates.extend(
        [
            root / "vggt-omega" / f"{stem}.pt",
            root / "vggt-omega" / f"{checkpoint}.pt",
            root / "vggt" / f"{checkpoint}.pt",
            root / f"{checkpoint}.pt",
        ]
    )
    # Fall back to whatever VGGT-Omega checkpoint is actually on disk, but never
    # silently: ``resolve_checkpoint`` reports the substitution to the caller.
    for directory in (root / "vggt-omega", root / "vggt", root):
        candidates.extend(directory / name for name in GENERIC_CHECKPOINT_FILENAMES)
    candidates.extend([root / "vggt-omega", root / "vggt"])
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
        if candidate.is_file() and candidate.suffix in {".pt", ".pth", ".safetensors", ".ckpt"}:
            return candidate
    return None


def resolve_checkpoint(weights_root: str | Path, checkpoint: str) -> tuple[Path | None, str | None]:
    """Resolve ``checkpoint`` and say which file was actually used.

    Returns ``(path, substituted_name)`` where ``substituted_name`` is ``None``
    when the requested checkpoint was found, or the file name of the checkpoint
    that was used instead. Callers must surface a substitution - running the 512
    checkpoint while the configuration asks for the 416 reproduction is allowed,
    but it has to be visible in the logs, the runner summary and the ablation
    table, never silently.
    """
    path = find_checkpoint(weights_root, checkpoint)
    if path is None:
        return None, None
    if path.is_file():
        if path.name in CHECKPOINT_FILENAMES.get(checkpoint, ()) or path.stem == checkpoint:
            return path, None
        return path, path.name
    if path.name == checkpoint or checkpoint in str(path):
        return path, None
    return path, path.name


def require(third_party: str | Path, weights_root: str | Path) -> BackendStatus:
    return require_backend(VGGT_SPEC, third_party=Path(third_party), weights_root=Path(weights_root))


def run_window(
    request: VggtWindowRequest,
    *,
    invocation: BackendInvocation,
    third_party: str | Path,
    weights_root: str | Path,
    out_dir: str | Path,
    num_frames: int,
    frames_dir: str | Path | None = None,
    window: int = 200,
    overlap: int = 40,
    log_path: Path | None = None,
    selection: WindowSelection | None = None,
    skip_existing: bool = False,
    precision: str | None = None,
) -> list[Path]:
    """Run the VGGT-Omega backend and return the window files it wrote.

    Model loading dominates the runtime, so the backend is invoked **once** for
    the clip and writes every window into ``out_dir``; the schedule is
    re-derived and every written file is validated before it is trusted.

    ``selection`` lets one shard request only its windows. Because VGGT windows
    are independent, this *is* the parallel unit: N shards on N GPUs see exactly
    the same schedule as one whole-clip run. ``skip_existing`` reuses a shard
    whose windows and provenance marker are already on disk.

    ``precision`` is forwarded to the runner (``auto``/``fp32``/``fp16``); fp16
    halves the 1B checkpoint's resident memory (4.26 -> 2.13 GiB), which is how
    Phase 3 fits on a 12 GB GPU that another job is sharing.

    In mock mode this executes ``backends/mock_backend.py vggt``.

    Raises:
        BackendNotAvailableError: the backend env/weights are missing.
        BackendExecutionError: the runner failed or timed out.
        StageIOError: a written window is missing or malformed.
    """
    if not invocation.is_mock:
        require(third_party, weights_root)

    active = selection or WindowSelection()
    spec, prefix = invocation.resolve("vggt", mock_subcommand="vggt")
    args = [
        *prefix,
        "--out-dir",
        str(out_dir),
        *(["--frames", str(frames_dir)] if frames_dir is not None else []),
        "--num-frames",
        str(num_frames),
        "--window",
        str(window),
        "--overlap",
        str(overlap),
        "--resolution",
        str(request.resolution),
        "--checkpoint",
        request.checkpoint,
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
    payload = run_runner(spec, args, log_path=log_path)
    logger.info("VGGT-Omega runner reported %s", json.dumps(payload, sort_keys=True))
    LAST_RUN.clear()
    LAST_RUN.update(payload)

    from .window import make_windows

    target = Path(out_dir)
    expected = active.select(make_windows(num_frames, window=window, overlap=overlap))
    if not expected:
        raise StageIOError(
            f"selection '{active.describe()}' matched no VGGT window; the clip has "
            f"{len(make_windows(num_frames, window=window, overlap=overlap))} window(s)"
        )
    paths: list[Path] = []
    for rng in expected:
        path = target / f"{rng.start:06d}_{rng.end - 1:06d}.npz"
        if not path.is_file():
            raise StageIOError(
                f"the camera backend did not produce window {path.name}; it reported "
                f"{payload.get('windows', 'no window list')}"
            )
        load_camera_window(path)  # validates shapes before the file is trusted
        paths.append(path)
    return paths


def empty_window(start: int, end: int, *, height: int, width: int, intrinsics: np.ndarray) -> CameraWindow:
    """A window placeholder used by tests and by dry runs of the stitcher."""
    num = end - start
    return CameraWindow(
        window=WindowRange(index=0, start=start, end=end),
        rotation_c2w=np.broadcast_to(np.eye(3), (num, 3, 3)).copy(),
        translation_c2w=np.zeros((num, 3)),
        intrinsics=np.broadcast_to(intrinsics, (num, 3, 3)).copy(),
        depth=np.full((num, height, width), np.nan),
    )


def camera_window_from_output(
    *,
    start: int,
    end: int,
    rotation_c2w: np.ndarray,
    translation_c2w: np.ndarray,
    intrinsics: np.ndarray,
    depth: np.ndarray,
    depth_confidence: np.ndarray | None = None,
) -> CameraWindow:
    """Validate a backend's raw window output and wrap it as a :class:`CameraWindow`.

    This is the single place where "what the model returned" becomes "what the
    pipeline trusts": shapes, frame count and the window range are checked here,
    so a backend that silently truncates or transposes a depth map fails at the
    boundary instead of producing a plausible-looking wrong trajectory.

    Raises:
        StageIOError: on shape mismatch or an inconsistent frame count.
    """
    num = int(np.asarray(depth).shape[0])
    if end - start != num:
        raise StageIOError(
            f"window [{start}, {end}) has {end - start} frames but the backend returned {num}"
        )
    window = CameraWindow(
        window=WindowRange(index=0, start=start, end=end),
        rotation_c2w=np.asarray(rotation_c2w, dtype=np.float64),
        translation_c2w=np.asarray(translation_c2w, dtype=np.float64),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        depth=np.asarray(depth, dtype=np.float64),
        depth_confidence=None if depth_confidence is None else np.asarray(depth_confidence, dtype=np.float64),
    )
    if not np.isfinite(window.depth).any():
        logger.warning("window %s has no finite depth values", window.name)
    return window

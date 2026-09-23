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
) -> list[Path]:
    """Run the VGGT-Omega backend and return the window files it wrote.

    Model loading dominates the runtime, so the backend is invoked **once** for
    the whole clip and writes every window into ``out_dir``; the schedule is
    re-derived and every written file is validated before it is trusted.

    In mock mode this executes ``backends/mock_backend.py vggt``.

    Raises:
        BackendNotAvailableError: the backend env/weights are missing.
        BackendExecutionError: the runner failed or timed out.
        StageIOError: a written window is missing or malformed.
    """
    if not invocation.is_mock:
        require(third_party, weights_root)

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
    payload = run_runner(spec, args, log_path=log_path)
    logger.info("VGGT-Omega runner reported %s", json.dumps(payload, sort_keys=True))

    from .window import make_windows

    target = Path(out_dir)
    paths: list[Path] = []
    for rng in make_windows(num_frames, window=window, overlap=overlap):
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

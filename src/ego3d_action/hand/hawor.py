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
    log_path: Path | None = None,
) -> list[Path]:
    """Reconstruct every window of a clip and return the written files.

    In mock mode this executes ``backends/mock_backend.py hawor``; in real mode it
    requires the checkout/weights and runs ``backends/hawor_runner.py`` in the
    configured interpreter (normally ``ego3d_hawor``). Every written window is
    validated before it is trusted.

    Raises:
        BackendNotAvailableError: the checkpoint/env is not present.
        BackendExecutionError: the runner failed or timed out.
        StageIOError: a window is missing or malformed.
    """
    if not invocation.is_mock:
        require(third_party, weights_root)

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
    if detection_path is not None:
        args += ["--detection", str(detection_path)]
    payload = run_runner(spec, args, log_path=log_path)
    logger.info("HaWoR runner reported %s", json.dumps(payload, sort_keys=True))

    target = Path(out_dir)
    paths: list[Path] = []
    for start, end in request.ranges():
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


def blend_windows(windows: list[HandWindow]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend overlapping windows (the model-free half of Phase 2)."""
    result = blend_hand_windows(windows)
    return result.joints_camera, result.valid, result.confidence, result.root_rot

"""HaWoR backend adapter (Phase 2).

HaWoR reconstructs 16-frame windows with an 8-frame overlap. The backend runs
in its own environment (``ego3d_hawor``: Python 3.10 / torch 1.13 / CUDA 11.7,
with DROID-SLAM and Metric3D available but unused here, because the whole point
of this project is to replace them with VGGT-Omega).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import BackendInvocationNotImplemented
from ..runtime.backend import BackendSpec, BackendStatus, probe_backend, require_backend
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
class HaworWindowRequest:
    """One HaWoR window request, always resolved against the on-disk frames."""

    start: int
    end: int
    frames_dir: Path
    boxes: np.ndarray  # [n, 2, 4]
    valid: np.ndarray  # [n, 2]
    confidence: np.ndarray  # [n, 2]
    device: str = "auto"

    @property
    def num_frames(self) -> int:
        return self.end - self.start


def probe(third_party: str | Path, weights_root: str | Path) -> BackendStatus:
    return probe_backend(HAWOR_SPEC, third_party=Path(third_party), weights_root=Path(weights_root))


def require(third_party: str | Path, weights_root: str | Path) -> BackendStatus:
    return require_backend(HAWOR_SPEC, third_party=Path(third_party), weights_root=Path(weights_root))


def run_window(
    request: HaworWindowRequest,
    *,
    third_party: str | Path,
    weights_root: str | Path,
) -> HandWindow:
    """Reconstruct one window in camera space.

    Raises:
        BackendNotAvailableError: the checkpoint/env is not present, or the
            backend invocation has not been wired up yet (GPU server step).
    """
    status = require(third_party, weights_root)
    raise BackendInvocationNotImplemented(
        "HaWoR",
        f"window reconstruction is executed by the '{HAWOR_SPEC.env}' environment on the GPU "
        f"server; request frames [{request.start}, {request.end}) with "
        f"{int(np.count_nonzero(request.valid))} valid hand-frames "
        f"(checkout={status.checkout}, weights={status.weights}).",
    )


def blend_windows(windows: list[HandWindow]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend overlapping windows (the model-free half of Phase 2)."""
    result = blend_hand_windows(windows)
    return result.joints_camera, result.valid, result.confidence, result.root_rot

"""VGGT-Omega backend adapter (Phase 3).

The reference configuration uses the 416 px / 200-frame / 40-overlap setup with
the official ``VGGT-Omega-1B-416-Reproduction`` checkpoint. That checkpoint is
published as a *reproduction* checkpoint: even with correct code there is no
guarantee of reproducing the blog's 52.0435 mm, so the pipeline never treats
that number as a target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import BackendInvocationNotImplemented, StageIOError
from ..runtime.backend import BackendSpec, BackendStatus, probe_backend, require_backend
from .window import CameraWindow, WindowRange

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
    third_party: str | Path,
    weights_root: str | Path,
) -> CameraWindow:
    """Run one VGGT-Omega window and return metric poses + depth.

    Raises:
        BackendNotAvailableError: the backend env/weights are missing, or the
            invocation has not been wired to the GPU server yet.
    """
    status = require(third_party, weights_root)
    raise BackendInvocationNotImplemented(
        "VGGT-Omega",
        f"window inference runs in the '{VGGT_SPEC.env}' environment on the GPU server; "
        f"requested frames [{request.start}, {request.end}) at {request.resolution} px with "
        f"checkpoint {request.checkpoint} "
        f"(checkout={status.checkout}, weights={status.weights}).",
    )


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

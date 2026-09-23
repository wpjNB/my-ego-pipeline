"""WiLoR backend adapter (Phase 1).

WiLoR is treated as an external detector: the pipeline never forks it. This
adapter owns three responsibilities:

1. report whether the checkout/weights are present (`probe`),
2. convert raw detections into the pipeline's :class:`HandDetection` records,
3. run the conservative tracker on top of them.

The detector call itself is delegated to the backend environment
(``ego3d_wilor``) because WiLoR pins an older torch/CUDA stack than the
orchestrator env.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..errors import BackendInvocationNotImplemented, StageIOError
from ..runtime.backend import BackendSpec, BackendStatus, probe_backend, require_backend
from .tracker import HandDetection, Track, conservative_track_both, tracks_to_detection_arrays

logger = logging.getLogger(__name__)

WILOR_SPEC = BackendSpec(
    name="WiLoR",
    env="ego3d_wilor",
    install_hint=(
        "Clone WiLoR into third_party/WiLoR and place its checkpoint in weights/wilor, then "
        "install the backend env (see environment-wilor.yml)."
    ),
    repository="https://github.com/rolpotamias/WiLoR",
)


@dataclass(frozen=True)
class RawDetection:
    """Detector output before handedness/tracking decisions."""

    bbox: np.ndarray  # [4] x1, y1, x2, y2
    confidence: float
    right_score: float = 0.0
    left_score: float = 0.0
    track_hint: int = -1


def probe(third_party: str | Path, weights_root: str | Path) -> BackendStatus:
    """Check whether the WiLoR checkout and weights are available."""
    return probe_backend(WILOR_SPEC, third_party=Path(third_party), weights_root=Path(weights_root))


def require(third_party: str | Path, weights_root: str | Path) -> BackendStatus:
    """Raise :class:`BackendNotAvailableError` when WiLoR cannot be used."""
    return require_backend(WILOR_SPEC, third_party=Path(third_party), weights_root=Path(weights_root))


def to_hand_detections(
    frame_detections: Sequence[Sequence[RawDetection]],
    *,
    handedness_threshold: float = 0.0,
) -> list[list[HandDetection]]:
    """Convert raw detector output into pipeline detections.

    A detection is assigned ``handedness = 1`` (right) when its right-hand score
    beats the left-hand score, otherwise ``0`` (left).
    """
    converted: list[list[HandDetection]] = []
    for frame, detections in enumerate(frame_detections):
        frame_out: list[HandDetection] = []
        for det in detections:
            handedness = 1 if (det.right_score - det.left_score) > handedness_threshold else 0
            frame_out.append(
                HandDetection(
                    frame_id=frame,
                    bbox=np.asarray(det.bbox, dtype=np.float64),
                    confidence=float(det.confidence),
                    handedness=handedness,
                )
            )
        converted.append(frame_out)
    return converted


def track_clip(
    frame_detections: Sequence[Sequence[RawDetection]],
    *,
    min_confidence: float = 0.75,
    max_gap: int = 4,
    iou_threshold: float = 0.20,
) -> dict[str, np.ndarray]:
    """Run the conservative tracker over a whole clip.

    Returns the arrays written to ``detection/detection.npz``.
    """
    detections = to_hand_detections(frame_detections)
    tracks: dict[int, Track] = conservative_track_both(
        detections,
        num_frames=len(detections),
        min_confidence=min_confidence,
        max_gap=max_gap,
        iou_threshold=iou_threshold,
    )
    logger.info(
        "tracking: left %.1f%%, right %.1f%% coverage",
        100.0 * tracks[0].coverage,
        100.0 * tracks[1].coverage,
    )
    return tracks_to_detection_arrays(tracks)


def detect_clip(
    frames_dir: str | Path,
    *,
    third_party: str | Path,
    weights_root: str | Path,
    device: str = "auto",
    batch_size: int = 4,
) -> list[list[RawDetection]]:
    """Run the WiLoR detector over a decoded frame folder.

    Raises:
        BackendNotAvailableError: when the checkout/weights are missing.
        StageIOError: when ``frames_dir`` holds no frames.
    """
    status = require(third_party, weights_root)
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise StageIOError(f"no frames found in {frames_dir}")
    raise BackendInvocationNotImplemented(
        "WiLoR",
        f"detector invocation is wired to the '{WILOR_SPEC.env}' environment and runs on the "
        f"GPU server; {len(frames)} frames are ready at {frames_dir} "
        f"(checkout={status.checkout}, weights={status.weights}). "
        "This adapter intentionally does not fall back to a CPU re-implementation.",
    )

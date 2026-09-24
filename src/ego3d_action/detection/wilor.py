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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..errors import StageIOError
from ..runtime.backend import BackendSpec, BackendStatus, probe_backend, require_backend
from ..runtime.subprocess_backend import BackendInvocation, run_runner
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


def raw_detections_from_arrays(arrays: Mapping[str, np.ndarray]) -> list[list[RawDetection]]:
    """Convert the runner's ``boxes/confidence/count`` arrays into detections."""
    if "boxes" not in arrays or "confidence" not in arrays:
        raise StageIOError("raw detections need at least 'boxes' and 'confidence'")
    boxes = np.asarray(arrays["boxes"], dtype=np.float64)
    confidence = np.asarray(arrays["confidence"], dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise StageIOError(f"boxes must be [T, K, 4], got {boxes.shape}")
    if confidence.shape != boxes.shape[:2]:
        raise StageIOError(f"confidence must be {boxes.shape[:2]}, got {confidence.shape}")
    right = arrays.get("right_score")
    left = arrays.get("left_score")
    count = arrays.get("count")
    if count is None:
        count = np.full(boxes.shape[0], boxes.shape[1], dtype=np.int64)
    count = np.asarray(count, dtype=np.int64).reshape(-1)
    if count.shape[0] != boxes.shape[0]:
        raise StageIOError(f"count must have length {boxes.shape[0]}, got {count.shape}")

    frames: list[list[RawDetection]] = []
    for frame in range(boxes.shape[0]):
        detections: list[RawDetection] = []
        for slot in range(int(count[frame])):
            detections.append(
                RawDetection(
                    bbox=boxes[frame, slot],
                    confidence=float(confidence[frame, slot]),
                    right_score=float(right[frame, slot]) if right is not None else 0.0,
                    left_score=float(left[frame, slot]) if left is not None else 0.0,
                )
            )
        frames.append(detections)
    return frames


def load_raw_detections(path: str | Path) -> list[list[RawDetection]]:
    """Read detections exported by a WiLoR runner."""
    from ..io.serialization import load_npz

    return raw_detections_from_arrays(load_npz(path, required=("boxes", "confidence")))


def build_raw_detection_arrays(
    frames: Sequence[Sequence[RawDetection]],
    *,
    frame_ids: Sequence[int] | None = None,
    num_frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Pack detector output into the ``boxes/confidence/count`` artefact.

    This is the conversion the runner uses: whatever shape WiLoR's API returns,
    it ends up as compacted per-frame candidate slots plus an explicit
    ``count``, so "no detection in this frame" is representable without a
    zero-confidence placeholder box. An empty detection list is legal as long as
    ``num_frames`` says how long the clip is - "no hands anywhere" is a real
    outcome, not an error.

    Raises:
        StageIOError: when frames do not cover ``num_frames``.
    """
    ids = list(frame_ids) if frame_ids is not None else list(range(len(frames)))
    total = num_frames if num_frames is not None else (max(ids) + 1 if ids else 0)
    if total <= 0:
        raise StageIOError("cannot build detection arrays for an empty clip")
    if len(ids) != len(frames):
        raise StageIOError(f"{len(ids)} frame ids for {len(frames)} detection lists")

    slots = max(1, max((len(detections) for detections in frames), default=1))
    boxes = np.zeros((total, slots, 4), dtype=np.float64)
    confidence = np.zeros((total, slots), dtype=np.float64)
    right_score = np.zeros((total, slots), dtype=np.float64)
    left_score = np.zeros((total, slots), dtype=np.float64)
    count = np.zeros(total, dtype=np.int64)
    for frame_id, detections in zip(ids, frames, strict=False):
        if not 0 <= frame_id < total:
            raise StageIOError(f"frame id {frame_id} is outside [0, {total})")
        count[frame_id] = len(detections)
        for slot, detection in enumerate(detections):
            boxes[frame_id, slot] = np.asarray(detection.bbox, dtype=np.float64)
            confidence[frame_id, slot] = float(detection.confidence)
            if detection.right_score:
                right_score[frame_id, slot] = float(detection.right_score)
            if detection.left_score:
                left_score[frame_id, slot] = float(detection.left_score)
    return {
        "boxes": boxes,
        "confidence": confidence,
        "right_score": right_score,
        "left_score": left_score,
        "count": count,
    }


def detections_from_predictions(predictions: object, *, frame: int) -> list[RawDetection]:
    """Convert one frame of WiLoR output into :class:`RawDetection` records.

    WiLoR returns boxes, scores and left/right scores. The conversion is
    forgiving about field names (``boxes``/``pred_boxes``,
    ``scores``/``conf``/``pred_scores``, ``left``/``right``) but never guesses:
    an unrecognised layout raises with the keys it actually saw, so a backend
    upgrade fails loudly instead of producing empty detections.

    Raises:
        NotImplementedError: the output layout is not recognised.
    """

    def pick(*names: str) -> object | None:
        for name in names:
            if isinstance(predictions, dict) and name in predictions:
                return predictions[name]
            if hasattr(predictions, name):
                return getattr(predictions, name)
        return None

    boxes = pick("boxes", "pred_boxes", "det_boxes")
    scores = pick("scores", "conf", "pred_scores", "box_scores")
    if boxes is None or scores is None:
        available = (
            sorted(predictions.keys()) if isinstance(predictions, dict) else dir(predictions)
        )
        raise NotImplementedError(
            "cannot map WiLoR output to detections: expected box and score fields. "
            f"Available: {available}"
        )
    left = pick("left", "left_scores", "is_left")
    right = pick("right", "right_scores", "is_right")
    boxes_array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    left_array = (
        np.asarray(left, dtype=np.float64).reshape(-1)
        if left is not None
        else np.zeros_like(scores_array)
    )
    right_array = (
        np.asarray(right, dtype=np.float64).reshape(-1)
        if right is not None
        else np.zeros_like(scores_array)
    )
    return [
        RawDetection(
            bbox=box,
            confidence=float(score),
            left_score=float(left_array[index]) if index < left_array.size else 0.0,
            right_score=float(right_array[index]) if index < right_array.size else 0.0,
        )
        for index, (box, score) in enumerate(zip(boxes_array, scores_array, strict=False))
    ]


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
    out_path: str | Path,
    *,
    invocation: BackendInvocation,
    third_party: str | Path,
    weights_root: str | Path,
    device: str = "auto",
    batch_size: int = 4,
    detector_confidence: float = 0.1,
    width: int | None = None,
    height: int | None = None,
    image_format: str = "jpg",
    num_frames: int | None = None,
    log_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Run the WiLoR detector over a decoded frame folder.

    In mock mode this executes ``backends/mock_backend.py``; in real mode it
    requires the checkout/weights and runs ``backends/wilor_runner.py`` inside
    the configured interpreter (normally ``ego3d_wilor``).

    Returns:
        The detection arrays written to ``out_path``.

    Raises:
        BackendNotAvailableError: checkout/weights missing in real mode.
        StageIOError: no frames to process.
        BackendExecutionError: the runner failed or timed out.
    """
    if not invocation.is_mock:
        require(third_party, weights_root)
    frames = sorted(Path(frames_dir).glob(f"*.{image_format}"))
    if not frames and num_frames is None:
        raise StageIOError(f"no frames found in {frames_dir}")

    spec, prefix = invocation.resolve("wilor", mock_subcommand="wilor")
    args = [
        *prefix,
        "--frames",
        str(frames_dir),
        "--out",
        str(out_path),
        "--image-format",
        image_format,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--third-party",
        str(third_party),
        "--weights",
        str(weights_root),
        # Keep the detector threshold below the tracker's anchor threshold
        # (0.75) so gap-recovery candidates survive the detector.
        "--conf",
        str(detector_confidence),
    ]
    if num_frames is not None:
        args += ["--num-frames", str(num_frames)]
    if width is not None:
        args += ["--width", str(width)]
    if height is not None:
        args += ["--height", str(height)]

    payload = run_runner(spec, args, log_path=log_path)
    logger.info("WiLoR runner reported %s", json.dumps(payload, sort_keys=True))
    from ..io.serialization import load_npz

    return load_npz(out_path, required=("boxes", "confidence"))

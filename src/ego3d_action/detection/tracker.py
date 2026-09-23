"""Conservative hand tracking (Phase 1).

Macrodata's final system does **not** use the upstream HaWoR
``detect_track(..., thresh=0.2)`` heuristic. It uses a conservative rule set:

===========================  ==========================================
high confidence              ``confidence >= 0.75``
gap recovery window          same-side gap ``<= 4`` frames
recovery matching            interpolated-box IoU ``>= 0.20``
===========================  ==========================================

Only gaps *between two high-confidence anchors* are considered, and a low
confidence detection is accepted only when its box overlaps the interpolated
anchor box. Everything else stays missing - the pipeline never invents 3D pose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray

LEFT = 0
RIGHT = 1
SIDES: tuple[int, ...] = (LEFT, RIGHT)


@dataclass(frozen=True)
class HandDetection:
    """A single hand detection in one frame."""

    frame_id: int
    bbox: Array  # [x1, y1, x2, y2]
    confidence: float
    handedness: int  # 0 left, 1 right

    def __post_init__(self) -> None:
        box = np.asarray(self.bbox, dtype=np.float64)
        if box.shape != (4,):
            raise StageIOError(f"bbox must have shape [4], got {box.shape}")
        if not np.all(np.isfinite(box)):
            raise StageIOError("bbox contains non-finite coordinates")
        if box[2] < box[0] or box[3] < box[1]:
            raise StageIOError(f"bbox has negative extent: {box.tolist()}")
        if self.handedness not in SIDES:
            raise StageIOError(f"handedness must be 0 (left) or 1 (right), got {self.handedness}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise StageIOError(f"confidence must lie in [0, 1], got {self.confidence}")
        object.__setattr__(self, "bbox", box)
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "frame_id", int(self.frame_id))

    @property
    def area(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1]))


@dataclass(frozen=True)
class Track:
    """A single-hand track over the whole clip."""

    side: int
    boxes: Array  # [T, 4], NaN where invalid
    confidence: Array  # [T], 0 where invalid
    valid: Array  # [T] bool
    track_id: Array  # [T] int, -1 where invalid
    recovered: Array  # [T] bool, True where a low-confidence detection was accepted

    @property
    def num_frames(self) -> int:
        return int(self.valid.shape[0])

    @property
    def coverage(self) -> float:
        return float(np.mean(self.valid)) if self.num_frames else 0.0

    @property
    def num_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    def to_arrays(self) -> dict[str, Array]:
        return {
            "boxes": self.boxes,
            "confidence": self.confidence,
            "valid": self.valid,
            "track_id": self.track_id,
            "recovered": self.recovered,
        }


def box_iou(box_a: Array, box_b: Array) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def interpolate_box(box_a: Array, box_b: Array, alpha: float) -> Array:
    """Linear interpolation between two boxes (``alpha`` in ``[0, 1]``)."""
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    return (1.0 - float(alpha)) * a + float(alpha) * b


def conservative_track(
    detections: Sequence[Sequence[HandDetection]],
    *,
    side: int,
    num_frames: int | None = None,
    min_confidence: float = 0.75,
    max_gap: int = 4,
    iou_threshold: float = 0.20,
) -> Track:
    """Track one hand side across a clip with the conservative rule set.

    Args:
        detections: ``detections[t]`` is the list of candidate
            :class:`HandDetection` objects for frame ``t``. Candidates whose
            ``handedness`` differs from ``side`` are ignored.
        side: ``0`` for the left hand, ``1`` for the right hand.
        num_frames: clip length; inferred from ``detections`` when omitted.
        min_confidence: anchor threshold (spec: ``0.75``).
        max_gap: largest number of missing frames that may be recovered
            (spec: ``4``).
        iou_threshold: required IoU against the interpolated anchor box
            (spec: ``0.20``).

    Returns:
        A :class:`Track` where frames that fail every rule stay ``valid=False``.

    Raises:
        StageIOError: on an invalid ``side``, negative ``max_gap``, or a
            detection list shorter than ``num_frames``.
    """
    if side not in SIDES:
        raise StageIOError(f"side must be 0 (left) or 1 (right), got {side}")
    if max_gap < 0:
        raise StageIOError(f"max_gap must be >= 0, got {max_gap}")
    if not 0.0 <= iou_threshold <= 1.0:
        raise StageIOError(f"iou_threshold must lie in [0, 1], got {iou_threshold}")
    if not 0.0 <= min_confidence <= 1.0:
        raise StageIOError(f"min_confidence must lie in [0, 1], got {min_confidence}")

    total = num_frames if num_frames is not None else len(detections)
    if total <= 0:
        raise StageIOError("num_frames must be positive")
    if len(detections) < total:
        raise StageIOError(f"detections has {len(detections)} frames, expected at least {total}")

    boxes = np.full((total, 4), np.nan, dtype=np.float64)
    confidence = np.zeros(total, dtype=np.float64)
    valid = np.zeros(total, dtype=bool)
    recovered = np.zeros(total, dtype=bool)
    track_id = np.full(total, -1, dtype=np.int64)

    # Best candidate per frame (highest confidence, then largest area).
    candidates: list[HandDetection | None] = []
    for frame in range(total):
        pool = [d for d in detections[frame] if d.handedness == side]
        if not pool:
            candidates.append(None)
            continue
        candidates.append(max(pool, key=lambda d: (d.confidence, d.area)))

    anchors = [
        frame
        for frame in range(total)
        if candidates[frame] is not None and candidates[frame].confidence >= min_confidence  # type: ignore[union-attr]
    ]

    for frame in anchors:
        det = candidates[frame]
        assert det is not None  # guaranteed by the anchors filter
        boxes[frame] = det.bbox
        confidence[frame] = det.confidence
        valid[frame] = True

    for left_anchor, right_anchor in zip(anchors, anchors[1:], strict=False):
        gap = right_anchor - left_anchor - 1
        if gap <= 0 or gap > max_gap:
            continue
        box_a = candidates[left_anchor].bbox  # type: ignore[union-attr]
        box_b = candidates[right_anchor].bbox  # type: ignore[union-attr]
        span = right_anchor - left_anchor
        for frame in range(left_anchor + 1, right_anchor):
            det = candidates[frame]
            if det is None or det.confidence >= min_confidence:
                continue
            expected = interpolate_box(box_a, box_b, (frame - left_anchor) / span)
            iou = box_iou(det.bbox, expected)
            if iou >= iou_threshold:
                boxes[frame] = det.bbox
                confidence[frame] = det.confidence
                valid[frame] = True
                recovered[frame] = True
                logger.debug(
                    "side %d: recovered frame %d (conf=%.3f, IoU=%.3f)", side, frame, det.confidence, iou
                )

    track_id[valid] = 0
    return Track(
        side=side,
        boxes=boxes,
        confidence=confidence,
        valid=valid,
        track_id=track_id,
        recovered=recovered,
    )


def conservative_track_both(
    detections: Sequence[Sequence[HandDetection]],
    *,
    num_frames: int | None = None,
    min_confidence: float = 0.75,
    max_gap: int = 4,
    iou_threshold: float = 0.20,
) -> dict[int, Track]:
    """Run :func:`conservative_track` for both hands and return ``{0: left, 1: right}``."""
    return {
        side: conservative_track(
            detections,
            side=side,
            num_frames=num_frames,
            min_confidence=min_confidence,
            max_gap=max_gap,
            iou_threshold=iou_threshold,
        )
        for side in SIDES
    }


def tracks_to_detection_arrays(tracks: dict[int, Track]) -> dict[str, Array]:
    """Stack per-side tracks into the ``detection/*.npy`` style arrays.

    Shapes: ``boxes [T, 2, 4]``, ``confidence [T, 2]``, ``valid [T, 2]``,
    ``track_id [T, 2]``, ``recovered [T, 2]`` with left at index 0.
    """
    if sorted(tracks) != [LEFT, RIGHT]:
        raise StageIOError(f"expected tracks for sides 0 and 1, got {sorted(tracks)}")
    left, right = tracks[LEFT], tracks[RIGHT]
    if left.num_frames != right.num_frames:
        raise StageIOError(
            f"left/right tracks have different lengths: {left.num_frames} vs {right.num_frames}"
        )
    return {
        "boxes": np.stack([left.boxes, right.boxes], axis=1),
        "confidence": np.stack([left.confidence, right.confidence], axis=1),
        "valid": np.stack([left.valid, right.valid], axis=1),
        "track_id": np.stack([left.track_id, right.track_id], axis=1),
        "recovered": np.stack([left.recovered, right.recovered], axis=1),
    }

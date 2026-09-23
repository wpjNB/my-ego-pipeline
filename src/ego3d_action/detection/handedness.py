"""Handedness utilities for WiLoR output.

WiLoR emits a left/right score per detection. Two things matter downstream:

1. the score has to become a hard ``0/1`` label, and
2. an egocentric clip must stay anatomically consistent - a hand that is "left"
   in frame 0 cannot silently become "right" in frame 100.
"""

from __future__ import annotations

import logging

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray


def scores_to_handedness(right_scores: Array, *, threshold: float = 0.0) -> Array:
    """Convert ``right``-hand scores into ``0`` (left) / ``1`` (right) labels.

    ``threshold`` is applied to the *log-ratio* ``score_right - score_left``
    when a ``[..., 2]`` score array is supplied, or directly to the score when a
    ``[...]`` array of right-hand scores is supplied.
    """
    scores = np.asarray(right_scores, dtype=np.float64)
    if scores.shape[-1:] == (2,):
        logits = scores[..., 1] - scores[..., 0]
    else:
        logits = scores
    return (logits > threshold).astype(np.int64)


def flip_handedness(labels: Array) -> Array:
    """Map ``0 -> 1`` and ``1 -> 0``."""
    arr = np.asarray(labels, dtype=np.int64)
    if not np.all(np.isin(arr, (0, 1))):
        raise StageIOError("handedness labels must be 0 or 1")
    return 1 - arr


def majority_handedness(labels: Array, *, weights: Array | None = None) -> int:
    """Return the dominant label of a sequence, ``-1`` when it is empty."""
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return -1
    if not np.all(np.isin(arr, (0, 1))):
        raise StageIOError("handedness labels must be 0 or 1")
    if weights is None:
        score = np.bincount(arr, minlength=2)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != arr.shape[0]:
            raise StageIOError(f"weights length {w.shape[0]} != labels length {arr.shape[0]}")
        score = np.zeros(2, dtype=np.float64)
        for label in (0, 1):
            score[label] = float(np.sum(w[arr == label]))
    return int(np.argmax(score))


def enforce_side_consistency(
    labels: Array,
    *,
    confidence: Array | None = None,
    min_frames: int = 5,
) -> Array:
    """Relabel a short sequence so that one physical hand keeps one label.

    If the weighted majority of the frames agrees with a single label, any
    minority frames are flipped, because in an egocentric recording the left
    hand never becomes the right hand.

    Args:
        labels: ``[T]`` array of ``0/1`` labels.
        confidence: optional ``[T]`` detection confidences used as vote weights.
        min_frames: sequences shorter than this are returned unchanged.

    Returns:
        A corrected copy of ``labels``.
    """
    arr = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
    if arr.size < min_frames:
        return arr
    dominant = majority_handedness(arr, weights=confidence)
    if dominant < 0:
        return arr
    flipped = int(np.count_nonzero(arr != dominant))
    if flipped:
        logger.info(
            "handedness: enforcing dominant label %d, flipped %d/%d frames", dominant, flipped, arr.size
        )
    arr[:] = dominant
    return arr

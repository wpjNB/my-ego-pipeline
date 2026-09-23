"""Coverage metrics: missing hand-frames stay missing.

The reference system reports lower coverage than the original HaWoR run
(81.23 % vs 87.11 %) because the conservative tracker refuses to invent poses.
Coverage is therefore an explicit, first-class metric here.
"""

from __future__ import annotations

import numpy as np

from ..errors import StageIOError

Array = np.ndarray


def coverage_ratio(valid: Array) -> float:
    """Fraction of ``True`` entries in a ``[T, 2]`` (or ``[T]``) mask."""
    mask = np.asarray(valid, dtype=bool)
    if mask.size == 0:
        raise StageIOError("coverage is undefined for an empty mask")
    return float(np.mean(mask))


def coverage_by_hand(valid: Array) -> Array:
    """Per-hand coverage for a ``[T, 2]`` mask."""
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != 2:
        raise StageIOError(f"valid must be [T, 2], got {mask.shape}")
    if mask.shape[0] == 0:
        raise StageIOError("coverage is undefined for an empty sequence")
    return np.mean(mask, axis=0)


def missing_runs(valid: Array) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` runs of missing frames in a ``[T]`` mask."""
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if mask.size == 0:
        return []
    padded = np.concatenate([[True], mask, [True]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2], strict=False)]

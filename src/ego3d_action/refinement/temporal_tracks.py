"""Short-gap interpolation and rotation-aware temporal filtering, stages P2/P3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError
from ..geometry.transforms import rotation_slerp

Array = np.ndarray


@dataclass(frozen=True)
class TrackFillResult:
    """Filled track and masks describing recovered samples."""

    values: Array
    valid: Array
    interpolated: Array
    gaps_filled: int
    frames_filled: int


def _missing_runs(valid: Array) -> list[tuple[int, int]]:
    missing = ~np.asarray(valid, dtype=bool).reshape(-1)
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def interpolate_vector_gaps(values: Array, valid: Array, *, max_gap: int) -> TrackFillResult:
    """Linearly fill short interior gaps in a [T,...] vector-valued track."""
    x = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if x.ndim < 2 or x.shape[0] == 0:
        raise StageIOError(f"values must be [T, ...] with T > 0, got {x.shape}")
    if mask.shape != (x.shape[0],):
        raise StageIOError(f"valid must be [{x.shape[0]}], got {mask.shape}")
    if max_gap < 1:
        raise StageIOError(f"max_gap must be >= 1, got {max_gap}")

    row_finite = np.isfinite(x).reshape(x.shape[0], -1).all(axis=1)
    anchors = mask & row_finite
    out = x.copy()
    interpolated = np.zeros(x.shape[0], dtype=bool)
    gaps_filled = 0
    for start, end in _missing_runs(anchors):
        if start == 0 or end == x.shape[0] or end - start > max_gap:
            continue
        left, right = start - 1, end
        if not anchors[left] or not anchors[right]:
            continue
        weights = (np.arange(start, end, dtype=np.float64) - left) / float(right - left)
        broadcast = weights.reshape((weights.size,) + (1,) * (x.ndim - 1))
        out[start:end] = x[left][None, ...] + broadcast * (x[right] - x[left])[None, ...]
        interpolated[start:end] = True
        gaps_filled += 1

    valid_out = anchors | interpolated
    return TrackFillResult(
        values=out,
        valid=valid_out,
        interpolated=interpolated,
        gaps_filled=gaps_filled,
        frames_filled=int(interpolated.sum()),
    )


def interpolate_rotation_gaps(
    rotations: Array, valid: Array, *, max_gap: int
) -> TrackFillResult:
    """Fill short interior rotation gaps by shortest-path SLERP.

    Rotation tracks have shape [T,...,3,3] and validity has shape [T,...].
    Leading/trailing and over-limit holes remain missing.
    """
    r = np.asarray(rotations, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if r.ndim < 3 or r.shape[-2:] != (3, 3) or r.shape[0] == 0:
        raise StageIOError(f"rotations must be [T, ..., 3, 3], got {r.shape}")
    expected = r.shape[:-2]
    if mask.shape != expected:
        raise StageIOError(f"valid must be {expected}, got {mask.shape}")
    if max_gap < 1:
        raise StageIOError(f"max_gap must be >= 1, got {max_gap}")

    tracks = int(np.prod(r.shape[1:-2])) if r.ndim > 3 else 1
    frames = r.shape[0]
    flat_r = r.reshape(frames, tracks, 3, 3)
    flat_mask = mask.reshape(frames, tracks)
    out = flat_r.copy()
    interp = np.zeros((frames, tracks), dtype=bool)
    gaps_filled = 0
    for track in range(tracks):
        finite = np.isfinite(flat_r[:, track]).all(axis=(1, 2))
        anchors = flat_mask[:, track] & finite
        for start, end in _missing_runs(anchors):
            if start == 0 or end == frames or end - start > max_gap:
                continue
            left, right = start - 1, end
            if not anchors[left] or not anchors[right]:
                continue
            alpha = (np.arange(start, end, dtype=np.float64) - left) / float(right - left)
            a = np.broadcast_to(flat_r[left, track], (alpha.size, 3, 3))
            b = np.broadcast_to(flat_r[right, track], (alpha.size, 3, 3))
            out[start:end, track] = rotation_slerp(a, b, alpha)
            interp[start:end, track] = True
            gaps_filled += 1

    finite_out = np.isfinite(flat_r).all(axis=(2, 3))
    valid_out = (flat_mask & finite_out) | interp
    return TrackFillResult(
        values=out.reshape(r.shape),
        valid=valid_out.reshape(mask.shape),
        interpolated=interp.reshape(mask.shape),
        gaps_filled=gaps_filled,
        frames_filled=int(interp.sum()),
    )


def smooth_rotation_tracks(
    rotations: Array,
    valid: Array,
    *,
    strength: float = 0.25,
    passes: int = 1,
) -> Array:
    """Apply centered geodesic SLERP smoothing to [T,...,3,3] tracks.

    Each interior frame moves a fraction strength toward the SLERP midpoint of
    its neighbours. It stays on SO(3), preserves endpoints, and does not cross
    invalid frames.
    """
    x = np.asarray(rotations, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if x.ndim < 3 or x.shape[-2:] != (3, 3):
        raise StageIOError(f"rotations must be [T, ..., 3, 3], got {x.shape}")
    if mask.shape != x.shape[:-2]:
        raise StageIOError(f"valid must be {x.shape[:-2]}, got {mask.shape}")
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise StageIOError(f"strength must be in [0, 1], got {strength}")
    if passes < 0:
        raise StageIOError(f"passes must be >= 0, got {passes}")

    tracks = int(np.prod(x.shape[1:-2])) if x.ndim > 3 else 1
    frames = x.shape[0]
    out = x.reshape(frames, tracks, 3, 3).copy()
    track_valid = mask.reshape(frames, tracks) & np.isfinite(out).all(axis=(2, 3))
    if passes == 0 or strength == 0.0:
        return out.reshape(x.shape)

    for _ in range(passes):
        previous = out.copy()
        for track in range(tracks):
            for frame in range(1, frames - 1):
                if not np.all(track_valid[frame - 1 : frame + 2, track]):
                    continue
                neighbor_mid = rotation_slerp(
                    previous[frame - 1, track], previous[frame + 1, track], 0.5
                )
                out[frame, track] = rotation_slerp(
                    previous[frame, track], neighbor_mid, strength
                )
    return out.reshape(x.shape)

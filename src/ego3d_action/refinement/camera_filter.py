"""Phase 6.1: 3-frame binomial filter for camera translation.

The reference system filters only the *camera translation* with a 3-tap
binomial kernel:

``t'_i = 0.25 * t_{i-1} + 0.5 * t_i + 0.25 * t_{i+1}``

Wide-window smoothing of the hand trajectory is explicitly avoided (it makes
Action-MPJPE worse), so this module never touches the hands.
"""

from __future__ import annotations

import logging

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray

KERNEL = np.array([0.25, 0.5, 0.25], dtype=np.float64)


def binomial_3_filter(translation: Array, *, passes: int = 1) -> Array:
    """Apply the 3-tap binomial kernel along the time axis.

    Boundaries use edge replication (``t_{-1} = t_0``, ``t_T = t_{T-1}``),
    which keeps the filter a proper weighted average and is exact for constant
    signals.

    Args:
        translation: ``[T, 3]`` (or ``[T, ...]``) translation sequence.
        passes: number of times the kernel is applied.

    Raises:
        StageIOError: on an empty sequence or a non-positive ``passes``.
    """
    values = np.asarray(translation, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise StageIOError(f"translation must be [T, ...] with T > 0, got {values.shape}")
    if passes <= 0:
        raise StageIOError(f"passes must be positive, got {passes}")

    out = values
    for _ in range(passes):
        padded = np.pad(out, [(1, 1)] + [(0, 0)] * (out.ndim - 1), mode="edge")
        out = (
            KERNEL[0] * padded[:-2]
            + KERNEL[1] * padded[1:-1]
            + KERNEL[2] * padded[2:]
        )
    return out


def filter_camera_translation(
    translation: Array,
    *,
    valid: Array | None = None,
    passes: int = 1,
) -> Array:
    """Filter a camera translation sequence, leaving invalid frames untouched.

    Invalid frames are interpolated linearly before filtering and restored
    afterwards, so a gap in the camera track cannot bleed a spurious value into
    its neighbours.
    """
    values = np.asarray(translation, dtype=np.float64)
    if valid is None:
        return binomial_3_filter(values, passes=passes)

    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if mask.shape[0] != values.shape[0]:
        raise StageIOError(f"valid length {mask.shape[0]} != {values.shape[0]}")
    if not mask.any():
        logger.warning("camera translation filter skipped: no valid frames")
        return values.copy()

    filled = values.copy()
    index = np.arange(values.shape[0], dtype=np.float64)
    valid_idx = index[mask]
    for axis in range(values.shape[1]):
        filled[:, axis] = np.interp(index, valid_idx, values[mask, axis])

    filtered = binomial_3_filter(filled, passes=passes)
    out = values.copy()
    out[mask] = filtered[mask]
    return out

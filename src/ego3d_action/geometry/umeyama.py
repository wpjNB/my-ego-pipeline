"""Weighted Umeyama similarity estimation."""

from __future__ import annotations

import logging

import numpy as np

from ..errors import InsufficientDataError

logger = logging.getLogger(__name__)

Array = np.ndarray


def weighted_umeyama(
    src: Array,
    dst: Array,
    weights: Array | None = None,
    *,
    with_scale: bool = True,
) -> tuple[float, Array, Array]:
    """Least-squares similarity ``dst ~= s * R @ src + t``.

    Args:
        src: ``[N, 3]`` source points.
        dst: ``[N, 3]`` destination points, one-to-one with ``src``.
        weights: optional ``[N]`` non-negative weights.
        with_scale: if ``False`` the scale is forced to ``1.0``.

    Returns:
        ``(scale, rotation, translation)``.

    Raises:
        InsufficientDataError: if fewer than three points are supplied, the
            weights are invalid, or the source cloud is degenerate.
    """
    s = np.asarray(src, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    if s.ndim != 2 or s.shape[-1] != 3:
        raise InsufficientDataError(f"src must have shape [N, 3], got {s.shape}")
    if d.shape != s.shape:
        raise InsufficientDataError(f"dst shape {d.shape} does not match src shape {s.shape}")
    if s.shape[0] < 3:
        raise InsufficientDataError(f"weighted Umeyama needs >= 3 points, got {s.shape[0]}")

    if weights is None:
        w = np.ones(s.shape[0], dtype=np.float64)
        n_eff = float(s.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != s.shape[0]:
            raise InsufficientDataError(f"weights length {w.shape[0]} != {s.shape[0]}")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise InsufficientDataError("weights must be finite and non-negative")
        n_eff = float(np.count_nonzero(w > 0.0))

    if n_eff < 3:
        raise InsufficientDataError(f"weighted Umeyama needs >= 3 non-zero weights, got {n_eff}")

    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise InsufficientDataError("weights sum to zero")
    w_n = w / w_sum

    mu_src = np.einsum("n,ni->i", w_n, s)
    mu_dst = np.einsum("n,ni->i", w_n, d)
    src_c = s - mu_src
    dst_c = d - mu_dst

    cov = np.einsum("n,ni,nj->ij", w_n, dst_c, src_c)
    var_src = float(np.einsum("n,ni,ni->", w_n, src_c, src_c))
    if var_src <= 1e-12:
        raise InsufficientDataError(f"source point cloud is degenerate (variance={var_src:.3e})")

    u, sv, vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign[2] = -1.0
    rotation = u @ np.diag(sign) @ vt
    if np.linalg.det(rotation) < 0.0:  # pragma: no cover - defensive
        raise InsufficientDataError("Umeyama produced an improper rotation")

    scale = 1.0
    if with_scale:
        scale = float(np.sum(sv * sign) / var_src)
        if not np.isfinite(scale) or scale <= 0.0:
            raise InsufficientDataError(f"estimated scale is not positive: {scale}")

    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def umeyama_residuals(
    src: Array,
    dst: Array,
    scale: float,
    rotation: Array,
    translation: Array,
) -> Array:
    """Per-point Euclidean residuals of a similarity fit."""
    s = np.asarray(src, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    pred = scale * np.einsum("ij,nj->ni", rotation, s) + translation
    return np.linalg.norm(pred - d, axis=-1)


"""Phase 6.3: ray-constrained wrist-depth optimisation.

Monocular RGB constrains the x/y projection of the wrist well, but its depth is
the least reliable quantity. Instead of smoothing the wrist, the reference
system moves it **along the original camera ray**, so the 2D projection is
preserved exactly:

::

    camera
       \\
        * original wrist      <- same ray
         \\
          * optimised wrist   <- identical 2D projection

The optimised variables are the per-frame wrist depths ``d_1 .. d_T`` and the
objective is

``L(d) = L_stay_close + lambda * L_acc``   with ``lambda = 0.2``

where the stay-close weight is derived from the detection confidence
(``confidence / median -> clamp[0.5, 1.5] -> **8``): high-confidence frames stay
near the HaWoR depth, low-confidence frames are free to move.

Because both terms are quadratic in ``d``, the optimum is obtained by solving a
single sparse linear system instead of running an iterative optimiser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..errors import StageIOError
from ..geometry.transforms import project_points

logger = logging.getLogger(__name__)

Array = np.ndarray

WRIST_JOINT = 0
DEFAULT_LAMBDA = 0.2
CONFIDENCE_CLAMP = (0.5, 1.5)
CONFIDENCE_POWER = 8


@dataclass(frozen=True)
class WristDepthResult:
    """Result of :func:`optimize_wrist_depth`."""

    joints_camera: Array  # [T, 2, 21, 3]
    depth_before: Array  # [T, 2]
    depth_after: Array  # [T, 2]
    weights: Array  # [T, 2]
    segments: int


def confidence_weights(confidence: Array, valid: Array) -> Array:
    """``confidence / median -> clamp[0.5, 1.5] -> **8`` over valid frames."""
    conf = np.asarray(confidence, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if conf.shape != mask.shape:
        raise StageIOError(f"confidence {conf.shape} and valid {mask.shape} disagree")
    weights = np.ones_like(conf)
    finite = mask & np.isfinite(conf) & (conf > 0.0)
    if not finite.any():
        logger.warning("no finite confidences; wrist-depth weights default to 1.0")
        return weights
    median = float(np.median(conf[finite]))
    if median <= 0.0:
        logger.warning("median confidence is %.3f; weights default to 1.0", median)
        return weights
    scaled = np.clip(conf / median, CONFIDENCE_CLAMP[0], CONFIDENCE_CLAMP[1])
    return scaled**CONFIDENCE_POWER


def _ray_matrix(intrinsics: Array, wrist: Array) -> tuple[Array, Array]:
    """Return ``(rays [N, 3], depths [N])`` with ``wrist = depth * ray``."""
    k = np.asarray(intrinsics, dtype=np.float64)
    points = np.asarray(wrist, dtype=np.float64)
    if k.shape != (points.shape[0], 3, 3):
        raise StageIOError(f"intrinsics must be [{points.shape[0]}, 3, 3], got {k.shape}")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise StageIOError(f"wrist must be [N, 3], got {points.shape}")
    uv = project_points(k, points)
    if not np.all(np.isfinite(uv)):
        raise StageIOError("wrist has non-finite projections; cannot build camera rays")
    k_inv = np.linalg.inv(k)
    homog = np.concatenate([uv, np.ones_like(uv[..., :1])], axis=-1)
    scaled_rays = np.einsum("nij,nj->ni", k_inv, homog)
    depths = points[:, 2]
    if np.any(np.abs(depths) < 1e-9):
        raise StageIOError("wrist depth is (near) zero; the ray parameterisation is undefined")
    rays = scaled_rays / scaled_rays[:, 2:3]
    return rays, depths


def _acceleration_operator(rays: Array) -> sparse.csr_matrix:
    """Second-difference operator acting on ``p = s * ray``.

    Rows are stacked per axis, so ``A`` has shape ``[3T, T]`` and the
    acceleration energy is ``||A s||^2``.

    Boundaries replicate the edge point (``p_{-1} = p_0``), giving the one-sided
    second difference ``p_1 - p_0`` at the first and last sample.
    """
    num = rays.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for axis in range(3):
        for t in range(num):
            row = axis * num + t
            if num == 1:
                continue
            if t == 0:
                rows.extend([row, row])
                cols.extend([1, 0])
                data.extend([rays[1, axis], -rays[0, axis]])
            elif t == num - 1:
                rows.extend([row, row])
                cols.extend([num - 1, num - 2])
                data.extend([rays[num - 1, axis], -rays[num - 2, axis]])
            else:
                rows.extend([row, row, row])
                cols.extend([t + 1, t, t - 1])
                data.extend([rays[t + 1, axis], -2.0 * rays[t, axis], rays[t - 1, axis]])
    return sparse.csr_matrix((data, (rows, cols)), shape=(3 * num, num))


def _solve_segment(
    rays: Array,
    depths: Array,
    weights: Array,
    lam: float,
) -> Array:
    """Solve one contiguous valid segment for the optimal depths."""
    num = rays.shape[0]
    if num == 1:
        return depths.copy()
    accel = _acceleration_operator(rays)
    system = sparse.diags(weights, format="csr") + lam * (accel.T @ accel)
    rhs = weights * depths
    try:
        solution = spsolve(system.tocsc(), rhs)
    except Exception as exc:  # pragma: no cover - scipy raises several types
        raise StageIOError(f"wrist-depth linear solve failed: {exc}") from exc
    if not np.all(np.isfinite(solution)):
        raise StageIOError("wrist-depth linear solve returned non-finite values")
    return solution


def optimize_wrist_depth(
    joints_camera: Array,
    intrinsics: Array,
    confidence: Array,
    valid: Array,
    *,
    lam: float = DEFAULT_LAMBDA,
    max_depth_change: float | None = None,
) -> WristDepthResult:
    """Optimise wrist depth along the original camera ray.

    Args:
        joints_camera: ``[T, 2, 21, 3]`` camera-space joints.
        intrinsics: ``[T, 3, 3]`` per-frame intrinsics.
        confidence: ``[T, 2]`` detection confidences.
        valid: ``[T, 2]`` validity mask; invalid frames are never modified.
        lam: acceleration weight (spec: ``0.2``).
        max_depth_change: optional hard bound (metres) on ``|d - d0|``.

    Returns:
        :class:`WristDepthResult`; the whole hand is translated by the wrist
        delta so intra-hand geometry is preserved exactly.

    Raises:
        StageIOError: on shape mismatches or a negative ``lam``.
    """
    joints = np.asarray(joints_camera, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise StageIOError(f"joints_camera must be [T, 2, 21, 3], got {joints.shape}")
    total = joints.shape[0]
    if k.shape != (total, 3, 3):
        raise StageIOError(f"intrinsics must be [{total}, 3, 3], got {k.shape}")
    if mask.shape != (total, 2):
        raise StageIOError(f"valid must be [{total}, 2], got {mask.shape}")
    if conf.shape != (total, 2):
        raise StageIOError(f"confidence must be [{total}, 2], got {conf.shape}")
    if lam < 0.0:
        raise StageIOError(f"lam must be >= 0, got {lam}")

    weights = confidence_weights(conf, mask)
    out = joints.copy()
    depth_before = np.full((total, 2), np.nan, dtype=np.float64)
    depth_after = np.full((total, 2), np.nan, dtype=np.float64)
    segments = 0

    for hand in range(2):
        valid_frames = mask[:, hand] & np.isfinite(joints[:, hand]).all(axis=(1, 2))
        if not valid_frames.any():
            logger.info("wrist-depth: hand %d has no valid frames, skipped", hand)
            continue

        # Split into contiguous valid runs so a gap never links two segments.
        runs = _contiguous_runs(valid_frames)
        for start, end in runs:
            frames = np.arange(start, end, dtype=np.int64)
            if frames.size < 2:
                continue
            wrist = joints[frames, hand, WRIST_JOINT, :]
            ray, depths = _ray_matrix(k[frames], wrist)
            w = weights[frames, hand]
            if not np.all(np.isfinite(w)) or w.sum() <= 0.0:
                logger.warning(
                    "wrist-depth: hand %d frames %d-%d have degenerate weights, segment skipped",
                    hand,
                    start,
                    end - 1,
                )
                continue
            solved = _solve_segment(ray, depths, w, lam)
            if max_depth_change is not None:
                solved = depths + np.clip(solved - depths, -max_depth_change, max_depth_change)
            delta = ray * (solved - depths)[:, None]
            out[frames, hand] = joints[frames, hand] + delta[:, None, :]
            depth_before[frames, hand] = depths
            depth_after[frames, hand] = solved
            segments += 1

    changed = np.isfinite(depth_after)
    if changed.any():
        diff = np.abs(depth_after[changed] - depth_before[changed])
        logger.info(
            "wrist-depth optimisation (lambda=%.2f): %d segments, mean |dd|=%.2f mm, max=%.2f mm",
            lam,
            segments,
            float(1000.0 * np.mean(diff)),
            float(1000.0 * np.max(diff)),
        )
    else:
        logger.warning("wrist-depth optimisation produced no updated segments")

    return WristDepthResult(
        joints_camera=out,
        depth_before=depth_before,
        depth_after=depth_after,
        weights=weights,
        segments=segments,
    )


def _contiguous_runs(mask: Array) -> list[tuple[int, int]]:
    """Return half-open ``(start, end)`` runs of ``True`` in a 1-D mask."""
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    if flags.size == 0:
        return []
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2], strict=False)]

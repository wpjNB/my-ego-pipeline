"""Sim(3) similarity transform: ``X_dst = s * R @ X_src + t``.

This is the module used by Phase 4 to align two overlapping VGGT-Omega windows.
The estimate is always **depth-derived**: correspondences come from
back-projected depth maps, never from camera centres.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..errors import InsufficientDataError
from .umeyama import umeyama_residuals, weighted_umeyama

logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class Sim3:
    """A 7-DoF similarity transform."""

    scale: float
    rotation: Array
    translation: Array

    def __post_init__(self) -> None:
        rot = np.asarray(self.rotation, dtype=np.float64)
        tr = np.asarray(self.translation, dtype=np.float64)
        if rot.shape != (3, 3):
            raise InsufficientDataError(f"Sim3.rotation must be [3, 3], got {rot.shape}")
        if tr.shape != (3,):
            raise InsufficientDataError(f"Sim3.translation must be [3], got {tr.shape}")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise InsufficientDataError(f"Sim3.scale must be finite and > 0, got {self.scale}")
        if not np.all(np.isfinite(rot)) or not np.all(np.isfinite(tr)):
            raise InsufficientDataError("Sim3 contains non-finite values")
        object.__setattr__(self, "rotation", rot)
        object.__setattr__(self, "translation", tr)
        object.__setattr__(self, "scale", float(self.scale))

    # ------------------------------------------------------------------ maths
    def transform(self, points: Array) -> Array:
        """Apply the similarity to ``[..., 3]`` points."""
        p = np.asarray(points, dtype=np.float64)
        if p.shape[-1] != 3:
            raise InsufficientDataError(f"points must have trailing dim 3, got {p.shape}")
        return self.scale * np.einsum("ij,...j->...i", self.rotation, p) + self.translation

    def inverse(self) -> "Sim3":
        r_inv = self.rotation.T
        s_inv = 1.0 / self.scale
        t_inv = -s_inv * (r_inv @ self.translation)
        return Sim3(scale=s_inv, rotation=r_inv, translation=t_inv)

    def compose(self, other: "Sim3") -> "Sim3":
        """Return ``self ∘ other`` (i.e. apply ``other`` first, then ``self``)."""
        scale = self.scale * other.scale
        rotation = self.rotation @ other.rotation
        translation = self.scale * (self.rotation @ other.translation) + self.translation
        return Sim3(scale=scale, rotation=rotation, translation=translation)

    def __matmul__(self, other: "Sim3") -> "Sim3":
        if not isinstance(other, Sim3):
            raise TypeError(f"Sim3 @ {type(other).__name__} is not supported")
        return self.compose(other)

    def transform_poses(self, rotation_c2w: Array, translation_c2w: Array) -> tuple[Array, Array]:
        """Map a camera trajectory into the destination frame.

        A c2w pose is handled exactly like a point transform applied to the
        rotation part: ``R' = R_sim @ R``, ``t' = s * R_sim @ t + t_sim``.
        """
        rot = np.asarray(rotation_c2w, dtype=np.float64)
        tr = np.asarray(translation_c2w, dtype=np.float64)
        if rot.shape[-2:] != (3, 3):
            raise InsufficientDataError(f"rotation_c2w must be [..., 3, 3], got {rot.shape}")
        if tr.shape[-1] != 3:
            raise InsufficientDataError(f"translation_c2w must be [..., 3], got {tr.shape}")
        new_rot = np.einsum("ij,...jk->...ik", self.rotation, rot)
        new_tr = self.scale * np.einsum("ij,...j->...i", self.rotation, tr) + self.translation
        return new_rot, new_tr

    @property
    def matrix(self) -> Array:
        """The 4x4 homogeneous matrix (last row ``[0, 0, 0, 1]``)."""
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation
        return m

    @classmethod
    def from_matrix(cls, matrix: Array) -> "Sim3":
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise InsufficientDataError(f"matrix must be [4, 4], got {m.shape}")
        if not np.allclose(m[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
            raise InsufficientDataError("last row of a Sim(3) matrix must be [0, 0, 0, 1]")
        linear = m[:3, :3]
        scale = float(np.cbrt(np.linalg.det(linear)))
        if not np.isfinite(scale) or scale <= 0.0:
            raise InsufficientDataError(f"non-positive scale in matrix: {scale}")
        return cls(scale=scale, rotation=linear / scale, translation=m[:3, 3])

    @classmethod
    def identity(cls) -> "Sim3":
        return cls(scale=1.0, rotation=np.eye(3), translation=np.zeros(3))

    def as_dict(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }


def estimate_sim3(
    src_points: Array,
    dst_points: Array,
    weights: Array | None = None,
    *,
    with_scale: bool = True,
) -> Sim3:
    """Least-squares Sim(3) taking ``src_points`` onto ``dst_points``."""
    scale, rotation, translation = weighted_umeyama(
        src_points, dst_points, weights, with_scale=with_scale
    )
    return Sim3(scale=scale, rotation=rotation, translation=translation)


@dataclass(frozen=True)
class RobustSim3Result:
    """Result of :func:`estimate_sim3_robust`."""

    sim3: Sim3
    inlier_mask: Array
    inlier_ratio: float
    inlier_rmse: float
    threshold: float


def estimate_sim3_robust(
    src_points: Array,
    dst_points: Array,
    weights: Array | None = None,
    *,
    with_scale: bool = True,
    inlier_threshold: float | None = None,
    ransac_iterations: int = 128,
    sample_size: int = 3,
    min_inliers: int = 12,
    refine_iterations: int = 2,
    random_state: int | None = 0,
) -> RobustSim3Result:
    """RANSAC + re-fit Sim(3) with robust correspondence filtering.

    Depth-derived correspondences always contain outliers (specular surfaces,
    depth edges, occlusions). The pipeline therefore solves Sim(3) with a
    minimal-set RANSAC followed by weighted least-squares re-fits on the inlier
    set.

    Args:
        src_points: ``[N, 3]`` correspondences in the source window frame.
        dst_points: ``[N, 3]`` matching points in the destination window frame.
        weights: optional ``[N]`` confidence weights.
        with_scale: solve for the metric scale (should stay ``True``).
        inlier_threshold: absolute residual threshold in metres. When ``None``
            it is derived from the residuals of an initial least-squares fit.
        ransac_iterations: number of minimal-set hypotheses.
        sample_size: points per hypothesis (``>= 3``).
        min_inliers: below this the fit is rejected.
        refine_iterations: rounds of inlier re-fit.
        random_state: seed for reproducibility.

    Raises:
        InsufficientDataError: if there are too few points or no acceptable
            hypothesis exists.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[-1] != 3:
        raise InsufficientDataError(f"src_points must be [N, 3], got {src.shape}")
    if dst.shape != src.shape:
        raise InsufficientDataError(f"dst shape {dst.shape} != src shape {src.shape}")
    n = src.shape[0]
    if n < max(3, min_inliers):
        raise InsufficientDataError(f"robust Sim(3) needs >= {max(3, min_inliers)} points, got {n}")
    if sample_size < 3:
        raise InsufficientDataError(f"sample_size must be >= 3, got {sample_size}")
    sample_size = min(sample_size, n)

    w = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if w is not None and w.shape[0] != n:
        raise InsufficientDataError(f"weights length {w.shape[0]} != {n}")

    if inlier_threshold is None:
        try:
            seed = estimate_sim3(src, dst, w, with_scale=with_scale)
            residuals = umeyama_residuals(src, dst, seed.scale, seed.rotation, seed.translation)
            threshold = max(1e-3, 3.0 * float(np.median(residuals)))
        except InsufficientDataError as exc:  # degenerate seed fit - fall back to a fixed scale
            logger.warning("Sim(3) seed fit failed (%s); falling back to a 1 cm threshold", exc)
            threshold = 0.01
    else:
        threshold = float(inlier_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise InsufficientDataError(f"inlier_threshold must be positive, got {threshold}")

    rng = np.random.default_rng(random_state)
    best_mask: Array | None = None
    best_score = -np.inf
    best_rmse = np.inf

    for _ in range(max(1, ransac_iterations)):
        idx = rng.choice(n, size=sample_size, replace=False)
        try:
            hyp = estimate_sim3(src[idx], dst[idx], None if w is None else w[idx], with_scale=with_scale)
        except InsufficientDataError:
            continue
        residuals = umeyama_residuals(src, dst, hyp.scale, hyp.rotation, hyp.translation)
        mask = residuals <= threshold
        if w is not None:
            score = float(np.sum(w[mask]))
        else:
            score = float(np.count_nonzero(mask))
        if score > best_score:
            best_score = score
            best_mask = mask
            best_rmse = float(np.sqrt(np.mean(residuals[mask] ** 2))) if mask.any() else np.inf

    if best_mask is None:
        raise InsufficientDataError("RANSAC found no valid Sim(3) hypothesis")

    mask = best_mask
    fit: Sim3 | None = None
    for _ in range(max(1, refine_iterations)):
        if np.count_nonzero(mask) < max(3, min_inliers):
            break
        fit = estimate_sim3(src[mask], dst[mask], None if w is None else w[mask], with_scale=with_scale)
        residuals = umeyama_residuals(src, dst, fit.scale, fit.rotation, fit.translation)
        new_mask = residuals <= threshold
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    n_inliers = int(np.count_nonzero(mask))
    if fit is None or n_inliers < min_inliers:
        raise InsufficientDataError(
            f"robust Sim(3) rejected: only {n_inliers} inliers (need >= {min_inliers})"
        )

    residuals = umeyama_residuals(src, dst, fit.scale, fit.rotation, fit.translation)
    inlier_rmse = float(np.sqrt(np.mean(residuals[mask] ** 2)))
    inlier_ratio = n_inliers / n
    logger.info(
        "Sim(3) robust fit: inliers=%d/%d (%.1f%%), rmse=%.4f m, threshold=%.4f m, scale=%.4f",
        n_inliers,
        n,
        100.0 * inlier_ratio,
        inlier_rmse,
        threshold,
        fit.scale,
    )
    return RobustSim3Result(
        sim3=fit,
        inlier_mask=mask,
        inlier_ratio=inlier_ratio,
        inlier_rmse=inlier_rmse,
        threshold=threshold,
    )

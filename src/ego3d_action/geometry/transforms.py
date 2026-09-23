"""Rigid-transform and camera-projection helpers.

Conventions used everywhere in this project:

* Camera extrinsics are **c2w**: ``R_c2w`` and ``t_c2w`` map a point expressed
  in the camera frame to the world frame: ``p_w = R_c2w @ p_c + t_c2w``.
* Camera intrinsics ``K`` are 3x3 and act on homogeneous pixel coordinates with
  the OpenCV convention (x right, y down, z forward, metres).
* Joint arrays are ``[T, H, 21, 3]`` with ``H = 2`` (left, right).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation

from ..errors import InsufficientDataError

logger = logging.getLogger(__name__)

Array = np.ndarray


def as_rotation_matrix(rot: Array, *, name: str = "rotation") -> Array:
    """Validate and return a ``[..., 3, 3]`` rotation array.

    Raises:
        InsufficientDataError: if the shape is wrong or a matrix is not a
            proper rotation (determinant far from ``+1``).
    """
    arr = np.asarray(rot, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[-2:] != (3, 3):
        raise InsufficientDataError(f"{name} must have shape [..., 3, 3], got {arr.shape}")
    det = np.linalg.det(arr)
    if not np.allclose(det, 1.0, atol=1e-4):
        worst = float(np.max(np.abs(det - 1.0)))
        raise InsufficientDataError(f"{name} contains non-rotation matrices (max |det-1|={worst:.3e})")
    return arr


def apply_rigid(rotation: Array, translation: Array, points: Array) -> Array:
    """Apply ``p @ rotation.T + translation`` to ``[..., 3]`` points."""
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    return np.einsum("...ij,...j->...i", r, p) + t


def invert_rigid(rotation: Array, translation: Array) -> tuple[Array, Array]:
    """Return the inverse rigid transform ``(R.T, -R.T @ t)``."""
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    r_inv = np.swapaxes(r, -1, -2)
    t_inv = -np.einsum("...ij,...j->...i", r_inv, t)
    return r_inv, t_inv


def rotation_slerp(rot_a: Array, rot_b: Array, alpha: Array | float) -> Array:
    """Shortest-path SLERP between two rotation arrays.

    ``alpha`` is broadcast against ``rot_a.shape[:-2]``; ``alpha=0`` returns
    ``rot_a`` and ``alpha=1`` returns ``rot_b``. Near-identical rotations fall
    back to (renormalised) linear interpolation, which is numerically stable and
    identical to SLERP to second order.
    """
    ra = np.asarray(rot_a, dtype=np.float64)
    rb = np.asarray(rot_b, dtype=np.float64)
    if ra.shape != rb.shape:
        raise InsufficientDataError(f"rotation shapes differ: {ra.shape} vs {rb.shape}")
    if ra.shape[-2:] != (3, 3):
        raise InsufficientDataError(f"expected [..., 3, 3] rotations, got {ra.shape}")

    a = np.asarray(alpha, dtype=np.float64)
    lead = ra.shape[:-2]
    if not np.broadcast_shapes(a.shape, lead) == lead and a.shape != ():
        raise InsufficientDataError(f"alpha shape {a.shape} is not broadcastable to {lead}")

    qa = Rotation.from_matrix(ra.reshape(-1, 3, 3)).as_quat()
    qb = Rotation.from_matrix(rb.reshape(-1, 3, 3)).as_quat()

    dot = np.sum(qa * qb, axis=-1, keepdims=True)
    qb = np.where(dot < 0.0, -qb, qb)
    dot = np.abs(dot)

    a_flat = np.broadcast_to(a, lead).reshape(-1, 1)
    q = _quat_slerp(qa, qb, dot, a_flat)
    out = Rotation.from_quat(q).as_matrix().reshape(ra.shape)
    return out


def _quat_slerp(qa: Array, qb: Array, dot: Array, alpha: Array) -> Array:
    """Vectorised quaternion SLERP with a linear fallback for close rotations."""
    eps = 1e-6
    qa = qa.copy()
    qb = qb.copy()
    # Force the first component positive to avoid a sign flip around w == 0.
    flip = qa[:, :1] < 0.0
    qa = np.where(flip, -qa, qa)
    qb = np.where(flip, -qb, qb)
    dot = np.sum(qa * qb, axis=-1, keepdims=True)
    qb = np.where(dot < 0.0, -qb, qb)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    near = sin_theta_0 < eps
    # SLERP branch
    theta = theta_0 * alpha
    s0 = np.sin(theta_0 - theta) / np.where(near, 1.0, sin_theta_0)
    s1 = np.sin(theta) / np.where(near, 1.0, sin_theta_0)
    slerp = s0 * qa + s1 * qb
    # Linear branch (theta_0 ~ 0)
    linear = (1.0 - alpha) * qa + alpha * qb
    out = np.where(near, linear, slerp)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise InsufficientDataError("degenerate quaternion interpolation (zero-norm result)")
    return out / norm


def rotation_angle(rotation: Array) -> Array:
    """Geodesic angle (radians) of a rotation array ``[..., 3, 3]``."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(r, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def project_points(intrinsics: Array, points_cam: Array, *, eps: float = 1e-9) -> Array:
    """Project ``[..., 3]`` camera-frame points to ``[..., 2]`` pixels.

    Points behind / on the camera plane produce ``inf`` rather than silently
    clamping, so callers must mask them explicitly.
    """
    k = np.asarray(intrinsics, dtype=np.float64)
    p = np.asarray(points_cam, dtype=np.float64)
    z = p[..., 2:3]
    safe_z = np.where(np.abs(z) < eps, np.nan, z)
    xy = np.einsum("...ij,...j->...i", k, p) / safe_z
    return xy[..., :2]


def unproject_pixels(intrinsics: Array, pixels: Array, depth: Array) -> Array:
    """Back-project ``[..., 2]`` pixels with ``[...]`` depth into ``[..., 3]``.

    ``depth`` is the z-coordinate (metres), matching ``project_points``.
    """
    k = np.asarray(intrinsics, dtype=np.float64)
    uv = np.asarray(pixels, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    ones = np.ones(uv.shape[:-1] + (1,), dtype=np.float64)
    homog = np.concatenate([uv, ones], axis=-1)
    k_inv = np.linalg.inv(k)
    rays = np.einsum("...ij,...j->...i", k_inv, homog)
    return rays * d[..., None]


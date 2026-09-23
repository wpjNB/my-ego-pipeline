"""Camera-pose helpers (Phase 4/5).

After stitching, all poses live in ``World-0``. The reference system defines
``World-0`` as *the camera of the first frame*, so :func:`normalize_to_first_camera`
turns the stitched trajectory into the canonical world frame used by the final
``trajectory.npz``.
"""

from __future__ import annotations

import logging

import numpy as np

from ..errors import InsufficientDataError, StageIOError
from ..geometry.transforms import apply_rigid, invert_rigid, rotation_slerp

logger = logging.getLogger(__name__)

Array = np.ndarray


def pose_matrices(rotation_c2w: Array, translation_c2w: Array) -> Array:
    """Pack a pose sequence into ``[T, 4, 4]`` homogeneous matrices."""
    rot = np.asarray(rotation_c2w, dtype=np.float64)
    tr = np.asarray(translation_c2w, dtype=np.float64)
    if rot.shape[:-2] != tr.shape[:-1]:
        raise StageIOError(f"rotation shape {rot.shape} and translation shape {tr.shape} disagree")
    if rot.shape[-2:] != (3, 3) or tr.shape[-1] != 3:
        raise StageIOError(f"expected [..., 3, 3] and [..., 3], got {rot.shape}, {tr.shape}")
    out = np.zeros(rot.shape[:-2] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = rot
    out[..., :3, 3] = tr
    out[..., 3, 3] = 1.0
    return out


def matrices_to_pose(matrices: Array) -> tuple[Array, Array]:
    """Unpack ``[..., 4, 4]`` homogeneous matrices into ``(R, t)``."""
    m = np.asarray(matrices, dtype=np.float64)
    if m.shape[-2:] != (4, 4):
        raise StageIOError(f"matrices must end with [4, 4], got {m.shape}")
    return m[..., :3, :3], m[..., :3, 3]


def normalize_to_first_camera(
    rotation_c2w: Array,
    translation_c2w: Array,
    valid: Array | None = None,
) -> tuple[Array, Array]:
    """Re-express a pose sequence with the first valid camera at the origin.

    Args:
        rotation_c2w: ``[T, 3, 3]``.
        translation_c2w: ``[T, 3]``.
        valid: optional ``[T]`` mask; the first ``True`` frame becomes the
            world origin.

    Raises:
        InsufficientDataError: when no frame is valid.
    """
    rot = np.asarray(rotation_c2w, dtype=np.float64)
    tr = np.asarray(translation_c2w, dtype=np.float64)
    mask = np.ones(rot.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    if mask.shape[0] != rot.shape[0]:
        raise StageIOError(f"valid mask length {mask.shape[0]} != {rot.shape[0]}")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise InsufficientDataError("cannot normalize a camera trajectory with no valid frame")
    anchor = int(indices[0])

    r_inv, t_inv = invert_rigid(rot[anchor], tr[anchor])
    new_rot = np.einsum("ij,tjk->tik", r_inv, rot)
    new_tr = np.einsum("ij,tj->ti", r_inv, tr) + t_inv
    logger.info("camera world frame anchored at frame %d", anchor)
    return new_rot, new_tr


def world_frame_alignment(
    rotation_c2w: Array,
    translation_c2w: Array,
    valid: Array | None = None,
) -> tuple[Array, Array]:
    """The ``(R_g, t_g)`` that re-anchors a world frame onto the first valid camera.

    Applying ``p' = R_g (p - t_g)``, ``R' = R_g R`` and ``t' = R_g (t - t_g)``
    to *everything* (camera poses and world points alike) is a pure gauge
    change: it leaves every camera-relative quantity - and therefore
    Action-MPJPE - unchanged, while restoring the documented "``World-0`` is the
    first frame's camera" invariant after the camera translation filter has
    moved frame 0.

    Raises:
        InsufficientDataError: when no frame is valid.
    """
    rot = np.asarray(rotation_c2w, dtype=np.float64)
    tr = np.asarray(translation_c2w, dtype=np.float64)
    mask = (
        np.ones(rot.shape[0], dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool).reshape(-1)
    )
    if mask.shape[0] != rot.shape[0]:
        raise StageIOError(f"valid mask length {mask.shape[0]} != {rot.shape[0]}")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise InsufficientDataError("cannot anchor a trajectory with no valid frame")
    anchor = int(indices[0])
    return rot[anchor].T, tr[anchor]


def camera_to_world(points_camera: Array, rotation_c2w: Array, translation_c2w: Array) -> Array:
    """Batch ``p_w = R @ p_c + t`` for ``[T, 3, 3]`` / ``[T, 3]``."""
    return apply_rigid(rotation_c2w, translation_c2w, np.asarray(points_camera, dtype=np.float64))


def interpolate_pose(
    rotation_a: Array,
    translation_a: Array,
    rotation_b: Array,
    translation_b: Array,
    alpha: Array | float,
) -> tuple[Array, Array]:
    """Linear translation / SLERP rotation interpolation between two poses."""
    a = np.asarray(alpha, dtype=np.float64)
    trans = (1.0 - a)[..., None] * np.asarray(translation_a, dtype=np.float64) + a[
        ..., None
    ] * np.asarray(translation_b, dtype=np.float64)
    rot = rotation_slerp(np.asarray(rotation_a, dtype=np.float64), np.asarray(rotation_b, dtype=np.float64), a)
    return rot, trans

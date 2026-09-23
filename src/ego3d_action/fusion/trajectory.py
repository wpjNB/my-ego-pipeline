"""Phase 5: hand + camera -> world frame.

``p_w = R_c2w @ p_c + t_c2w`` with the first frame's camera defining the world
frame. The output is the project's main artefact and is always written through
:func:`ego3d_action.io.serialization.save_trajectory` so the section-7 contract
is validated on the way out.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray


def camera_joints_to_world(
    joints_camera: Array,
    camera_rotation_c2w: Array,
    camera_translation_c2w: Array,
    *,
    hand_valid: Array | None = None,
) -> Array:
    """Apply ``p_w = R p_c + t`` to ``[T, 2, 21, 3]`` joints.

    Invalid hand-frames are returned as ``NaN`` (they stay missing, they are
    never filled in with a fabricated pose).
    """
    joints = np.asarray(joints_camera, dtype=np.float64)
    rot = np.asarray(camera_rotation_c2w, dtype=np.float64)
    trans = np.asarray(camera_translation_c2w, dtype=np.float64)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise StageIOError(f"joints_camera must be [T, 2, 21, 3], got {joints.shape}")
    if rot.shape != (joints.shape[0], 3, 3):
        raise StageIOError(f"camera_rotation_c2w must be [{joints.shape[0]}, 3, 3], got {rot.shape}")
    if trans.shape != (joints.shape[0], 3):
        raise StageIOError(f"camera_translation_c2w must be [{joints.shape[0]}, 3], got {trans.shape}")

    world = np.einsum("tij,thnj->thni", rot, joints) + trans[:, None, None, :]

    if hand_valid is not None:
        valid = np.asarray(hand_valid, dtype=bool)
        if valid.shape != joints.shape[:2]:
            raise StageIOError(f"hand_valid must be {joints.shape[:2]}, got {valid.shape}")
        world = np.where(valid[:, :, None, None], world, np.nan)
    return world


def trajectory_metadata(
    *,
    fps: float,
    world_frame: int = 0,
    num_frames: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Section-7 metadata block for a ``trajectory.npz``."""
    meta: dict[str, Any] = {
        "fps": float(fps),
        "world_frame": int(world_frame),
        "hand_representation": "21_joints_metric_xyz",
        "camera_convention": "c2w",
        "units": "meter",
    }
    if num_frames is not None:
        meta["num_frames"] = int(num_frames)
    if extra:
        meta.update(dict(extra))
    return meta


def build_trajectory(
    *,
    joints_camera: Array,
    hand_valid: Array,
    hand_confidence: Array,
    camera_rotation_c2w: Array,
    camera_translation_c2w: Array,
    camera_intrinsics: Array,
    bbox: Array,
    track_id: Array,
    fps: float,
    mano_root_rot: Array | None = None,
    mano_hand_pose: Array | None = None,
    mano_betas: Array | None = None,
    postprocess_valid: Array | None = None,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Assemble every field of the final ``trajectory.npz`` contract.

    Raises:
        StageIOError: when array shapes do not agree with ``[T, 2, ...]``.
    """
    joints = np.asarray(joints_camera, dtype=np.float64)
    num_frames = joints.shape[0]
    valid = np.asarray(hand_valid, dtype=bool)
    confidence = np.asarray(hand_confidence, dtype=np.float64)
    if valid.shape != (num_frames, 2):
        raise StageIOError(f"hand_valid must be [{num_frames}, 2], got {valid.shape}")
    if confidence.shape != (num_frames, 2):
        raise StageIOError(f"hand_confidence must be [{num_frames}, 2], got {confidence.shape}")

    world = camera_joints_to_world(joints, camera_rotation_c2w, camera_translation_c2w, hand_valid=valid)

    arrays: dict[str, Array] = {
        "frames": np.arange(num_frames, dtype=np.int64),
        "timestamps": np.arange(num_frames, dtype=np.float64) / float(fps),
        "hand_xyz_world": world,
        "hand_xyz_camera": joints,
        "hand_valid": valid,
        "hand_confidence": confidence,
        "camera_R_c2w": np.asarray(camera_rotation_c2w, dtype=np.float64),
        "camera_t_c2w": np.asarray(camera_translation_c2w, dtype=np.float64),
        "camera_K": np.asarray(camera_intrinsics, dtype=np.float64),
        "bbox": np.asarray(bbox, dtype=np.float64),
        "track_id": np.asarray(track_id, dtype=np.int64),
        "mano_root_rot": np.asarray(
            mano_root_rot
            if mano_root_rot is not None
            else np.broadcast_to(np.eye(3), (num_frames, 2, 3, 3)),
            dtype=np.float64,
        ),
        "mano_hand_pose": np.asarray(
            mano_hand_pose
            if mano_hand_pose is not None
            else np.broadcast_to(np.eye(3), (num_frames, 2, 15, 3, 3)),
            dtype=np.float64,
        ),
        "mano_betas": np.asarray(
            mano_betas if mano_betas is not None else np.zeros((num_frames, 2, 10)),
            dtype=np.float64,
        ),
        "postprocess_valid": np.asarray(
            postprocess_valid if postprocess_valid is not None else valid,
            dtype=bool,
        ),
    }
    metadata = trajectory_metadata(fps=fps, world_frame=0, num_frames=num_frames)
    logger.info(
        "trajectory: T=%d, hands valid %.1f%% (left) / %.1f%% (right)",
        num_frames,
        100.0 * float(np.mean(valid[:, 0])),
        100.0 * float(np.mean(valid[:, 1])),
    )
    return arrays, metadata

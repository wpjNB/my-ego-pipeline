"""Convert a LeRobot HOT3D episode into the project's trajectory contract.

What the sample actually provides, and what that means for the contract:

======================================  ==========================================
dataset field                           trajectory field
======================================  ==========================================
``extrinsics_w2c`` (16)                 ``camera_R_c2w`` / ``camera_t_c2w``
                                        (inverted: ``p_cam = R p_world + t``)
``intrinsics`` (9)                      ``camera_K``
``left/right_transl_world`` (3)         ``hand_xyz_world[t, h, 0, :]`` (the wrist)
``left/right_orient_world`` (9)         ``mano_root_rot[t, h]``
``left/right_hand_pose`` (135)          ``mano_hand_pose[t, h]`` (15 rotations)
``observation.state`` (61 per hand)     ``mano_betas[t, h]``
``state_mask`` & ``*_kept``             ``hand_valid[t, h]``
======================================  ==========================================

The dataset carries a *root* (wrist) pose plus per-joint rotations, not 21 joint
positions. Turning those into fingertip positions needs the MANO mesh model
(``v_template``/``shapedirs``/``J_regressor``/``weights``), which is not present
on this machine and is licence-gated. Rather than invent joints, the conversion
writes the wrist exactly and leaves joints 1..20 as ``NaN``, recording
``hand_joints: "wrist_only"`` in the metadata. The evaluation understands that
and reduces to a wrist Action-MPJPE instead of silently averaging over
fabricated joints.

The layout of ``observation.state`` (45 axis-angle values, then 3, then 3, then
10 shared betas) was inferred from the sample itself; it is recorded in the
metadata as ``state_layout`` so a future reader can audit it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..datasets.lerobot import LeRobotDataset, LeRobotEpisode
from ..errors import StageIOError
from ..camera.camera_pose import world_frame_alignment
from ..geometry.transforms import invert_rigid
from ..hand.mano_model import (
    ManoModel,
    forward_kinematics,
    mirror_pose,
    validate_landmark_mapping,
)
from ..io.serialization import save_json, save_npz

logger = logging.getLogger(__name__)

Array = np.ndarray

STATE_PER_HAND = 61
STATE_LAYOUT = "45x axis-angle (15 joints), 3, 3, 10 shared betas"

#: Models whose landmark mapping has already been validated in this process
#: (the check is pose-independent but only needs to run once per model).
_MAPPING_VALIDATED: dict[int, bool] = {}


@dataclass(frozen=True)
class Hot3dEpisode:
    """The converted episode, in the project's trajectory contract."""

    arrays: dict[str, Array]
    metadata: dict[str, object]
    num_frames: int
    fps: float
    width: int
    height: int
    video_path: object
    task: str | None


def _stack_rotation_matrices(values: Array, *, count: int, name: str) -> Array:
    """``[T, 9 * count]`` -> ``[T, count, 3, 3]`` with a proper-rotation check."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9 * count:
        raise StageIOError(f"{name} must be [T, {9 * count}], got {arr.shape}")
    rotations = arr.reshape(arr.shape[0], count, 3, 3)
    determinant = np.linalg.det(rotations)
    if not np.allclose(determinant, 1.0, atol=1e-4):
        worst = float(np.max(np.abs(determinant - 1.0)))
        raise StageIOError(f"{name} contains non-rotation matrices (max |det-1|={worst:.2e})")
    return rotations


def convert_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    *,
    fps_override: float | None = None,
    mano_models: dict[str, ManoModel] | None = None,
) -> Hot3dEpisode:
    """Build the trajectory arrays + metadata for one episode.

    Args:
        dataset: the LeRobot v3 dataset.
        episode_index: which episode to convert.
        fps_override: frame rate override (defaults to the dataset's).
        mano_models: optional ``{"left": ManoModel, "right": ManoModel}``. When
            given, the 21 joint positions are computed by MANO forward
            kinematics instead of leaving joints 1..20 empty.

    Raises:
        StageIOError: on a missing/oddly shaped column or a malformed rotation.
    """
    episode: LeRobotEpisode = dataset.load_episode(episode_index)
    total = episode.length
    fps = float(fps_override or dataset.info.fps)

    extrinsics = np.asarray(episode.column("extrinsics_w2c"), dtype=np.float64).reshape(total, 4, 4)
    if not np.allclose(extrinsics[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise StageIOError("extrinsics_w2c is not a homogeneous matrix stack")
    rotation_c2w, translation_c2w = invert_rigid(extrinsics[:, :3, :3], extrinsics[:, :3, 3])
    intrinsics = np.asarray(episode.column("intrinsics"), dtype=np.float64).reshape(total, 3, 3)

    # The dataset lives in the HOT3D world frame; the pipeline's contract is
    # World-0 = the camera of the first frame. Re-anchoring here (and recording
    # the original anchor below) makes the reference directly comparable with a
    # prediction while keeping the HOT3D world recoverable.
    anchor_rotation, anchor_translation = world_frame_alignment(rotation_c2w, translation_c2w)
    hot3d_anchor_rotation = rotation_c2w[0].copy()
    hot3d_anchor_translation = translation_c2w[0].copy()
    rotation_c2w = np.einsum("ij,tjk->tik", anchor_rotation, rotation_c2w)
    translation_c2w = np.einsum("ij,tj->ti", anchor_rotation, translation_c2w - anchor_translation)

    valid = np.zeros((total, 2), dtype=bool)
    state_mask = np.asarray(episode.column("state_mask"), dtype=bool).reshape(total, 2)
    valid |= state_mask
    for hand, name in enumerate(("left_kept", "right_kept")):
        if episode.has(name):
            valid[:, hand] &= np.asarray(episode.column(name), dtype=bool).reshape(total)

    joints = np.full((total, 2, 21, 3), np.nan, dtype=np.float64)
    root_rot = np.broadcast_to(np.eye(3), (total, 2, 3, 3)).copy()
    hand_pose = np.broadcast_to(np.eye(3), (total, 2, 15, 3, 3)).copy()
    betas = np.zeros((total, 2, 10), dtype=np.float64)
    confidence = np.where(valid, 1.0, 0.0)

    state = np.asarray(episode.column("observation.state"), dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 2 * STATE_PER_HAND:
        raise StageIOError(
            f"observation.state must be [T, {2 * STATE_PER_HAND}], got {state.shape}"
        )

    for hand, side in enumerate(("left", "right")):
        wrist_world = np.asarray(episode.column(f"{side}_transl_world"), dtype=np.float64)
        if wrist_world.shape != (total, 3):
            raise StageIOError(f"{side}_transl_world must be [{total}, 3], got {wrist_world.shape}")
        joints[:, hand, 0, :] = np.einsum(
            "ij,tj->ti", anchor_rotation, wrist_world - anchor_translation
        )
        root_rot[:, hand] = _stack_rotation_matrices(
            episode.column(f"{side}_orient_world"), count=1, name=f"{side}_orient_world"
        )[:, 0]
        hand_pose[:, hand] = _stack_rotation_matrices(
            episode.column(f"{side}_hand_pose"), count=15, name=f"{side}_hand_pose"
        )
        block = state[:, hand * STATE_PER_HAND : (hand + 1) * STATE_PER_HAND]
        betas[:, hand] = block[:, 51:61]
        # Fingertip positions are unknown without the MANO model - they stay NaN.
        joints[~valid[:, hand], hand, :, :] = np.nan

    if np.allclose(betas[:, 0], betas[:, 1]):
        logger.info("observation.state: both hands share the same shape betas (as stored)")

    bbox = np.full((total, 2, 4), np.nan, dtype=np.float64)
    track_id = np.where(valid, 0, -1).astype(np.int64)
    joints_mode = "wrist_only"
    if mano_models:
        joints, joints_mode = _joints_from_mano(
            mano_models, betas, hand_pose, root_rot, joints, valid
        )
    arrays: dict[str, Array] = {
        "frames": np.arange(total, dtype=np.int64),
        "timestamps": np.arange(total, dtype=np.float64) / fps,
        "hand_xyz_world": joints,
        "hand_xyz_camera": np.einsum(
            "tji,thkj->thki", rotation_c2w, joints - translation_c2w[:, None, None, :]
        ),
        "hand_valid": valid,
        "hand_confidence": confidence,
        "hand_interpolated": np.zeros(valid.shape, dtype=bool),
        "camera_R_c2w": rotation_c2w,
        "camera_t_c2w": translation_c2w,
        "camera_K": intrinsics,
        "bbox": bbox,
        "track_id": track_id,
        "mano_root_rot": root_rot,
        "mano_hand_pose": hand_pose,
        "mano_betas": betas,
        "postprocess_valid": valid,
    }

    height, width = 0, 0
    feature = dataset.info.features[dataset.video_key]
    shape = feature.get("shape", [])
    if len(shape) == 3:
        height, width = int(shape[0]), int(shape[1])
    metadata: dict[str, object] = {
        "fps": fps,
        "world_frame": 0,
        "hand_representation": "21_joints_metric_xyz",
        "camera_convention": "c2w",
        "units": "meter",
        "num_frames": total,
        "hot3d_world_anchor_rotation": hot3d_anchor_rotation.tolist(),
        "hot3d_world_anchor_translation": hot3d_anchor_translation.tolist(),
        "hot3d_world_anchor_note": (
            "p_hot3d_world = anchor_rotation @ p_world0 + anchor_translation"
        ),
        "source": f"{dataset.root.name}#episode{episode_index}",
        "source_dataset": str(dataset.info.raw.get("_source_dataset", "hot3d")),
        "dataset_root": str(dataset.root),
        "video": str(episode.video_path),
        "task": episode.task,
        "hand_joints": joints_mode,
        "hand_joints_note": (
            "21 joint positions from MANO forward kinematics"
            if joints_mode.startswith("mano_fk")
            else "the sample stores a wrist pose plus 15 joint rotations; 21 joint positions "
            "need the MANO mesh model (absent, licence-gated), so joints 1..20 are NaN"
        ),
        "mano_model": (
            {hands: model.source for hands, model in mano_models.items()} if mano_models else None
        ),
        "mano_mirrored": (
            {hands: bool(model.mirrored) for hands, model in mano_models.items()}
            if mano_models
            else None
        ),
        "state_layout": STATE_LAYOUT,
        "state_layout_inferred": True,
        "bbox_available": False,
        "validity_source": "state_mask & *_kept",
        "coverage_left": float(np.mean(valid[:, 0])),
        "coverage_right": float(np.mean(valid[:, 1])),
        "width": width,
        "height": height,
    }
    logger.info(
        "episode %d: %d frames @ %.1f fps, coverage left %.1f%% right %.1f%% (%s)",
        episode_index,
        total,
        fps,
        100.0 * metadata["coverage_left"],
        100.0 * metadata["coverage_right"],
        joints_mode,
    )
    return Hot3dEpisode(
        arrays=arrays,
        metadata=metadata,
        num_frames=total,
        fps=fps,
        width=width,
        height=height,
        video_path=episode.video_path,
        task=episode.task,
    )


def _joints_from_mano(
    mano_models: dict[str, ManoModel],
    betas: Array,
    hand_pose: Array,
    root_rot: Array,
    reference_wrist: Array,
    valid: Array,
) -> tuple[Array, str]:
    """Fill all 21 joints with MANO forward kinematics.

    The wrist stays exactly where the dataset says it is (``align='wrist'``), so
    the 21-joint reference remains consistent with the wrist-only one it
    replaces. A ``False`` from :func:`landmarks_match_topology` is logged as a
    warning - it usually means the landmark mapping disagrees with the model.
    """
    joints = reference_wrist.copy()
    mirrored_any = False
    for hand, side in enumerate(("left", "right")):
        model = mano_models.get(side)
        if model is None:
            logger.warning("no %s MANO model supplied; hand %d stays wrist-only", side, hand)
            continue
        if model.mirrored:
            mirrored_any = True
            pose, root = mirror_pose(hand_pose[:, hand], root_rot[:, hand])
        else:
            pose, root = hand_pose[:, hand], root_rot[:, hand]
        landmarks = forward_kinematics(
            model,
            betas[:, hand],
            pose,
            root_rotation=root,
            root_translation=reference_wrist[:, hand, 0, :],
        )
        frame_valid = valid[:, hand]
        joints[:, hand] = np.where(frame_valid[:, None, None], landmarks, np.nan)
        # The mapping is validated once, in the model's rest pose: a real curled
        # hand legitimately brings its fingertips closer to the wrist than a
        # proximal joint, so a per-frame check would only produce false alarms.
        if hand not in _MAPPING_VALIDATED:
            ok, detail = validate_landmark_mapping(model)
            _MAPPING_VALIDATED[hand] = ok
            if ok:
                logger.info("hand %d: MANO landmark mapping verified (%s)", hand, detail)
            else:
                logger.warning("hand %d: %s", hand, detail)
    mode = "mano_fk_mirrored" if mirrored_any else "mano_fk"
    return joints, mode


def write_episode(
    episode: Hot3dEpisode,
    trajectory_path: str | Path,
    metadata_path: str | Path,
) -> tuple[Path, Path]:
    """Persist the converted episode (validating the array contract first)."""
    from ..io.serialization import validate_trajectory

    problems = validate_trajectory(episode.arrays, strict=True)
    if problems:
        raise StageIOError("converted ground truth is invalid: " + "; ".join(problems))
    return (
        save_npz(trajectory_path, **episode.arrays),
        save_json(metadata_path, episode.metadata),
    )

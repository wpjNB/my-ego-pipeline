"""Generate a MANO preview mesh that follows the cleaned camera-space hand track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError
from .motion_outlier import rotation_matrix_valid
from .temporal_tracks import interpolate_rotation_gaps, interpolate_vector_gaps

Array = np.ndarray


@dataclass(frozen=True)
class ProcessedManoPreview:
    """Rigidly re-anchored/interpolated camera-space mesh for the viewer."""

    vertices_camera: Array
    valid: Array
    interpolated: Array


def build_processed_mano_preview(
    vertices_camera: Array,
    raw_joints_camera: Array,
    processed_joints_camera: Array,
    raw_valid: Array,
    processed_valid: Array,
    outlier: Array,
    *,
    raw_root_rot: Array | None = None,
    processed_root_rot: Array | None = None,
    max_gap: int = 12,
) -> ProcessedManoPreview:
    """Interpolate short mesh holes and align each mesh to cleaned wrist/root pose.

    The source mesh retains HaWoR's articulated hand shape. The pipeline does
    not invent finger articulation: it interpolates vertex tracks across short
    invalid spans, applies the cleaned root rotation when available, then
    translates the mesh to the processed wrist.
    """
    vertices = np.asarray(vertices_camera, dtype=np.float64)
    raw_joints = np.asarray(raw_joints_camera, dtype=np.float64)
    joints = np.asarray(processed_joints_camera, dtype=np.float64)
    raw_mask = np.asarray(raw_valid, dtype=bool)
    clean_mask = np.asarray(processed_valid, dtype=bool)
    rejected = np.asarray(outlier, dtype=bool)
    if vertices.ndim != 4 or vertices.shape[:2] != (raw_joints.shape[0], 2) or vertices.shape[-1] != 3:
        raise StageIOError(f"vertices_camera must be [T, 2, V, 3], got {vertices.shape}")
    frames = vertices.shape[0]
    if raw_joints.shape != (frames, 2, 21, 3) or joints.shape != raw_joints.shape:
        raise StageIOError("raw and processed joints must both be [T, 2, 21, 3]")
    if raw_mask.shape != (frames, 2) or clean_mask.shape != (frames, 2) or rejected.shape != (frames, 2):
        raise StageIOError(f"hand masks must be [{frames}, 2]")
    if max_gap < 1:
        raise StageIOError(f"max_gap must be >= 1, got {max_gap}")

    if raw_root_rot is not None:
        raw_rot = np.asarray(raw_root_rot, dtype=np.float64)
        if raw_rot.shape != (frames, 2, 3, 3):
            raise StageIOError(f"raw_root_rot must be [{frames}, 2, 3, 3], got {raw_rot.shape}")
    else:
        raw_rot = None
    if processed_root_rot is not None:
        clean_rot = np.asarray(processed_root_rot, dtype=np.float64)
        if clean_rot.shape != (frames, 2, 3, 3):
            raise StageIOError(
                f"processed_root_rot must be [{frames}, 2, 3, 3], got {clean_rot.shape}"
            )
    else:
        clean_rot = None

    output = np.full_like(vertices, np.nan)
    valid_out = np.zeros((frames, 2), dtype=bool)
    interpolated = np.zeros((frames, 2), dtype=bool)
    for hand in range(2):
        vertex_anchor = (
            raw_mask[:, hand]
            & ~rejected[:, hand]
            & np.isfinite(vertices[:, hand]).all(axis=(1, 2))
        )
        wrist_anchor = vertex_anchor & np.isfinite(raw_joints[:, hand, 0]).all(axis=1)
        filled_vertices = interpolate_vector_gaps(
            vertices[:, hand], vertex_anchor, max_gap=max_gap
        )
        filled_wrist = interpolate_vector_gaps(
            raw_joints[:, hand, 0], wrist_anchor, max_gap=max_gap
        )

        if raw_rot is not None:
            root_anchor = vertex_anchor & rotation_matrix_valid(raw_rot[:, hand])
            filled_root = interpolate_rotation_gaps(
                raw_rot[:, hand], root_anchor, max_gap=max_gap
            )
        else:
            filled_root = None

        for frame in np.flatnonzero(clean_mask[:, hand] & filled_vertices.valid):
            if not filled_wrist.valid[frame] or not np.isfinite(joints[frame, hand, 0]).all():
                continue
            source_rotation = None
            target_rotation = None
            if filled_root is not None and filled_root.valid[frame]:
                source_rotation = filled_root.values[frame]
            if clean_rot is not None and rotation_matrix_valid(clean_rot[frame, hand]):
                target_rotation = clean_rot[frame, hand]
            mesh = filled_vertices.values[frame]
            if source_rotation is not None and target_rotation is not None:
                delta_rotation = target_rotation @ source_rotation.T
                mesh = (mesh - filled_wrist.values[frame]) @ delta_rotation.T
            else:
                mesh = mesh - filled_wrist.values[frame]
            output[frame, hand] = mesh + joints[frame, hand, 0]
            valid_out[frame, hand] = np.isfinite(output[frame, hand]).all()
            interpolated[frame, hand] = filled_vertices.interpolated[frame]

    return ProcessedManoPreview(
        vertices_camera=output,
        valid=valid_out,
        interpolated=interpolated,
    )

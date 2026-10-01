"""Phase 2: temporal blending of HaWoR windows.

HaWoR reconstructs 16-frame windows with an 8-frame overlap. Consecutive
windows already live in the *same* camera coordinate system, so no Sim(3) is
needed here - only a temporal blend on the shared frames. The blend weight is
the position within the overlap, exactly as in the reference configuration:

``p_t = (1 - alpha_t) * p_t^A + alpha_t * p_t^B``
``R_t = SLERP(R_t^A, R_t^B, alpha_t)`` with ``alpha_t`` ramping linearly from
``0`` to ``1`` across the shared frames.

Frames that only one window covers keep that window's value; frames covered by
no valid window stay missing (the pipeline never invents 3D pose).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError
from ..geometry.transforms import rotation_slerp

logger = logging.getLogger(__name__)

Array = np.ndarray
NUM_JOINTS = 21
NUM_HANDS = 2


@dataclass(frozen=True)
class HandWindow:
    """One HaWoR window of reconstructed hands, in camera space."""

    start: int
    joints_camera: Array  # [n, 2, 21, 3]
    valid: Array  # [n, 2] bool
    confidence: Array | None = None  # [n, 2]
    root_rot: Array | None = None  # [n, 2, 3, 3]
    betas: Array | None = None  # [n, 2, 10]
    vertices_camera: Array | None = None  # [n, 2, V, 3] (V = 778 for MANO)

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints_camera, dtype=np.float64)
        if joints.ndim != 4 or joints.shape[1:] != (NUM_HANDS, NUM_JOINTS, 3):
            raise StageIOError(
                f"joints_camera must have shape [n, 2, 21, 3], got {joints.shape}"
            )
        valid = np.asarray(self.valid, dtype=bool)
        if valid.shape != joints.shape[:2]:
            raise StageIOError(f"valid must have shape {joints.shape[:2]}, got {valid.shape}")
        if self.start < 0:
            raise StageIOError(f"window start must be >= 0, got {self.start}")
        if self.confidence is not None:
            conf = np.asarray(self.confidence, dtype=np.float64)
            if conf.shape != joints.shape[:2]:
                raise StageIOError(f"confidence must have shape {joints.shape[:2]}, got {conf.shape}")
            object.__setattr__(self, "confidence", conf)
        if self.root_rot is not None:
            rot = np.asarray(self.root_rot, dtype=np.float64)
            if rot.shape != (joints.shape[0], NUM_HANDS, 3, 3):
                raise StageIOError(f"root_rot must have shape [{joints.shape[0]}, 2, 3, 3], got {rot.shape}")
            object.__setattr__(self, "root_rot", rot)
        if self.betas is not None:
            betas = np.asarray(self.betas, dtype=np.float64)
            if betas.shape != (joints.shape[0], NUM_HANDS, 10):
                raise StageIOError(f"betas must have shape [{joints.shape[0]}, 2, 10], got {betas.shape}")
            object.__setattr__(self, "betas", betas)
        if self.vertices_camera is not None:
            verts = np.asarray(self.vertices_camera, dtype=np.float64)
            if verts.ndim != 4 or verts.shape[:2] != joints.shape[:2] or verts.shape[3] != 3:
                raise StageIOError(
                    "vertices_camera must have shape [n, 2, V, 3] with n, 2 matching "
                    f"joints_camera, got {verts.shape}"
                )
            object.__setattr__(self, "vertices_camera", verts)
        object.__setattr__(self, "joints_camera", joints)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "start", int(self.start))

    @property
    def num_frames(self) -> int:
        return int(self.joints_camera.shape[0])

    @property
    def end(self) -> int:
        """Exclusive end frame."""
        return self.start + self.num_frames

    def frame_ids(self) -> Array:
        return np.arange(self.start, self.end, dtype=np.int64)


@dataclass(frozen=True)
class HandBlendResult:
    """Blended camera-space hand trajectory."""

    joints_camera: Array  # [T, 2, 21, 3]
    valid: Array  # [T, 2] bool
    confidence: Array  # [T, 2]
    root_rot: Array  # [T, 2, 3, 3] (identity where unavailable)
    weight: Array  # [T, 2] number of windows that contributed
    has_root_rot: bool
    vertices_camera: Array | None = None  # [T, 2, V, 3] when the windows carry meshes


def blend_hand_windows(windows: list[HandWindow] | tuple[HandWindow, ...]) -> HandBlendResult:
    """Blend overlapping HaWoR windows into one camera-space trajectory.

    Args:
        windows: windows in temporal order; they may overlap arbitrarily.

    Returns:
        :class:`HandBlendResult` covering frames ``0 .. max(end) - 1``.

    Raises:
        StageIOError: if ``windows`` is empty, windows are unordered, or their
            joint arrays disagree in shape.
    """
    if not windows:
        raise StageIOError("blend_hand_windows requires at least one window")
    ordered = sorted(windows, key=lambda w: w.start)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur.start < prev.start:
            raise StageIOError("hand windows are not in temporal order")

    total = max(w.end for w in ordered)
    joints = np.zeros((total, NUM_HANDS, NUM_JOINTS, 3), dtype=np.float64)
    valid = np.zeros((total, NUM_HANDS), dtype=bool)
    confidence = np.zeros((total, NUM_HANDS), dtype=np.float64)
    weight = np.zeros((total, NUM_HANDS), dtype=np.float64)
    has_root_rot = any(w.root_rot is not None for w in ordered)
    root_rot = np.broadcast_to(np.eye(3, dtype=np.float64), (total, NUM_HANDS, 3, 3)).copy()
    rot_weight = np.zeros((total, NUM_HANDS), dtype=np.float64)
    has_vertices = any(w.vertices_camera is not None for w in ordered)
    vertices: Array | None = None
    vertex_weight: Array | None = None
    if has_vertices:
        num_verts = int(
            next(w.vertices_camera.shape[2] for w in ordered if w.vertices_camera is not None)
        )
        vertices = np.zeros((total, NUM_HANDS, num_verts, 3), dtype=np.float32)
        vertex_weight = np.zeros((total, NUM_HANDS), dtype=np.float64)

    for index, window in enumerate(ordered):
        alpha = (
            np.ones(window.num_frames, dtype=np.float64)
            if index == 0
            else overlap_alpha_ramp(window, ordered[index - 1])
        )
        for local in range(window.num_frames):
            frame = window.start + local
            a = float(alpha[local])
            for hand in range(NUM_HANDS):
                if not window.valid[local, hand]:
                    continue
                if window.confidence is not None:
                    confidence[frame, hand] = max(
                        confidence[frame, hand], float(window.confidence[local, hand])
                    )

                new_joints = window.joints_camera[local, hand]
                if not np.isfinite(new_joints).all():
                    # A window can mark a frame valid while its model output
                    # collapsed to NaN (e.g. a broken focal length upstream):
                    # such a frame must stay missing, never poison the blend.
                    continue
                if weight[frame, hand] <= 0.0:
                    joints[frame, hand] = new_joints
                    weight[frame, hand] = 1.0
                    valid[frame, hand] = True
                elif a <= 0.0:
                    # The older window owns this frame; count the contribution only.
                    weight[frame, hand] += 1.0
                else:
                    joints[frame, hand] = (1.0 - a) * joints[frame, hand] + a * new_joints
                    weight[frame, hand] += 1.0

                new_vertices = (
                    None
                    if window.vertices_camera is None
                    else window.vertices_camera[local, hand]
                )
                if vertices is not None and new_vertices is not None:
                    assert vertex_weight is not None  # for the type checker
                    if vertex_weight[frame, hand] <= 0.0:
                        vertices[frame, hand] = new_vertices
                        vertex_weight[frame, hand] = 1.0
                    elif a > 0.0:
                        vertices[frame, hand] = (
                            (1.0 - a) * vertices[frame, hand].astype(np.float64)
                            + a * new_vertices
                        )
                        vertex_weight[frame, hand] += 1.0

                if window.root_rot is not None:
                    candidate = window.root_rot[local, hand]
                    if rot_weight[frame, hand] <= 0.0:
                        root_rot[frame, hand] = candidate
                    elif a > 0.0:
                        root_rot[frame, hand] = rotation_slerp(
                            root_rot[frame, hand][None], candidate[None], a
                        )[0]
                    rot_weight[frame, hand] += 1.0

    num_missing = int(np.count_nonzero(~valid))
    if num_missing:
        logger.info(
            "hand blending: %d/%d hand-frames stay missing (kept as missing, never filled)",
            num_missing,
            total * NUM_HANDS,
        )
    else:
        logger.info("hand blending: full coverage over %d frames", total)

    return HandBlendResult(
        joints_camera=joints,
        valid=valid,
        confidence=confidence,
        root_rot=root_rot,
        weight=weight,
        has_root_rot=has_root_rot,
        vertices_camera=vertices,
    )


def overlap_alpha_ramp(window: HandWindow, previous: HandWindow) -> Array:
    """Per-frame blend weight of ``window`` inside its overlap with ``previous``.

    Returns ``1.0`` for frames that only ``window`` covers and a linear
    ``0 -> 1`` ramp across the shared frames.
    """
    alpha = np.ones(window.num_frames, dtype=np.float64)
    start = max(window.start, previous.start)
    end = min(window.end, previous.end)
    if end <= start:
        return alpha
    count = end - start
    ramp = np.linspace(0.0, 1.0, count) if count > 1 else np.ones(1, dtype=np.float64)
    alpha[start - window.start : end - window.start] = ramp
    return alpha


def overlap_alpha(index: int, count: int) -> float:
    """Linear blend weight for index ``index`` of a ``count``-frame overlap."""
    if count <= 0:
        raise StageIOError(f"overlap count must be positive, got {count}")
    if not 0 <= index < count:
        raise StageIOError(f"overlap index {index} out of range for count {count}")
    if count == 1:
        return 1.0
    return index / (count - 1)


def smooth_hand_trajectory(
    joints_camera: Array,
    valid: Array,
    *,
    vertices_camera: Array | None = None,
    passes: int = 1,
) -> tuple[Array, Array | None]:
    """Binomial [1, 2, 1] / 4 smoothing along time, per hand, within valid runs.

    The blended trajectory inherits the independent reconstruction noise of
    every overlapping HaWoR window (~8 mm high-frequency wrist residual on the
    real clips), which the mesh overlay makes visible as rapid wobble. This
    damps that noise while leaving real motion (17-20 mm/frame) mostly intact:
    one pass is a 3-tap filter, applied only where the frame, its predecessor
    and its successor are all valid, so gaps never bleed across and the run
    edges keep their original values. Vertices, when given, are smoothed with
    the same taps so the mesh stays consistent with the joints.

    Returns ``(joints, vertices)`` with the same shapes; ``vertices`` is None
    when no vertices were passed.
    """
    if passes < 0:
        raise StageIOError(f"smooth passes must be >= 0, got {passes}")
    joints = np.asarray(joints_camera, dtype=np.float64)
    verts = None if vertices_camera is None else np.asarray(vertices_camera, dtype=np.float64)
    for _ in range(passes):
        for hand in range(joints.shape[1]):
            for start, stop in _valid_runs(valid[:, hand]):
                if stop - start < 3:
                    continue
                # Read from a snapshot so the filter stays symmetric (an
                # in-place update would make it directional).
                joints_src = joints[start:stop, hand].copy()
                verts_src = None if verts is None else verts[start:stop, hand].copy()
                for local in range(1, stop - start - 1):
                    trio = joints_src[local - 1 : local + 2]
                    if np.isfinite(trio).all():
                        joints[start + local, hand] = (
                            0.25 * trio[0] + 0.5 * trio[1] + 0.25 * trio[2]
                        )
                    if verts_src is not None:
                        trio_v = verts_src[local - 1 : local + 2]
                        if np.isfinite(trio_v).all():
                            verts[start + local, hand] = (
                                0.25 * trio_v[0] + 0.5 * trio_v[1] + 0.25 * trio_v[2]
                            )
    return joints, verts


def _valid_runs(mask: Array) -> list[tuple[int, int]]:
    """Maximal ``[start, stop)`` runs of True in a boolean vector."""
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(np.asarray(mask, dtype=bool)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs

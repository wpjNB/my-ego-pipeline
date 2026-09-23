"""Deterministic synthetic clip: scene, camera, hands, detections, windows.

Everything is a pure function of ``(num_frames, seed)`` so that separate
processes (the mock WiLoR / HaWoR / VGGT runners) agree without sharing state.

Coordinate story used throughout:

* the scene is static in world coordinates;
* the camera trajectory is ``c2w`` and starts at the origin with ``R = I``, so
  ``World-0`` is literally the first frame's camera;
* the hand ground truth is a smooth world-space motion, which is then expressed
  in camera space the way a reconstruction backend would report it.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

from ..camera.window import CameraWindow, WindowRange
from ..hand.mano_model import MANO_FINGERTIP_VERTICES, ManoModel
from ..geometry.sim3 import Sim3
from ..geometry.transforms import invert_rigid, project_points

Array = np.ndarray

DEFAULT_FPS = 30.0
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_DEPTH_SIZE = (48, 64)  # (height, width) used by the mock VGGT backend


def make_intrinsics(width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, fov_deg: float = 60.0) -> Array:
    """Pinhole intrinsics with the standard OpenCV convention."""
    focal = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_camera_trajectory(num_frames: int, *, seed: int = 0) -> tuple[Array, Array]:
    """Smooth oscillatory c2w trajectory that keeps a scene at ~1.8 m in view."""
    _ = seed  # the trajectory is a fixed smooth function; kept for symmetry
    rotation = np.zeros((num_frames, 3, 3), dtype=np.float64)
    translation = np.zeros((num_frames, 3), dtype=np.float64)
    for index in range(num_frames):
        rotation[index] = Rotation.from_euler(
            "yx",
            [0.18 * np.sin(0.05 * index), 0.06 * np.sin(0.03 * index + 1.0)],
        ).as_matrix()
        translation[index] = [
            0.15 * np.sin(0.04 * index),
            0.10 * np.sin(0.03 * index + 0.4),
            0.10 * np.sin(0.02 * index + 1.7),
        ]
    return rotation, translation


def sample_scene_points(count: int = 300, *, seed: int = 7) -> Array:
    """A static point cloud in front of the camera."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-0.6, 0.6, count),
            rng.uniform(-0.5, 0.5, count),
            rng.uniform(1.2, 2.4, count),
        ],
        axis=-1,
    )


def world0_camera_trajectory(num_frames: int, *, seed: int = 0) -> tuple[Array, Array]:
    """Camera poses re-expressed so that frame 0 is the world origin."""
    rotation, translation = make_camera_trajectory(num_frames, seed=seed)
    r0_inv, t0_inv = invert_rigid(rotation[0], translation[0])
    rotation_w0 = np.einsum("ij,tjk->tik", r0_inv, rotation)
    translation_w0 = np.einsum("ij,tj->ti", r0_inv, translation) + t0_inv
    return rotation_w0, translation_w0


def to_world0(points_world: Array, num_frames: int, *, seed: int = 0) -> Array:
    """Re-express points of the raw world frame in ``World-0`` (frame 0 camera).

    The pipeline always reports results in ``World-0``, so a reference
    trajectory has to be expressed there too - otherwise the two differ by a
    rigid gauge transform that shows up as a constant offset in Action-MPJPE.
    """
    rotation, translation = make_camera_trajectory(num_frames, seed=seed)
    r0_inv, _ = invert_rigid(rotation[0], translation[0])
    return np.einsum("ij,...j->...i", r0_inv, np.asarray(points_world) - translation[0])


def world_to_camera_points(
    points_world: Array, rotation_c2w: Array, translation_c2w: Array
) -> Array:
    """Exact ``p_cam = R_c2w^T (p_world - t_c2w)`` with no injected noise."""
    points = np.asarray(points_world, dtype=np.float64)
    rotation = np.asarray(rotation_c2w, dtype=np.float64)
    translation = np.asarray(translation_c2w, dtype=np.float64)
    return np.einsum("tji,thkj->thki", rotation, points - translation[:, None, None, :])


def render_depth(
    points: Array,
    rotation_c2w: Array,
    translation_c2w: Array,
    intrinsics: Array,
    *,
    height: int,
    width: int,
    splat: int = 2,
) -> Array:
    """Z-buffer splat of a point cloud into a depth map (NaN where empty)."""
    depth = np.full((height, width), np.nan, dtype=np.float64)
    cam = (np.asarray(points) - translation_c2w) @ rotation_c2w
    z = cam[:, 2]
    visible = z > 1e-6
    if not np.any(visible):
        return depth
    cam = cam[visible]
    z = z[visible]
    u = intrinsics[0, 0] * cam[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * cam[:, 1] / z + intrinsics[1, 2]
    for uu, vv, zz in zip(u, v, z, strict=False):
        cu, cv = int(round(uu)), int(round(vv))
        if not (0 <= cu < width and 0 <= cv < height):
            continue
        r0, r1 = max(0, cv - splat), min(height, cv + splat + 1)
        c0, c1 = max(0, cu - splat), min(width, cu + splat + 1)
        patch = depth[r0:r1, c0:c1]
        depth[r0:r1, c0:c1] = np.where(np.isnan(patch) | (zz < patch), zz, patch)
    return depth


def relative_mismatch(index: int, *, seed: int = 0, scale: float = 1.35) -> Sim3:
    """Per-window Sim(3) disagreement injected by the mock camera backend.

    Window 0 is the world anchor and therefore exact; later windows drift in
    rotation, translation *and* metric scale, which is exactly what Phase 4 has
    to undo.
    """
    if index == 0:
        return Sim3.identity()
    rng = np.random.default_rng(seed + 1000 * index)
    rotation = Rotation.random(random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()
    return Sim3(
        scale=scale + 0.05 * rng.normal(),
        rotation=rotation,
        translation=rng.normal(0.0, 0.25, size=3),
    )


def build_camera_windows(
    num_frames: int,
    *,
    window: int = 200,
    overlap: int = 40,
    depth_size: tuple[int, int] = DEFAULT_DEPTH_SIZE,
    seed: int = 0,
    scene_points: int = 600,
) -> tuple[list[CameraWindow], Array, Array]:
    """Window outputs as a VGGT-Omega backend would produce them.

    Returns ``(windows, ground_truth_rotation_c2w, ground_truth_translation_c2w)``
    where the ground truth is in ``World-0``.
    """
    height, width = depth_size
    intrinsics = make_intrinsics(width, height)
    points = sample_scene_points(scene_points, seed=seed + 7)
    rotation, translation = make_camera_trajectory(num_frames, seed=seed)
    rotation_w0, translation_w0 = world0_camera_trajectory(num_frames, seed=seed)

    depth = np.stack(
        [
            render_depth(points, rotation[t], translation[t], intrinsics, height=height, width=width)
            for t in range(num_frames)
        ]
    )

    windows: list[CameraWindow] = []
    stride = window - overlap
    starts = list(range(0, max(1, num_frames - overlap), stride))
    for index, start in enumerate(starts):
        end = min(start + window, num_frames)
        if windows and end <= windows[-1].end:
            break
        mismatch = relative_mismatch(index, seed=seed)
        local_rotation, local_translation = mismatch.transform_poses(
            rotation_w0[start:end], translation_w0[start:end]
        )
        windows.append(
            CameraWindow(
                window=WindowRange(index=index, start=start, end=end),
                rotation_c2w=local_rotation,
                translation_c2w=local_translation,
                intrinsics=np.broadcast_to(intrinsics, (end - start, 3, 3)).copy(),
                depth=depth[start:end] * mismatch.scale,
            )
        )
    return windows, rotation_w0, translation_w0


def hand_world_trajectory(num_frames: int, *, seed: int = 0) -> Array:
    """Smooth world-space ground-truth hands ``[T, 2, 21, 3]``."""
    _ = seed
    times = np.arange(num_frames, dtype=np.float64)
    offsets = np.zeros((21, 3))
    offsets[:, 0] = np.linspace(0.0, 0.09, 21)
    offsets[:, 1] = 0.03 * np.sin(np.linspace(0.0, 3.0, 21))
    offsets[:, 2] = 0.01 * np.cos(np.linspace(0.0, 3.0, 21))

    joints = np.zeros((num_frames, 2, 21, 3), dtype=np.float64)
    for hand in range(2):
        base = np.stack(
            [
                -0.16 + 0.32 * hand + 0.05 * np.sin(0.07 * times),
                0.02 * np.cos(0.05 * times + hand),
                1.45 + 0.12 * np.sin(0.04 * times + 0.5 * hand),
            ],
            axis=-1,
        )
        joints[:, hand] = base[:, None, :] + offsets[None, :, :]
    return joints


def hand_validity(num_frames: int, *, seed: int = 0, right_coverage: float = 0.9) -> Array:
    """``[T, 2]`` validity with realistic gaps (missing stays missing).

    The right hand loses a 5-frame stretch and one longer 11-frame stretch; the
    left hand stays visible apart from a single short gap.
    """
    _ = seed
    valid = np.ones((num_frames, 2), dtype=bool)

    def blank(start: int, end: int, hand: int) -> None:
        lo, hi = min(start, num_frames), min(end, num_frames)
        if hi > lo:
            valid[lo:hi, hand] = False

    blank(20, 22, 0)
    blank(30, 35, 1)
    blank(100, 111, 1)
    if right_coverage < 1.0:
        keep = int(round(right_coverage * num_frames))
        valid[keep:, 1] = False
    return valid


def camera_space_hands(
    world_joints: Array,
    rotation_c2w: Array,
    translation_c2w: Array,
    *,
    seed: int = 0,
    bone_wobble: float = 0.12,
    depth_noise: float = 0.03,
) -> Array:
    """Express world hands in camera space with the artefacts a real model has.

    Two deliberate imperfections: the MANO shape wobbles per frame (the target
    of the bone-scale correction) and the wrist depth is noisy (the target of
    the wrist-depth optimisation). The x/y projection is left clean, matching
    the reference observation that monocular z is the weak axis.
    """
    rng = np.random.default_rng(seed + 99)
    num_frames = world_joints.shape[0]
    camera = np.zeros_like(world_joints)

    for hand in range(2):
        wrist_world = world_joints[:, hand, 0, :]
        relative = world_joints[:, hand] - wrist_world[:, None, :]
        scale = 1.0 + bone_wobble * np.sin(0.7 * np.arange(num_frames) + hand)
        relative = relative * scale[:, None, None]
        hand_world = wrist_world[:, None, :] + relative
        cam = np.einsum("tji,tkj->tki", rotation_c2w, hand_world - translation_c2w[:, None, :])
        cam[:, :, 2] += rng.normal(0.0, depth_noise, size=(num_frames, 21))
        camera[:, hand] = cam
    return camera


def synthetic_detections(
    num_frames: int,
    *,
    seed: int = 0,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    box_padding: int = 38,
) -> dict[str, Array]:
    """WiLoR-style detections for the mock backend.

    Layout: ``boxes [T, K, 4]``, ``confidence [T, K]``, ``right_score``,
    ``left_score`` and ``count [T]``. Candidates are compacted to the front of
    each frame's slots and ``count`` says how many are real, so "no detection"
    is representable without inventing a zero-confidence box.

    Two gaps are planted on purpose:

    * frames 18-21 of the right hand lose their anchor, but frame 20 keeps a
      low-confidence detection whose box matches the interpolated anchor box, so
      the conservative tracker must recover exactly that frame;
    * frames 30-39 of the right hand vanish entirely (10-frame hole > the
      4-frame rule), so they must stay missing.
    """
    intrinsics = make_intrinsics(width, height)
    rotation, translation = make_camera_trajectory(num_frames, seed=seed)
    world = hand_world_trajectory(num_frames, seed=seed)

    per_frame: list[list[tuple[Array, float, float, float]]] = []
    for frame in range(num_frames):
        camera = np.einsum(
            "ij,hkj->hki", rotation[frame], world[frame] - translation[frame]
        )
        pixels = project_points(intrinsics, camera)
        entries: list[tuple[Array, float, float, float]] = []
        for hand in range(2):
            if not np.isfinite(pixels[hand]).all():
                continue
            if hand == 0 and 20 <= frame < 22:
                continue  # left hand briefly invisible
            if hand == 1:
                if 30 <= frame < 40:
                    continue  # unrecoverable hole
                if 18 <= frame < 22 and frame != 20:
                    continue  # recoverable gap, only the middle frame survives
            cx, cy = float(np.mean(pixels[hand, :, 0])), float(np.mean(pixels[hand, :, 1]))
            half = box_padding / 2.0
            box = np.array(
                [
                    max(0.0, cx - half),
                    max(0.0, cy - half),
                    min(float(width - 1), cx + half),
                    min(float(height - 1), cy + half),
                ]
            )
            low_confidence = hand == 1 and frame == 20
            conf = 0.34 if low_confidence else 0.9
            right = 0.9 if hand == 1 else 0.05
            left = 0.05 if hand == 1 else 0.9
            entries.append((box, conf, right, left))
        per_frame.append(entries)

    slots = max(1, max(len(entries) for entries in per_frame))
    boxes = np.zeros((num_frames, slots, 4), dtype=np.float64)
    confidence = np.zeros((num_frames, slots), dtype=np.float64)
    right_score = np.zeros((num_frames, slots), dtype=np.float64)
    left_score = np.zeros((num_frames, slots), dtype=np.float64)
    count = np.zeros(num_frames, dtype=np.int64)
    for frame, entries in enumerate(per_frame):
        count[frame] = len(entries)
        for slot, (box, conf, right, left) in enumerate(entries):
            boxes[frame, slot] = box
            confidence[frame, slot] = conf
            right_score[frame, slot] = right
            left_score[frame, slot] = left
    return {
        "boxes": boxes,
        "confidence": confidence,
        "right_score": right_score,
        "left_score": left_score,
        "count": count,
    }


def synthetic_mano_joints() -> Array:
    """A plausible MANO skeleton: wrist at the origin, five chains along +x."""
    directions = {
        "index": np.array([1.0, 0.35, 0.05]),
        "middle": np.array([1.0, 0.10, 0.0]),
        "pinky": np.array([1.0, -0.35, 0.0]),
        "ring": np.array([1.0, -0.12, 0.02]),
        "thumb": np.array([1.0, 0.55, -0.45]),
    }
    joints = np.zeros((16, 3), dtype=np.float64)
    for finger, (a, b, c) in (("index", (1, 2, 3)), ("middle", (4, 5, 6)), ("pinky", (7, 8, 9)), ("ring", (10, 11, 12)), ("thumb", (13, 14, 15))):
        direction = directions[finger] / np.linalg.norm(directions[finger])
        for step, index in enumerate((a, b, c), start=1):
            joints[index] = direction * (0.03 * step)
    return joints


def mano_fingertip_positions(joints: Array | None = None) -> Array:
    """Fingertip positions in :data:`MANO_FINGERTIP_VERTICES` order."""
    base = synthetic_mano_joints() if joints is None else np.asarray(joints, dtype=np.float64)
    chain = {
        "thumb": (13, 14, 15),
        "index": (1, 2, 3),
        "middle": (4, 5, 6),
        "ring": (10, 11, 12),
        "pinky": (7, 8, 9),
    }
    out = np.zeros((5, 3), dtype=np.float64)
    for slot, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        _, middle, dip = chain[finger]
        direction = base[dip] - base[middle]
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        out[slot] = base[dip] + direction * 0.03
    return out


def make_synthetic_mano_model(
    *,
    num_vertices: int = 800,
    shape_scale: float = 0.0,
    pose_dirs_scale: float = 0.0,
) -> ManoModel:
    """A tiny MANO-shaped model with the same structure as the real asset.

    Every joint is a vertex (``J_regressor`` is a selection matrix) and every
    vertex is weighted to exactly one joint, so the forward kinematics has an
    exact, checkable answer - which is what makes it useful as a test fixture.
    """
    joints = synthetic_mano_joints()
    tips = mano_fingertip_positions(joints)

    template = np.zeros((num_vertices, 3), dtype=np.float64)
    template[:16] = joints
    for slot, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        template[MANO_FINGERTIP_VERTICES[finger]] = tips[slot]
    filler = np.arange(num_vertices)
    remainder = (filler >= 16) & ~np.isin(filler, list(MANO_FINGERTIP_VERTICES.values()))
    template[remainder] = joints[0] + 0.01 * np.stack(
        [np.cos(0.1 * filler[remainder]), np.sin(0.1 * filler[remainder]), np.zeros(remainder.sum())],
        axis=-1,
    )

    regressor = np.zeros((16, num_vertices), dtype=np.float64)
    for index in range(16):
        regressor[index, index] = 1.0

    weights = np.zeros((num_vertices, 16), dtype=np.float64)
    weights[:, 0] = 1.0
    for index in range(16):
        weights[index] = 0.0
        weights[index, index] = 1.0
    tip_to_dip = {"thumb": 15, "index": 3, "middle": 6, "ring": 12, "pinky": 9}
    for finger, vertex in MANO_FINGERTIP_VERTICES.items():
        weights[vertex] = 0.0
        weights[vertex, tip_to_dip[finger]] = 1.0

    shapedirs = np.zeros((num_vertices, 3, 10), dtype=np.float64)
    if shape_scale:
        shapedirs[:, :, 0] = template * shape_scale
    posedirs = np.zeros((num_vertices, 3, 9 * 15), dtype=np.float64)
    if pose_dirs_scale:
        posedirs[:16, :, 0] = pose_dirs_scale

    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    return ManoModel(
        v_template=template,
        shapedirs=shapedirs,
        j_regressor=regressor,
        weights=weights,
        posedirs=posedirs,
        faces=faces,
        hands="right",
        source="synthetic",
    )


def write_synthetic_mano_npz(path: str | Path, **kwargs: object) -> Path:
    """Persist a synthetic MANO model as the ``.npz`` the loader expects."""
    model = make_synthetic_mano_model(**kwargs)  # type: ignore[arg-type]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        v_template=model.v_template,
        shapedirs=model.shapedirs,
        j_regressor=model.j_regressor,
        weights=model.weights,
        posedirs=model.posedirs,
        f=model.faces,
    )
    return target

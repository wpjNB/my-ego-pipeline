"""Shared fixtures and synthetic-scene helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.camera.window import CameraWindow, WindowRange
from ego3d_action.geometry.sim3 import Sim3
from ego3d_action.geometry.transforms import invert_rigid


def make_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def make_pose_sequence(num_frames: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A smooth c2w camera trajectory that keeps a scene at z~1.8 m in view.

    The motion is oscillatory rather than drifting so that every frame of a
    360-frame clip still sees the synthetic point cloud, while still providing
    enough rotation and parallax for a well-posed Sim(3).
    """
    from scipy.spatial.transform import Rotation

    t = np.arange(num_frames, dtype=np.float64)
    rot = np.zeros((num_frames, 3, 3))
    trans = np.zeros((num_frames, 3))
    for index in range(num_frames):
        yaw = 0.18 * np.sin(0.05 * index)
        pitch = 0.06 * np.sin(0.03 * index + 1.0)
        rot[index] = Rotation.from_euler("yx", [yaw, pitch]).as_matrix()
        trans[index] = np.array(
            [
                0.15 * np.sin(0.04 * index),
                0.10 * np.sin(0.03 * index + 0.4),
                0.10 * np.sin(0.02 * index + 1.7),
            ]
        )
    _ = seed  # kept for API compatibility with callers that want determinism
    return rot, trans


def render_depth_from_points(
    points_world: np.ndarray,
    rotation_c2w: np.ndarray,
    translation_c2w: np.ndarray,
    intrinsics: np.ndarray,
    *,
    height: int,
    width: int,
    splat: int = 2,
) -> np.ndarray:
    """Splat a 3D point cloud into a depth map with a z-buffer."""
    depth = np.full((height, width), np.nan, dtype=np.float64)
    cam = (points_world - translation_c2w) @ rotation_c2w
    z = cam[:, 2]
    visible = z > 1e-6
    if not np.any(visible):
        return np.full((height, width), np.nan)
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


def make_window(
    *,
    index: int,
    start: int,
    num_frames: int,
    rotation: np.ndarray,
    translation: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> CameraWindow:
    return CameraWindow(
        window=WindowRange(index=index, start=start, end=start + num_frames),
        rotation_c2w=rotation,
        translation_c2w=translation,
        intrinsics=intrinsics,
        depth=depth,
    )


def build_two_windows(
    scene: dict[str, object],
    relative: Sim3,
    *,
    num_frames: int = 360,
) -> tuple[CameraWindow, CameraWindow, np.ndarray, np.ndarray]:
    """Two overlapping windows whose local frames differ by ``relative``.

    Window A covers frames ``[0, 200)`` with its own first camera as the local
    world origin; window B covers ``[160, 360)`` and is expressed in a local
    frame related to A's by ``relative`` (rotation, translation *and* scale -
    the depth of B is ``relative.scale`` times the depth of A, exactly as a
    differently-scaled reconstruction of the same scene would be).
    """
    points = np.asarray(scene["points"])
    rot = np.asarray(scene["rotation_c2w"])
    trans = np.asarray(scene["translation_c2w"])
    intrinsics = np.asarray(scene["intrinsics"])
    height = int(scene["height"])
    width = int(scene["width"])

    r0_inv, t0_inv = invert_rigid(rot[0], trans[0])
    rot_a = np.einsum("ij,tjk->tik", r0_inv, rot[:num_frames])
    trans_a = np.einsum("ij,tj->ti", r0_inv, trans[:num_frames]) + t0_inv

    depth_a = np.stack(
        [
            render_depth_from_points(
                points, rot[t], trans[t], intrinsics, height=height, width=width, splat=2
            )
            for t in range(num_frames)
        ]
    )
    rot_b, trans_b = relative.transform_poses(rot_a[160:360], trans_a[160:360])
    return (
        make_window(
            index=0,
            start=0,
            num_frames=200,
            rotation=rot_a[:200],
            translation=trans_a[:200],
            depth=depth_a[:200],
            intrinsics=np.broadcast_to(intrinsics, (200, 3, 3)).copy(),
        ),
        make_window(
            index=1,
            start=160,
            num_frames=200,
            rotation=rot_b,
            translation=trans_b,
            depth=depth_a[160:360] * relative.scale,
            intrinsics=np.broadcast_to(intrinsics, (200, 3, 3)).copy(),
        ),
        rot_a,
        trans_a,
    )


@pytest.fixture(scope="session")
def synthetic_scene() -> dict[str, object]:
    """A static 3D point cloud plus a camera trajectory."""
    rng = np.random.default_rng(7)
    num_points = 300
    points = np.stack(
        [
            rng.uniform(-0.6, 0.6, num_points),
            rng.uniform(-0.5, 0.5, num_points),
            rng.uniform(1.2, 2.4, num_points),
        ],
        axis=-1,
    )
    rot, trans = make_pose_sequence(360, seed=3)
    return {
        "points": points,
        "rotation_c2w": rot,
        "translation_c2w": trans,
        "intrinsics": make_intrinsics(96, 72),
        "height": 72,
        "width": 96,
    }


@pytest.fixture(scope="session")
def known_sim3() -> Sim3:
    from scipy.spatial.transform import Rotation

    return Sim3(
        scale=1.35,
        rotation=Rotation.from_euler("xyz", [12.0, -7.0, 25.0], degrees=True).as_matrix(),
        translation=np.array([0.4, -0.25, 1.1]),
    )

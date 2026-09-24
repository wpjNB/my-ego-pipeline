#!/usr/bin/env python
"""A GPU-free demo of the geometry half of the pipeline.

Builds a synthetic scene, renders two overlapping "VGGT-Omega windows" that
disagree by a known Sim(3), then runs the real stitcher and writes the Phase-4
debug visualisations to ``outputs/demo``:

    03_camera_stitch.png   ground-truth vs stitched camera trajectory
    01_detection.mp4       synthetic RGB + hand boxes
    02_hawor.mp4           synthetic RGB + projected 3D hand skeleton

This makes Phases 4-6 and the visualisation stack demonstrable on a laptop while
the model backends (Phases 1-3) wait for the GPU server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.stitch import stitch_camera_windows  # noqa: E402
from ego3d_action.camera.window import CameraWindow, WindowRange  # noqa: E402
from ego3d_action.geometry.sim3 import Sim3  # noqa: E402
from ego3d_action.geometry.transforms import invert_rigid, rotation_angle  # noqa: E402
from ego3d_action.visualization.overlay import (  # noqa: E402
    write_detection_video,
    write_hand_video,
)

WIDTH, HEIGHT = 320, 240
NUM_FRAMES = 360


def intrinsics(width: int = WIDTH, height: int = HEIGHT, fov_deg: float = 60.0) -> np.ndarray:
    focal = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def camera_path(num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """A smooth oscillatory c2w trajectory that keeps a scene at z~1.8 m in view."""
    from scipy.spatial.transform import Rotation

    rot = np.zeros((num_frames, 3, 3))
    trans = np.zeros((num_frames, 3))
    for index in range(num_frames):
        rot[index] = Rotation.from_euler(
            "yx", [0.18 * np.sin(0.05 * index), 0.06 * np.sin(0.03 * index + 1.0)]
        ).as_matrix()
        trans[index] = [
            0.15 * np.sin(0.04 * index),
            0.10 * np.sin(0.03 * index + 0.4),
            0.10 * np.sin(0.02 * index + 1.7),
        ]
    return rot, trans


def render_depth(
    points: np.ndarray, rot: np.ndarray, trans: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """Z-buffer splat of a point cloud into a depth map."""
    depth = np.full((HEIGHT, WIDTH), np.nan)
    cam = (points - trans) @ rot
    visible = cam[:, 2] > 1e-6
    cam = cam[visible]
    z = cam[:, 2]
    u = k[0, 0] * cam[:, 0] / z + k[0, 2]
    v = k[1, 1] * cam[:, 1] / z + k[1, 2]
    for uu, vv, zz in zip(u, v, z, strict=False):
        cu, cv = int(round(uu)), int(round(vv))
        if not (0 <= cu < WIDTH and 0 <= cv < HEIGHT):
            continue
        r0, r1 = max(0, cv - 6), min(HEIGHT, cv + 7)
        c0, c1 = max(0, cu - 6), min(WIDTH, cu + 7)
        patch = depth[r0:r1, c0:c1]
        depth[r0:r1, c0:c1] = np.where(np.isnan(patch) | (zz < patch), zz, patch)
    return depth


def build_windows(
    num_frames: int = NUM_FRAMES,
) -> tuple[list[CameraWindow], np.ndarray, np.ndarray, Sim3]:
    """Two overlapping windows; window B lives in a scaled local frame."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(11)
    points = np.stack(
        [
            rng.uniform(-0.6, 0.6, 600),
            rng.uniform(-0.5, 0.5, 600),
            rng.uniform(1.2, 2.4, 600),
        ],
        axis=-1,
    )
    k = intrinsics()
    rot, trans = camera_path(num_frames)
    r0_inv, t0_inv = invert_rigid(rot[0], trans[0])
    rot_a = np.einsum("ij,tjk->tik", r0_inv, rot)
    trans_a = np.einsum("ij,tj->ti", r0_inv, trans) + t0_inv

    depth_a = np.stack([render_depth(points, rot[t], trans[t], k) for t in range(num_frames)])

    # The second window "disagrees" by a known Sim(3): rotation, translation and
    # a 35 % metric-scale error - exactly what the stitcher has to undo.
    relative = Sim3(
        scale=1.35,
        rotation=Rotation.from_euler("xyz", [12.0, -7.0, 25.0], degrees=True).as_matrix(),
        translation=np.array([0.4, -0.25, 1.1]),
    )
    rot_b, trans_b = relative.transform_poses(rot_a[160:360], trans_a[160:360])

    window_a = CameraWindow(
        window=WindowRange(index=0, start=0, end=200),
        rotation_c2w=rot_a[:200],
        translation_c2w=trans_a[:200],
        intrinsics=np.broadcast_to(k, (200, 3, 3)).copy(),
        depth=depth_a[:200],
    )
    window_b = CameraWindow(
        window=WindowRange(index=1, start=160, end=360),
        rotation_c2w=rot_b,
        translation_c2w=trans_b,
        intrinsics=np.broadcast_to(k, (200, 3, 3)).copy(),
        depth=depth_a[160:360] * relative.scale,
    )
    return [window_a, window_b], rot_a, trans_a, relative


def draw_frames(directory: Path, num_frames: int) -> list[Path]:
    """Render a simple synthetic RGB sequence."""
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(num_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (0, HEIGHT - 90), (WIDTH, HEIGHT), (70, 60, 55), -1)
        shift = int(20 * np.sin(0.05 * index))
        cv2.circle(frame, (WIDTH // 2 + shift, HEIGHT // 2 - 30), 40, (110, 110, 110), -1)
        path = directory / f"{index:06d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def plot_stitch(stitched, ground_truth_translation: np.ndarray, out_path: Path) -> Path:
    """3D plot: ground truth vs stitched camera-centre trajectory."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection="3d")
    gt = ground_truth_translation
    axes.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="#2c3e50", lw=2.2, label="ground truth")
    axes.plot(
        stitched.translation_c2w[:, 0],
        stitched.translation_c2w[:, 1],
        stitched.translation_c2w[:, 2],
        color="#e67e22",
        lw=1.3,
        ls="--",
        label="stitched (depth-derived Sim(3))",
    )
    axes.scatter(
        *stitched.translation_c2w[0],
        color="#27ae60",
        s=45,
        label="World-0",
    )
    start = len(stitched.translation_c2w) - 200
    axes.scatter(
        *stitched.translation_c2w[start],
        color="#8e44ad",
        marker="^",
        s=50,
        label="window 1 first frame",
    )
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_zlabel("z [m]")
    axes.set_title("camera trajectory: stitched vs ground truth")
    axes.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="GPU-free synthetic demo")
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument("--frames", type=int, default=120, help="frames for the overlay videos")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows, rot_a, trans_a, relative = build_windows()
    print(f"window A: {windows[0].name}   window B: {windows[1].name}")
    print(
        f"injected mismatch: scale={relative.scale:.3f}, "
        f"rotation={np.degrees(rotation_angle(relative.rotation)):.1f} deg, "
        f"|t|={np.linalg.norm(relative.translation):.3f} m"
    )

    stitched = stitch_camera_windows(windows, num_frames=NUM_FRAMES, stride=8, random_state=0)
    for diag in stitched.diagnostics:
        print(
            f"  aligned {diag.src_window} -> {diag.dst_window}: scale={diag.scale:.4f}, "
            f"inliers={100.0 * diag.inlier_ratio:.1f}%, "
            f"rmse={1000.0 * diag.inlier_rmse:.3f} mm"
        )

    error = np.linalg.norm(stitched.translation_c2w - trans_a, axis=-1)
    rotation_error = np.degrees(
        rotation_angle(np.einsum("tji,tjk->tik", stitched.rotation_c2w, rot_a))
    )
    print(
        f"stitched vs ground truth: max translation error {1000.0 * error.max():.3f} mm, "
        f"max rotation error {rotation_error.max():.4f} deg"
    )

    plot_stitch(stitched, trans_a, out_dir / "03_camera_stitch.png")
    print(f"wrote {out_dir / '03_camera_stitch.png'}")

    from ego3d_action.visualization.overlay import require_cv2

    require_cv2()
    frames = draw_frames(out_dir / "frames", args.frames)

    rng = np.random.default_rng(5)
    boxes = np.zeros((args.frames, 2, 4))
    confidence = np.clip(0.75 + 0.2 * rng.random((args.frames, 2)), 0.0, 1.0)
    valid = np.ones((args.frames, 2), dtype=bool)
    valid[30:34, 1] = False  # a deliberate gap: missing stays missing
    for index in range(args.frames):
        for hand, (x, y) in enumerate(((60, 90), (200, 110))):
            jitter = 6 * np.sin(0.1 * index + hand)
            boxes[index, hand] = [x + jitter, y + jitter, x + 70 + jitter, y + 60 + jitter]
    write_detection_video(frames, boxes, confidence, valid, out_dir / "01_detection.mp4", fps=30.0)

    k = intrinsics()
    joints = np.zeros((args.frames, 2, 21, 3))
    offsets = np.zeros((21, 3))
    offsets[:, 0] = np.linspace(0.0, 0.09, 21)
    offsets[:, 1] = 0.03 * np.sin(np.linspace(0.0, 3.0, 21))
    for hand in range(2):
        joints[:, hand] = np.array([-0.12 + 0.24 * hand, 0.0, 1.3]) + offsets
    joints[..., 0] += 0.05 * np.sin(0.1 * np.arange(args.frames))[:, None, None]
    write_hand_video(
        frames,
        joints,
        np.broadcast_to(k, (args.frames, 3, 3)).copy(),
        valid,
        out_dir / "02_hawor.mp4",
        fps=30.0,
    )
    print(f"wrote {out_dir / '01_detection.mp4'} and {out_dir / '02_hawor.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

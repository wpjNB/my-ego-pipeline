#!/usr/bin/env python
"""Visualise a real VGGT-Omega Phase 3 run.

Turns the camera windows (and their Phase 4 stitching) into things you can
actually look at: a back-projected metric point cloud with the camera trajectory
and frusta, an RGB-vs-depth strip, a rotating 3D turntable, a depth-overlay
video, and a scale sanity check that fits the ground plane and reports how high
the camera ended up.

    python scripts/render_vggt_reconstruction.py \
        --windows data/vggt_smoke/camera/windows_overlap \
        --frames data/vggt_smoke/frames \
        --num-frames 24 --window 4 --overlap 2 \
        --out-dir outputs/vggt_real

Every number comes from the stored windows; nothing is synthesised. The scale
check is the useful part: VGGT depth is metric, so a fitted road plane that puts
the camera somewhere absurd (10 m, 5 cm) means the depth scale is wrong even
though the arrays look fine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.stitch import stitch_camera_windows  # noqa: E402
from ego3d_action.camera.window import load_camera_window, make_windows  # noqa: E402
from ego3d_action.visualization.overlay import transcode_to_h264  # noqa: E402


def camera_centre(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Camera centre in world coordinates from c2w extrinsics: -R^T t."""
    return -rotation.T @ translation


def backproject(
    window: object, local: int, *, row_fraction: tuple[float, float] = (0.0, 1.0),
    stride: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject one frame's depth into world space.

    ``p_cam = K^-1 [u, v, 1] * depth`` then ``p_world = R_c2w p_cam + t_c2w``.
    Depth and intrinsics already share the depth grid (the runner rescales the
    intrinsics), so no extra scaling is needed here.
    """
    height, width = window.depth.shape[1], window.depth.shape[2]
    r0 = int(round(row_fraction[0] * height))
    r1 = max(r0 + 1, int(round(row_fraction[1] * height)))
    depth = window.depth[local][r0:r1:stride, ::stride]
    vs, us = np.meshgrid(
        np.arange(r0, r1, stride), np.arange(0, width, stride), indexing="ij"
    )
    pixels = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], axis=1)
    rays = pixels @ np.linalg.inv(window.intrinsics[local]).T
    cam_points = rays * depth.ravel()[:, None]
    world = cam_points @ window.rotation_c2w[local].T + window.translation_c2w[local]
    return world, depth.ravel()


def collect_points(windows, ranges, *, stride: int, depth_range: tuple[float, float]):
    """Back-project every frame; also gather a 'ground' subset from low rows."""
    points, depths, ground = [], [], []
    for window, rng in zip(windows, ranges):
        for local in range(window.depth.shape[0]):
            world, depth = backproject(window, local, stride=stride)
            keep = (depth > depth_range[0]) & (depth < depth_range[1])
            points.append(world[keep])
            depths.append(depth[keep])
            low, low_depth = backproject(
                window, local, row_fraction=(0.72, 1.0), stride=max(2, stride)
            )
            low_keep = (low_depth > depth_range[0]) & (low_depth < depth_range[1])
            ground.append(low[low_keep])
    return (
        np.concatenate(points),
        np.concatenate(depths),
        np.concatenate(ground),
    )


def draw_frustum(axes, centre, rotation, intrinsics, *, length: float = 0.7) -> None:
    """Draw a camera frustum: 4 rays from the centre plus the far rectangle."""
    height, width = 320, 560
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    fx, fy, cx, cy = (
        intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2],
    )
    rays = np.stack(
        [(corners[:, 0] - cx) / fx, (corners[:, 1] - cy) / fy, np.ones(4)], axis=1
    )
    for ray in rays:
        apex_to_far = np.stack([centre, centre + rotation @ (ray * length)])
        axes.plot(*apex_to_far.T, color="#e74c3c", lw=0.7, alpha=0.7)
    loop = list(range(4)) + [0]
    far = np.stack([centre + rotation @ (rays[i] * length) for i in loop])
    axes.plot(far[:, 0], far[:, 1], far[:, 2], color="#e74c3c", lw=1.0, alpha=0.9)


def render_pointcloud(plt, points, depths, centres, rotations, intrinsics, out_path):
    """Static 3D view: point cloud + stitched trajectory + frusta."""
    sample = np.random.default_rng(0).choice(
        len(points), min(120_000, len(points)), replace=False
    )
    pts, dep = points[sample], depths[sample]

    figure = plt.figure(figsize=(16, 11))
    grid = figure.add_gridspec(2, 2, height_ratios=[2.1, 1])

    axes = figure.add_subplot(grid[0, :], projection="3d")
    scatter = axes.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2], c=dep, s=0.25, cmap="turbo",
        alpha=0.35, linewidths=0,
    )
    axes.plot(
        centres[:, 0], centres[:, 1], centres[:, 2], color="black", lw=2.2,
        marker="o", ms=3.5, label="camera centres (stitched)",
    )
    for index in range(0, len(centres), max(1, len(centres) // 5)):
        draw_frustum(axes, centres[index], rotations[index], intrinsics[index])
    axes.scatter(*centres[0], color="lime", s=140, marker="*", zorder=5, label="World-0")
    axes.set_xlabel("x [m]"); axes.set_ylabel("y [m]"); axes.set_zlabel("z [m]")
    axes.set_title(
        f"VGGT-Omega reconstruction — {len(pts):,} back-projected points "
        f"+ {len(centres)} stitched cameras", fontsize=13,
    )
    axes.legend(loc="upper right")
    axes.view_init(elev=18, azim=-62)
    figure.colorbar(scatter, ax=axes, shrink=0.55, label="depth [m]")

    top = figure.add_subplot(grid[1, 0])
    top.scatter(pts[:, 0], pts[:, 1], c=dep, s=0.15, cmap="turbo", alpha=0.25,
                linewidths=0)
    top.plot(centres[:, 0], centres[:, 1], color="black", lw=2, marker="o", ms=3)
    for index in range(0, len(centres), max(1, len(centres) // 5)):
        forward = rotations[index] @ np.array([0, 0, 1.0])
        top.arrow(centres[index, 0], centres[index, 1],
                  forward[0] * 0.25, forward[1] * 0.25,
                  color="#e74c3c", head_width=0.012, lw=1.1)
    top.scatter(*centres[0][:2], color="lime", s=120, marker="*", zorder=5)
    top.set_xlabel("x [m]"); top.set_ylabel("y [m]")
    top.set_title("top-down (XY): trajectory + viewing directions")
    top.set_aspect("equal"); top.grid(alpha=0.3)

    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    motion = figure.add_subplot(grid[1, 1])
    motion.bar(np.arange(1, len(steps) + 1), steps, color="#3498db")
    motion.axhline(steps.mean(), color="#e74c3c", ls="--",
                   label=f"mean {steps.mean():.4f} m")
    motion.set_xlabel("frame transition"); motion.set_ylabel("camera motion [m]")
    motion.set_title("per-frame camera displacement (continuity check)")
    motion.legend(); motion.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(out_path, dpi=115)
    plt.close(figure)
    return out_path


def render_turntable(plt, cv2, Image, points, depths, centres, rotations, intrinsics,
                     out_dir, *, frames: int = 60) -> list[Path]:
    """Rotating 3D view written as MP4 + GIF."""
    sample = np.random.default_rng(0).choice(
        len(points), min(45_000, len(points)), replace=False
    )
    pts, dep = points[sample], depths[sample]
    low, high = pts.min(0), pts.max(0)
    middle, span = (low + high) / 2, (high - low).max() * 0.62

    writer = cv2.VideoWriter(
        str(out_dir / "vggt_pointcloud_rotate.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), 18, (960, 720),
    )
    stills = []
    for index, azimuth in enumerate(np.linspace(-62, 298, frames)):
        figure = plt.figure(figsize=(9.6, 7.2), dpi=100)
        axes = figure.add_subplot(111, projection="3d")
        axes.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=dep, s=0.35, cmap="turbo",
                     alpha=0.5, linewidths=0)
        axes.plot(centres[:, 0], centres[:, 1], centres[:, 2], color="black",
                  lw=2.5, marker="o", ms=4)
        for i in range(0, len(centres), max(1, len(centres) // 5)):
            draw_frustum(axes, centres[i], rotations[i], intrinsics[i])
        axes.scatter(*centres[0], color="lime", s=180, marker="*", zorder=6)
        axes.set_xlim(middle[0] - span, middle[0] + span)
        axes.set_ylim(middle[1] - span, middle[1] + span)
        axes.set_zlim(middle[2] - span, middle[2] + span)
        axes.set_xlabel("x [m]"); axes.set_ylabel("y [m]"); axes.set_zlabel("z [m]")
        axes.set_title(
            f"VGGT-Omega reconstruction  ·  {len(points):,} pts  ·  "
            f"azimuth {azimuth:.0f}°", fontsize=11,
        )
        axes.view_init(elev=20, azim=azimuth)
        figure.tight_layout()
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if index % 3 == 0:
            stills.append(Image.fromarray(frame).resize((640, 480)))
        plt.close(figure)
    writer.release()
    transcode_to_h264(out_dir / "vggt_pointcloud_rotate.mp4")
    gif = out_dir / "vggt_pointcloud_rotate.gif"
    stills[0].save(gif, save_all=True, append_images=stills[1:], duration=180, loop=0)
    return [out_dir / "vggt_pointcloud_rotate.mp4", gif]


def render_depth_video(cv2, windows, ranges, frames_dir, num_frames, out_dir):
    """RGB | predicted depth | overlay, one panel per frame."""
    if cv2 is None:
        return None
    writer = cv2.VideoWriter(
        str(out_dir / "vggt_depth_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), 6, (560 * 3, 320),
    )
    for frame in range(num_frames):
        window = local = None
        for candidate, rng in zip(windows, ranges):
            if rng.start <= frame < rng.end:
                window, local = candidate, frame - rng.start
                break
        if window is None:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(frames_dir / f"{frame:06d}.jpg")),
                           cv2.COLOR_BGR2RGB)
        if rgb is None:
            continue
        rgb = cv2.resize(rgb, (560, 320))
        depth = cv2.resize(window.depth[local], (560, 320))
        # numpy>=2 removed ndarray.ptp(); the free function survives.
        span = max(float(np.ptp(depth)), 1e-9)
        heat = cv2.applyColorMap(
            ((depth - depth.min()) / span * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        blend = cv2.addWeighted(rgb, 0.45, heat, 0.55, 0)
        panel = np.hstack([rgb, heat, blend])
        for offset, text in (
            (12, f"frame {frame:06d}  RGB"),
            (572, f"predicted depth  {depth.min():.2f}-{depth.max():.2f} m"),
            (1132, "overlay"),
        ):
            cv2.putText(panel, text, (offset, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    writer.release()
    return transcode_to_h264(out_dir / "vggt_depth_overlay.mp4")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True, help="directory of stored windows")
    parser.add_argument("--frames", required=True, help="extracted RGB frames")
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--out-dir", default="outputs/vggt_real")
    parser.add_argument("--stride", type=int, default=3, help="depth subsampling")
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--no-video", action="store_true", help="skip the turntable/video")
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install 'ego3d-action[viz]'", file=sys.stderr)
        return 1

    cv2 = Image = None
    if not args.no_video:
        try:
            import cv2  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            print("opencv/pillow unavailable; skipping video output", file=sys.stderr)
            cv2 = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = Path(args.windows)
    ranges = make_windows(args.num_frames, window=args.window, overlap=args.overlap)
    windows = [
        load_camera_window(windows_dir / f"{r.start:06d}_{r.end - 1:06d}.npz")
        for r in ranges
    ]
    print(f"loaded {len(windows)} window(s) from {windows_dir}")

    stitched = stitch_camera_windows(windows, num_frames=args.num_frames)
    rotations, translations = stitched.rotation_c2w, stitched.translation_c2w
    centres = np.stack([camera_centre(rotations[i], translations[i])
                        for i in range(len(rotations))])
    print(f"stitched {stitched.num_frames} frames, coverage {stitched.coverage:.0%}")

    points, depths, ground = collect_points(
        windows, ranges, stride=args.stride, depth_range=(0.1, args.max_depth)
    )
    print(f"back-projected {len(points):,} points ({len(ground):,} ground candidates)")

    # --- scale sanity: a fitted ground plane says how high the camera really is
    if len(ground) > 1000:
        centroid = ground.mean(axis=0)
        normal = np.linalg.svd(ground - centroid, full_matrices=False)[2][-1]
        heights = np.array([abs(normal @ (c - centroid)) for c in centres])
        median = float(np.median(heights))
        plausible = 0.1 < median < 2.5
        print(
            f"ground plane normal {np.round(normal, 3)}, "
            f"camera height median {median:.3f} m "
            f"(range {heights.min():.3f}-{heights.max():.3f}) -> "
            f"{'plausible' if plausible else 'SUSPICIOUS: depth scale may be wrong'}"
        )

    written = [
        render_pointcloud(plt, points, depths, centres, rotations, stitched.intrinsics,
                          out_dir / "vggt_reconstruction.png")
    ]
    if cv2 is not None and Image is not None:
        written += render_turntable(plt, cv2, Image, points, depths, centres,
                                   rotations, stitched.intrinsics, out_dir)
        video = render_depth_video(cv2, windows, ranges, Path(args.frames),
                                   args.num_frames, out_dir)
        if video is not None:
            written.append(video)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
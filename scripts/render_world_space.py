#!/usr/bin/env python
"""World-space visualisation: camera trajectory + both hands (blog style).

Renders the reference system's "WORLD SPACE" panel: both MANO hands and the
camera trajectory in the stitched world frame, on a dark grid, in centimetres,
with wrist trails, drop lines, an axes triad, wrist-height labels and a 10 cm
scale bar. Static PNG by default; ``--video`` additionally animates the trails
growing over time with the current hands and camera pose per frame.

Everything comes from the on-disk trajectory contract
(``trajectory/trajectory.npz``: ``hand_xyz_world``, ``hand_valid``,
``camera_R_c2w``, ``camera_t_c2w``); ``--gt`` overlays the reference the same
way. No GPU and no backend involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.hand.mano import bone_pairs  # noqa: E402
from ego3d_action.io.serialization import load_npz  # noqa: E402
from ego3d_action.visualization.overlay import transcode_to_h264  # noqa: E402

BACKGROUND = "#111418"
PANE = "#181c22"
GRID = "#3a414b"
LEFT_COLOUR = "#4fd8e8"  # cyan, matching the reference panel
RIGHT_COLOUR = "#f0d848"  # yellow
CAMERA_COLOUR = "#e88ad2"  # pink camera path
GT_ALPHA = 0.45
SCALE_BAR_M = 0.10
AXIS_LEN_M = 0.20
TRAIL_STRIDE = 2  # subsample the trails (they are dense at 30 fps)
BONE_PAIRS = list(bone_pairs())
SKELETON_EVERY = 10  # hand-fan density along the trail


def _load(name: str, path: str | Path) -> dict[str, np.ndarray]:
    data = load_npz(
        path,
        required=("hand_xyz_world", "hand_valid", "camera_R_c2w", "camera_t_c2w"),
    )
    return {
        "joints": np.asarray(data["hand_xyz_world"], dtype=np.float64),
        "valid": np.asarray(data["hand_valid"], dtype=bool),
        "R": np.asarray(data["camera_R_c2w"], dtype=np.float64),
        "t": np.asarray(data["camera_t_c2w"], dtype=np.float64),
        "name": name,
    }


def _style_axes(axes) -> None:
    """Dark panes (set_pane_color is the only call that sticks on 3d axes)."""
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color((0.094, 0.11, 0.133, 1.0))
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)


def _ranges(points: list[np.ndarray]) -> tuple[float, float, float]:
    lows = np.min(np.stack([np.nanmin(p, axis=0) for p in points]), axis=0)
    highs = np.max(np.stack([np.nanmax(p, axis=0) for p in points]), axis=0)
    span = np.maximum(highs - lows, 0.3)
    centre = (highs + lows) / 2
    half = np.maximum(span / 2, 0.25)
    return tuple((centre[i] - half[i], centre[i] + half[i]) for i in range(3))


def _draw_scene(axes, tracks: list[dict], *, upto: int | None = None,
                ghosts: bool = True, drop_lines: bool = True) -> None:
    """Trails, skeletons, camera path, triad, scale bar — in centimetres."""
    floor_z = min(
        min(float(np.nanmin(tr["joints"][:upto, :, :, 2])),
            float(np.nanmin(tr["t"][:upto, 2]))) for tr in tracks
    ) - 0.05  # a little headroom below the lowest point

    for tr in tracks:
        alpha = GT_ALPHA if tr["name"] == "GT" else 1.0
        colours = {0: LEFT_COLOUR, 1: RIGHT_COLOUR}
        # camera path
        cam = tr["t"][:upto] * 100.0
        axes.plot(cam[:, 0], cam[:, 1], cam[:, 2], color=CAMERA_COLOUR,
                  lw=1.6, alpha=alpha)
        axes.scatter(*cam[-1], color=CAMERA_COLOUR, s=28, alpha=alpha, zorder=5)
        # wrist trails + ghost skeletons + drop lines per hand
        for hand, colour in colours.items():
            valid_t = np.where(tr["valid"][:upto, hand])[0]
            if len(valid_t) < 2:
                continue
            wrist = tr["joints"][valid_t, hand, 0, :] * 100.0
            axes.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2], color=colour,
                      lw=1.1, alpha=0.85 * alpha)
            if ghosts:
                # the hand fan: a skeleton every SKELETON_EVERY frames, clearly
                # visible - this is the "hand information" of the panel
                for t in valid_t[::SKELETON_EVERY]:
                    joints = tr["joints"][t, hand] * 100.0
                    for parent, child in BONE_PAIRS:
                        axes.plot(*zip(joints[parent], joints[child]),
                                  color=colour, lw=0.9, alpha=0.45 * alpha)
            # the hero: the most recent valid hand, thick, with joint dots
            last_t = valid_t[-1]
            joints = tr["joints"][last_t, hand] * 100.0
            for parent, child in BONE_PAIRS:
                axes.plot(*zip(joints[parent], joints[child]),
                          color=colour, lw=2.6, alpha=alpha, zorder=6)
            axes.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                         color=colour, s=10, alpha=alpha, zorder=7)
            axes.scatter(*joints[0], color=colour, s=90, facecolors="none",
                         edgecolors=colour, linewidths=1.6, alpha=alpha, zorder=7)
            # drop line from the last valid wrist to the floor
            last = tr["joints"][valid_t[-1], hand, 0, :] * 100.0
            if drop_lines:
                axes.plot([last[0], last[0]], [last[1], last[1]],
                          [last[2], floor_z * 100.0], color=colour, lw=0.7,
                          ls="--", alpha=0.6 * alpha)
            label = f"{'L' if hand == 0 else 'R'} {last[2]:.1f} cm"
            axes.text(last[0], last[1], last[2] + 3.0, label, color=colour,
                      fontsize=7, alpha=max(alpha, 0.8))
    # axes triad at the world origin
    for direction, colour, label in (
        ((1, 0, 0), "#ff5d5d", "+X"), ((0, 1, 0), "#7ee87e", "+Y"),
        ((0, 0, 1), "#6da8ff", "+Z"),
    ):
        tip = AXIS_LEN_M * 100.0 * np.array(direction)
        axes.plot(*zip(np.zeros(3), tip), color=colour, lw=1.4)
        axes.text(*tip, label, color=colour, fontsize=7)
    # scale bar: 10 cm along +x at the floor corner
    z0 = floor_z * 100.0
    axes.plot([0, SCALE_BAR_M * 100.0], [0, 0], [z0, z0], color="#c8ced6", lw=2.0)
    axes.text(SCALE_BAR_M * 50.0, 0, z0 + 2.0, "10 cm", color="#c8ced6", fontsize=7)


def _limits(tracks: list[dict], upto: int | None) -> tuple[tuple, tuple, tuple]:
    points = []
    for tr in tracks:
        points.append(tr["joints"][:upto][tr["valid"][:upto]].reshape(-1, 3) * 100.0)
        points.append(tr["t"][:upto] * 100.0)
    return _ranges(points)


def _wrist_error_stats(tracks: list[dict]) -> list[float]:
    """Median |PRED-GT| wrist distance in cm, when GT is overlaid."""
    if len(tracks) < 2 or {tr["name"] for tr in tracks} != {"PRED", "GT"}:
        return []
    pred, gt = tracks[0], tracks[1]
    total = min(pred["joints"].shape[0], gt["joints"].shape[0])
    common = pred["valid"][:total] & gt["valid"][:total]
    distances = []
    for hand in (0, 1):
        ok = common[:, hand]
        if ok.any():
            d = np.linalg.norm(
                pred["joints"][:total][ok, hand, 0] - gt["joints"][:total][ok, hand, 0],
                axis=1,
            )
            distances.extend((d * 100.0).tolist())
    return distances


def render_static(tracks: list[dict], out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    total = max(tr["joints"].shape[0] for tr in tracks)
    x_lim, y_lim, z_lim = _limits(tracks, None)
    figure = plt.figure(figsize=(13.5, 6.4), facecolor=BACKGROUND)
    for slot, (elev, azim) in enumerate(((22, -60), (90, -90))):
        axes = figure.add_subplot(1, 2, slot + 1, projection="3d", facecolor=BACKGROUND)
        _style_axes(axes)
        _draw_scene(axes, tracks)
        axes.set_xlim(*x_lim); axes.set_ylim(*y_lim); axes.set_zlim(*z_lim)
        span = (x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
        axes.set_box_aspect(span)
        axes.view_init(elev=elev, azim=azim)
        axes.set_title("3D" if slot == 0 else "top-down (XY)", color="#c8ced6", fontsize=9)
    header = " | ".join(tr["name"] for tr in tracks)
    stats = _wrist_error_stats(tracks)
    if stats:
        header += (
            f"  |  wrist |PRED-GT| median {np.median(stats):.1f} cm"
            f", p90 {np.percentile(stats, 90):.1f} cm"
        )
    figure.suptitle(
        f"WORLD SPACE - camera + both hands  [{header}]  |  cm  |  grid 5 cm",
        color="#e8ecf0", fontsize=11,
    )
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, facecolor=BACKGROUND)
    plt.close(figure)
    return out_path


def render_video(tracks: list[dict], out_path: Path, *, fps: float,
                 video_stride: int) -> Path:
    import cv2
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    total = max(tr["joints"].shape[0] for tr in tracks)
    x_lim, y_lim, z_lim = _limits(tracks, None)
    span = (x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
    size = (960, 720)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), size)
    if not writer.isOpened():
        raise StageIOError(f"cannot open a video writer for {out_path} (codec mp4v)")
    for t in range(1, total + 1, video_stride):
        figure = plt.figure(figsize=(9.6, 7.2), dpi=100, facecolor=BACKGROUND)
        axes = figure.add_subplot(111, projection="3d", facecolor=BACKGROUND)
        _style_axes(axes)
        _draw_scene(axes, tracks, upto=t)
        axes.set_xlim(*x_lim); axes.set_ylim(*y_lim); axes.set_zlim(*z_lim)
        axes.set_box_aspect(span)
        axes.view_init(elev=22, azim=-60)
        axes.set_title(f"WORLD SPACE  |  frame {t}/{total}  |  cm", color="#e8ecf0", fontsize=10)
        figure.tight_layout()
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        plt.close(figure)
    writer.release()
    return transcode_to_h264(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "world-space render")
    parser.add_argument("--prediction", default=None, help="trajectory npz (default: the clip's)")
    parser.add_argument("--gt", action="store_true", help="overlay the reference trails")
    parser.add_argument("--video", action="store_true", help="also render the time animation")
    parser.add_argument("--video-stride", type=int, default=2, help="every Nth frame in the video")
    args = parser.parse_args(argv)
    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        prediction = args.prediction or layout.trajectory_dir / "trajectory.npz"
        tracks = [_load("PRED", prediction)]
        if args.gt:
            tracks.append(_load("GT", layout.trajectory_dir / "ground_truth.npz"))
        viz = layout.visualization_dir
        png = render_static(tracks, viz / "world_space.png")
        print(f"wrote {png}")
        if args.video:
            fps = 30.0 / max(1, args.video_stride)
            video = render_video(tracks, viz / "world_space_time.mp4",
                                 fps=fps, video_stride=args.video_stride)
            print(f"wrote {video}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

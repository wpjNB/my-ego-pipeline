#!/usr/bin/env python
"""World-space visualisation: camera trajectory + both hands (blog style).

Renders the reference system's "WORLD SPACE" panel in the fixed world frame
(X right / Y forward / Z up), in centimetres: a 10 cm ground grid, the dashed
camera path with the current 视线/+Zc · 右/+Xc · 上/-Yc pose axes, the current
hands with drop lines and hand-to-camera distance labels, a legend, a frame
counter, a 10 cm scale bar and the X/Y/Z triad screen-anchored in the
top-right corner. Static PNG by default; ``--video`` additionally animates the
camera path growing over time with the current hands and camera pose per frame.

The trajectory contract's world frame is *the first frame's camera frame*
(CV convention: x right, y down, z forward), so the panel canonicalises it for
display with ``(x, y, z) -> (x, z, -y)``: forward becomes +Y and physical up
(-Y_cam) becomes +Z, matching the reference panel's 固定世界系 Z-up layout.

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
CAMERA_AXIS_COLOUR = "#f0a848"  # orange camera axes (right/up)
GT_ALPHA = 0.45
SCALE_BAR_M = 0.10
CAMERA_AXIS_CM = 12.0  # length of the current-pose camera axes
BONE_PAIRS = list(bone_pairs())


def _cjk_font() -> None:
    """The panel labels are Chinese; JP covers the simplified glyphs we use."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _to_fixed_frame(points: np.ndarray) -> np.ndarray:
    """Contract world frame (X右/Y下/Z前) -> fixed display frame (X右/Y前/Z上)."""
    return np.stack([points[..., 0], points[..., 2], -points[..., 1]], axis=-1)


def _load(name: str, path: str | Path) -> dict[str, np.ndarray]:
    data = load_npz(
        path,
        required=("hand_xyz_world", "hand_valid", "camera_R_c2w", "camera_t_c2w"),
    )
    # ``camera_R_c2w``'s columns are the camera axes *in the contract frame*;
    # rotating them the same way keeps 视线/+Zc etc. correct on screen.
    rotation = np.asarray(data["camera_R_c2w"], dtype=np.float64)
    return {
        "joints": _to_fixed_frame(np.asarray(data["hand_xyz_world"], dtype=np.float64)),
        "valid": np.asarray(data["hand_valid"], dtype=bool),
        "R": _to_fixed_frame(rotation),
        "t": _to_fixed_frame(np.asarray(data["camera_t_c2w"], dtype=np.float64)),
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


def _floor_grid(axes, tracks: list[dict], z0: float) -> None:
    """10 cm ground grid on the floor plane, spanning the fixed axes range."""
    x_lim, y_lim, _ = _limits(tracks, None)
    step = 10.0
    xs = np.arange(np.floor(x_lim[0] / step) * step, x_lim[1] + step, step)
    ys = np.arange(np.floor(y_lim[0] / step) * step, y_lim[1] + step, step)
    for x in xs:
        axes.plot([x, x], [y_lim[0], y_lim[1]], [z0, z0], color=GRID, lw=0.5, alpha=0.45)
    for y in ys:
        axes.plot([x_lim[0], x_lim[1]], [y, y], [z0, z0], color=GRID, lw=0.5, alpha=0.45)


def _camera_axes(axes, track: dict, cur: int) -> None:
    """The current camera pose: pink view axis (+Zc) and orange right/up."""
    c = track["t"][cur] * 100.0
    rot = track["R"][cur]
    for column, sign, colour, lw, label in (
        (2, 1.0, CAMERA_COLOUR, 2.2, "视线/+Zc"),
        (0, 1.0, CAMERA_AXIS_COLOUR, 1.3, "右/+Xc"),
        (1, -1.0, CAMERA_AXIS_COLOUR, 1.3, "上/-Yc"),
    ):
        direction = rot[:, column] * sign
        tip = c + CAMERA_AXIS_CM * direction / max(np.linalg.norm(direction), 1e-12)
        axes.plot([c[0], tip[0]], [c[1], tip[1]], [c[2], tip[2]], color=colour, lw=lw)
        axes.text(tip[0], tip[1], tip[2], label, color=colour, fontsize=7)


def _corner_triad(axes, *, anchor=(0.86, 0.86), length_px: float = 34.0) -> None:
    """The world-axis (X/Y/Z) triad, screen-anchored in the top-right corner.

    The 3D axis directions are projected through the live view and drawn in
    axes-fraction coordinates, so the triad reads the current orientation
    without sitting in the scene. Call after ``view_init`` and
    ``tight_layout`` (the transforms must be final).
    """
    from mpl_toolkits.mplot3d import proj3d

    to_display = axes.transData
    inverse = axes.transAxes.inverted()
    anchor_px = np.asarray(axes.transAxes.transform(anchor))
    proj = axes.get_proj()
    origin = np.zeros(3)
    ox, oy, _ = proj3d.proj_transform(*origin, proj)
    origin_px = np.asarray(to_display.transform((ox, oy)))
    for direction, colour, label in (
        ((1, 0, 0), "#ff5d5d", "X"), ((0, 1, 0), "#7ee87e", "Y"),
        ((0, 0, 1), "#6da8ff", "Z"),
    ):
        tx, ty, _ = proj3d.proj_transform(*(origin + np.asarray(direction, float)), proj)
        tip_px = np.asarray(to_display.transform((tx, ty)))
        delta = tip_px - origin_px
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            continue  # an axis pointing straight at the camera has no direction
        tip = inverse.transform(anchor_px + delta / norm * length_px)
        axes.annotate("", xy=tip, xytext=anchor, xycoords="axes fraction",
                      textcoords="axes fraction", annotation_clip=False,
                      arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.5,
                                  "shrinkA": 0.0, "shrinkB": 0.0})
        axes.text2D(float(tip[0]) + 0.02, float(tip[1]) + 0.01, label,
                    transform=axes.transAxes, color=colour, fontsize=8, fontweight="bold")


def _draw_scene(axes, tracks: list[dict], *, upto: int | None = None,
                drop_lines: bool = True) -> None:
    """Floor grid, current hands, camera path + pose axes, labels.

    Fixed-frame panel in the reference system's style: Z-up, 10 cm ground grid,
    dashed camera path with the current 视线/+Zc · 右/+Xc · 上/-Yc axes, the
    current hand skeletons with drop lines, hand-to-camera distance labels, a
    legend and a frame counter — all in centimetres. A hand is drawn only on
    frames where that side is valid, and hand trails are not drawn at all: on
    this footage they read as a wire snarl (the reference panel shows none).
    """
    _cjk_font()
    # the floor plane is fixed over the whole clip so it does not drift as the
    # video reveals lower points
    floor_z = min(
        min(float(np.nanmin(tr["joints"][..., 2])), float(np.nanmin(tr["t"][:, 2])))
        for tr in tracks
    ) - 0.05  # a little headroom below the lowest point
    z0 = floor_z * 100.0
    _floor_grid(axes, tracks, z0)
    total = max(tr["joints"].shape[0] for tr in tracks)
    cur = (upto if upto is not None else total) - 1

    for tr in tracks:
        alpha = GT_ALPHA if tr["name"] == "GT" else 1.0
        colours = {0: LEFT_COLOUR, 1: RIGHT_COLOUR}
        # camera path (dashed) and, for the primary track, the current pose axes
        cam = tr["t"][:upto] * 100.0
        axes.plot(cam[:, 0], cam[:, 1], cam[:, 2], color=CAMERA_COLOUR,
                  lw=1.4, ls="--", alpha=alpha)
        axes.scatter(*cam[-1], color=CAMERA_COLOUR, s=28, alpha=alpha, zorder=5)
        if tr is tracks[0]:
            _camera_axes(axes, tr, cur)
        # the current hand per side: thick skeleton + joint dots + drop line.
        # Strictly the *current* frame: during a dropout nothing is drawn -
        # no lingering hand from the last valid frame.
        for hand, colour in colours.items():
            if cur >= tr["valid"].shape[0] or not tr["valid"][cur, hand]:
                continue
            joints = tr["joints"][cur, hand] * 100.0
            if not np.isfinite(joints).all():
                continue
            for parent, child in BONE_PAIRS:
                axes.plot(*zip(joints[parent], joints[child]),
                          color=colour, lw=2.6, alpha=alpha, zorder=6)
            axes.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                         color=colour, s=10, alpha=alpha, zorder=7)
            axes.scatter(*joints[0], color=colour, s=90, facecolors="none",
                         edgecolors=colour, linewidths=1.6, alpha=alpha, zorder=7)
            # drop line from the wrist to the floor
            wrist = joints[0]
            if drop_lines:
                axes.plot([wrist[0], wrist[0]], [wrist[1], wrist[1]],
                          [wrist[2], z0], color=colour, lw=0.7,
                          ls="--", alpha=0.6 * alpha)
            dist = np.linalg.norm(wrist - tr["t"][cur] * 100.0)
            label = f"{'L' if hand == 0 else 'R'} {dist:.1f} cm"
            axes.text(wrist[0], wrist[1], wrist[2] + 3.0, label, color=colour,
                      fontsize=7, alpha=max(alpha, 0.8))
    # scale bar: 10 cm along +x at a free floor corner
    x_lim, y_lim, _ = _limits(tracks, None)
    bx, by = x_lim[0] + 2.0, y_lim[0] + 2.0
    axes.plot([bx, bx + SCALE_BAR_M * 100.0], [by, by], [z0, z0], color="#c8ced6", lw=2.0)
    axes.text(bx + SCALE_BAR_M * 50.0, by, z0 + 2.0, "10 cm", color="#c8ced6", fontsize=7)
    # legend and frame counter, screen-anchored like the reference panel
    for i, (name, colour) in enumerate((("左手", LEFT_COLOUR), ("右手", RIGHT_COLOUR),
                                        ("相机", CAMERA_COLOUR))):
        axes.text2D(0.02, 0.17 - i * 0.05, f"● {name}", transform=axes.transAxes,
                    color=colour, fontsize=8)
    axes.text2D(0.02, 0.03, f"帧 {cur + 1} / {total}", transform=axes.transAxes,
                color="#c8ced6", fontsize=8)


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

    x_lim, y_lim, z_lim = _limits(tracks, None)
    figure = plt.figure(figsize=(13.5, 6.4), facecolor=BACKGROUND)
    panels = []
    for slot, (elev, azim) in enumerate(((22, -60), (90, -90))):
        axes = figure.add_subplot(1, 2, slot + 1, projection="3d", facecolor=BACKGROUND)
        panels.append(axes)
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
        f"固定世界系 Z-up（X右 / Y前 / Z上）  [{header}]  |  cm  |  格 10 cm",
        color="#e8ecf0", fontsize=11,
    )
    figure.tight_layout()
    for axes in panels:
        _corner_triad(axes)
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
        axes.set_title(f"固定世界系 Z-up  |  帧 {t}/{total}  |  cm", color="#e8ecf0", fontsize=10)
        figure.tight_layout()
        _corner_triad(axes)
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
    parser.add_argument(
        "--output",
        default=None,
        help="write the PNG here (default: <clip>/visualization/world_space.png; "
        "the --video animation goes to <stem>_time.mp4 next to it)",
    )
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
        target = Path(args.output) if args.output else viz / "world_space.png"
        png = render_static(tracks, target)
        print(f"wrote {png}")
        if args.video:
            fps = 30.0 / max(1, args.video_stride)
            video = render_video(tracks, target.with_name(target.stem + "_time.mp4"),
                                 fps=fps, video_stride=args.video_stride)
            print(f"wrote {video}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

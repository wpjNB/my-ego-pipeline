#!/usr/bin/env python
"""Combined EGO | WORLD viewer, in the style of the reference system's panel.

One video, two synchronised panels per frame:

* left  - EGO VIEW: the RGB frame with the MANO mesh reprojected plus the
  21-joint skeleton on top (as ``02_hawor.mp4``) and left/right validity badges;
* right - WORLD SPACE: the stitched world frame with the camera path and both
  hands (as ``render_world_space.py``), trails growing up to the current frame.

Everything comes from the on-disk artefacts (``hand_camera.npz`` for the ego
panel, ``trajectory.npz`` for the world panel); ``--gt`` adds the reference
trails to the world panel. The robot-hand third panel of the reference viewer
is intentionally absent - it needs ``wuji-retargeting`` (MIT) plus the Wuji
URDF in MuJoCo, tracked in the docs as a follow-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_world_space as world  # the WORLD SPACE drawing reuse
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.io.serialization import load_npz  # noqa: E402
from ego3d_action.visualization.overlay import (  # noqa: E402
    SKELETON_COLOUR,
    draw_hand_mesh,
    draw_hand_projection,
    require_cv2,
    transcode_to_h264,
)


def _mano_faces(context) -> list[np.ndarray]:
    """MANO triangle indices per hand (left winding mirrored)."""
    from ego3d_action.hand.mano_model import load_mano_models

    models = load_mano_models(context.path("paths.mano_model"))
    return [
        np.asarray(models["left"].faces, dtype=np.int64),
        np.asarray(models["right"].faces, dtype=np.int64),
    ]

EGO_HEADER_BG = (28, 30, 34)  # BGR strip behind the panel headers
HEADER_HEIGHT = 26


def _ego_intrinsics(viz_dir: Path, frames_dir: Path, size: tuple[int, int]):
    """The overlay intrinsics: the clip's canonical estimate (median across
    the Phase 3 windows) - the exact matrix run_hand.resolve_focal hands to
    the reconstruction, never the reference calibration and never a single
    (possibly outlier) window."""
    from ego3d_action.camera.depth import canonical_intrinsics

    K = canonical_intrinsics(viz_dir.parent / "camera" / "windows", size)
    if K is not None:
        return K
    from ego3d_action.testing.synthetic import make_intrinsics

    return make_intrinsics(*size)


def _ego_panel(cv2, frame_path: Path, vertices, joints, intrinsics, valid, faces, *,
               target_total: int) -> np.ndarray:
    """RGB frame + MANO mesh + skeleton + validity badges, resized to panel height."""
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise StageIOError(f"cannot decode {frame_path}")
    hand_valid = np.asarray(valid, dtype=bool) & np.isfinite(vertices).all(axis=(1, 2))
    canvas = draw_hand_mesh(frame, vertices, faces, intrinsics, hand_valid)
    canvas = draw_hand_projection(canvas, joints, intrinsics, hand_valid, colour=SKELETON_COLOUR)
    badges = f"left: {'yes' if hand_valid[0] else 'NO '}   right: {'yes' if hand_valid[1] else 'NO '}"
    strip = np.full((HEADER_HEIGHT, canvas.shape[1], 3), EGO_HEADER_BG, dtype=np.uint8)
    cv2.putText(strip, f"EGO VIEW - MANO mesh + skeleton   {badges}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    canvas = np.vstack([strip, canvas])
    # the world panel (strip + panel_h) sets the total height; match it
    scale = target_total / canvas.shape[0]
    return cv2.resize(canvas, (int(canvas.shape[1] * scale), target_total))


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "EGO | WORLD combined viewer")
    parser.add_argument("--gt", action="store_true", help="overlay the reference trails (world panel)")
    parser.add_argument("--video-stride", type=int, default=2, help="every Nth frame in the video")
    parser.add_argument("--fps", type=float, default=None, help="output fps (default: 30/stride)")
    args = parser.parse_args(argv)
    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        cv2 = require_cv2()
        viz = layout.visualization_dir
        hand = load_npz(layout.hand_path, required=("vertices_camera", "joints_camera", "valid"))
        vertices = np.asarray(hand["vertices_camera"], dtype=np.float64)
        joints = np.asarray(hand["joints_camera"], dtype=np.float64)
        hand_valid = np.asarray(hand["valid"], dtype=bool)
        tracks = [world._load("PRED", layout.trajectory_dir / "trajectory.npz")]
        if args.gt:
            tracks.append(world._load("GT", layout.trajectory_dir / "ground_truth.npz"))
        faces = _mano_faces(context)
        frames = sorted(Path(layout.frames_dir).glob("*.jpg"))
        if not frames:
            return fail("no frames found")
        total = min(len(frames), vertices.shape[0], tracks[0]["joints"].shape[0])
        # The ego panel must project with the *frame's* resolution: a hardcoded
        # (512, 512) drew 1408 px clips at 512-scale intrinsics (mesh shrunk to
        # 1/2.75 and anchored near the image origin - the "mesh floating on the
        # wall" artefact).
        sample = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        if sample is None:
            return fail(f"cannot decode {frames[0]}")
        height, width = sample.shape[:2]
        size = (width, height)
        intrinsics = _ego_intrinsics(viz, frames[0].parent, size)

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        x_lim, y_lim, z_lim = world._limits(tracks, None)
        span = (x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
        panel_w, panel_h = 760, 570
        out_path = viz / "viewer.mp4"
        writer = None
        for t in range(1, total + 1, max(1, args.video_stride)):
            left = _ego_panel(cv2, frames[t - 1], vertices[t - 1], joints[t - 1], intrinsics,
                              hand_valid[t - 1], faces,
                              target_total=panel_h + HEADER_HEIGHT)
            figure = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100,
                                facecolor=world.BACKGROUND)
            axes = figure.add_subplot(111, projection="3d", facecolor=world.BACKGROUND)
            world._style_axes(axes)
            world._draw_scene(axes, tracks, upto=t)
            axes.set_xlim(*x_lim); axes.set_ylim(*y_lim); axes.set_zlim(*z_lim)
            axes.set_box_aspect(span)
            axes.view_init(elev=22, azim=-60)
            axes.set_title("WORLD SPACE - camera + both hands  |  cm", color="#e8ecf0", fontsize=9)
            figure.tight_layout()
            world._corner_triad(axes)
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
            plt.close(figure)
            world_rgb = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
            world_rgb = cv2.resize(world_rgb, (panel_w, panel_h))
            strip = np.full((HEADER_HEIGHT, panel_w, 3), EGO_HEADER_BG, dtype=np.uint8)
            cv2.putText(strip, "WORLD SPACE - PRED" + (" + GT" if args.gt else ""),
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            world_rgb = np.vstack([strip, world_rgb])
            combined = np.hstack([left, world_rgb])
            if writer is None:
                # size from the first composed frame: the ego panel's width
                # after scaling is not a round number
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps or 30.0 / max(1, args.video_stride)),
                    (combined.shape[1], combined.shape[0]),
                )
                if not writer.isOpened():
                    raise StageIOError(f"cannot open a video writer for {out_path}")
            writer.write(combined)
        if writer is None:
            return fail("no frames rendered")
        writer.release()
        final = transcode_to_h264(out_path)
        print(f"wrote {final}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

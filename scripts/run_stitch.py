#!/usr/bin/env python
"""Phase 4: depth-derived Sim(3) stitching of stored camera windows."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.stitch import (  # noqa: E402
    save_sim3_transforms,
    save_stitched_camera,
    stitch_camera_windows,
)
from ego3d_action.camera.window import load_camera_window, make_windows  # noqa: E402
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "stitch")
    parser.add_argument("--num-frames", type=int, default=None, help="clip length override")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")

        num_frames = args.num_frames
        if num_frames is None:
            from ego3d_action.io.artefacts import clip_metadata

            num_frames = int(clip_metadata(layout)["num_frames"])

        ranges = make_windows(
            num_frames,
            window=int(context.config.get("camera.window", 200)),
            overlap=int(context.config.get("camera.overlap", 40)),
        )
        paths = [layout.window_path(w.start, w.end) for w in ranges]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise StageIOError(
                f"{len(missing)} camera window artefact(s) are missing, e.g. {missing[0]}. "
                "Run scripts/run_camera.py first."
            )
        windows = [load_camera_window(p) for p in paths]
        print(f"stitching {len(windows)} window(s) covering {num_frames} frames")

        if args.dry_run:
            for window in windows:
                print(f"  window {window.name}: {window.window.num_frames} frames")
            return 0

        stitched = stitch_camera_windows(
            windows,
            num_frames=num_frames,
            stride=int(context.config.get("stitch.pixel_stride", 8)),
            min_depth=float(context.config.get("stitch.min_depth", 1e-3)),
            max_depth=context.config.get("stitch.max_depth", None),
            min_confidence=context.config.get("stitch.depth_confidence_floor", None),
            inlier_threshold=context.config.get("stitch.inlier_threshold", None),
            ransac_iterations=int(context.config.get("stitch.ransac_iterations", 128)),
            blend=bool(context.config.get("stitch.blend", True)),
            normalize=bool(context.config.get("stitch.normalize_world", True)),
        )
        save_stitched_camera(
            layout.stitched_camera_path,
            stitched,
            metadata={
                "stage": "phase4_stitch",
                "num_frames": num_frames,
                "num_windows": len(windows),
                "coverage": stitched.coverage,
                "diagnostics": [d.as_dict() for d in stitched.diagnostics],
            },
        )
        save_sim3_transforms(layout.sim3_path, stitched.sim3)
        for diag in stitched.diagnostics:
            print(
                f"  {diag.src_window} -> {diag.dst_window}: "
                f"scale={diag.scale:.5f} rmse={diag.inlier_rmse:.4f} m "
                f"inliers={100.0 * diag.inlier_ratio:.1f}% "
                f"rot={diag.rotation_deg:.2f} deg"
            )
        print(
            f"stitched camera: coverage {100.0 * stitched.coverage:.1f}% "
            f"-> {layout.stitched_camera_path}"
        )
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Phase 2: HaWoR 16/8 window reconstruction + temporal blending."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.hand import hawor  # noqa: E402
from ego3d_action.hand.temporal_blend import blend_hand_windows  # noqa: E402
from ego3d_action.io.artefacts import clip_metadata, load_detection, save_hand  # noqa: E402
from ego3d_action.io.frames import load_frame_set  # noqa: E402
from ego3d_action.runtime.subprocess_backend import BackendInvocation  # noqa: E402
from ego3d_action.visualization.overlay import write_hand_video  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "hand reconstruction")
    parser.add_argument("--reuse-windows", action="store_true", help="reuse windows already on disk")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        third_party = context.path("paths.third_party")
        weights = context.path("paths.weights")
        invocation = BackendInvocation.from_config(context.config)

        status = hawor.probe(third_party, weights)
        print(f"HaWoR backend: {status.format()}  [mode={invocation.mode}]")

        detection = load_detection(layout)
        num_frames = int(detection["valid"].shape[0])
        window = int(context.config.get("hand.window", 16))
        overlap = int(context.config.get("hand.overlap", 8))
        request = hawor.HaworClipRequest(
            num_frames=num_frames,
            frames_dir=layout.frames_dir,
            window=window,
            overlap=overlap,
            device=context.device,
        )
        spans = request.ranges()

        if args.dry_run:
            print(
                f"would run HaWoR over {len(spans)} window(s) of {window} frames "
                f"(overlap {overlap}) covering {num_frames} frames [mode={invocation.mode}]"
            )
            return 0

        layout.hand_windows_dir.mkdir(parents=True, exist_ok=True)
        expected = [
            layout.hand_windows_dir / f"{start:06d}_{end - 1:06d}.npz" for start, end in spans
        ]
        if args.reuse_windows and all(path.is_file() for path in expected):
            paths = expected
            print(f"reusing {len(paths)} HaWoR window(s) from {layout.hand_windows_dir}")
        else:
            paths = hawor.run_windows(
                request,
                layout.hand_windows_dir,
                invocation=invocation,
                third_party=third_party,
                weights_root=weights,
                detection_path=layout.detection_path,
                log_path=layout.hand_dir / "hawor_runner.log",
            )
        windows = [hawor.load_hand_window(path) for path in paths]

        blended = blend_hand_windows(windows)
        save_hand(
            layout,
            {
                "joints_camera": blended.joints_camera,
                "valid": blended.valid,
                "confidence": blended.confidence,
                "root_rot": blended.root_rot,
                "betas": np.zeros((blended.joints_camera.shape[0], 2, 10)),
            },
            metadata={
                "stage": "phase2_hand",
                "backend_mode": invocation.mode,
                "window": window,
                "overlap": overlap,
                "num_windows": len(windows),
                "coverage": float(blended.valid.mean()),
                "world_frame": "camera",
                "units": "meter",
            },
        )
        print(f"hand: coverage {100.0 * blended.valid.mean():.1f}% -> {layout.hand_path}")

        if bool(context.config.get("visualization.enabled", True)):
            frames = load_frame_set(layout.data_root, layout.clip)
            intrinsics = np.broadcast_to(
                _camera_intrinsics(layout), (blended.joints_camera.shape[0], 3, 3)
            ).copy()
            write_hand_video(
                frames.paths,
                blended.joints_camera,
                intrinsics,
                blended.valid,
                layout.visualization_dir / "02_hawor.mp4",
                fps=float(context.config.get("visualization.fps") or frames.fps),
            )
            print(f"wrote {layout.visualization_dir / '02_hawor.mp4'}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


def _camera_intrinsics(layout) -> np.ndarray:
    """Intrinsics for the debug overlay, scaled to the RGB frame size."""
    from ego3d_action.camera.depth import scale_intrinsics
    from ego3d_action.io.serialization import load_npz

    metadata = clip_metadata(layout)
    width = int(metadata.get("width", 320))
    height = int(metadata.get("height", 240))
    for path in sorted(layout.camera_windows_dir.glob("*.npz")):
        data = load_npz(path, required=("intrinsics", "depth"))
        depth = np.asarray(data["depth"])
        source = (int(depth.shape[2]), int(depth.shape[1]))  # (width, height)
        return scale_intrinsics(
            np.asarray(data["intrinsics"])[0], source_size=source, target_size=(width, height)
        )
    from ego3d_action.testing.synthetic import make_intrinsics

    return make_intrinsics(width, height)


if __name__ == "__main__":
    raise SystemExit(main())

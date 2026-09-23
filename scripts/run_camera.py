#!/usr/bin/env python
"""Phase 3: VGGT-Omega window inference (416 px / 200 frames / 40 overlap).

The backend is invoked once for the whole clip (model loading dominates) and
writes every window into ``camera/windows/``; each file is validated before the
pipeline trusts it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.vggt_omega import VggtWindowRequest, probe, run_window  # noqa: E402
from ego3d_action.camera.window import load_camera_window, make_windows  # noqa: E402
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.io.artefacts import clip_metadata  # noqa: E402
from ego3d_action.runtime.subprocess_backend import BackendInvocation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "camera reconstruction")
    parser.add_argument("--num-frames", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        third_party = context.path("paths.third_party")
        weights = context.path("paths.weights")
        invocation = BackendInvocation.from_config(context.config)

        status = probe(third_party, weights)
        print(f"VGGT-Omega backend: {status.format()}  [mode={invocation.mode}]")

        num_frames = args.num_frames or int(clip_metadata(layout)["num_frames"])
        window = int(context.config.get("camera.window", 200))
        overlap = int(context.config.get("camera.overlap", 40))
        resolution = int(context.config.get("camera.resolution", 416))
        checkpoint = str(context.config.require("camera.checkpoint"))
        ranges = make_windows(num_frames, window=window, overlap=overlap)

        if args.dry_run:
            print(f"would run {len(ranges)} VGGT-Omega window(s) at {resolution} px:")
            for rng in ranges:
                print(f"  {rng.start:06d}_{rng.end - 1:06d}")
            return 0

        request = VggtWindowRequest(
            start=0,
            end=num_frames,
            resolution=resolution,
            checkpoint=checkpoint,
            device=context.device,
            use_depth_confidence=bool(context.config.get("camera.use_depth_confidence", True)),
        )
        paths = run_window(
            request,
            invocation=invocation,
            third_party=third_party,
            weights_root=weights,
            out_dir=layout.camera_windows_dir,
            frames_dir=layout.frames_dir,
            num_frames=num_frames,
            window=window,
            overlap=overlap,
            log_path=layout.camera_dir / "vggt_runner.log",
        )
        for path in paths:
            camera_window = load_camera_window(path)
            print(f"window {camera_window.name}: {camera_window.window.num_frames} frames")
        print(f"camera: {len(paths)} window(s) -> {layout.camera_windows_dir}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

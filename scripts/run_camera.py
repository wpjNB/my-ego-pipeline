#!/usr/bin/env python
"""Phase 3: VGGT-Omega window inference (416 px / 200 frames / 40 overlap)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.vggt_omega import VggtWindowRequest, probe, run_window  # noqa: E402
from ego3d_action.camera.window import make_windows, save_camera_window  # noqa: E402
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.io.artefacts import clip_metadata  # noqa: E402


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

        status = probe(third_party, weights)
        print(f"VGGT-Omega backend: {status.format()}")

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

        for rng in ranges:
            request = VggtWindowRequest(
                start=rng.start,
                end=rng.end,
                resolution=resolution,
                checkpoint=checkpoint,
                device=context.device,
                use_depth_confidence=bool(
                    context.config.get("camera.use_depth_confidence", True)
                ),
            )
            camera_window = run_window(request, third_party=third_party, weights_root=weights)
            save_camera_window(camera_window, layout.window_path(rng.start, rng.end))
            print(f"window {camera_window.name}: written")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Phase 2: HaWoR 16/8 window reconstruction + temporal blending."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.hand import hawor  # noqa: E402
from ego3d_action.hand.temporal_blend import blend_hand_windows  # noqa: E402
from ego3d_action.io.artefacts import clip_metadata, load_detection, save_hand  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "hand reconstruction")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        third_party = context.path("paths.third_party")
        weights = context.path("paths.weights")

        status = hawor.probe(third_party, weights)
        print(f"HaWoR backend: {status.format()}")

        detection = load_detection(layout)
        num_frames = int(detection["valid"].shape[0])
        window = int(context.config.get("hand.window", 16))
        overlap = int(context.config.get("hand.overlap", 8))
        starts = list(range(0, max(1, num_frames - overlap), window - overlap))

        if args.dry_run:
            print(
                f"would run {len(starts)} HaWoR window(s) of {window} frames "
                f"(overlap {overlap}) over {num_frames} frames"
            )
            return 0

        windows = []
        for start in starts:
            end = min(start + window, num_frames)
            request = hawor.HaworWindowRequest(
                start=start,
                end=end,
                frames_dir=layout.frames_dir,
                boxes=detection["boxes"][start:end],
                valid=np.asarray(detection["valid"][start:end], dtype=bool),
                confidence=detection["confidence"][start:end],
                device=context.device,
            )
            windows.append(hawor.run_window(request, third_party=third_party, weights_root=weights))

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
                "window": window,
                "overlap": overlap,
                "num_windows": len(windows),
                "coverage": float(blended.valid.mean()),
                "world_frame": "camera",
                "units": "meter",
            },
        )
        print(f"hand: coverage {100.0 * blended.valid.mean():.1f}% -> {layout.hand_path}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

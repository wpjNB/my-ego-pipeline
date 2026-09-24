#!/usr/bin/env python
"""Phase 1: WiLoR detection + conservative tracking -> detection/detection.npz.

Two modes:

* ``--raw-detections FILE.npz``: run the *tracker* on detections exported
  earlier. This half of Phase 1 is model-free and runs anywhere.
* default: invoke the WiLoR backend through the runner protocol
  (``backends.mode: real`` -> ``ego3d_wilor``, ``mock`` -> deterministic
  stand-in from ``backends/mock_backend.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.detection import wilor  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.io.artefacts import save_detection  # noqa: E402
from ego3d_action.io.frames import load_frame_set  # noqa: E402
from ego3d_action.runtime.subprocess_backend import BackendInvocation  # noqa: E402
from ego3d_action.visualization.overlay import write_detection_video  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "detection")
    parser.add_argument("--raw-detections", default=None, help="npz with exported WiLoR detections")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        third_party = context.path("paths.third_party")
        weights = context.path("paths.weights")
        invocation = BackendInvocation.from_config(context.config)

        status = wilor.probe(third_party, weights)
        print(f"WiLoR backend: {status.format()}  [mode={invocation.mode}]")

        if args.raw_detections:
            raw = wilor.load_raw_detections(args.raw_detections)
            source = f"file {args.raw_detections}"
        elif args.dry_run:
            print(
                f"would invoke the WiLoR backend (mode={invocation.mode}) on {layout.frames_dir}"
            )
            return 0
        else:
            frames = load_frame_set(layout.data_root, layout.clip)
            from ego3d_action.visualization.overlay import require_cv2

            sample = require_cv2().imread(str(frames.paths[0]), require_cv2().IMREAD_COLOR)
            if sample is None:
                return fail(f"cannot decode {frames.paths[0]}")
            frame_height, frame_width = sample.shape[:2]
            raw_arrays = wilor.detect_clip(
                layout.frames_dir,
                layout.detection_dir / "raw_detections.npz",
                invocation=invocation,
                third_party=third_party,
                weights_root=weights,
                device=context.device,
                batch_size=int(context.config.get("hand.batch_size", 4)),
                detector_confidence=float(
                    context.config.get("detection.detector_confidence", 0.1)
                ),
                num_frames=frames.num_frames,
                width=int(frame_width),
                height=int(frame_height),
                image_format=str(context.config.get("preprocess.image_format", "jpg")),
                log_path=layout.detection_dir / "wilor_runner.log",
            )
            raw = wilor.raw_detections_from_arrays(raw_arrays)
            source = f"WiLoR runner (mode={invocation.mode})"

        arrays = wilor.track_clip(
            raw,
            min_confidence=float(context.config.get("detection.min_confidence", 0.75)),
            max_gap=int(context.config.get("detection.max_gap", 4)),
            iou_threshold=float(context.config.get("detection.iou_threshold", 0.20)),
        )
        if args.dry_run:
            print(f"would write {layout.detection_path} (valid={arrays['valid'].shape})")
            return 0
        save_detection(
            layout,
            arrays,
            metadata={
                "stage": "phase1_detection",
                "backend_mode": invocation.mode,
                "source": source,
                "min_confidence": float(context.config.get("detection.min_confidence", 0.75)),
                "max_gap": int(context.config.get("detection.max_gap", 4)),
                "iou_threshold": float(context.config.get("detection.iou_threshold", 0.20)),
                "num_frames": int(arrays["valid"].shape[0]),
            },
        )
        print(
            f"detection: {arrays['valid'].shape[0]} frames, coverage "
            f"left={100.0 * arrays['valid'][:, 0].mean():.1f}% "
            f"right={100.0 * arrays['valid'][:, 1].mean():.1f}% -> {layout.detection_path}"
        )

        if bool(context.config.get("visualization.enabled", True)):
            frames = load_frame_set(layout.data_root, layout.clip)
            write_detection_video(
                frames.paths,
                arrays["boxes"],
                arrays["confidence"],
                arrays["valid"],
                layout.visualization_dir / "01_detection.mp4",
                fps=float(context.config.get("visualization.fps") or frames.fps),
                draw_confidence=bool(context.config.get("visualization.draw_confidence", True)),
            )
            print(f"wrote {layout.visualization_dir / '01_detection.mp4'}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

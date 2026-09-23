#!/usr/bin/env python
"""Phase 1: WiLoR detection + conservative tracking -> detection/detection.npz.

Two modes:

* ``--raw-detections FILE.npz``: run the *tracker* on detections exported by the
  WiLoR backend env. This half of Phase 1 is model-free and runs anywhere.
* default: invoke the WiLoR backend directly (GPU server only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.detection import wilor  # noqa: E402
from ego3d_action.detection.tracker import HandDetection  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.io.artefacts import save_detection  # noqa: E402
from ego3d_action.io.serialization import load_npz  # noqa: E402


def raw_detections_from_npz(path: Path) -> list[list[wilor.RawDetection]]:
    """Read ``boxes [T, K, 4] / confidence [T, K] / right_score [T, K]`` arrays."""
    data = load_npz(path, required=("boxes", "confidence"))
    boxes = np.asarray(data["boxes"], dtype=np.float64)
    confidence = np.asarray(data["confidence"], dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise StageIOError(f"boxes must be [T, K, 4], got {boxes.shape}")
    if confidence.shape != boxes.shape[:2]:
        raise StageIOError(f"confidence must be {boxes.shape[:2]}, got {confidence.shape}")
    right = data.get("right_score")
    left = data.get("left_score")
    count = data.get("count")
    if count is None:
        count = np.full(boxes.shape[0], boxes.shape[1], dtype=np.int64)
    count = np.asarray(count, dtype=np.int64).reshape(-1)
    if count.shape[0] != boxes.shape[0]:
        raise StageIOError(f"count must have length {boxes.shape[0]}, got {count.shape}")

    frames: list[list[wilor.RawDetection]] = []
    for frame in range(boxes.shape[0]):
        detections: list[wilor.RawDetection] = []
        for slot in range(int(count[frame])):
            detections.append(
                wilor.RawDetection(
                    bbox=boxes[frame, slot],
                    confidence=float(confidence[frame, slot]),
                    right_score=float(right[frame, slot]) if right is not None else 0.0,
                    left_score=float(left[frame, slot]) if left is not None else 0.0,
                )
            )
        frames.append(detections)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "detection")
    parser.add_argument("--raw-detections", default=None, help="npz with exported WiLoR detections")
    parser.add_argument("--video", default=None, help="video to decode before detection (GPU mode)")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        third_party = context.path("paths.third_party")
        weights = context.path("paths.weights")

        status = wilor.probe(third_party, weights)
        print(f"WiLoR backend: {status.format()}")

        if args.raw_detections:
            raw = raw_detections_from_npz(Path(args.raw_detections))
        else:
            if args.dry_run:
                print("would invoke the WiLoR backend env on the GPU server")
                return 0
            if not args.video:
                frames_dir = layout.frames_dir
            else:
                frames_dir = layout.frames_dir
            raw = wilor.detect_clip(
                frames_dir,
                third_party=third_party,
                weights_root=weights,
                device=context.device,
                batch_size=int(context.config.get("hand.batch_size", 4)),
            )

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
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

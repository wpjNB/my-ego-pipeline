#!/usr/bin/env python
"""WiLoR detection runner: executed inside the detector environment.

    python backends/wilor_runner.py --check
    python backends/wilor_runner.py --frames data/clip/frames --out raw.npz \
        --third-party third_party --weights weights --device cuda --conf 0.1

What this stage actually is
---------------------------

Reading the sources settled a design question: this pipeline needs WiLoR's
**detector**, not its 3D model.

* WiLoR's demo splits the two: ``detector = YOLO('./pretrained_models/detector.pt')``
  for boxes + handedness, and ``model = load_wilor(wilor_final.ckpt, ...)`` for
  the 3D hand (its MANO layer needs the licence-gated MANO model).
* HaWoR's own ``detect_track`` uses the very same arrangement:
  ``YOLO('./weights/external/detector.pt')``.

The reference system takes hand *tracking* from WiLoR and hand *reconstruction*
from HaWoR, so only the detector (51 MiB, ``detector.pt``) is required here -
``wilor_final.ckpt`` and WiLoR's MANO are only needed if you also want WiLoR's
own 3D output, which this pipeline deliberately does not use.

The detector threshold is deliberately low (default 0.1): the conservative
tracker in Phase 1 anchors on ``confidence >= 0.75`` and only recovers a gap from
a *low-confidence* detection whose box matches the interpolated anchor box, so
those candidates have to survive the detector first.

The conversion from detector output into the pipeline's artefact lives in
``ego3d_action.detection.wilor`` and is unit-tested; the detector call itself
needs the environment and the weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.detection.wilor import (  # noqa: E402
    RawDetection,
    build_raw_detection_arrays,
)
from ego3d_action.io.serialization import save_npz  # noqa: E402
from ego3d_action.runtime.checkpoints import (  # noqa: E402
    allow_trusted_checkpoint_globals,
)
from ego3d_action.runtime.checkpoints import report as report_checkpoint_globals  # noqa: E402

DETECTION_MODULE = "ultralytics"
#: Where the detector may live, most specific first. ``external/detector.pt`` is
#: HaWoR's copy of the same YOLO hand detector and works as a stand-in.
DETECTOR_CANDIDATES = (
    "wilor/detector.pt",
    "external/detector.pt",
    "detector.pt",
)
#: YOLO class index -> our handedness (0 left, 1 right), as used by WiLoR's demo
#: (``is_right = det.boxes.cls``).
RIGHT_CLASS = 1


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def find_detector(weights_root: str | Path) -> Path | None:
    root = Path(weights_root)
    for relative in DETECTOR_CANDIDATES:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def backend_available(third_party: Path, weights_root: Path) -> tuple[bool, str]:
    """Whether the detection stage can run: detector weights + ultralytics."""
    _ = third_party  # the detector does not need the WiLoR checkout
    detector = find_detector(weights_root)
    if detector is None:
        return False, (
            f"detector.pt not found under {weights_root} (looked for "
            f"{list(DETECTOR_CANDIDATES)}) - run ./scripts/download_weights.sh --only wilor"
        )
    try:
        from ultralytics import __version__ as ultralytics_version  # noqa: PLC0415
    except ImportError as exc:
        return False, (
            f"'{DETECTION_MODULE}' is not installed in this interpreter ({exc}); the detector "
            "environment needs it (see environment-wilor.yml)"
        )
    return True, f"YOLO detector at {detector}, ultralytics {ultralytics_version}"


def detections_from_yolo(result: object, *, frame: int) -> list[RawDetection]:
    """Convert one Ultralytics result into :class:`RawDetection` records.

    Raises:
        NotImplementedError: the result object is not a YOLO boxes result.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        raise NotImplementedError(
            "expected an Ultralytics result with `.boxes.xyxy`; got "
            f"{type(result).__name__} with attributes {sorted(dir(result))[:12]}"
        )
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy().reshape(-1)
    classes = (
        boxes.cls.detach().cpu().numpy().reshape(-1).astype(int)
        if boxes.cls is not None
        else np.zeros(len(confidence), dtype=int)
    )
    detections: list[RawDetection] = []
    for box, score, klass in zip(xyxy, confidence, classes, strict=False):
        detections.append(
            RawDetection(
                bbox=np.asarray(box, dtype=np.float64),
                confidence=float(score),
                right_score=1.0 if int(klass) == RIGHT_CLASS else 0.0,
                left_score=0.0 if int(klass) == RIGHT_CLASS else 1.0,
            )
        )
    _ = frame
    return detections


def run_model(args: argparse.Namespace, frames: list[Path]) -> dict[str, np.ndarray]:
    """Detect hands on every frame and pack the result into the raw artefact."""
    import cv2  # noqa: PLC0415 - optional at import time
    from ultralytics import YOLO  # noqa: PLC0415 - backend import

    detector_path = find_detector(args.weights)
    if detector_path is None:
        raise FileNotFoundError(
            f"detector.pt not found under {args.weights}; run "
            "./scripts/download_weights.sh --only wilor"
        )
    allowed = allow_trusted_checkpoint_globals(detector_path, label="WiLoR detector")
    report_checkpoint_globals(detector_path, allowed)
    detector = YOLO(str(detector_path))
    if args.device not in {"auto", "cpu"}:
        detector.to(args.device if args.device != "auto" else "cuda")
    print(f"YOLO detector {detector_path.name} loaded", file=sys.stderr)

    per_frame: list[list[RawDetection]] = []
    for frame_id, path in enumerate(frames):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode {path}")
        results = detector.predict(
            image, conf=args.conf, verbose=False, device=None if args.device == "auto" else args.device
        )
        if not results:
            per_frame.append([])
            continue
        per_frame.append(detections_from_yolo(results[0], frame=frame_id))
    return build_raw_detection_arrays(per_frame, num_frames=args.num_frames or len(frames))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WiLoR detection runner")
    parser.add_argument("--check", action="store_true", help="report availability and exit")
    parser.add_argument("--frames", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--third-party", default="third_party")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-format", default="jpg")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="detector confidence floor; keep it below the tracker's 0.75 anchor "
        "threshold so gap-recovery candidates survive",
    )
    # Phase 1 detection is frame-coupled: the tracker recovers a same-side gap
    # across frames, so a frame slice would see less context than a whole-clip run
    # and could silently change the result. The flags are accepted (so the batch
    # scheduler's contract is uniform) but anything other than "the whole clip"
    # is rejected with a reason - never silently ignored.
    parser.add_argument("--shard", default=None, metavar="INDEX/COUNT")
    parser.add_argument("--window-range", default=None, metavar="A-B")
    parser.add_argument("--frame-range", default=None, metavar="A-B")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available, detail = backend_available(Path(args.third_party), Path(args.weights))
    if args.check:
        emit(
            {
                "status": "ok" if available else "error",
                "backend": "wilor",
                "available": available,
                "detail": detail,
                "device": args.device,
            }
        )
        return 0 if available else 1

    try:
        from ego3d_action.runtime.sharding import FrameRange, Shard

        shard = Shard.parse(args.shard)
        frame_range = FrameRange.parse(args.frame_range, num_frames=args.num_frames)
        if not shard.is_whole or frame_range is not None or args.window_range:
            raise ValueError(
                "WiLoR detection is frame-coupled (the tracker recovers gaps across frames); "
                "sharding it would change its output, so this runner refuses "
                f"--shard={args.shard!r}, --window-range={args.window_range!r}, "
                f"--frame-range={args.frame_range!r}. Shard the hand/camera window stages instead; "
                "the detection frame slicer (plus context padding) lands in M2."
            )
        frames = sorted(Path(args.frames).glob(f"*.{args.image_format}"))
        if not frames:
            raise FileNotFoundError(f"no *.{args.image_format} frames in {args.frames}")
        if args.num_frames is not None and len(frames) != args.num_frames:
            raise ValueError(f"expected {args.num_frames} frames, found {len(frames)}")
        arrays = run_model(args, frames)
        save_npz(args.out, **arrays)
        emit(
            {
                "status": "ok",
                "backend": "wilor",
                "num_frames": len(frames),
                "detections": int(np.count_nonzero(arrays["count"])),
                "conf": args.conf,
                "output": str(args.out),
                "device": args.device,
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - reported through the runner protocol
        emit({"status": "error", "backend": "wilor", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

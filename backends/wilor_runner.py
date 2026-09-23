#!/usr/bin/env python
"""WiLoR runner: executed **inside** the WiLoR environment (``ego3d_wilor``).

Protocol (see ``runtime/subprocess_backend.py``): parse arguments, write the
artefacts, print one JSON object as the last stdout line.

    python backends/wilor_runner.py --check
    python backends/wilor_runner.py --frames data/clip/frames --out raw.npz \
        --third-party third_party --weights weights --device cuda

Implemented here: argument handling, frame discovery, the artefact format and
the availability check. Still to fill in on the GPU server: :func:`run_model` -
a handful of lines calling WiLoR's own detector. It raises rather than
pretending, so a half-configured server can never emit a silently empty
detection file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.io.serialization import save_npz  # noqa: E402

DETECTION_MODULE = "wilor"  # WiLoR's python package once the repo is on sys.path


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def backend_available(third_party: Path) -> tuple[bool, str]:
    """Report whether WiLoR itself can be imported from this interpreter."""
    checkout = third_party / "WiLoR"
    if not checkout.is_dir():
        return False, f"WiLoR checkout not found at {checkout}"
    sys.path.insert(0, str(checkout))
    try:
        __import__(DETECTION_MODULE)
    except ImportError as exc:
        return False, f"cannot import '{DETECTION_MODULE}' from {checkout}: {exc}"
    return True, f"'{DETECTION_MODULE}' importable from {checkout}"


def run_model(args: argparse.Namespace, frames: list[Path]) -> dict[str, np.ndarray]:
    """Call the WiLoR detector on ``frames``.

    Fill this in on the GPU server with WiLoR's own detection entry point; it
    must return ``boxes [T, K, 4]``, ``confidence [T, K]``, ``right_score``,
    ``left_score`` and ``count [T]`` (= how many of the K slots are real).
    """
    raise NotImplementedError(
        "WiLoR detection is not wired to the backend API yet. Implement run_model() with "
        f"WiLoR's detector (checkout={args.third_party}/WiLoR, weights={args.weights}, "
        f"device={args.device}) so it returns boxes/confidence/right_score/left_score/count. "
        f"{len(frames)} frames are ready to process."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WiLoR runner")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available, detail = backend_available(Path(args.third_party))
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

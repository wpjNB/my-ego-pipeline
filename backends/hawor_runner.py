#!/usr/bin/env python
"""HaWoR runner: executed **inside** the HaWoR environment (``ego3d_hawor``).

    python backends/hawor_runner.py --check
    python backends/hawor_runner.py --frames data/clip/frames --detection detection.npz \
        --window-start 0 --window-end 16 --out hand/windows/000000_000015.npz \
        --third-party third_party --weights weights --device cuda

Implemented: argument handling, window/detection plumbing, artefact format and
the availability check. To fill in: :func:`run_model`, which must call HaWoR on
the window and return camera-space MANO joints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.io.serialization import load_npz, save_npz  # noqa: E402

BACKEND_MODULE = "hawor"


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def backend_available(third_party: Path) -> tuple[bool, str]:
    checkout = third_party / "HaWoR"
    if not checkout.is_dir():
        return False, f"HaWoR checkout not found at {checkout}"
    sys.path.insert(0, str(checkout))
    try:
        __import__(BACKEND_MODULE)
    except ImportError as exc:
        return False, f"cannot import '{BACKEND_MODULE}' from {checkout}: {exc}"
    return True, f"'{BACKEND_MODULE}' importable from {checkout}"


def run_model(args: argparse.Namespace, window: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reconstruct the window in camera space.

    Must return ``joints_camera [n, 2, 21, 3]`` (metres, camera frame),
    ``valid [n, 2]``, ``confidence [n, 2]`` and optionally ``root_rot``
    ``[n, 2, 3, 3]`` and ``betas [n, 2, 10]``. Frames without a usable hand
    must stay ``NaN``/invalid - never interpolated into existence.
    """
    raise NotImplementedError(
        "HaWoR window reconstruction is not wired to the backend API yet. Implement "
        f"run_model() with HaWoR's inference entry point (checkout={args.third_party}/HaWoR, "
        f"weights={args.weights}, device={args.device}) for frames "
        f"[{args.window_start}, {args.window_end}) with "
        f"{int(np.count_nonzero(window['valid']))} valid hand-frames."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HaWoR runner")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--frames", default=None)
    parser.add_argument("--detection", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--window-start", type=int, default=None)
    parser.add_argument("--window-end", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--third-party", default="third_party")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available, detail = backend_available(Path(args.third_party))
    if args.check:
        emit(
            {
                "status": "ok" if available else "error",
                "backend": "hawor",
                "available": available,
                "detail": detail,
                "device": args.device,
            }
        )
        return 0 if available else 1

    try:
        if args.window_start is None or args.window_end is None:
            raise ValueError("--window-start and --window-end are required")
        detection = load_npz(args.detection, required=("boxes", "confidence", "valid"))
        window = {
            "boxes": np.asarray(detection["boxes"])[args.window_start : args.window_end],
            "confidence": np.asarray(detection["confidence"])[
                args.window_start : args.window_end
            ],
            "valid": np.asarray(detection["valid"])[args.window_start : args.window_end],
        }
        if window["boxes"].shape[0] != args.window_end - args.window_start:
            raise ValueError(
                f"detection covers {window['boxes'].shape[0]} frames but the window is "
                f"[{args.window_start}, {args.window_end})"
            )
        result = run_model(args, window)
        payload: dict[str, np.ndarray] = {
            "start": np.array([args.window_start], dtype=np.int64),
            "joints_camera": np.asarray(result["joints_camera"], dtype=np.float64),
            "valid": np.asarray(result["valid"], dtype=bool),
            "confidence": np.asarray(result["confidence"], dtype=np.float64),
        }
        for optional in ("root_rot", "betas"):
            if optional in result:
                payload[optional] = np.asarray(result[optional], dtype=np.float64)
        save_npz(args.out, **payload)
        emit(
            {
                "status": "ok",
                "backend": "hawor",
                "start": args.window_start,
                "end": args.window_end,
                "valid_frames": int(np.count_nonzero(payload["valid"])),
                "output": str(args.out),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - reported through the runner protocol
        emit({"status": "error", "backend": "hawor", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

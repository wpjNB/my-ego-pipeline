#!/usr/bin/env python
"""VGGT-Omega runner: executed **inside** the VGGT environment (``ego3d_vggt``).

    python backends/vggt_runner.py --check
    python backends/vggt_runner.py --out-dir data/clip/camera/windows --num-frames 600 \
        --window 200 --overlap 40 --resolution 416 \
        --checkpoint VGGT-Omega-1B-416-Reproduction \
        --third-party third_party --weights weights --device cuda

Implemented: the window schedule, the checkpoint policy, the artefact format and
the availability check. To fill in: :func:`run_model`, one call into VGGT-Omega
per window returning camera poses, intrinsics and metric depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.window import CameraWindow, WindowRange, make_windows, save_camera_window  # noqa: E402

BACKEND_MODULE = "vggt"
SUPPORTED_CHECKPOINTS = (
    "VGGT-Omega-1B-416-Reproduction",
    "VGGT-Omega-1B-512",
    "VGGT-Omega-1B-256-Text-Alignment",
)


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def backend_available(third_party: Path, weights: Path) -> tuple[bool, str]:
    checkout = third_party / "VGGT-Omega"
    if not checkout.is_dir():
        return False, f"VGGT-Omega checkout not found at {checkout}"
    if not weights.is_dir() or not any(weights.iterdir()):
        return False, f"no checkpoint files found in {weights}"
    sys.path.insert(0, str(checkout))
    try:
        __import__(BACKEND_MODULE)
    except ImportError as exc:
        return False, f"cannot import '{BACKEND_MODULE}' from {checkout}: {exc}"
    return True, f"'{BACKEND_MODULE}' importable from {checkout}, checkpoints in {weights}"


def run_model(
    args: argparse.Namespace, rng: WindowRange
) -> dict[str, np.ndarray]:
    """Run VGGT-Omega on one window.

    Must return ``rotation_c2w [n, 3, 3]``, ``translation_c2w [n, 3]`` (the
    window's own first camera defines its frame), ``intrinsics [n, 3, 3]`` and
    metric ``depth [n, H, W]`` - optionally with ``depth_confidence``.
    """
    raise NotImplementedError(
        "VGGT-Omega window inference is not wired to the backend API yet. Implement run_model() "
        f"with VGGT-Omega's inference entry point (checkout={args.third_party}/VGGT-Omega, "
        f"checkpoint={args.checkpoint} in {args.weights}, resolution={args.resolution}, "
        f"device={args.device}) for frames [{rng.start}, {rng.end})."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT-Omega runner")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=416)
    parser.add_argument("--checkpoint", default="VGGT-Omega-1B-416-Reproduction")
    parser.add_argument("--third-party", default="third_party")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available, detail = backend_available(Path(args.third_party), Path(args.weights))
    if args.check:
        emit(
            {
                "status": "ok" if available else "error",
                "backend": "vggt",
                "available": available,
                "detail": detail,
                "checkpoint": args.checkpoint,
                "device": args.device,
            }
        )
        return 0 if available else 1

    try:
        if args.checkpoint not in SUPPORTED_CHECKPOINTS:
            raise ValueError(
                f"checkpoint '{args.checkpoint}' is not one of {list(SUPPORTED_CHECKPOINTS)}; "
                "record any custom checkpoint explicitly in the ablation table"
            )
        ranges = make_windows(args.num_frames, window=args.window, overlap=args.overlap)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for rng in ranges:
            result = run_model(args, rng)
            window = CameraWindow(
                window=rng,
                rotation_c2w=np.asarray(result["rotation_c2w"], dtype=np.float64),
                translation_c2w=np.asarray(result["translation_c2w"], dtype=np.float64),
                intrinsics=np.asarray(result["intrinsics"], dtype=np.float64),
                depth=np.asarray(result["depth"], dtype=np.float64),
                depth_confidence=result.get("depth_confidence"),
            )
            path = out_dir / f"{rng.start:06d}_{rng.end - 1:06d}.npz"
            save_camera_window(window, path)
            written.append(path.name)
        emit(
            {
                "status": "ok",
                "backend": "vggt",
                "resolution": args.resolution,
                "checkpoint": args.checkpoint,
                "windows": written,
                "output_dir": str(out_dir),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - reported through the runner protocol
        emit({"status": "error", "backend": "vggt", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

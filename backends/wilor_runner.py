#!/usr/bin/env python
"""WiLoR runner: executed **inside** the WiLoR environment (``ego3d_wilor``).

Protocol (see ``runtime/subprocess_backend.py``): parse arguments, write the
artefacts, print one JSON object as the last stdout line.

    python backends/wilor_runner.py --check
    python backends/wilor_runner.py --frames data/clip/frames --out raw.npz \
        --third-party third_party --weights weights --device cuda

The conversion from "whatever the detector returned" to the pipeline's
compacted ``boxes/confidence/count`` artefact lives in
``ego3d_action.detection.wilor`` and is unit-tested; what is *not* verified here
is the detector call itself, which cannot run without the checkout, the weights
and a GPU. Every step raises with the exact missing piece rather than writing an
empty artefact.
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
    detections_from_predictions,
)
from ego3d_action.io.serialization import save_npz  # noqa: E402

DETECTION_MODULE = "wilor"  # WiLoR's python package once the repo is on sys.path

CHECKPOINT_CANDIDATES = ("wilor_final.ckpt", "wilor.ckpt", "checkpoints/wilor_final.ckpt")
CFG_CANDIDATES = ("model_config.yaml", "pretrained_models/model_config.yaml")


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def backend_available(third_party: Path) -> tuple[bool, str]:
    """Report whether WiLoR (and its MANO asset) can be imported."""
    checkout = third_party / "WiLoR"
    if not checkout.is_dir():
        return False, f"WiLoR checkout not found at {checkout}"
    # wilor/models/__init__.py sets MANO.MODEL_PATH='./mano_data/', so WiLoR
    # cannot construct its MANO layer without the licence-gated model there.
    mano = checkout / "mano_data" / "MANO_RIGHT.pkl"
    if not mano.is_file():
        return False, (
            f"MANO_RIGHT.pkl not found at {mano} - WiLoR's MANO layer needs it. "
            "Get it from https://mano.is.tue.mpg.de/ and run "
            "scripts/install_mano.sh --from <mano dir>"
        )
    sys.path.insert(0, str(checkout))
    try:
        __import__(DETECTION_MODULE)
    except ImportError as exc:
        return False, f"cannot import '{DETECTION_MODULE}' from {checkout}: {exc}"
    return True, f"'{DETECTION_MODULE}' importable from {checkout}, MANO at {mano}"


def _first_existing(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative in candidates:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def load_detector(args: argparse.Namespace) -> tuple[object, Path]:
    """Import WiLoR and load its checkpoint.

    WiLoR is loaded through its documented entry point
    (``wilor.models.load_wilor``); the checkpoint and config are located by
    convention inside ``--weights``.

    Raises:
        FileNotFoundError: checkpoint/config missing.
        ImportError: WiLoR (or torch) is not importable in this interpreter.
    """
    weights = Path(args.weights)
    checkpoint = _first_existing(weights, CHECKPOINT_CANDIDATES)
    cfg_path = _first_existing(weights, CFG_CANDIDATES) or _first_existing(
        Path(args.third_party) / "WiLoR", CFG_CANDIDATES
    )
    if checkpoint is None:
        raise FileNotFoundError(
            f"no WiLoR checkpoint in {weights} (looked for {list(CHECKPOINT_CANDIDATES)})"
        )
    if cfg_path is None:
        raise FileNotFoundError(
            f"no WiLoR model config found under {weights} or {args.third_party}/WiLoR"
        )
    from wilor.models import load_wilor  # noqa: PLC0415 - backend import

    model, _cfg = load_wilor(checkpoint_path=str(checkpoint), cfg_path=str(cfg_path))
    device = args.device if args.device != "auto" else ("cuda" if _cuda() else "cpu")
    model = model.to(device).eval()
    return model, Path(str(device))


def _cuda() -> bool:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def run_model(args: argparse.Namespace, frames: list[Path]) -> dict[str, np.ndarray]:
    """Detect hands on every frame and pack the result into the raw artefact."""
    import cv2  # noqa: PLC0415 - optional at import time

    model, device = load_detector(args)
    print(f"WiLoR loaded on {device}", file=sys.stderr)
    from wilor.utils import process_image  # noqa: PLC0415 - backend import

    per_frame: list[list[RawDetection]] = []
    for frame_id, path in enumerate(frames):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode {path}")
        with _no_grad():
            predictions = model(process_image(image))
        per_frame.append(detections_from_predictions(predictions, frame=frame_id))
    return build_raw_detection_arrays(
        per_frame, num_frames=args.num_frames or len(frames)
    )


def _no_grad() -> object:
    import contextlib

    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - only reachable without the backend
        return contextlib.nullcontext()
    return torch.no_grad()


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

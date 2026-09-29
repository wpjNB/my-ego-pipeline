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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ego3d_action.camera.depth import scale_intrinsics  # noqa: E402
from ego3d_action.camera.vggt_omega import (  # noqa: E402
    CHECKPOINT_FILENAMES,
    camera_window_from_output,
    resolve_checkpoint,
)
from ego3d_action.camera.window import make_windows, save_camera_window  # noqa: E402
from ego3d_action.errors import StageIOError  # noqa: E402
from ego3d_action.geometry.transforms import invert_rigid  # noqa: E402

from _shard_cli import (  # noqa: E402
    add_shard_arguments,
    partition_is_reusable,
    record_partition,
    select_windows,
    selection_from_args,
)

#: The checkout is the VGGT-Omega repo, whose importable package is
#: ``vggt_omega`` (class ``VGGTOmega``), not the original VGGT's ``vggt``.
BACKEND_MODULE = "vggt_omega"
SUPPORTED_CHECKPOINTS = (
    "VGGT-Omega-1B-416-Reproduction",
    "VGGT-Omega-1B-512",
    "VGGT-Omega-1B-256-Text-Alignment",
)

_MODEL_CACHE: dict[tuple[str, str], object] = {}
#: Which checkpoint was actually resolved, so the JSON summary can report a
#: substitution (e.g. only the 512 file is on disk while 416 was requested).
_RESOLVED: dict[str, object] = {"path": None, "substituted": None}


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


def load_model(args: argparse.Namespace) -> object:
    """Load VGGT-Omega once per process (model loading dominates the runtime)."""
    precision = getattr(args, "precision", "auto")
    key = (str(args.weights), str(args.checkpoint), precision)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    weights = Path(args.weights)
    checkpoint_dir, substituted = resolve_checkpoint(weights, args.checkpoint)
    if checkpoint_dir is None:
        raise FileNotFoundError(
            f"no checkpoint for '{args.checkpoint}' under {weights}; run "
            "scripts/download_weights.sh --only vggt (see doc_auto/setup.md)"
        )
    if substituted is not None:
        print(
            f"WARNING: requested checkpoint '{args.checkpoint}' was not found; using "
            f"'{substituted}' instead. This is a DIFFERENT checkpoint - record it in the "
            "ablation table (mixing 416/512 numbers is not comparable).",
            file=sys.stderr,
        )
    _RESOLVED["path"] = str(checkpoint_dir)
    _RESOLVED["substituted"] = substituted
    # The real package is `vggt_omega` (class VGGTOmega); see its demo_gradio.py:
    #   model = VGGTOmega().eval(); model.load_state_dict(torch.load(ckpt))
    # VGGT-Omega refuses to run without CUDA (the demo raises outright), so there
    # is no CPU fallback for Phase 3 - say that plainly instead of failing later.
    import torch  # noqa: PLC0415
    from vggt_omega.models import VGGTOmega  # noqa: PLC0415 - backend import

    if args.device not in {"cuda", "auto"} and not str(args.device).startswith("cuda"):
        raise RuntimeError(
            f"VGGT-Omega requires CUDA, but device='{args.device}' was requested"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VGGT-Omega requires CUDA; torch.cuda.is_available() is False. Run Phase 3 on the "
            "GPU server (the rest of the pipeline has no such requirement)."
        )
    weights_file = (
        checkpoint_dir
        if checkpoint_dir.is_file()
        else next(checkpoint_dir.glob("*.pt"), None)
    )
    if weights_file is None:
        raise FileNotFoundError(
            f"no .pt checkpoint inside {checkpoint_dir}; expected "
            f"{CHECKPOINT_FILENAMES.get(args.checkpoint, ('*.pt',))}"
        )
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(str(weights_file), map_location="cpu"))
    # ``fp16`` halves the *aggregator* - the DINOv3 backbone plus the alternating
    # attention stack, i.e. the bulk of the 1B parameters and the reason the fp32
    # model needs ~4.3 GiB before a single activation - while leaving the heads in
    # fp32. ``VGGTOmega.forward`` runs the heads inside
    # ``autocast(..., enabled=False)``, so casting them to half fails with
    # "expected scalar type Float but found Half"; the aggregator is called
    # *inside* autocast, so half weights there are fine on any modern torch.
    if precision == "fp16":
        model.aggregator = model.aggregator.half()
    device = "cuda" if args.device == "auto" else args.device
    model = model.to(device)
    print(
        f"VGGT-Omega {weights_file.name} loaded on {device} (precision={precision})",
        file=sys.stderr,
    )
    _MODEL_CACHE[key] = model
    return model


def decode_predictions(
    predictions: object, *, image_size: tuple[int, int], resolution: int
) -> dict[str, np.ndarray]:
    """Turn VGGT's raw output into window arrays in **this project's** convention.

    VGGT reports world-to-camera extrinsics and pixel intrinsics for the input
    image, while the depth map may live on a coarser grid - so the intrinsics are
    rescaled to the depth resolution here, which is the pair the stitcher
    actually uses.
    """
    def pick(*names: str) -> object | None:
        for name in names:
            if isinstance(predictions, dict) and name in predictions:
                return predictions[name]
            if hasattr(predictions, name):
                return getattr(predictions, name)
        return None

    pose_enc = pick("pose_enc", "pose_encoding", "camera_pose")
    depth = pick("depth", "depth_map")
    if pose_enc is None or depth is None:
        available = sorted(predictions.keys()) if isinstance(predictions, dict) else dir(predictions)
        raise NotImplementedError(
            "cannot decode VGGT output: expected a pose encoding and a depth map. "
            f"Available: {available}"
        )
    try:
        from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - backend import
        raise ImportError(
            "vggt_omega.utils.pose_enc.encoding_to_camera is required to decode the camera "
            "(that is the VGGT-Omega decoder; the original VGGT called it "
            "pose_encoding_to_extri_intri)"
        ) from exc

    import torch  # noqa: PLC0415

    # The demo decodes against the *preprocessed* image size, which is what the
    # intrinsics refer to (depth may live on a coarser grid - rescaled below).
    processed = pick("images")
    if processed is not None:
        image_size = (int(processed.shape[-2]), int(processed.shape[-1]))
    extrinsics, intrinsics = encoding_to_camera(
        torch.as_tensor(pose_enc),
        image_size,
    )
    extrinsics = np.asarray(extrinsics.detach().cpu(), dtype=np.float64).reshape(-1, 3, 4)
    intrinsics = np.asarray(intrinsics.detach().cpu(), dtype=np.float64).reshape(-1, 3, 3)
    # VGGT-Omega returns depth as (batch=1, views, H, W, 1). The demo strips the
    # trailing channel itself (`depth_map[..., 0]`) and drops the leading batch
    # dim, leaving (views, H, W) - which is this project's convention.
    depth_array = np.asarray(depth.detach().cpu(), dtype=np.float64)
    if depth_array.ndim >= 4 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if depth_array.ndim == 4 and depth_array.shape[0] == 1:
        depth_array = depth_array[0]
    if depth_array.ndim != 3:
        raise StageIOError(
            f"backend returned depth with shape {np.shape(depth)}; expected "
            "(batch, views, H, W, 1) or (views, H, W)"
        )
    confidence = pick("depth_conf", "depth_confidence")
    confidence_array = None
    if confidence is not None:
        confidence_array = np.asarray(confidence.detach().cpu(), dtype=np.float64)
        if confidence_array.ndim == 4 and confidence_array.shape[0] == 1:
            confidence_array = confidence_array[0]

    if depth_array.shape[0] != intrinsics.shape[0]:
        raise StageIOError(
            f"backend returned {intrinsics.shape[0]} camera(s) but "
            f"{depth_array.shape[0]} depth map(s)"
        )

    rotation_c2w, translation_c2w = invert_rigid(extrinsics[:, :3, :3], extrinsics[:, :3, 3])
    depth_size = (int(depth_array.shape[2]), int(depth_array.shape[1]))  # (w, h)
    image_wh = (int(image_size[1]), int(image_size[0]))
    scaled = scale_intrinsics(
        intrinsics, source_size=image_wh, target_size=depth_size
    )
    return {
        "rotation_c2w": rotation_c2w,
        "translation_c2w": translation_c2w,
        "intrinsics": scaled,
        "depth": depth_array,
        **({"depth_confidence": confidence_array} if confidence_array is not None else {}),
        "resolution": np.array([resolution]),
    }


def run_model(args: argparse.Namespace, rng: object) -> dict[str, np.ndarray]:
    """Run VGGT-Omega on one window and return window arrays (poses + depth)."""
    import torch  # noqa: PLC0415
    from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: PLC0415

    frames_dir = Path(args.frames)
    image_paths = [
        str(frames_dir / f"{index:06d}.jpg") for index in range(rng.start, rng.end)
    ]
    missing = [path for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frame(s) missing, e.g. {missing[0]}")

    model = load_model(args)
    images = load_and_preprocess_images(image_paths, image_resolution=args.resolution)
    device = "cuda" if args.device == "auto" else args.device
    images = images.to(device)
    # Images stay fp32: the aggregator autocasts internally, and the fp32 heads
    # want fp32 tensors.
    with torch.inference_mode():
        predictions = model(images)
    height, width = int(images.shape[-2]), int(images.shape[-1])
    return decode_predictions(predictions, image_size=(height, width), resolution=args.resolution)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT-Omega runner")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=416)
    parser.add_argument("--checkpoint", default="VGGT-Omega-1B-416-Reproduction")
    parser.add_argument("--third-party", default="third_party")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="weight precision: fp16 halves the 1B checkpoint (needed on a shared 12 GB GPU)",
    )
    add_shard_arguments(parser)
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
        selection = selection_from_args(args)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "num_frames": args.num_frames,
            "window": args.window,
            "overlap": args.overlap,
            "resolution": args.resolution,
            "checkpoint": args.checkpoint,
            # Precision changes the numbers, so it belongs in the unit identity:
            # an fp16 window must never be reused as if it were an fp32 one.
            "precision": args.precision,
        }
        chosen = select_windows(ranges, selection)
        if args.skip_existing:
            if partition_is_reusable(
                out_dir,
                stage="camera",
                ranges=chosen,
                params=params,
                selection=selection,
            ):
                emit(
                    {
                        "status": "ok",
                        "backend": "vggt",
                        "resolution": args.resolution,
                        "checkpoint": args.checkpoint,
                        "precision": args.precision,
                        "windows": [f"{r.start:06d}_{r.end - 1:06d}.npz" for r in chosen],
                        "output_dir": str(out_dir),
                        "reused": True,
                    }
                )
                return 0
        written: list[str] = []
        for rng in chosen:
            result = run_model(args, rng)
            window = camera_window_from_output(
                start=rng.start,
                end=rng.end,
                rotation_c2w=result["rotation_c2w"],
                translation_c2w=result["translation_c2w"],
                intrinsics=result["intrinsics"],
                depth=result["depth"],
                depth_confidence=result.get("depth_confidence"),
            )
            path = out_dir / f"{rng.start:06d}_{rng.end - 1:06d}.npz"
            save_camera_window(window, path)
            written.append(path.name)
        record_partition(
            out_dir,
            stage="camera",
            ranges=chosen,
            params=params,
            selection=selection,
        )
        emit(
            {
                "status": "ok",
                "backend": "vggt",
                "resolution": args.resolution,
                "checkpoint": args.checkpoint,
                "checkpoint_path": _RESOLVED["path"],
                "checkpoint_substituted": _RESOLVED["substituted"],
                "precision": args.precision,
                "windows": written,
                "output_dir": str(out_dir),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - reported through the runner protocol
        import traceback  # noqa: PLC0415

        traceback.print_exc()  # one-line JSON stays; the log keeps the stack
        emit({"status": "error", "backend": "vggt", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

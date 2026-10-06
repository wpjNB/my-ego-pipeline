#!/usr/bin/env python
"""Deterministic stand-in for the three model backends.

It speaks exactly the same runner protocol as the real backends (argparse CLI +
a single JSON summary line on stdout + artefacts on disk), so the orchestration
in ``src/ego3d_action`` can be exercised end to end without a GPU.

This is **not** a model and must never be mistaken for one: every artefact it
writes is derived from :mod:`ego3d_action.testing.synthetic`, the run is
requested explicitly through ``backends.mode: mock``, and the pipeline records
the mode in its stage metadata.

    python backends/mock_backend.py wilor  --frames DIR --out detection.npz --num-frames 120
    python backends/mock_backend.py hawor  --detection detection.npz --window-start 0 --window-end 16 --out w.npz --num-frames 120
    python backends/mock_backend.py vggt   --out-dir data/clip01/camera/windows --num-frames 120
    python backends/mock_backend.py truth  --out data/clip01/trajectory/trajectory.npz --num-frames 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.window import save_camera_window  # noqa: E402
from ego3d_action.fusion.trajectory import build_trajectory  # noqa: E402
from ego3d_action.io.serialization import save_json, save_npz  # noqa: E402
from ego3d_action.testing import synthetic  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shard_cli import (  # noqa: E402
    add_shard_arguments,
    partition_is_reusable,
    record_partition,
    select_windows,
    selection_from_args,
)


def _emit(payload: dict[str, object]) -> None:
    """Print the single machine-readable summary line."""
    print(json.dumps(payload, sort_keys=True))


def command_wilor(args: argparse.Namespace) -> int:
    num_frames = args.num_frames
    if num_frames is None:
        frames = sorted(Path(args.frames).glob(f"*.{args.image_format}"))
        if not frames:
            raise SystemExit(f"no frames found in {args.frames}")
        num_frames = len(frames)
    arrays = synthetic.synthetic_detections(
        num_frames, seed=args.seed, width=args.width, height=args.height
    )
    save_npz(args.out, **arrays)
    _emit(
        {
            "status": "ok",
            "backend": "mock-wilor",
            "num_frames": num_frames,
            "output": str(args.out),
        }
    )
    return 0


def command_hawor(args: argparse.Namespace) -> int:
    """Write every 16/8 window of the clip, exactly like the real runner."""
    from ego3d_action.hand.hawor import HaworClipRequest

    if args.detection:
        detection = np.load(args.detection, allow_pickle=False)
        num_frames = args.num_frames or int(detection["boxes"].shape[0])
    else:
        if args.num_frames is None:
            raise SystemExit("--num-frames is required when no detection artefact is given")
        num_frames = args.num_frames

    rotation, translation = synthetic.make_camera_trajectory(num_frames, seed=args.seed)
    world = synthetic.hand_world_trajectory(num_frames, seed=args.seed)
    camera = synthetic.camera_space_hands(world, rotation, translation, seed=args.seed)
    # A real backend only reconstructs hands the tracker kept, so the detection
    # artefact owns the validity mask whenever one is supplied.
    if args.detection:
        valid = np.asarray(detection["valid"], dtype=bool)
    else:
        valid = synthetic.hand_validity(num_frames, seed=args.seed)
    camera = np.where(valid[:, :, None, None], camera, np.nan)
    confidence = np.where(valid, 0.9, 0.0)

    request = HaworClipRequest(
        num_frames=num_frames,
        frames_dir=Path(args.frames or "."),
        window=args.window,
        overlap=args.overlap,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = selection_from_args(args)
    ranges = select_windows(request.ranges(), selection)
    params = {
        "num_frames": num_frames,
        "window": args.window,
        "overlap": args.overlap,
        "seed": args.seed,
    }
    if args.skip_existing and partition_is_reusable(
        out_dir, stage="hand", ranges=ranges, params=params, selection=selection
    ):
        _emit(
            {
                "status": "ok",
                "backend": "mock-hawor",
                "num_frames": num_frames,
                "windows": [f"{s:06d}_{e - 1:06d}.npz" for s, e in ranges],
                "output_dir": str(out_dir),
                "reused": True,
            }
        )
        return 0

    written: list[str] = []
    for start, end in ranges:
        path = out_dir / f"{start:06d}_{end - 1:06d}.npz"
        save_npz(
            path,
            start=np.array([start], dtype=np.int64),
            joints_camera=camera[start:end],
            valid=valid[start:end],
            confidence=confidence[start:end],
            root_rot=np.broadcast_to(np.eye(3), (end - start, 2, 3, 3)).copy(),
            betas=np.zeros((end - start, 2, 10)),
        )
        written.append(path.name)
    record_partition(
        out_dir, stage="hand", ranges=ranges, params=params, selection=selection
    )
    _emit(
        {
            "status": "ok",
            "backend": "mock-hawor",
            "num_frames": num_frames,
            "windows": written,
            "output_dir": str(out_dir),
        }
    )
    return 0


def command_vggt(args: argparse.Namespace) -> int:
    windows, _, _ = synthetic.build_camera_windows(
        args.num_frames,
        window=args.window,
        overlap=args.overlap,
        depth_size=(args.depth_height, args.depth_width),
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = selection_from_args(args)
    windows = select_windows(windows, selection)
    params = {
        "num_frames": args.num_frames,
        "window": args.window,
        "overlap": args.overlap,
        "resolution": args.resolution,
        "seed": args.seed,
    }
    if args.skip_existing and partition_is_reusable(
        out_dir, stage="camera", ranges=windows, params=params, selection=selection
    ):
        _emit(
            {
                "status": "ok",
                "backend": "mock-vggt",
                "resolution": args.resolution,
                "windows": [f"{w.start:06d}_{w.end - 1:06d}.npz" for w in windows],
                "output_dir": str(out_dir),
                "reused": True,
            }
        )
        return 0

    written: list[str] = []
    for window in windows:
        path = out_dir / f"{window.start:06d}_{window.end - 1:06d}.npz"
        save_camera_window(window, path)
        written.append(path.name)
    record_partition(
        out_dir, stage="camera", ranges=windows, params=params, selection=selection
    )
    _emit(
        {
            "status": "ok",
            "backend": "mock-vggt",
            "resolution": args.resolution,
            "windows": written,
            "output_dir": str(out_dir),
        }
    )
    return 0


def command_truth(args: argparse.Namespace) -> int:
    """Write the mock ground truth in the project's trajectory format.

    The reference is built exactly like a perfect backend would report it -
    camera-space joints plus the camera poses - and then handed to the same
    fusion code as the prediction, so the two are guaranteed to be expressed in
    the same ``World-0`` frame.
    """
    num_frames = args.num_frames
    rotation_raw, translation_raw = synthetic.make_camera_trajectory(num_frames, seed=args.seed)
    rotation, translation = synthetic.world0_camera_trajectory(num_frames, seed=args.seed)
    world = synthetic.hand_world_trajectory(num_frames, seed=args.seed)
    camera = synthetic.world_to_camera_points(world, rotation_raw, translation_raw)
    valid = synthetic.hand_validity(num_frames, seed=args.seed)
    intrinsics = np.broadcast_to(
        synthetic.make_intrinsics(args.depth_width, args.depth_height), (num_frames, 3, 3)
    ).copy()

    arrays, metadata = build_trajectory(
        joints_camera=np.where(valid[:, :, None, None], camera, np.nan),
        hand_valid=valid,
        hand_confidence=np.where(valid, 0.9, 0.0),
        camera_rotation_c2w=rotation,
        camera_translation_c2w=translation,
        camera_intrinsics=intrinsics,
        bbox=np.zeros((num_frames, 2, 4)),
        track_id=np.zeros((num_frames, 2), dtype=np.int64),
        fps=args.fps,
    )
    save_npz(args.out, **arrays)
    save_json(
        Path(args.out).with_name("metadata.json"),
        {**metadata, "source": "mock_backend.truth", "seed": args.seed},
    )
    _emit(
        {
            "status": "ok",
            "backend": "mock-truth",
            "num_frames": num_frames,
            "output": str(args.out),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="deterministic mock model backend")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_backend_args(sub_parser: argparse.ArgumentParser) -> None:
        """Flags the real runners accept and the mock simply ignores."""
        sub_parser.add_argument("--device", default="auto")
        sub_parser.add_argument("--third-party", default="third_party")
        sub_parser.add_argument("--weights", default="weights")

    wilor = sub.add_parser("wilor", help="synthetic WiLoR detections")
    wilor.add_argument("--seed", type=int, default=0)
    wilor.add_argument("--frames", default=None, help="frame directory (to infer the length)")
    wilor.add_argument("--out", required=True)
    wilor.add_argument("--num-frames", type=int, default=None)
    wilor.add_argument("--width", type=int, default=synthetic.DEFAULT_WIDTH)
    wilor.add_argument("--height", type=int, default=synthetic.DEFAULT_HEIGHT)
    wilor.add_argument("--image-format", default="jpg")
    wilor.add_argument("--batch-size", type=int, default=4)
    wilor.add_argument("--conf", type=float, default=0.1)
    add_backend_args(wilor)
    wilor.set_defaults(func=command_wilor)

    hawor = sub.add_parser("hawor", help="synthetic camera-space hand window")
    hawor.add_argument("--seed", type=int, default=0)
    hawor.add_argument("--detection", default=None)
    hawor.add_argument("--out-dir", required=True)
    hawor.add_argument("--num-frames", type=int, default=None)
    hawor.add_argument("--window", type=int, default=16)
    hawor.add_argument("--overlap", type=int, default=8)
    hawor.add_argument("--frames", default=None)
    # CLI parity with the real HaWoR runner; synthetic hands do not consume these.
    hawor.add_argument("--focal", type=float, default=None)
    hawor.add_argument("--camera-windows", default=None)
    hawor.add_argument("--precision", default=None)
    hawor.add_argument("--crop-size", type=int, default=None)
    hawor.add_argument("--box-pad", type=float, default=None)
    add_shard_arguments(hawor)
    add_backend_args(hawor)
    hawor.set_defaults(func=command_hawor)

    vggt = sub.add_parser("vggt", help="synthetic VGGT-Omega windows")
    vggt.add_argument("--seed", type=int, default=0)
    vggt.add_argument("--out-dir", required=True)
    vggt.add_argument("--num-frames", type=int, required=True)
    vggt.add_argument("--window", type=int, default=200)
    vggt.add_argument("--overlap", type=int, default=40)
    vggt.add_argument("--resolution", type=int, default=416)
    vggt.add_argument("--depth-height", type=int, default=synthetic.DEFAULT_DEPTH_SIZE[0])
    vggt.add_argument("--depth-width", type=int, default=synthetic.DEFAULT_DEPTH_SIZE[1])
    vggt.add_argument("--checkpoint", default="VGGT-Omega-1B-416-Reproduction")
    vggt.add_argument("--frames", default=None)
    add_shard_arguments(vggt)
    add_backend_args(vggt)
    vggt.set_defaults(func=command_vggt)

    truth = sub.add_parser("truth", help="synthetic ground-truth trajectory")
    truth.add_argument("--seed", type=int, default=0)
    truth.add_argument("--out", required=True)
    truth.add_argument("--num-frames", type=int, required=True)
    truth.add_argument("--fps", type=float, default=synthetic.DEFAULT_FPS)
    truth.add_argument("--depth-height", type=int, default=synthetic.DEFAULT_DEPTH_SIZE[0])
    truth.add_argument("--depth-width", type=int, default=synthetic.DEFAULT_DEPTH_SIZE[1])
    truth.set_defaults(func=command_truth)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - the runner protocol reports failures as JSON
        _emit({"status": "error", "backend": f"mock-{args.command}", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

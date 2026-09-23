#!/usr/bin/env python
"""HaWoR runner: executed **inside** the HaWoR environment (``ego3d_hawor``).

    python backends/hawor_runner.py --check
    python backends/hawor_runner.py --detection detection.npz --out-dir hand/windows \
        --frames data/clip/frames --num-frames 450 --window 16 --overlap 8 \
        --third-party third_party --weights weights --device cuda

How this differs from HaWoR's own demo
--------------------------------------

``demo.py`` runs ``detect_track_video`` (WiLoR + HaWoR's own ``thresh=0.2``
rule set), then ``hawor_motion_estimation`` -> ``hawor_slam`` ->
``hawor_infiller``. This runner replaces the first step: the tracking decision
comes from **our** conservative tracker via ``detection.npz``, written into the
``model_tracks.npy`` structure ``hawor_motion_estimation`` reads. HaWoR then
reconstructs exactly the frames Phase 1 kept.

Two facts discovered by reading HaWoR's source, both load-bearing here:

1. ``hawor_infiller`` hard-depends on ``SLAM/hawor_slam_w_scale_*.npz`` and
   produces hands in **HaWoR's SLAM world frame** (the comment in its source
   says "camera space to world space"). So the runner also calls ``hawor_slam``
   and then converts the hands back into **camera space** with the SLAM poses -
   which is what Phase 2 must hand to VGGT-based fusion. VGGT still owns the
   metric world trajectory; SLAM is only a coordinate carrier here.
2. ``run_mano``/``run_mano_left`` are what turn ``(trans, rot, hand_pose,
   betas)`` into 21 landmarks, and they need the MANO model inside the HaWoR
   checkout.

The conversion of 21 camera-space joints into our 16/8 window artefacts lives in
``ego3d_action.hand.hawor`` and is unit-tested. What cannot be verified without
the checkpoint and a GPU is the HaWoR call sequence itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.hand.hawor import (  # noqa: E402
    hawor_tracks_from_detection,
    hand_windows_from_joints,
    save_hawor_tracks,
)
from ego3d_action.io.serialization import load_npz  # noqa: E402

BACKEND_MODULE = "hawor"
MANO_CANDIDATES = ("_DATA/data/mano/MANO_RIGHT.pkl", "_DATA/data/mano/MANO_RIGHT.npz")


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def backend_available(third_party: Path) -> tuple[bool, str]:
    """Report whether HaWoR (and its MANO asset) can be imported."""
    checkout = third_party / "HaWoR"
    if not checkout.is_dir():
        return False, f"HaWoR checkout not found at {checkout}"
    mano = next((checkout / rel for rel in MANO_CANDIDATES if (checkout / rel).exists()), None)
    if mano is None:
        return False, f"no MANO model under {checkout}/_DATA/data/mano"
    sys.path.insert(0, str(checkout))
    try:
        __import__(BACKEND_MODULE)
    except ImportError as exc:
        return False, f"cannot import '{BACKEND_MODULE}' from {checkout}: {exc}"
    return True, f"'{BACKEND_MODULE}' importable from {checkout}, MANO at {mano}"


def build_hawor_args(args: argparse.Namespace, seq_folder: Path, frames_dir: Path) -> object:
    """Assemble the namespace HaWoR's functions expect.

    HaWoR derives ``seq_folder`` from ``video_path`` (``<dir>/<stem>``) and reads
    the frames from ``<seq_folder>/extracted_images``. We point it at our own
    Phase-0 frames instead of letting it re-extract them.
    """
    images = seq_folder / "extracted_images"
    images.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(sorted(frames_dir.glob("*.jpg"))):
        target = images / f"{index:04d}.jpg"
        if target.exists():
            continue
        try:
            os.link(frame, target)
        except OSError:  # different filesystem - fall back to a symlink
            target.symlink_to(frame.resolve())
    return argparse.Namespace(
        video_path=str(seq_folder.parent / f"{seq_folder.name}.mp4"),
        input_type="file",
        checkpoint=str(Path(args.weights) / "hawor" / "checkpoints" / "hawor.ckpt"),
        infiller_weight=str(Path(args.weights) / "hawor" / "checkpoints" / "infiller.pt"),
        img_focal=args.focal,
        vis_mode="cam",
    )


def run_model(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Run HaWoR over the clip and return camera-space hands for every frame."""
    third_party = Path(args.third_party)
    frames_dir = Path(args.frames)
    seq_folder = Path(args.out_dir).parent / "hawor_seq"
    seq_folder.mkdir(parents=True, exist_ok=True)

    # 1. our tracking becomes HaWoR's model_tracks.npy
    detection = load_npz(args.detection, required=("boxes", "confidence", "valid"))
    boxes = np.asarray(detection["boxes"], dtype=np.float64)
    confidence = np.asarray(detection["confidence"], dtype=np.float64)
    valid = np.asarray(detection["valid"], dtype=bool)
    model_boxes, tracks = hawor_tracks_from_detection(
        boxes=boxes, confidence=confidence, valid=valid
    )
    start_idx, end_idx = 0, int(boxes.shape[0])
    save_hawor_tracks(
        seq_folder / f"tracks_{start_idx}_{end_idx}", model_boxes, tracks
    )
    print(
        f"HaWoR will reconstruct {int(valid.sum())} tracked hand-frames "
        f"(left {int(valid[:, 0].sum())}, right {int(valid[:, 1].sum())})",
        file=sys.stderr,
    )

    # 2. HaWoR itself
    sys.path.insert(0, str(third_party / "HaWoR"))
    from hawor.utils.process import run_mano, run_mano_left  # noqa: PLC0415
    from lib.eval_utils.custom_utils import load_slam_cam  # noqa: PLC0415
    from scripts.scripts_test_video.hawor_slam import hawor_slam  # noqa: PLC0415
    from scripts.scripts_test_video.hawor_video import (  # noqa: PLC0415
        hawor_infiller,
        hawor_motion_estimation,
    )
    import torch  # noqa: PLC0415

    hawor_args = build_hawor_args(args, seq_folder, frames_dir)
    frame_chunks_all, _img_focal = hawor_motion_estimation(
        hawor_args, start_idx, end_idx, seq_folder
    )
    hawor_slam(hawor_args, start_idx, end_idx)
    slam_path = seq_folder / f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    if not slam_path.is_file():
        raise FileNotFoundError(
            f"HaWoR's SLAM step did not produce {slam_path}; the infiller cannot run without it"
        )
    r_w2c, t_w2c, _, _ = load_slam_cam(str(slam_path))
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        hawor_args, start_idx, end_idx, frame_chunks_all
    )

    # 3. MANO landmarks, then back to camera space
    torch.set_grad_enabled(False)
    landmarks = np.zeros((2, end_idx, 21, 3), dtype=np.float64)
    for hand, run in ((0, run_mano_left), (1, run_mano)):
        sl = slice(hand, hand + 1)
        mano = run(pred_trans[sl], pred_rot[sl], pred_hand_pose[sl], betas=pred_betas[sl])
        joints = mano["joints"] if isinstance(mano, dict) else mano
        joints = np.asarray(getattr(joints, "cpu", lambda: joints)())
        landmarks[hand] = np.asarray(joints, dtype=np.float64).reshape(end_idx, 21, 3)

    camera_space = np.einsum(
        "tji,thnj->thni", np.asarray(r_w2c, dtype=np.float64), landmarks
    ) + np.asarray(t_w2c, dtype=np.float64)[:, None, None, :]
    hand_valid = np.asarray(pred_valid, dtype=np.float64).T > 0.5
    all_valid = hand_valid & valid
    return {
        "joints_camera": np.where(all_valid[:, :, None, None], camera_space, np.nan),
        "valid": all_valid,
        "confidence": np.where(all_valid, confidence, 0.0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HaWoR runner")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--detection", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--focal", type=float, default=None, help="pixel focal length for HaWoR")
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
        if not args.detection or not args.out_dir or not args.frames:
            raise ValueError("--detection, --out-dir and --frames are required")
        result = run_model(args)
        written = hand_windows_from_joints(
            result["joints_camera"],
            result["valid"],
            result["confidence"],
            out_dir=args.out_dir,
            window=args.window,
            overlap=args.overlap,
        )
        emit(
            {
                "status": "ok",
                "backend": "hawor",
                "num_frames": int(result["joints_camera"].shape[0]),
                "valid_frames": int(np.count_nonzero(result["valid"])),
                "windows": [path.name for path in written],
                "output_dir": str(args.out_dir),
                "frame": "camera (converted from HaWoR's SLAM world with its own poses)",
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - reported through the runner protocol
        emit({"status": "error", "backend": "hawor", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Reference-versus-prediction viewer.

Projects the reference wrist on the egocentric RGB using the reference camera,
optionally together with a prediction, and writes a video plus a few stills.
``--skeleton`` additionally draws the reference's 21 joints (it needs a
MANO-built reference); both the wrist and the joints go through the same
world -> camera transform, so what you see is the trajectory contract, not an
ad-hoc projection:

    # check that the imported ground truth lands on the real hand
    python scripts/render_gt_vs_pred.py --config configs/hot3d.yaml --clip hot3d_ep000 \
        --data-root data/hot3d \
        --prediction data/hot3d/hot3d_ep000/trajectory/ground_truth.npz \
        --ground-truth data/hot3d/hot3d_ep000/trajectory/ground_truth.npz

    # compare a real prediction against it
    python scripts/render_gt_vs_pred.py --config configs/hot3d.yaml --clip hot3d_ep000 \
        --data-root data/hot3d \
        --prediction data/hot3d/hot3d_ep000/trajectory/trajectory.npz \
        --ground-truth data/hot3d/hot3d_ep000/trajectory/ground_truth.npz

Both trajectories must already share one world frame; the pipeline writes
``World-0`` (the first frame's camera) and the LeRobot importer re-anchors the
HOT3D reference to the same convention, so no extra alignment is applied here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.evaluation.dataset import load_trajectory  # noqa: E402
from ego3d_action.evaluation.gt_align import apply_offset, camera_frame_offset  # noqa: E402
from ego3d_action.io.frames import load_frame_set  # noqa: E402
from ego3d_action.visualization.overlay import write_wrist_comparison_video  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "GT vs prediction viewer")
    parser.add_argument("--prediction", default=None, help="predicted trajectory (optional)")
    parser.add_argument("--ground-truth", required=True, help="reference trajectory")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stills", type=int, default=4, help="number of single-frame PNGs to save")
    parser.add_argument(
        "--output",
        default=None,
        help="write the video here (default: <clip>/visualization/gt_vs_pred.mp4; "
        "single-frame stills go to <stem>_stills/)",
    )
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="also draw the reference's 21 joints (needs a MANO-built reference)",
    )
    parser.add_argument(
        "--align-gt",
        action="store_true",
        help="align the GT hands with a per-hand constant camera-frame translation "
        "estimated against the prediction (removes the mirror GT's rigid hand-eye "
        "offset, so the overlay lands on the visible hands)",
    )
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        if args.stride <= 0:
            return fail("--stride must be positive")

        reference = load_trajectory(args.ground_truth, fps=args.fps)
        prediction = load_trajectory(args.prediction, fps=args.fps) if args.prediction else None
        if prediction is not None and prediction.num_frames != reference.num_frames:
            raise StageIOError(
                f"prediction has {prediction.num_frames} frames but the reference has "
                f"{reference.num_frames}"
            )
        if args.align_gt:
            if prediction is None:
                return fail("--align-gt needs --prediction to estimate the offset from")
            offsets = camera_frame_offset(
                prediction.joints_world,
                reference.joints_world,
                pred_rotation_c2w=prediction.rotation_c2w,
                pred_translation_c2w=prediction.translation_c2w,
                gt_rotation_c2w=reference.rotation_c2w,
                gt_translation_c2w=reference.translation_c2w,
                pred_valid=prediction.valid,
                gt_valid=reference.valid,
            )
            print(
                "GT translation alignment (camera frame, per hand): "
                f"left {np.round(offsets[0] * 100.0, 1)} cm, right {np.round(offsets[1] * 100.0, 1)} cm"
            )
            from dataclasses import replace  # noqa: PLC0415

            reference = replace(
                reference,
                joints_world=apply_offset(
                    reference.joints_world,
                    reference.rotation_c2w,
                    reference.translation_c2w,
                    offsets,
                ),
            )

        frames = load_frame_set(layout.data_root, layout.clip)
        if frames.num_frames != reference.num_frames:
            raise StageIOError(
                f"clip has {frames.num_frames} frames but the reference has "
                f"{reference.num_frames}; re-import the episode"
            )

        total = args.max_frames or frames.num_frames
        indices = list(range(0, min(total, frames.num_frames), args.stride))
        subset = [frames.paths[i] for i in indices]
        valid = reference.valid[indices]
        if prediction is not None:
            valid = valid & prediction.valid[indices]
        stills = [indices[min(len(indices) - 1, k * max(1, len(indices) // max(1, args.stills)))]
                  for k in range(args.stills)] if args.stills else []
        still_steps = sorted({indices.index(s) for s in stills if s in indices})

        out_path = (
            Path(args.output) if args.output else layout.visualization_dir / "gt_vs_pred.mp4"
        )
        write_wrist_comparison_video(
            subset,
            reference.joints_world[indices],
            reference.rotation_c2w[indices],
            reference.translation_c2w[indices],
            _intrinsics_for(frames, layout),
            valid,
            out_path,
            prediction_world=None if prediction is None else prediction.joints_world[indices],
            draw_skeleton=args.skeleton,
            fps=float(args.fps or frames.fps) / float(args.stride),
            still_indices=still_steps,
            note="GT aligned to prediction (constant wrist offset)" if args.align_gt else None,
        )
        print(f"wrote {out_path}")

        if prediction is not None:
            both = reference.valid & prediction.valid
            error = np.linalg.norm(
                reference.joints_world[:, :, 0, :] - prediction.joints_world[:, :, 0, :], axis=-1
            )
            finite = both & np.isfinite(error)
            if finite.any():
                print(
                    f"wrist error: mean {1000.0 * float(error[finite].mean()):.2f} mm, "
                    f"median {1000.0 * float(np.median(error[finite])):.2f} mm, "
                    f"over {int(finite.sum())} hand-frames"
                )
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


def _intrinsics_for(frames, layout) -> np.ndarray:
    """Clip intrinsics: from the reference trajectory when present, else the camera.

    A reference whose ``camera_K`` is not finite (the HOT3D-Clips mirror ships
    no intrinsics) falls back to Phase 3's first camera window instead of
    feeding NaN into every projection.
    """
    ground_truth = layout.trajectory_dir / "ground_truth.npz"
    if ground_truth.is_file():
        from ego3d_action.io.serialization import load_npz

        candidate = np.asarray(
            load_npz(ground_truth, required=("camera_K",))["camera_K"], dtype=np.float64
        )
        if np.isfinite(candidate).all():
            return candidate
    for path in sorted(layout.camera_windows_dir.glob("*.npz")):
        from ego3d_action.camera.depth import scale_intrinsics

        data = load_npz(path, required=("intrinsics", "depth"))
        depth = np.asarray(data["depth"])
        scaled = scale_intrinsics(
            np.asarray(data["intrinsics"])[0],
            source_size=(int(depth.shape[2]), int(depth.shape[1])),
            target_size=(frames.width, frames.height),
        )
        return np.broadcast_to(scaled, (frames.num_frames, 3, 3)).copy()
    from ego3d_action.testing.synthetic import make_intrinsics

    base = make_intrinsics(frames.width, frames.height)
    return np.broadcast_to(base, (frames.num_frames, 3, 3)).copy()


if __name__ == "__main__":
    raise SystemExit(main())

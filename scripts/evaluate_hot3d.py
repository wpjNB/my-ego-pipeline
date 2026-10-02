#!/usr/bin/env python
"""Phase 7: HOT3D Action-MPJPE / Coverage / FPS.

    python scripts/evaluate_hot3d.py \\
        --prediction data/demo01/trajectory/trajectory.npz \\
        --ground-truth data/hot3d/demo01/trajectory.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.evaluation.action_mpjpe import action_mpjpe  # noqa: E402
from ego3d_action.evaluation.benchmark import (  # noqa: E402
    EvaluationReport,
    camera_pose_error_mm,
    measure_fps,
    stopwatch,
)
from ego3d_action.evaluation.dataset import load_trajectory  # noqa: E402
from ego3d_action.evaluation.gt_align import apply_offset, camera_frame_offset  # noqa: E402
from ego3d_action.io.serialization import save_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "HOT3D evaluation")
    parser.add_argument("--prediction", required=True, help="predicted trajectory.npz")
    parser.add_argument("--ground-truth", required=True, help="reference trajectory.npz")
    parser.add_argument("--fps", type=float, default=None, help="override the clip fps")
    parser.add_argument(
        "--pipeline-seconds",
        type=float,
        default=None,
        help="wall time of the full pipeline; used for the FPS column",
    )
    parser.add_argument("--json", dest="json_out", default=None, help="write the report as JSON")
    parser.add_argument(
        "--align-gt",
        action="store_true",
        help="align the GT hands with a per-hand constant camera-frame translation "
        "before scoring. The HOT3D-mirror hand GT carries a rigid hand-eye offset "
        "(constant 1.7-6.9 cm vs 2.4-3.7 cm per-frame residual, measured 2026-10-01); "
        "this removes it by construction, so the numbers measure time-varying "
        "agreement, not absolute placement. Camera error is unaffected.",
    )
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        chunk_seconds = float(context.config.get("evaluation.chunk_seconds", 1.0))

        prediction = load_trajectory(args.prediction, fps=args.fps)
        ground_truth = load_trajectory(args.ground_truth, fps=args.fps)
        if prediction.num_frames != ground_truth.num_frames:
            return fail(
                f"frame count mismatch: prediction has {prediction.num_frames}, "
                f"ground truth has {ground_truth.num_frames}"
            )
        if args.align_gt:
            offsets = camera_frame_offset(
                prediction.joints_world,
                ground_truth.joints_world,
                pred_rotation_c2w=prediction.rotation_c2w,
                pred_translation_c2w=prediction.translation_c2w,
                gt_rotation_c2w=ground_truth.rotation_c2w,
                gt_translation_c2w=ground_truth.translation_c2w,
                pred_valid=prediction.valid,
                gt_valid=ground_truth.valid,
            )
            print(
                "GT translation alignment (camera frame, per hand): "
                f"left {np.round(offsets[0] * 100.0, 1)} cm, right {np.round(offsets[1] * 100.0, 1)} cm"
            )
            from dataclasses import replace  # noqa: PLC0415

            aligned_world = apply_offset(
                ground_truth.joints_world,
                ground_truth.rotation_c2w,
                ground_truth.translation_c2w,
                offsets,
            )
            ground_truth = replace(ground_truth, joints_world=aligned_world)

        with stopwatch() as timer:
            result = action_mpjpe(
                prediction.joints_world,
                ground_truth.joints_world,
                prediction_rotation_c2w=prediction.rotation_c2w,
                prediction_translation_c2w=prediction.translation_c2w,
                ground_truth_rotation_c2w=ground_truth.rotation_c2w,
                ground_truth_translation_c2w=ground_truth.translation_c2w,
                fps=prediction.fps,
                chunk_seconds=chunk_seconds,
                prediction_valid=prediction.valid,
                ground_truth_valid=ground_truth.valid,
            )

        usable = prediction.valid & ground_truth.valid
        coverage = float(np.mean(usable)) if usable.size else 0.0
        valid_frames = np.asarray(
            prediction.valid.any(axis=1) & ground_truth.valid.any(axis=1), dtype=bool
        )
        camera_error = camera_pose_error_mm(
            prediction.translation_c2w,
            ground_truth.translation_c2w,
            valid=valid_frames,
        )

        # The headline numbers include gap-filled frames; when the prediction
        # carries an interpolation mask, report the predicted-only numbers next
        # to them so the two can never be confused.
        extra: dict[str, float] = {}
        interpolated = prediction.interpolated_mask
        if interpolated.any():
            real_valid = prediction.valid & ~interpolated
            real_only = action_mpjpe(
                prediction.joints_world,
                ground_truth.joints_world,
                prediction_rotation_c2w=prediction.rotation_c2w,
                prediction_translation_c2w=prediction.translation_c2w,
                ground_truth_rotation_c2w=ground_truth.rotation_c2w,
                ground_truth_translation_c2w=ground_truth.translation_c2w,
                fps=prediction.fps,
                chunk_seconds=chunk_seconds,
                prediction_valid=real_valid,
                ground_truth_valid=ground_truth.valid,
            )
            extra = {
                "coverage_predicted_only": float(np.mean(real_valid & ground_truth.valid)),
                "coverage_interpolated": float(np.mean(prediction.valid & interpolated)),
                "action_mpjpe_predicted_only_mm": real_only.action_mpjpe_mm,
            }

        if args.pipeline_seconds is not None:
            fps = measure_fps(prediction.num_frames, float(args.pipeline_seconds))
            fps_source = "pipeline"
        else:
            fps = measure_fps(prediction.num_frames, float(timer["elapsed"]))
            fps_source = "evaluation"

        report = EvaluationReport.from_result(
            result,
            coverage=coverage,
            fps=fps,
            num_frames=prediction.num_frames,
            camera_error=camera_error,
            extra=extra,
        )
        payload = report.as_dict()
        payload["fps_source"] = fps_source  # type: ignore[assignment]
        payload["joint_coverage"] = result.joint_coverage
        print(report.format())
        print(
            f"referenced joints : {100.0 * result.joint_coverage:.2f} % of "
            f"(frame, hand, joint) terms"
        )
        if str(ground_truth.metadata.get("hand_joints", "")) == "wrist_only":
            print(
                "note: the reference trajectory carries the wrist only (no MANO mesh model "
                "available to derive finger joints), so every number above is a "
                "wrist-level Action-MPJPE"
            )
        if payload["fps_source"] != "pipeline":
            print(
                "note: FPS above is the evaluation throughput; pass --pipeline-seconds to report "
                "end-to-end pipeline FPS instead"
            )
        if args.json_out:
            save_json(args.json_out, payload)
            print(f"report written to {args.json_out}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

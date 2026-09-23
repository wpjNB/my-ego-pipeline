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
        )
        payload = report.as_dict()
        payload["fps_source"] = fps_source  # type: ignore[assignment]
        print(report.format())
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

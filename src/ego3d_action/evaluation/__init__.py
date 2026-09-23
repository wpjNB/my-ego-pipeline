"""Phase 7: HOT3D evaluation (Action-MPJPE / coverage / FPS)."""

from __future__ import annotations

from .action_mpjpe import ActionMPJPEResult, action_mpjpe
from .benchmark import EvaluationReport, camera_pose_error_mm, measure_fps, stopwatch
from .coverage import coverage_by_hand, coverage_ratio, missing_runs
from .dataset import EvalTrajectory, load_trajectory

__all__ = [
    "ActionMPJPEResult",
    "EvaluationReport",
    "EvalTrajectory",
    "action_mpjpe",
    "camera_pose_error_mm",
    "coverage_by_hand",
    "coverage_ratio",
    "load_trajectory",
    "measure_fps",
    "missing_runs",
    "stopwatch",
]

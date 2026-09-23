"""Reporting: Action-MPJPE / Coverage / FPS plus the auxiliary error columns."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..errors import StageIOError
from .action_mpjpe import ActionMPJPEResult

Array = np.ndarray


@dataclass
class EvaluationReport:
    """The numbers printed by ``scripts/evaluate_hot3d.py``."""

    action_mpjpe_mm: float
    coverage: float
    fps: float
    num_frames: int
    camera_error: float = float("nan")
    wrist_error_mm: float = float("nan")
    depth_error_mm: float = float("nan")
    per_hand_mm: tuple[float, float] = (float("nan"), float("nan"))
    num_chunks: int = 0
    num_terms: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        result: ActionMPJPEResult,
        *,
        coverage: float,
        fps: float,
        num_frames: int,
        camera_error: float = float("nan"),
        extra: dict[str, float] | None = None,
    ) -> "EvaluationReport":
        return cls(
            action_mpjpe_mm=result.action_mpjpe_mm,
            coverage=coverage,
            fps=fps,
            num_frames=num_frames,
            camera_error=camera_error,
            wrist_error_mm=result.wrist_mm,
            depth_error_mm=result.depth_mm,
            per_hand_mm=(float(result.per_hand_mm[0]), float(result.per_hand_mm[1])),
            num_chunks=result.num_chunks,
            num_terms=result.num_terms,
            extra=dict(extra or {}),
        )

    def format(self) -> str:
        return "\n".join(
            [
                "===============================",
                "HOT3D Evaluation",
                "===============================",
                f"Action MPJPE : {self.action_mpjpe_mm:.4f} mm",
                f"Coverage     : {100.0 * self.coverage:.2f} %",
                f"FPS          : {self.fps:.2f}",
                "",
                f"Camera error : {self.camera_error:.4f}",
                f"Wrist error  : {self.wrist_error_mm:.4f}",
                f"Depth error  : {self.depth_error_mm:.4f}",
                "===============================",
            ]
        )

    def as_dict(self) -> dict[str, float]:
        payload = {
            "action_mpjpe_mm": self.action_mpjpe_mm,
            "coverage": self.coverage,
            "fps": self.fps,
            "num_frames": float(self.num_frames),
            "num_chunks": float(self.num_chunks),
            "num_terms": float(self.num_terms),
            "camera_error": self.camera_error,
            "wrist_error_mm": self.wrist_error_mm,
            "depth_error_mm": self.depth_error_mm,
            "left_mm": self.per_hand_mm[0],
            "right_mm": self.per_hand_mm[1],
        }
        payload.update(self.extra)
        return payload


def measure_fps(num_frames: int, elapsed_seconds: float) -> float:
    """Frames per second, guarding against a zero/nonsense timer."""
    if num_frames < 0:
        raise StageIOError(f"num_frames must be >= 0, got {num_frames}")
    if elapsed_seconds <= 0.0:
        raise StageIOError(f"elapsed_seconds must be positive, got {elapsed_seconds}")
    return num_frames / elapsed_seconds


@contextmanager
def stopwatch() -> Iterator[dict[str, float]]:
    """Time a block: ``with stopwatch() as sw: ...`` then ``sw["elapsed"]``."""
    record: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record.setdefault("elapsed", time.perf_counter() - start)


def camera_pose_error_mm(
    prediction_translation_c2w: Array,
    ground_truth_translation_c2w: Array,
    *,
    valid: Array | None = None,
) -> float:
    """Mean camera-centre distance (millimetres) for the report."""
    pred_t = np.asarray(prediction_translation_c2w, dtype=np.float64)
    gt_t = np.asarray(ground_truth_translation_c2w, dtype=np.float64)
    if pred_t.shape != gt_t.shape:
        raise StageIOError(f"camera translation shapes differ: {pred_t.shape} vs {gt_t.shape}")
    err = np.linalg.norm(pred_t - gt_t, axis=-1)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool).reshape(-1)
        if mask.shape[0] != err.shape[0]:
            raise StageIOError(f"valid length {mask.shape[0]} != {err.shape[0]}")
        if not mask.any():
            return float("nan")
        err = err[mask]
    return float(1000.0 * np.mean(err))


def ablation_row(
    name: str,
    report: EvaluationReport,
) -> dict[str, float | str]:
    """One row of the target ablation table (section 10 of the spec)."""
    return {
        "pipeline": name,
        "mpjpe_mm": report.action_mpjpe_mm,
        "coverage": report.coverage,
        "fps": report.fps,
    }

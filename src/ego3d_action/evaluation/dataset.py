"""Loading the trajectory artefacts used for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import StageIOError
from ..io.serialization import load_json, load_npz, validate_trajectory

Array = np.ndarray


@dataclass(frozen=True)
class EvalTrajectory:
    """The subset of ``trajectory.npz`` that evaluation needs."""

    joints_world: Array  # [T, 2, 21, 3]
    valid: Array  # [T, 2]
    rotation_c2w: Array  # [T, 3, 3]
    translation_c2w: Array  # [T, 3]
    fps: float
    source: Path
    num_frames: int

    @property
    def coverage(self) -> float:
        return float(np.mean(self.valid)) if self.valid.size else 0.0


def load_trajectory(
    path: str | Path, *, fps: float | None = None, strict: bool = True
) -> EvalTrajectory:
    """Load and validate a trajectory artefact.

    Args:
        path: ``trajectory.npz``.
        fps: override; otherwise read from the sibling ``metadata.json``.
        strict: also validate array shapes against the section-7 contract.

    Raises:
        StageIOError: on a missing/misshaped artefact or an unknown fps.
    """
    source = Path(path)
    data = load_npz(source)
    if strict:
        problems = validate_trajectory(data, strict=False)
        if problems:
            raise StageIOError(
                f"{source} is not a valid trajectory artefact: {'; '.join(problems)}"
            )
    for name in ("hand_xyz_world", "hand_valid", "camera_R_c2w", "camera_t_c2w"):
        if name not in data:
            raise StageIOError(f"{source} is missing the field '{name}'")

    if fps is None:
        metadata_path = source.with_name("metadata.json")
        if not metadata_path.is_file():
            # Ablation runs write <stem>.json next to the artefact.
            metadata_path = source.with_suffix(".json")
        if not metadata_path.is_file():
            raise StageIOError(
                f"cannot determine fps: {metadata_path} does not exist; pass --fps explicitly"
            )
        metadata = load_json(metadata_path)
        if "fps" not in metadata:
            raise StageIOError(f"{metadata_path} has no 'fps' field; pass --fps explicitly")
        fps = float(metadata["fps"])

    joints = np.asarray(data["hand_xyz_world"], dtype=np.float64)
    valid = np.asarray(data["hand_valid"], dtype=bool)
    return EvalTrajectory(
        joints_world=joints,
        valid=valid,
        rotation_c2w=np.asarray(data["camera_R_c2w"], dtype=np.float64),
        translation_c2w=np.asarray(data["camera_t_c2w"], dtype=np.float64),
        fps=float(fps),
        source=source,
        num_frames=int(joints.shape[0]),
    )

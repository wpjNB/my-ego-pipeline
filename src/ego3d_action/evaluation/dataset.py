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
    metadata: dict[str, object]
    interpolated: Array | None = None  # [T, 2] gap-filled frames (None = artefact predates P2)

    @property
    def coverage(self) -> float:
        return float(np.mean(self.valid)) if self.valid.size else 0.0

    @property
    def interpolated_mask(self) -> Array:
        """``[T, 2]`` bool mask of gap-filled frames (all ``False`` if absent)."""
        if self.interpolated is None:
            return np.zeros((self.num_frames, 2), dtype=bool)
        return np.asarray(self.interpolated, dtype=bool)


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

    metadata_path = source.with_name("metadata.json")
    if not metadata_path.is_file():
        # Reference/ablation runs write <stem>.json next to the artefact.
        metadata_path = source.with_suffix(".json")
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        metadata = load_json(metadata_path)
    if fps is None:
        if not metadata_path.is_file():
            raise StageIOError(
                f"cannot determine fps: neither {source.with_name('metadata.json')} nor "
                f"{source.with_suffix('.json')} exists; pass --fps explicitly"
            )
        if "fps" not in metadata:
            raise StageIOError(f"{metadata_path} has no 'fps' field; pass --fps explicitly")
        fps = float(metadata["fps"])

    joints = np.asarray(data["hand_xyz_world"], dtype=np.float64)
    valid = np.asarray(data["hand_valid"], dtype=bool)
    interpolated = (
        np.asarray(data["hand_interpolated"], dtype=bool)
        if "hand_interpolated" in data
        else None
    )
    return EvalTrajectory(
        joints_world=joints,
        valid=valid,
        rotation_c2w=np.asarray(data["camera_R_c2w"], dtype=np.float64),
        translation_c2w=np.asarray(data["camera_t_c2w"], dtype=np.float64),
        fps=float(fps),
        source=source,
        num_frames=int(joints.shape[0]),
        metadata=metadata,
        interpolated=interpolated,
    )

"""On-disk layout of one clip's artefacts.

This mirrors section 5 of the project spec exactly::

    data/<clip>/
    ├── frames/                 000000.jpg ...
    ├── detection/              boxes.npy confidence.npy handedness.npy track_ids.npy valid.npy
    ├── hand/                   camera_space_mano.npz joints_camera.npy ...
    ├── camera/windows/         000000_000199.npz ...
    ├── camera/                 stitched_camera.npz
    ├── stitched/               sim3_transforms.npz
    ├── trajectory/             trajectory.npz metadata.json
    └── visualization/          01_detection.mp4 ...

Keeping the layout in one place means a stage can never disagree with another
about where an artefact lives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import StageIOError
from .serialization import load_json, load_npz, save_json, save_npz

logger = logging.getLogger(__name__)

Array = np.ndarray

DETECTION_FIELDS = ("boxes", "confidence", "valid", "track_id")


@dataclass(frozen=True)
class ClipLayout:
    """Path resolver for one clip."""

    data_root: Path
    clip: str

    @property
    def root(self) -> Path:
        return self.data_root / self.clip

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def detection_dir(self) -> Path:
        return self.root / "detection"

    @property
    def hand_dir(self) -> Path:
        return self.root / "hand"

    @property
    def camera_dir(self) -> Path:
        return self.root / "camera"

    @property
    def camera_windows_dir(self) -> Path:
        return self.camera_dir / "windows"

    @property
    def stitched_dir(self) -> Path:
        return self.root / "stitched"

    @property
    def trajectory_dir(self) -> Path:
        return self.root / "trajectory"

    @property
    def visualization_dir(self) -> Path:
        return self.root / "visualization"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def detection_path(self) -> Path:
        return self.detection_dir / "detection.npz"

    @property
    def hand_path(self) -> Path:
        return self.hand_dir / "hand_camera.npz"

    @property
    def stitched_camera_path(self) -> Path:
        return self.camera_dir / "stitched_camera.npz"

    @property
    def sim3_path(self) -> Path:
        """Per-window local->World0 transforms (``scales``/``rotations``/``translations``).

        Stored as ``.npz`` rather than a single ``.npy`` because three arrays
        have to travel together; the name keeps the documented location.
        """
        return self.stitched_dir / "sim3_transforms.npz"

    @property
    def trajectory_path(self) -> Path:
        return self.trajectory_dir / "trajectory.npz"

    @property
    def trajectory_raw_path(self) -> Path:
        return self.trajectory_dir / "trajectory_raw.npz"

    @property
    def trajectory_metadata_path(self) -> Path:
        return self.trajectory_dir / "metadata.json"

    @property
    def world_joints_raw_path(self) -> Path:
        return self.trajectory_dir / "world_joints_raw.npy"

    @property
    def world_joints_refined_path(self) -> Path:
        return self.trajectory_dir / "world_joints_refined.npy"

    def window_path(self, start: int, end: int) -> Path:
        """``camera/windows/000000_000199.npz`` (``end`` exclusive -> inclusive name)."""
        if end <= start:
            raise StageIOError(f"invalid window range [{start}, {end})")
        return self.camera_windows_dir / f"{start:06d}_{end - 1:06d}.npz"

    def ensure_dirs(self) -> None:
        for path in (
            self.frames_dir,
            self.detection_dir,
            self.hand_dir,
            self.camera_windows_dir,
            self.stitched_dir,
            self.trajectory_dir,
            self.visualization_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Phase 1 artefacts
# --------------------------------------------------------------------------


def save_detection(layout: ClipLayout, arrays: dict[str, Array], metadata: dict[str, object] | None = None) -> Path:
    """Write ``detection/detection.npz`` plus the flattened ``*.npy`` mirrors."""
    missing = [name for name in DETECTION_FIELDS if name not in arrays]
    if missing:
        raise StageIOError(f"detection artefact is missing fields {missing}")
    valid = np.asarray(arrays["valid"])
    if valid.ndim != 2 or valid.shape[1] != 2:
        raise StageIOError(f"'valid' must be [T, 2], got {valid.shape}")
    path = save_npz(layout.detection_path, **arrays)
    layout.detection_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(layout.detection_dir / f"{name}.npy", value)
    if metadata is not None:
        save_json(layout.detection_dir / "metadata.json", metadata)
    return path


def load_detection(layout: ClipLayout) -> dict[str, Array]:
    """Read the Phase-1 artefact with its required fields enforced."""
    return load_npz(layout.detection_path, required=DETECTION_FIELDS)


HAND_FIELDS = ("joints_camera", "valid", "confidence", "root_rot", "betas")


def save_hand(layout: ClipLayout, arrays: dict[str, Array], metadata: dict[str, object] | None = None) -> Path:
    """Write ``hand/hand_camera.npz``."""
    missing = [name for name in HAND_FIELDS if name not in arrays]
    if missing:
        raise StageIOError(f"hand artefact is missing fields {missing}")
    path = save_npz(layout.hand_path, **arrays)
    if metadata is not None:
        save_json(layout.hand_dir / "metadata.json", metadata)
    return path


def load_hand(layout: ClipLayout) -> dict[str, Array]:
    return load_npz(layout.hand_path, required=HAND_FIELDS)


def clip_metadata(layout: ClipLayout) -> dict[str, object]:
    """Phase-0 metadata for a clip, with a clear error if it is missing."""
    return load_json(layout.metadata_path)

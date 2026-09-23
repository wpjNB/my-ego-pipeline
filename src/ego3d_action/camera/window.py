"""VGGT-Omega window bookkeeping (Phase 3).

The final configuration is 200-frame windows with a 40-frame overlap, i.e. a
stride of 160. For a 600-frame clip that yields:

======  ==========
window  frames
======  ==========
0       0-199
1       160-359
2       320-519
3       480-599
======  ==========

The trailing window is truncated, exactly as in the reference table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class WindowRange:
    """Half-open frame range ``[start, end)`` with an index."""

    index: int
    start: int
    end: int

    @property
    def num_frames(self) -> int:
        return self.end - self.start

    def frames(self) -> Array:
        return np.arange(self.start, self.end, dtype=np.int64)

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.end


@dataclass(frozen=True)
class CameraWindow:
    """Per-window VGGT-Omega output, expressed in the window's own frame.

    ``depth`` is metric (VGGT-Omega outputs metric depth with the 416
    reproduction checkpoint) and shares the resolution of ``intrinsics``.
    """

    window: WindowRange
    rotation_c2w: Array  # [n, 3, 3] - window-local world
    translation_c2w: Array  # [n, 3]
    intrinsics: Array  # [n, 3, 3]
    depth: Array  # [n, H, W]
    depth_confidence: Array | None = None  # [n, H, W]
    pose_confidence: Array | None = None  # [n]

    def __post_init__(self) -> None:
        n = self.window.num_frames
        rotation = np.asarray(self.rotation_c2w, dtype=np.float64)
        translation = np.asarray(self.translation_c2w, dtype=np.float64)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        depth = np.asarray(self.depth, dtype=np.float64)
        if rotation.shape != (n, 3, 3):
            raise StageIOError(f"rotation_c2w must be [{n}, 3, 3], got {rotation.shape}")
        if translation.shape != (n, 3):
            raise StageIOError(f"translation_c2w must be [{n}, 3], got {translation.shape}")
        if intrinsics.shape != (n, 3, 3):
            raise StageIOError(f"intrinsics must be [{n}, 3, 3], got {intrinsics.shape}")
        if depth.ndim != 3 or depth.shape[0] != n:
            raise StageIOError(f"depth must be [{n}, H, W], got {depth.shape}")
        if self.depth_confidence is not None:
            conf = np.asarray(self.depth_confidence, dtype=np.float64)
            if conf.shape != depth.shape:
                raise StageIOError(f"depth_confidence must be {depth.shape}, got {conf.shape}")
            object.__setattr__(self, "depth_confidence", conf)
        if self.pose_confidence is not None:
            pc = np.asarray(self.pose_confidence, dtype=np.float64).reshape(-1)
            if pc.shape[0] != n:
                raise StageIOError(f"pose_confidence must have length {n}, got {pc.shape}")
            object.__setattr__(self, "pose_confidence", pc)
        object.__setattr__(self, "rotation_c2w", rotation)
        object.__setattr__(self, "translation_c2w", translation)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "depth", depth)

    @property
    def start(self) -> int:
        return self.window.start

    @property
    def end(self) -> int:
        return self.window.end

    @property
    def name(self) -> str:
        return f"{self.start:06d}_{self.end - 1:06d}"


def make_windows(num_frames: int, *, window: int = 200, overlap: int = 40) -> list[WindowRange]:
    """Build the window schedule for ``num_frames``.

    Raises:
        StageIOError: on non-positive sizes or ``overlap >= window``.
    """
    if num_frames <= 0:
        raise StageIOError(f"num_frames must be positive, got {num_frames}")
    if window <= 0:
        raise StageIOError(f"window must be positive, got {window}")
    if overlap < 0:
        raise StageIOError(f"overlap must be non-negative, got {overlap}")
    if overlap >= window:
        raise StageIOError(f"overlap ({overlap}) must be smaller than window ({window})")

    stride = window - overlap
    ranges: list[WindowRange] = []
    start = 0
    index = 0
    while start < num_frames:
        end = min(start + window, num_frames)
        if ranges and end <= ranges[-1].end:
            # A trailing window that adds no new frames is pure recomputation.
            logger.info(
                "window schedule: dropping fully contained window [%d, %d) after [%d, %d)",
                start,
                end,
                ranges[-1].start,
                ranges[-1].end,
            )
            break
        ranges.append(WindowRange(index=index, start=start, end=end))
        index += 1
        start += stride
    return ranges


def window_ranges(num_frames: int, *, window: int = 200, overlap: int = 40) -> list[tuple[int, int]]:
    """Convenience wrapper returning ``(start, end)`` tuples."""
    return [(w.start, w.end) for w in make_windows(num_frames, window=window, overlap=overlap)]


def window_coverage(num_frames: int, windows: list[WindowRange]) -> Array:
    """Per-frame number of windows covering each frame."""
    counts = np.zeros(num_frames, dtype=np.int64)
    for w in windows:
        counts[w.start : w.end] += 1
    return counts


def save_camera_window(window: CameraWindow, path: str | Path) -> Path:
    """Persist one window to ``camera/windows/000000_000199.npz``."""
    from ..io.serialization import save_npz

    payload: dict[str, Array] = {
        "start": np.array([window.start], dtype=np.int64),
        "end": np.array([window.end], dtype=np.int64),
        "rotation_c2w": window.rotation_c2w,
        "translation_c2w": window.translation_c2w,
        "intrinsics": window.intrinsics,
        "depth": window.depth,
    }
    if window.depth_confidence is not None:
        payload["depth_confidence"] = window.depth_confidence
    if window.pose_confidence is not None:
        payload["pose_confidence"] = window.pose_confidence
    return save_npz(path, **payload)


def load_camera_window(path: str | Path) -> CameraWindow:
    """Load a persisted window, validating the stage contract."""
    from ..io.serialization import load_npz

    data = load_npz(
        path,
        required=("start", "end", "rotation_c2w", "translation_c2w", "intrinsics", "depth"),
    )
    start = int(np.asarray(data["start"]).reshape(-1)[0])
    end = int(np.asarray(data["end"]).reshape(-1)[0])
    return CameraWindow(
        window=WindowRange(index=0, start=start, end=end),
        rotation_c2w=data["rotation_c2w"],
        translation_c2w=data["translation_c2w"],
        intrinsics=data["intrinsics"],
        depth=data["depth"],
        depth_confidence=data.get("depth_confidence"),
        pose_confidence=data.get("pose_confidence"),
    )

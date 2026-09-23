"""Phase 0: ``video.mp4 -> frames/ + metadata.json``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import StageIOError
from .serialization import save_json
from .video import extract_frames, probe_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameSet:
    """A decoded frame sequence on disk."""

    clip: str
    directory: Path
    paths: tuple[Path, ...]
    fps: float
    width: int
    height: int

    @property
    def num_frames(self) -> int:
        return len(self.paths)

    def path(self, frame_id: int) -> Path:
        if not 0 <= frame_id < self.num_frames:
            raise StageIOError(
                f"frame {frame_id} out of range for clip '{self.clip}' (0..{self.num_frames - 1})"
            )
        return self.paths[frame_id]

    def timestamps(self) -> np.ndarray:
        return np.arange(self.num_frames, dtype=np.float64) / self.fps


def preprocess_video(
    video: str | Path,
    data_root: str | Path,
    *,
    clip: str | None = None,
    image_format: str = "jpg",
    overwrite: bool = False,
) -> FrameSet:
    """Decode ``video`` into ``data_root/<clip>/frames`` and write metadata.

    The clip layout matches the on-disk contract of the agent spec:
    ``data/<clip>/frames/000000.jpg`` plus ``data/<clip>/metadata.json``.

    Raises:
        StageIOError: if decoding fails or the resulting frame sequence is empty.
    """
    src = Path(video)
    clip_name = clip or src.stem
    if not clip_name:
        raise StageIOError(f"cannot derive a clip name from {src}")

    clip_root = Path(data_root) / clip_name
    frames_dir = clip_root / "frames"

    info = probe_video(src)
    paths = extract_frames(src, frames_dir, image_format=image_format, overwrite=overwrite)
    if not paths:
        raise StageIOError(f"no frames available for clip '{clip_name}'")

    frames = FrameSet(
        clip=clip_name,
        directory=frames_dir,
        paths=tuple(paths),
        fps=info.fps,
        width=info.width,
        height=info.height,
    )

    metadata = {
        "clip": clip_name,
        "source_video": str(src.resolve()),
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "num_frames": frames.num_frames,
        "duration": info.duration,
        "codec": info.codec,
        "image_format": image_format,
        "frame_pattern": "%06d." + image_format,
        "stage": "phase0_preprocess",
        "probed_num_frames": info.num_frames,
    }
    save_json(clip_root / "metadata.json", metadata)
    logger.info("clip '%s': %d frames @ %.3f fps -> %s", clip_name, frames.num_frames, info.fps, frames_dir)
    return frames


def load_frame_set(data_root: str | Path, clip: str, *, image_format: str = "jpg") -> FrameSet:
    """Re-open a previously preprocessed clip."""
    from .serialization import load_json

    clip_root = Path(data_root) / clip
    metadata_path = clip_root / "metadata.json"
    metadata = load_json(metadata_path)
    frames_dir = clip_root / "frames"
    paths = tuple(sorted(frames_dir.glob(f"*.{image_format}")))
    if not paths:
        raise StageIOError(f"no frames found in {frames_dir}")
    return FrameSet(
        clip=clip,
        directory=frames_dir,
        paths=paths,
        fps=float(metadata["fps"]),
        width=int(metadata["width"]),
        height=int(metadata["height"]),
    )


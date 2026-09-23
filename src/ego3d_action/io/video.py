"""Video probing and frame extraction (Phase 0 backend).

Decoding is delegated to the system ``ffmpeg`` / ``ffprobe`` binaries so that
the pipeline does not depend on a particular OpenCV build. Both tools are
invoked through :mod:`subprocess` with explicit exit-code handling: a failure
raises :class:`~ego3d_action.errors.StageIOError` instead of producing a partial
frame folder silently.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ..errors import StageIOError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    num_frames: int
    duration: float
    codec: str

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


def _require_binary(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise StageIOError(
            f"required binary '{name}' not found on PATH; install ffmpeg to use video IO"
        )
    return found


def _run(cmd: list[str], *, stage: str) -> subprocess.CompletedProcess[str]:
    logger.debug("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise StageIOError(f"{stage} failed (exit={proc.returncode}): " + " | ".join(tail))
    return proc


def probe_video(path: str | Path) -> VideoInfo:
    """Return :class:`VideoInfo` for ``path``.

    Raises:
        StageIOError: if the file does not exist, ffprobe fails, or the stream
            has no decodable video track.
    """
    video = Path(path)
    if not video.is_file():
        raise StageIOError(f"video not found: {video}")
    ffprobe = _require_binary("ffprobe")
    proc = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        stage=f"ffprobe {video.name}",
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise StageIOError(f"ffprobe returned unparsable JSON for {video}: {exc}") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise StageIOError(f"no video stream found in {video}")
    stream = streams[0]

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if fps is None or fps <= 0.0:
        raise StageIOError(f"cannot determine a positive frame rate for {video}")

    num_frames = _parse_int(stream.get("nb_frames"))
    duration = _parse_float(stream.get("duration")) or _parse_float(
        (payload.get("format") or {}).get("duration")
    )
    if num_frames is None:
        if duration is None:
            raise StageIOError(f"cannot determine frame count or duration for {video}")
        num_frames = int(round(duration * fps))
        logger.warning(
            "%s has no nb_frames metadata; inferred %d frames from duration %.3fs @ %.3f fps",
            video.name,
            num_frames,
            duration,
            fps,
        )
    if duration is None:
        duration = num_frames / fps

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise StageIOError(f"invalid resolution {width}x{height} for {video}")

    return VideoInfo(
        path=video,
        fps=float(fps),
        width=width,
        height=height,
        num_frames=int(num_frames),
        duration=float(duration),
        codec=str(stream.get("codec_name") or "unknown"),
    )


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    *,
    image_format: str = "jpg",
    quality: int = 2,
    overwrite: bool = False,
) -> list[Path]:
    """Decode every frame of ``video`` into ``out_dir`` as ``%06d.<fmt>``.

    Existing frames are reused unless ``overwrite`` is set, which makes long
    jobs resumable.

    Returns:
        Sorted list of written frame paths.

    Raises:
        StageIOError: on a missing input, ffmpeg failure, or when the number of
            decoded frames does not match the probed frame count.
    """
    info = probe_video(video)
    target = Path(out_dir)

    if overwrite and target.exists():
        for stale in sorted(target.glob(f"*.{image_format}")):
            stale.unlink()
    target.mkdir(parents=True, exist_ok=True)

    existing = sorted(target.glob(f"*.{image_format}"))
    if len(existing) == info.num_frames and not overwrite:
        logger.info("%s already holds %d frames; skipping extraction", target, len(existing))
        return existing
    if existing and not overwrite:
        logger.warning(
            "%s holds %d of %d frames; re-extracting into it",
            target,
            len(existing),
            info.num_frames,
        )

    ffmpeg = _require_binary("ffmpeg")
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(info.path),
            "-start_number",
            "0",
            "-vsync",
            "0",
            "-qscale:v",
            str(quality),
            str(target / f"%06d.{image_format}"),
        ],
        stage=f"ffmpeg frame extraction for {info.path.name}",
    )

    frames = sorted(target.glob(f"*.{image_format}"))
    if not frames:
        raise StageIOError(f"ffmpeg produced no frames for {info.path}")
    if len(frames) != info.num_frames:
        logger.warning(
            "decoded %d frames but ffprobe reported %d for %s",
            len(frames),
            info.num_frames,
            info.path.name,
        )
    logger.info("extracted %d frames to %s", len(frames), target)
    return frames


def _parse_rate(value: object) -> float | None:
    if value in (None, "", "0/0"):
        return None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None


def _parse_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _parse_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None

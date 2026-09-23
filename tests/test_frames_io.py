"""Phase 0: video -> frames + metadata, exercised through the real ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.io.frames import load_frame_set, preprocess_video
from ego3d_action.io.video import probe_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "demo01.mp4"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"cannot synthesise a test video: {proc.stderr.strip()[-200:]}")
    return path


def test_probe_video_reads_metadata(sample_video: Path) -> None:
    info = probe_video(sample_video)
    assert info.width == 64 and info.height == 48
    assert info.fps == pytest.approx(10.0, abs=0.1)
    assert info.num_frames == pytest.approx(10, abs=1)
    assert info.resolution == (64, 48)


def test_probe_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(StageIOError):
        probe_video(tmp_path / "nope.mp4")


def test_preprocess_video_writes_frames_and_metadata(sample_video: Path, tmp_path: Path) -> None:
    frames = preprocess_video(sample_video, tmp_path, clip="demo01")
    assert frames.clip == "demo01"
    assert frames.num_frames == 10
    assert (tmp_path / "demo01" / "metadata.json").is_file()
    assert frames.path(0).name == "000000.jpg"
    assert frames.path(9).name == "000009.jpg"
    assert np_allclose(frames.timestamps(), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def test_preprocess_is_resumable(sample_video: Path, tmp_path: Path) -> None:
    first = preprocess_video(sample_video, tmp_path, clip="demo01")
    second = preprocess_video(sample_video, tmp_path, clip="demo01")
    assert first.num_frames == second.num_frames


def test_load_frame_set_roundtrip(sample_video: Path, tmp_path: Path) -> None:
    written = preprocess_video(sample_video, tmp_path, clip="demo01")
    loaded = load_frame_set(tmp_path, "demo01")
    assert loaded.num_frames == written.num_frames
    assert loaded.fps == pytest.approx(written.fps)
    assert (loaded.width, loaded.height) == (written.width, written.height)


def test_frame_path_out_of_range_raises(sample_video: Path, tmp_path: Path) -> None:
    frames = preprocess_video(sample_video, tmp_path, clip="demo01")
    with pytest.raises(StageIOError):
        frames.path(frames.num_frames)


def np_allclose(actual, expected, *, atol: float = 1e-6) -> bool:
    import numpy as np

    return bool(np.allclose(np.asarray(actual), np.asarray(expected), atol=atol))

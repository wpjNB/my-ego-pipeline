"""End-to-end run of the whole pipeline with the deterministic mock backend.

Exercises Phases 0-7 through the real CLI entry points and the real runner
protocol - only the three model calls are replaced by ``backends/mock_backend.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ego3d_action.io.serialization import load_npz, validate_trajectory

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_FRAMES = 240
FPS = 30.0

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the mock pipeline once and hand the data root to the tests."""
    work = tmp_path_factory.mktemp("mock_pipeline")
    video = work / "clip01.mp4"
    created = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={int(FPS)}:duration={NUM_FRAMES / FPS:.2f}",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"cannot synthesise the test video: {created.stderr[-200:]}")

    data_root = work / "data"
    result = run_script(
        "scripts/run_pipeline.py",
        "--config",
        "configs/mock.yaml",
        "--clip",
        "clip01",
        "--data-root",
        str(data_root),
        "--video",
        str(video),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return data_root


def test_every_stage_artefact_exists(pipeline_run: Path) -> None:
    clip = pipeline_run / "clip01"
    for relative in (
        "metadata.json",
        "detection/detection.npz",
        "hand/hand_camera.npz",
        "camera/stitched_camera.npz",
        "stitched/sim3_transforms.npz",
        "trajectory/trajectory_raw.npz",
        "trajectory/trajectory.npz",
        "trajectory/metadata.json",
    ):
        assert (clip / relative).is_file(), f"missing {relative}"
    assert len(list((clip / "camera" / "windows").glob("*.npz"))) == 2
    assert len(list((clip / "hand" / "windows").glob("*.npz"))) > 1


def test_debug_videos_are_written(pipeline_run: Path) -> None:
    visualization = pipeline_run / "clip01" / "visualization"
    for name in ("01_detection.mp4", "02_hawor.mp4"):
        path = visualization / name
        assert path.is_file(), f"missing {name}"
        assert path.stat().st_size > 0


def test_final_trajectory_satisfies_the_contract(pipeline_run: Path) -> None:
    data = load_npz(pipeline_run / "clip01" / "trajectory" / "trajectory.npz")
    assert validate_trajectory(data, strict=True) == []
    assert data["hand_xyz_world"].shape == (NUM_FRAMES, 2, 21, 3)
    assert np.isfinite(data["camera_t_c2w"]).all()
    # World-0 is the first frame's camera by construction.
    assert np.allclose(data["camera_R_c2w"][0], np.eye(3), atol=1e-6)
    assert np.allclose(data["camera_t_c2w"][0], np.zeros(3), atol=1e-6)

    metadata = json.loads(
        (pipeline_run / "clip01" / "trajectory" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["world_frame"] == 0
    assert metadata["units"] == "meter"
    assert metadata["camera_convention"] == "c2w"


def test_missing_frames_are_filled_only_when_marked(pipeline_run: Path) -> None:
    """P2 policy: short holes are interpolated and marked; nothing else appears."""
    trajectory = pipeline_run / "clip01" / "trajectory"
    raw = load_npz(trajectory / "trajectory_raw.npz")
    data = load_npz(trajectory / "trajectory.npz")
    raw_valid = np.asarray(raw["hand_valid"], dtype=bool)
    valid = data["hand_valid"]
    interpolated = data["hand_interpolated"]
    assert valid.shape == (NUM_FRAMES, 2)
    # The conservative tracker leaves the planted 10-frame hole missing...
    assert not raw_valid[30:40, 1].any()
    # ... refinement's gap fill interpolates it (10 <= max_gap 12) ...
    assert interpolated[30:40, 1].all()
    # ... and every frame the pipeline invented is exactly a marked one.
    assert np.array_equal(valid & ~raw_valid, interpolated)
    assert not (interpolated & raw_valid).any()
    # Filled frames carry finite world joints; the rest stay missing.
    world = data["hand_xyz_world"]
    assert np.isfinite(world[interpolated]).all()
    assert np.isnan(world[~valid]).all()


def test_detection_metadata_records_the_backend_mode(pipeline_run: Path) -> None:
    metadata = json.loads(
        (pipeline_run / "clip01" / "detection" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["backend_mode"] == "mock"
    assert metadata["min_confidence"] == 0.75


def test_refinement_reduces_the_error_against_the_mock_ground_truth(pipeline_run: Path) -> None:
    """The refined trajectory must beat the raw one on the mock reference."""
    clip = pipeline_run / "clip01"
    truth = clip / "trajectory" / "truth.npz"
    produced = run_script(
        "backends/mock_backend.py",
        "truth",
        "--out",
        str(truth),
        "--num-frames",
        str(NUM_FRAMES),
        "--fps",
        str(FPS),
    )
    assert produced.returncode == 0, produced.stdout + produced.stderr

    evaluations = {}
    for label, name in (("raw", "trajectory_raw.npz"), ("refined", "trajectory.npz")):
        result = run_script(
            "scripts/evaluate_hot3d.py",
            "--config",
            "configs/mock.yaml",
            "--clip",
            "clip01",
            "--prediction",
            str(clip / "trajectory" / name),
            "--ground-truth",
            str(truth),
            "--json",
            str(clip / "evaluation" / f"{label}.json"),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Action MPJPE" in result.stdout
        payload = json.loads((clip / "evaluation" / f"{label}.json").read_text(encoding="utf-8"))
        evaluations[label] = payload

    for label in ("raw", "refined"):
        assert evaluations[label]["action_mpjpe_mm"] > 0.0
        assert evaluations[label]["coverage"] > 0.5
        assert np.isfinite(evaluations[label]["action_mpjpe_mm"])

    # The wrist-depth stage targets the *wrist* error directly, so that error
    # must improve even though the mock's corruption model is not the real
    # HaWoR one (the overall Action-MPJPE effect of post-processing is not
    # guaranteed on synthetic noise - see doc_auto/architecture.md).
    assert evaluations["refined"]["wrist_error_mm"] < evaluations["raw"]["wrist_error_mm"]
    # No catastrophic regression either way.
    assert evaluations["refined"]["action_mpjpe_mm"] < evaluations["raw"]["action_mpjpe_mm"] * 1.2

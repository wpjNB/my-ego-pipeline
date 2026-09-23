"""End-to-end CLI checks: phase scripts and the orchestrator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ego3d_action.camera.stitch import load_stitched_camera
from ego3d_action.camera.window import save_camera_window
from ego3d_action.geometry.sim3 import Sim3

from conftest import build_two_windows

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def stitched_clip(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_scene: dict[str, object],
    known_sim3: Sim3,
) -> Path:
    """A data root holding Phase-0 metadata and two camera windows."""
    data_root = tmp_path_factory.mktemp("data")
    clip_root = data_root / "clip01"
    (clip_root / "camera" / "windows").mkdir(parents=True, exist_ok=True)
    (clip_root / "metadata.json").write_text(
        json.dumps({"clip": "clip01", "fps": 30.0, "num_frames": 360, "width": 96, "height": 72}),
        encoding="utf-8",
    )
    window_a, window_b, _, _ = build_two_windows(synthetic_scene, known_sim3)
    save_camera_window(window_a, clip_root / "camera" / "windows" / "000000_000199.npz")
    save_camera_window(window_b, clip_root / "camera" / "windows" / "000160_000359.npz")
    return data_root


def test_run_stitch_cli_produces_a_world_zero_trajectory(stitched_clip: Path) -> None:
    result = run_script(
        "scripts/run_stitch.py",
        "--config",
        "configs/macrodata_final.yaml",
        "--clip",
        "clip01",
        "--data-root",
        str(stitched_clip),
    )
    assert result.returncode == 0, result.stderr
    assert "scale=" in result.stdout

    clip_root = stitched_clip / "clip01"
    stitched = load_stitched_camera(clip_root / "camera" / "stitched_camera.npz")
    assert stitched.valid.all()
    assert np.allclose(stitched.translation_c2w[0], np.zeros(3), atol=1e-6)
    assert (clip_root / "stitched" / "sim3_transforms.npz").is_file()

    metadata = json.loads((clip_root / "camera" / "stitched_camera.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "phase4_stitch"
    assert metadata["num_windows"] == 2


def test_run_stitch_cli_reports_missing_windows(stitched_clip: Path, tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    clip_root = empty_root / "clip01"
    clip_root.mkdir(parents=True)
    (clip_root / "metadata.json").write_text(
        json.dumps({"fps": 30.0, "num_frames": 360}), encoding="utf-8"
    )
    result = run_script(
        "scripts/run_stitch.py",
        "--config",
        "configs/macrodata_final.yaml",
        "--clip",
        "clip01",
        "--data-root",
        str(empty_root),
    )
    assert result.returncode != 0
    assert "missing" in (result.stdout + result.stderr)


def test_pipeline_dry_run_lists_every_stage(stitched_clip: Path) -> None:
    result = run_script(
        "scripts/run_pipeline.py",
        "--config",
        "configs/macrodata_final.yaml",
        "--clip",
        "clip01",
        "--data-root",
        str(stitched_clip),
        "--video",
        "demo.mp4",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    for stage in ("phase0-preprocess", "phase4-stitch", "phase6-refine"):
        assert stage in result.stdout
    assert "environment:" in result.stdout
    assert "blocked" in result.stdout  # backends are absent in the base env


def test_pipeline_resumes_from_a_stage(stitched_clip: Path) -> None:
    result = run_script(
        "scripts/run_pipeline.py",
        "--config",
        "configs/macrodata_final.yaml",
        "--clip",
        "clip01",
        "--data-root",
        str(stitched_clip),
        "--from-stage",
        "phase4-stitch",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "phase0-preprocess" not in result.stdout
    assert "phase4-stitch" in result.stdout


def test_evaluate_cli_reports_missing_artefacts(tmp_path: Path) -> None:
    result = run_script(
        "scripts/evaluate_hot3d.py",
        "--config",
        "configs/hot3d.yaml",
        "--clip",
        "clip01",
        "--prediction",
        str(tmp_path / "nope.npz"),
        "--ground-truth",
        str(tmp_path / "nope.npz"),
    )
    assert result.returncode != 0
    assert "not found" in (result.stdout + result.stderr).lower()

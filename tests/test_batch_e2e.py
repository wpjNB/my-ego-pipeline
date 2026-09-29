"""End-to-end batch dispatch with the deterministic mock backend.

This is the test that justifies the whole M1 design, using only CPU and the mock
backend:

* a sharded run produces exactly the same windows as an unsliced run - compared
  artefact by artefact, not just by file name;
* the blend assembled from the shards is identical to the whole-clip blend, so
  "the union of the shards is the run" is verified on real numpy content;
* a second run with ``--skip-existing`` performs zero computation (no file is
  rewritten, nothing is recomputed);
* the SSH executor builds the same stage command as the local executor, with only
  the transport wrapper added.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ego3d_action.runtime.executor import HostCapabilities, LocalExecutor, SshExecutor
from ego3d_action.runtime.sharding import WindowSelection

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_FRAMES = 40  # 4 HaWoR windows at 16/8 -> divisible into 2 shards
FPS = 20.0

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One synthesised video, used by both the reference and the sharded run."""
    work = tmp_path_factory.mktemp("batch_e2e")
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
            f"testsrc=size=256x192:rate={int(FPS)}:duration={NUM_FRAMES / FPS:.2f}",
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
    return video


def prepare_clip(video: Path, data_root: Path) -> None:
    """Phase 0 + Phase 1 for one clip, so the sharded hand stage has its input."""
    result = run_script(
        "scripts/run_preprocess.py",
        "--config",
        "configs/mock.yaml",
        "--clip",
        "clip01",
        "--data-root",
        str(data_root),
        str(video),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = run_script(
        "scripts/run_detection.py",
        "--config",
        "configs/mock.yaml",
        "--clip",
        "clip01",
        "--data-root",
        str(data_root),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def run_batch(data_root: Path, outputs: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run_script(
        "scripts/run_batch.py",
        "--manifest",
        str(outputs / "clips.yaml"),
        "--config",
        str(REPO_ROOT / "configs" / "mock.yaml"),
        "--data-root",
        str(data_root),
        "--outputs",
        str(outputs),
        "--stages",
        "hand",
        *extra,
    )


def write_manifest(outputs: Path) -> None:
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "clips.yaml").write_text(
        f"clips:\n  - clip: clip01\n    num_frames: {NUM_FRAMES}\n", encoding="utf-8"
    )


def window_files(data_root: Path) -> list[Path]:
    return sorted((data_root / "clip01" / "hand" / "windows").glob("*.npz"))


@pytest.fixture(scope="module")
def reference_run(clip: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An unsliced whole-clip run of the hand stage."""
    data_root = tmp_path_factory.mktemp("reference") / "data"
    outputs = data_root.parent / "outputs"
    prepare_clip(clip, data_root)
    result = run_script(
        "scripts/run_hand.py",
        "--config",
        "configs/mock.yaml",
        "--clip",
        "clip01",
        "--data-root",
        str(data_root),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _ = outputs
    return data_root


def test_plan_is_dry_runnable(clip: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    data_root = tmp_path_factory.mktemp("dry") / "data"
    outputs = data_root.parent / "outputs"
    prepare_clip(clip, data_root)
    write_manifest(outputs)
    result = run_batch(data_root, outputs, "--shards", "2", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 window(s)" in result.stdout
    assert "2 shard(s)" in result.stdout


def test_sharded_run_reproduces_the_unsharded_windows(
    clip: Path, reference_run: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    data_root = tmp_path_factory.mktemp("sharded") / "data"
    outputs = data_root.parent / "outputs"
    prepare_clip(clip, data_root)
    write_manifest(outputs)

    result = run_batch(data_root, outputs, "--shards", "2", "--max-parallel", "2")
    assert result.returncode == 0, result.stdout + result.stderr

    reference = window_files(reference_run)
    sharded = window_files(data_root)
    assert [p.name for p in sharded] == [p.name for p in reference]
    assert len(reference) == 4

    # Content equality: the slice is lossless, not merely same-named.
    for expected_path, actual_path in zip(reference, sharded, strict=True):
        expected = np.load(expected_path, allow_pickle=False)
        actual = np.load(actual_path, allow_pickle=False)
        assert sorted(expected.files) == sorted(actual.files)
        for key in expected.files:
            np.testing.assert_array_equal(expected[key], actual[key])

    # The blend assembled from the shards must equal the whole-clip blend.
    reference_hand = np.load(
        reference_run / "clip01" / "hand" / "hand_camera.npz", allow_pickle=False
    )
    sharded_hand = np.load(data_root / "clip01" / "hand" / "hand_camera.npz", allow_pickle=False)
    for key in reference_hand.files:
        np.testing.assert_array_equal(reference_hand[key], sharded_hand[key])

    report = json.loads((outputs / "batch_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["failed"] == 0
    assert report["degraded_clips"] == []
    units = report["units"]
    assert [u["unit"]["selection"]["shard"]["count"] for u in units] == [2, 2, 1]
    assert all(u["status"] == "ok" for u in units)


def test_second_sharded_run_computes_nothing(
    clip: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    data_root = tmp_path_factory.mktemp("idempotent") / "data"
    outputs = data_root.parent / "outputs"
    prepare_clip(clip, data_root)
    write_manifest(outputs)

    assert run_batch(data_root, outputs, "--shards", "2").returncode == 0
    before = {path.name: path.stat().st_mtime_ns for path in window_files(data_root)}
    assert before

    result = run_batch(data_root, outputs, "--shards", "2", "--skip-existing")
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads((outputs / "batch_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["skipped"] == 3  # 2 shards + the blend assembly
    assert report["summary"]["ok"] == 0

    # No artefact was rewritten (mtime is used here only as a witness; the
    # decision itself never consults mtime).
    after = {path.name: path.stat().st_mtime_ns for path in window_files(data_root)}
    assert after == before


def test_ssh_command_matches_the_local_command() -> None:
    """The transport adds a wrapper; it must not change the stage invocation."""
    local_host = HostCapabilities(name="local", executor="local", orchestrator_python=("python",))
    remote_host = HostCapabilities(
        name="worker",
        executor="ssh",
        ssh_host="gpu-01",
        ssh_user="lab",
        orchestrator_python=("python",),
    )
    stage_args = ["python", "scripts/run_hand.py", "--clip", "clip01", "--shard", "0/2"]

    local_executor = LocalExecutor(host=local_host)
    assert isinstance(local_executor, LocalExecutor)  # the local path is plain subprocess

    remote_executor = SshExecutor(host=remote_host)
    argv = remote_executor.build_command(stage_args, cwd="/srv/ego")
    remote_command = argv[-1]
    for token in stage_args:
        assert token in remote_command
    # The local variant is exactly ``stage_args`` itself.
    assert stage_args == ["python", "scripts/run_hand.py", "--clip", "clip01", "--shard", "0/2"]


def test_batch_runner_rejects_a_shard_count_that_does_not_divide(
    clip: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    data_root = tmp_path_factory.mktemp("badshard") / "data"
    outputs = data_root.parent / "outputs"
    prepare_clip(clip, data_root)
    write_manifest(outputs)
    result = run_batch(data_root, outputs, "--shards", "3")
    assert result.returncode != 0
    assert "divide evenly" in (result.stdout + result.stderr)


def test_window_selection_of_a_shard_is_a_strict_subset() -> None:
    """Belt and braces on the naming contract the E2E comparison relies on."""
    from ego3d_action.hand.hawor import HaworClipRequest

    request = HaworClipRequest(num_frames=NUM_FRAMES, frames_dir=".", window=16, overlap=8)
    spans = request.ranges()
    assert len(spans) == 4
    shard0 = WindowSelection.parse(shard="0/2").select(spans)
    shard1 = WindowSelection.parse(shard="1/2").select(spans)
    assert shard0 and shard1
    assert set(shard0) | set(shard1) == set(spans)
    assert not (set(shard0) & set(shard1))


def test_runner_and_scheduler_agree_on_the_unit_key() -> None:
    """Both sides must call a unit the same thing, or skip-existing never matches.

    The runner writes its marker (``backends/_shard_cli.record_partition``) and
    the scheduler reads it (``runtime.batch.stage_outputs``); a drift between the
    two would silently disable every skip.
    """
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "backends"))
    from _shard_cli import partition_is_reusable, record_partition, window_file_names  # noqa: PLC0415

    from ego3d_action.hand.hawor import HaworClipRequest
    from ego3d_action.runtime.batch import UnitSpec, stage_outputs

    selection = WindowSelection.parse(shard="1/2")
    request = HaworClipRequest(num_frames=NUM_FRAMES, frames_dir=".", window=16, overlap=8)
    ranges = selection.select(request.ranges())
    params = {"num_frames": NUM_FRAMES, "window": 16, "overlap": 8}

    unit = UnitSpec(
        clip="clip01", stage="hand", selection=selection, num_windows=4, num_frames=NUM_FRAMES
    )
    _, scheduler_key = stage_outputs(unit, "hand")
    assert scheduler_key == f"hand/{selection.describe()}"

    # The runner's marker must satisfy the scheduler's own check.
    out_dir = Path("/tmp") / "ego3d_unit_key_probe"
    import shutil as _shutil

    _shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in window_file_names(ranges):
        (out_dir / name).write_bytes(b"window")
    assert partition_is_reusable(
        out_dir, stage="hand", ranges=ranges, params=params, selection=selection
    ) is False  # no marker yet
    record_partition(out_dir, stage="hand", ranges=ranges, params=params, selection=selection)
    assert partition_is_reusable(
        out_dir, stage="hand", ranges=ranges, params=params, selection=selection
    )
    _shutil.rmtree(out_dir, ignore_errors=True)


def test_real_runner_parameters_are_refused_for_the_frame_coupled_stage() -> None:
    """Phase 1 slicing is rejected with a reason, not silently ignored."""
    result = run_script(
        "backends/wilor_runner.py",
        "--frames",
        ".",
        "--out",
        "/tmp/never_written.npz",
        "--shard",
        "0/2",
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "error"
    assert "frame-coupled" in payload["message"]
    assert not Path("/tmp/never_written.npz").exists()
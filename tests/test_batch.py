"""Executor and batch scheduling: capability matching, retries, no fake output.

The batch runner is the one component that can turn a single worker failure into
a silently wrong dataset, so most of these tests are about what happens when
things go wrong:

* a failed unit is retried exactly ``retries + 1`` times,
* a unit that never succeeds marks its clip ``degraded`` and writes no artefact
  or marker that a later stage could mistake for real output,
* a second run with ``--skip-existing`` performs zero computation.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import yaml

from ego3d_action.errors import ConfigError
from ego3d_action.runtime.batch import (
    BatchPlan,
    BatchRunner,
    ClipSpec,
    UnitSpec,
    build_plan,
    load_manifest,
    num_windows_for,
    stage_sequence,
    unit_argument_signature,
    unit_command,
)
from ego3d_action.runtime.executor import (
    CapabilityMismatch,
    HostCapabilities,
    LocalExecutor,
    SshExecutor,
    build_executor,
    load_hosts,
    select_host,
)
from ego3d_action.runtime.provenance import read_marker
from ego3d_action.runtime.sharding import WindowSelection


def host(name: str = "cpu", **kwargs: object) -> HostCapabilities:
    base: dict[str, object] = {
        "name": name,
        "executor": "local",
        "repo_root": ".",
    }
    base.update(kwargs)
    return HostCapabilities(**base)  # type: ignore[arg-type]


def gpu_host(name: str = "gpu", backend: str = "wilor", **kwargs: object) -> HostCapabilities:
    """A local host that declares one GPU backend and enough VRAM for it."""
    return host(
        name,
        backends=(backend,),
        python={backend: ("python",)},
        cuda="12.1",
        gpu_count=1,
        gpu_memory_gb=24.0,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Host capabilities and selection
# --------------------------------------------------------------------------


def test_host_requires_python_for_every_declared_backend() -> None:
    with pytest.raises(ConfigError):
        host(backends=("hawor",))


def test_host_rejects_unknown_executor() -> None:
    with pytest.raises(ConfigError):
        host(executor="slurm")


def test_ssh_host_requires_a_target() -> None:
    with pytest.raises(ConfigError):
        host(executor="ssh")


def test_capability_shortfall_names_the_reason() -> None:
    cpu = host("cpu-only", cuda=None, gpu_count=0)
    reason = cpu.explain_shortfall({"backends": ("vggt",), "gpu": True})
    assert reason is not None
    assert "missing backend(s) ['vggt']" in reason

    gpu = host(
        "gpu-small",
        backends=("vggt",),
        python={"vggt": ("python",)},
        cuda="12.1",
        gpu_count=1,
        gpu_memory_gb=8.0,
    )
    reason = gpu.explain_shortfall({"backends": ("vggt",), "gpu": True, "min_gpu_memory_gb": 16.0})
    assert reason is not None and "VRAM" in reason
    assert gpu.explain_shortfall({"backends": ("vggt",), "gpu": True, "min_gpu_memory_gb": 8.0}) is None


def test_select_host_picks_a_capable_host() -> None:
    cpu = host("cpu", cuda=None, gpu_count=0)
    gpu = gpu_host(backend="hawor")
    chosen = select_host([cpu, gpu], {"backends": ("hawor",), "gpu": True})
    assert chosen.name == "gpu"


def test_select_host_raises_with_every_reason() -> None:
    hosts = [host("a", cuda=None), host("b", cuda="12.1", gpu_count=1)]
    with pytest.raises(CapabilityMismatch) as excinfo:
        select_host(hosts, {"backends": ("vggt",)})
    assert "a" in str(excinfo.value) and "b" in str(excinfo.value)


def test_select_host_honours_a_preference_and_checks_it() -> None:
    cpu = host("cpu", cuda=None)
    gpu = gpu_host(backend="vggt")
    assert select_host([cpu, gpu], {"backends": ("vggt",)}, preferred="gpu").name == "gpu"
    with pytest.raises(CapabilityMismatch):
        select_host([cpu, gpu], {"backends": ("vggt",)}, preferred="cpu")


def test_load_hosts_rejects_an_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "hosts.yaml"
    path.write_text("hosts:\n  - name: a\n    gpu_mem_gb: 24\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_hosts(path)
    assert "unknown keys" in str(excinfo.value)


def test_load_hosts_reads_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "hosts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "hosts": [
                    {
                        "name": "worker-a",
                        "backends": ["wilor"],
                        "python": {"wilor": ["conda", "run", "-n", "ego3d_wilor", "python"]},
                        "gpu_count": 2,
                        "gpu_memory_gb": 24,
                        "max_parallel": 2,
                        "orchestrator_python": ["python"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    hosts = load_hosts(path)
    assert len(hosts) == 1
    assert hosts[0].python_command("wilor") == ("conda", "run", "-n", "ego3d_wilor", "python")
    assert hosts[0].stage_python == ("python",)


# --------------------------------------------------------------------------
# Executors
# --------------------------------------------------------------------------


def test_local_executor_reports_a_failure_without_raising() -> None:
    result = LocalExecutor(host=host()).run(["python", "-c", "import sys; sys.exit(3)"])
    assert result.returncode == 3
    assert not result.ok


def test_local_executor_missing_binary_is_an_error_result() -> None:
    result = LocalExecutor(host=host()).run(["definitely-not-a-real-binary-xyz"])
    assert result.returncode == 127
    assert "cannot launch" in result.stderr


def test_ssh_executor_builds_a_batch_mode_command() -> None:
    remote = host("worker", executor="ssh", ssh_host="gpu-01", ssh_user="lab", ssh_port=2222)
    executor = SshExecutor(host=remote)
    argv = executor.build_command(
        ["python", "scripts/run_hand.py", "--clip", "clip01"],
        cwd="/srv/ego",
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert "-p" in argv and "2222" in argv
    target = argv[-2]
    assert target == "lab@gpu-01"
    remote_command = argv[-1]
    assert "cd /srv/ego" in remote_command
    assert "CUDA_VISIBLE_DEVICES=0" in remote_command
    assert "python scripts/run_hand.py --clip clip01" in remote_command


def test_build_executor_rejects_an_unimplemented_kind() -> None:
    broken = HostCapabilities(name="x", executor="local")
    assert isinstance(build_executor(broken), LocalExecutor)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def test_num_windows_matches_the_real_schedules() -> None:
    # Mirrors camera.window.make_windows for 600 frames: 4 windows of 200/40.
    assert num_windows_for("camera", 600, window=200, overlap=40) == 4
    assert num_windows_for("hand", 240, window=16, overlap=8) == 29


def test_manifest_accepts_names_and_mappings(tmp_path: Path) -> None:
    path = tmp_path / "clips.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "clips": [
                    "clip01",
                    {"clip": "clip02", "video": "videos/clip02.mp4", "num_frames": 240},
                ]
            }
        ),
        encoding="utf-8",
    )
    clips = load_manifest(path)
    assert [c.clip for c in clips] == ["clip01", "clip02"]
    assert clips[1].num_frames == 240


def test_manifest_rejects_a_typo_key_and_duplicates(tmp_path: Path) -> None:
    bad_key = tmp_path / "a.yaml"
    bad_key.write_text("clips:\n  - clip: c1\n    num_frame: 120\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_manifest(bad_key)

    duplicate = tmp_path / "b.yaml"
    duplicate.write_text("clips:\n  - c1\n  - c1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_manifest(duplicate)


def test_stage_sequence_respects_from_stage() -> None:
    clip = ClipSpec(clip="c1", from_stage="camera")
    assert stage_sequence(clip)[0] == "camera"
    assert "preprocess" not in stage_sequence(clip)
    with pytest.raises(ValueError):
        ClipSpec(clip="c1", from_stage="nope")


def test_build_plan_emits_one_unit_per_shard_plus_an_assembly() -> None:
    plan = build_plan(
        [ClipSpec(clip="c1", num_frames=600)],
        config={"camera": {"window": 200, "overlap": 40}},
        shards=2,
        stages=["camera"],
    )
    shards = [u for u in plan.units if not u.selection.is_whole]
    assembly = [u for u in plan.units if u.selection.is_whole]
    assert len(shards) == 2
    assert len(assembly) == 1
    # The assembly must come after every shard.
    assert plan.units[-1] is assembly[0]


def test_build_plan_without_sharding_adds_no_assembly() -> None:
    plan = build_plan(
        [ClipSpec(clip="c1", num_frames=600)],
        config={"camera": {"window": 200, "overlap": 40}},
        shards=1,
        stages=["camera"],
    )
    assert len(plan.units) == 1
    assert plan.units[0].selection.is_whole


def test_build_plan_requires_a_divisible_shard_count() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        build_plan(
            [ClipSpec(clip="c1", num_frames=600)],
            config={"camera": {"window": 200, "overlap": 40}},
            shards=3,
            stages=["camera"],
        )


def test_build_plan_needs_a_length_for_sharded_stages() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_plan([ClipSpec(clip="c1")], config={}, shards=2, stages=["camera"])


def test_unit_command_shape() -> None:
    unit = UnitSpec(
        clip="c1",
        stage="camera",
        selection=WindowSelection.parse(shard="1/2"),
        num_windows=4,
        num_frames=600,
    )
    command = unit_command(unit, config_path="cfg.yaml", data_root="data", extra=[])
    assert "--shard" in command and "1/2" in command
    assert command[0] == "python"

    assembly = UnitSpec(
        clip="c1", stage="hand", selection=WindowSelection(), num_windows=29, mode="blend-only"
    )
    blend = unit_command(assembly, config_path="cfg.yaml", data_root="data", extra=[])
    assert "--blend-only" in blend and "--skip-existing" in blend

    # A whole-clip hand unit that is NOT the sharded plan's join must run the
    # model (the batch.py build_command regression: --blend-only on a full run
    # made every unsharded batch fail with "HaWoR window(s) are missing").
    full = UnitSpec(clip="c1", stage="hand", selection=WindowSelection(), num_windows=29)
    command = unit_command(full, config_path="cfg.yaml", data_root="data", extra=[])
    assert "--blend-only" not in command
    assert unit_argument_signature(full)["mode"] == "full"
    assert unit_argument_signature(assembly)["mode"] == "blend-only"
    with pytest.raises(ValueError, match="mode"):
        UnitSpec(clip="c1", stage="hand", selection=WindowSelection(), mode="wat")


def test_unit_command_preprocess_requires_a_video() -> None:
    unit = UnitSpec(clip="c1", stage="preprocess", selection=WindowSelection())
    with pytest.raises(ValueError, match="video path"):
        unit_command(unit, config_path="cfg.yaml", data_root="data", extra=[])


# --------------------------------------------------------------------------
# Execution: failure injection and idempotency
# --------------------------------------------------------------------------


class RecordingExecutor:
    """An executor that fails a fixed number of times, then succeeds."""

    def __init__(self, host: HostCapabilities, fail_times: int = 0, write: Path | None = None) -> None:
        self.host = host
        self.fail_times = fail_times
        self.calls = 0
        self.write = write

    def run(self, command, *, cwd=None, env=None, timeout=None):  # type: ignore[no-untyped-def]
        from ego3d_action.runtime.executor import ExecResult

        self.calls += 1
        if self.calls <= self.fail_times:
            return ExecResult(returncode=1, stderr="injected failure", host=self.host.name)
        if self.write is not None:
            self.write.parent.mkdir(parents=True, exist_ok=True)
            self.write.write_bytes(b"artefact")
        return ExecResult(returncode=0, stdout="ok", host=self.host.name)

    def describe(self) -> str:
        return f"recording:{self.host.name}"


def make_runner(tmp_path: Path, hosts: list[HostCapabilities], **kwargs: object) -> BatchRunner:
    data_root = tmp_path / "data"
    (data_root / "c1").mkdir(parents=True, exist_ok=True)
    return BatchRunner(
        hosts=hosts,
        config_path="configs/mock.yaml",
        data_root=str(data_root),
        layout_root=data_root,
        repo_root=str(tmp_path),
        **kwargs,  # type: ignore[arg-type]
    )


def test_failed_unit_is_retried_then_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executor = RecordingExecutor(host=host())
    monkeypatch.setattr(
        "ego3d_action.runtime.batch.build_executor", lambda *a, **k: executor
    )
    runner = make_runner(tmp_path, [gpu_host()], retries=2)
    unit = UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1)
    executor.fail_times = 99

    result = runner.run_unit(unit, skip_existing=False)
    assert result.status == "failed"
    assert result.attempts == 3  # 1 try + 2 retries
    assert executor.calls == 3

    # The clip is marked degraded, in the ledger and in its metadata.
    assert runner.degraded_clips() == ["c1"]
    metadata = json.loads((tmp_path / "data" / "c1" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["degraded"] is True
    assert "detection failed" in metadata["degraded_reason"]


def test_concurrent_units_spread_across_pinned_hosts(tmp_path: Path) -> None:
    """Three max_parallel=1 GPU hosts must serve three concurrent units on
    three different cards - a scheduler that stacks them on one host turns a
    3-GPU box into a 1-GPU box with OOMs."""
    runner = make_runner(
        tmp_path,
        [gpu_host(f"p100-{i}", backend="vggt", max_parallel=1) for i in range(3)],
    )
    unit = UnitSpec(clip="c1", stage="camera", selection=WindowSelection(), num_windows=1)
    acquired: list[str] = []
    barrier = threading.Barrier(3)
    lock = threading.Lock()

    def grab() -> None:
        barrier.wait()
        name = runner._acquire_host(unit).name
        with lock:
            acquired.append(name)

    threads = [threading.Thread(target=grab) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(acquired) == ["p100-0", "p100-1", "p100-2"]


def test_unit_waits_for_a_free_slot_when_all_hosts_are_busy(tmp_path: Path) -> None:
    """A unit whose every capable host is at max_parallel waits instead of
    over-subscribing a card."""
    runner = make_runner(
        tmp_path,
        [gpu_host("only-gpu", backend="vggt", max_parallel=1)],
    )
    unit = UnitSpec(clip="c1", stage="camera", selection=WindowSelection(), num_windows=1)
    first = runner._acquire_host(unit)
    assert first.name == "only-gpu"

    done = threading.Event()

    def grab() -> None:
        runner._acquire_host(unit)
        done.set()

    waiter = threading.Thread(target=grab, daemon=True)
    waiter.start()
    assert not done.wait(timeout=1.0), "second unit started while the host was full"
    runner._release_host()
    assert done.wait(timeout=30)
    # the waiter now owns the slot; hand it back so the runner stays balanced
    runner._release_host()


def test_failed_unit_writes_no_marker_or_artefact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure must not leave anything a later stage could mistake for output."""
    executor = RecordingExecutor(host=host(), fail_times=99)
    monkeypatch.setattr("ego3d_action.runtime.batch.build_executor", lambda *a, **k: executor)
    runner = make_runner(tmp_path, [gpu_host()])
    unit = UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1)
    runner.run_unit(unit, skip_existing=False)

    assert not (tmp_path / "data" / "c1" / "detection" / "detection.npz").exists()
    assert read_marker(tmp_path / "data" / "c1" / "detection", "detection") is None


def test_retry_then_success_is_recorded_with_the_attempt_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artefact = tmp_path / "data" / "c1" / "detection" / "detection.npz"
    executor = RecordingExecutor(host=host(), fail_times=1, write=artefact)
    monkeypatch.setattr("ego3d_action.runtime.batch.build_executor", lambda *a, **k: executor)
    runner = make_runner(tmp_path, [gpu_host()], retries=2)
    unit = UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1)

    result = runner.run_unit(unit, skip_existing=False)
    assert result.status == "ok"
    assert result.attempts == 2
    assert artefact.is_file()
    assert read_marker(tmp_path / "data" / "c1", "detection") is not None
    assert runner.degraded_clips() == []


def test_skip_existing_performs_zero_computation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artefact = tmp_path / "data" / "c1" / "detection" / "detection.npz"
    executor = RecordingExecutor(host=host(), write=artefact)
    monkeypatch.setattr("ego3d_action.runtime.batch.build_executor", lambda *a, **k: executor)
    hosts = [gpu_host()]
    unit = UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1)

    first = make_runner(tmp_path, hosts, retries=0)
    assert first.run_unit(unit, skip_existing=True).status == "ok"
    assert executor.calls == 1

    # A brand new runner (as a re-run would be) must skip entirely.
    second = make_runner(tmp_path, hosts, retries=0)
    result = second.run_unit(unit, skip_existing=True)
    assert result.status == "skipped"
    assert executor.calls == 1  # unchanged: no second invocation


def test_skip_existing_recomputes_when_the_parameters_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artefact = tmp_path / "data" / "c1" / "detection" / "detection.npz"
    executor = RecordingExecutor(host=host(), write=artefact)
    monkeypatch.setattr("ego3d_action.runtime.batch.build_executor", lambda *a, **k: executor)
    hosts = [gpu_host()]

    first = make_runner(tmp_path, hosts, retries=0)
    first.run_unit(UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1), skip_existing=True)
    assert executor.calls == 1

    # Same stage, different num_frames -> different identity -> must recompute.
    changed = UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1, num_frames=999)
    second = make_runner(tmp_path, hosts, retries=0)
    assert second.run_unit(changed, skip_existing=True).status == "ok"
    assert executor.calls == 2


def test_dry_run_computes_nothing(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, [host()])
    unit = UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1)
    result = runner.run_unit(unit, skip_existing=False, dry_run=True)
    assert result.status == "planned"


def test_run_plan_stops_a_clips_later_stages_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Later stages must not run on missing inputs (which would crash or lie)."""
    calls: list[str] = []

    class AlwaysFail:
        host = gpu_host()

        def run(self, command, *, cwd=None, env=None, timeout=None):  # type: ignore[no-untyped-def]
            from ego3d_action.runtime.executor import ExecResult

            calls.append(Path(str(command[-2])).name if len(command) > 1 else "")
            return ExecResult(returncode=1, stderr="nope", host="cpu")

        def describe(self) -> str:
            return "always-fail"

    monkeypatch.setattr("ego3d_action.runtime.batch.build_executor", lambda *a, **k: AlwaysFail())
    runner = make_runner(
        tmp_path,
        [gpu_host()],
        retries=0,
    )
    plan = BatchPlan(
        units=[
            UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1),
            UnitSpec(clip="c1", stage="stitch", selection=WindowSelection(), num_windows=1),
        ]
    )
    results = runner.run_plan(plan, skip_existing=False)
    assert [r.status for r in results] == ["failed"]
    assert len(calls) == 1  # stitch was never attempted


def test_report_is_written_even_with_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executor = RecordingExecutor(host=host(), fail_times=99)
    monkeypatch.setattr("ego3d_action.runtime.batch.build_executor", lambda *a, **k: executor)
    runner = make_runner(
        tmp_path,
        [gpu_host()],
        retries=0,
    )
    runner.run_unit(
        UnitSpec(clip="c1", stage="detection", selection=WindowSelection(), num_windows=1),
        skip_existing=False,
    )
    path = runner.write_report(tmp_path / "outputs" / "batch_report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["failed"] == 1
    assert payload["degraded_clips"] == ["c1"]
    assert payload["units"][0]["unit"]["clip"] == "c1"
    assert payload["hosts"]


def test_runner_rejects_bad_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_runner(tmp_path, [], retries=0)
    with pytest.raises(ValueError):
        make_runner(tmp_path, [host()], retries=-1)
    with pytest.raises(ValueError):
        make_runner(tmp_path, [host()], max_parallel=0)
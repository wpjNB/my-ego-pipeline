"""The runner protocol: resolution, JSON summaries and failure handling."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from ego3d_action.config import Config, load_config
from ego3d_action.errors import BackendExecutionError, ConfigError
from ego3d_action.runtime.subprocess_backend import (
    BackendInvocation,
    RunnerSpec,
    parse_summary,
    run_runner,
)


def write_script(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_invocation_from_config_real_mode() -> None:
    invocation = BackendInvocation.from_config(load_config("configs/macrodata_final.yaml"))
    assert invocation.mode == "real"
    assert invocation.python_commands["hawor"] == ("conda", "run", "-n", "ego3d_hawor", "python")


def test_invocation_from_config_mock_mode() -> None:
    invocation = BackendInvocation.from_config(load_config("configs/mock.yaml"))
    assert invocation.is_mock
    assert invocation.seed == 0


def test_invocation_rejects_unknown_mode() -> None:
    config = Config(data={"backends": {"mode": "banana", "timeout_seconds": 10}})
    with pytest.raises(ConfigError, match="backends.mode"):
        BackendInvocation.from_config(config)


def test_invocation_accepts_a_string_command() -> None:
    config = Config(
        data={
            "backends": {"mode": "real", "python": {"wilor": "conda run -n env python"}},
            "paths": {"backends": "backends"},
        }
    )
    invocation = BackendInvocation.from_config(config)
    assert invocation.python_commands["wilor"] == ("conda", "run", "-n", "env", "python")


def test_resolve_mock_uses_the_orchestrator_interpreter() -> None:
    invocation = BackendInvocation(
        mode="mock", backends_dir=Path("backends"), python_commands={}, timeout_seconds=30
    )
    spec, prefix = invocation.resolve("vggt", mock_subcommand="vggt")
    assert spec.mode == "mock"
    assert spec.command == (sys.executable,)
    assert spec.script.name == "mock_backend.py"
    assert prefix == ("vggt", "--seed", "0")


def test_resolve_real_requires_a_configured_interpreter() -> None:
    invocation = BackendInvocation(
        mode="real", backends_dir=Path("backends"), python_commands={}, timeout_seconds=30
    )
    with pytest.raises(ConfigError, match="backends.python.wilor"):
        invocation.resolve("wilor")


def test_resolve_real_reports_a_missing_runner(tmp_path: Path) -> None:
    invocation = BackendInvocation(
        mode="real",
        backends_dir=tmp_path,
        python_commands={"wilor": ("python",)},
        timeout_seconds=30,
    )
    with pytest.raises(ConfigError, match="runner script not found"):
        invocation.resolve("wilor")


def test_parse_summary_reads_the_last_json_line() -> None:
    stdout = 'noise\n{"status": "ok", "a": 1}\nwarning\n{"status": "ok", "b": 2}\n'
    assert parse_summary(stdout) == {"status": "ok", "b": 2}
    assert parse_summary("nothing here") is None
    assert parse_summary("not json at all") is None


OK_SCRIPT = """
import json
import sys
print(json.dumps({"status": "ok", "argv": sys.argv[1:]}))
"""

FAIL_SCRIPT = """
import json
print(json.dumps({"status": "error", "message": "boom"}))
raise SystemExit(3)
"""

SILENT_SCRIPT = "print('nothing to report')"

SLOW_SCRIPT = """
import time
time.sleep(30)
"""


def test_run_runner_returns_the_payload(tmp_path: Path) -> None:
    script = write_script(tmp_path, "ok.py", OK_SCRIPT)
    spec = RunnerSpec(name="demo", mode="mock", script=script, command=(sys.executable,))
    log = tmp_path / "runner.log"
    payload = run_runner(spec, ["--x", "1"], log_path=log)
    assert payload["status"] == "ok"
    assert payload["argv"] == ["--x", "1"]
    assert log.is_file()
    assert "--x" in log.read_text(encoding="utf-8")


def test_run_runner_reports_a_failing_runner(tmp_path: Path) -> None:
    script = write_script(tmp_path, "fail.py", FAIL_SCRIPT)
    spec = RunnerSpec(name="demo", mode="real", script=script, command=(sys.executable,))
    with pytest.raises(BackendExecutionError) as excinfo:
        run_runner(spec, [])
    assert excinfo.value.exit_code == 3
    assert "boom" in str(excinfo.value)


def test_run_runner_rejects_a_silent_runner(tmp_path: Path) -> None:
    script = write_script(tmp_path, "silent.py", SILENT_SCRIPT)
    spec = RunnerSpec(name="demo", mode="real", script=script, command=(sys.executable,))
    with pytest.raises(BackendExecutionError, match="no JSON summary"):
        run_runner(spec, [])


def test_run_runner_times_out(tmp_path: Path) -> None:
    script = write_script(tmp_path, "slow.py", SLOW_SCRIPT)
    spec = RunnerSpec(
        name="demo", mode="real", script=script, command=(sys.executable,), timeout_seconds=0.5
    )
    with pytest.raises(BackendExecutionError, match="timeout"):
        run_runner(spec, [])


def test_run_runner_reports_launch_failures(tmp_path: Path) -> None:
    spec = RunnerSpec(
        name="demo",
        mode="real",
        script=tmp_path / "nope.py",
        command=("/definitely/not/an/interpreter",),
        timeout_seconds=5,
    )
    with pytest.raises(BackendExecutionError, match="cannot launch"):
        run_runner(spec, [])


def test_real_runners_report_unavailable_checkouts(tmp_path: Path) -> None:
    """Every real runner supports ``--check`` and answers in JSON."""
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    for name in ("wilor", "hawor", "vggt"):
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "backends" / (name + "_runner.py")),
                "--check",
                "--third-party",
                str(tmp_path / "third_party"),
                "--weights",
                str(tmp_path / "weights"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1, result.stderr
        payload = parse_summary(result.stdout)
        assert payload is not None
        assert payload["backend"] == name
        assert payload["available"] is False

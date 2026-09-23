"""The environment/asset audit tool."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ego3d_action.runtime.doctor import (
    Check,
    check_backends,
    check_configuration,
    check_data,
    check_host_tools,
    check_python_dependencies,
    run_all,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_host_tools_reports_ffmpeg() -> None:
    checks = check_host_tools()
    names = {check.name for check in checks}
    assert names == {"ffmpeg", "ffprobe"}
    for check in checks:
        assert check.status in {"ok", "missing"}
        if check.status == "missing":
            assert check.fix


def test_python_dependency_checks_are_informative() -> None:
    checks = check_python_dependencies()
    by_name = {check.name: check for check in checks}
    assert by_name["python:numpy"].status == "ok"
    missing = check_python_dependencies(
        dependencies=[("definitely_not_a_module_xyz", "test")], required_for="gpu", fix="do the thing"
    )
    assert missing[0].status == "missing"
    assert missing[0].required_for == "gpu"
    assert missing[0].fix == "do the thing"


def test_configuration_check(tmp_path: Path) -> None:
    ok = check_configuration("configs/mock.yaml")
    assert ok[0].status == "ok"
    assert "mock" in ok[0].detail

    missing = check_configuration(tmp_path / "nope.yaml")
    assert missing[0].status == "missing"
    assert missing[0].fix

    broken = tmp_path / "broken.yaml"
    broken.write_text("hand: [1, 2\n", encoding="utf-8")
    assert check_configuration(broken)[0].status == "missing"


def test_backend_checks_name_the_missing_pieces(tmp_path: Path) -> None:
    checks = check_backends(tmp_path / "third_party", tmp_path / "weights")
    assert len(checks) == 3
    for check in checks:
        assert check.status == "missing"
        assert check.required_for == "gpu"
        assert "third_party" in check.fix and "clone" in check.fix


def test_data_checks_flag_a_missing_mano(tmp_path: Path) -> None:
    checks = check_data("configs/mock.yaml", clip="nope")
    by_name = {check.name: check for check in checks}
    assert by_name["MANO model"].status == "warn"
    assert by_name["clip:frames"].status == "warn"
    assert by_name["clip:detection"].fix


def test_summarise_verdicts_and_exit_codes() -> None:
    ok_cpu = [Check(name="a", status="ok", detail="fine")]
    text, code = summarise(ok_cpu)
    assert code == 0
    assert "CPU path: ready" in text

    missing_cpu = [Check(name="a", status="missing", detail="gone", fix="install it")]
    text, code = summarise(missing_cpu)
    assert code == 1
    assert "NOT ready" in text and "fix: install it" in text

    missing_gpu = [
        Check(name="a", status="missing", detail="gone", fix="download", required_for="gpu")
    ]
    assert summarise(missing_gpu)[1] == 0  # the CPU path is unaffected
    assert summarise(missing_gpu, strict=True)[1] == 1
    assert "GPU path: not complete" in summarise(missing_gpu)[0]

    warnings = [Check(name="a", status="warn", detail="meh")]
    assert summarise(warnings)[1] == 0
    assert summarise(warnings, strict=True)[1] == 1


def test_run_all_never_raises_and_covers_every_section() -> None:
    checks = run_all("configs/mock.yaml", clip="demo01")
    sections = {check.section for check in checks}
    assert {"host", "python", "config", "backends", "data"} <= sections
    text, code = summarise(checks)
    assert isinstance(text, str) and code in {0, 1}


def test_doctor_cli_json_output() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/doctor.py", "--config", "configs/mock.yaml", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert payload["exit_code"] in {0, 1}
    names = {item["name"] for item in payload["checks"]}
    assert "python:numpy" in names
    assert any(name.startswith("backend:") for name in names)
    # The mock config needs no weights, so the CPU path must be reported ready.
    assert result.returncode == 0


def test_doctor_cli_strict_can_fail_on_the_bundled_machine() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/doctor.py", "--config", "configs/mock.yaml", "--strict"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # No weights are bundled here, so strict mode is expected to fail - but the
    # report must still be printed rather than an exception raised.
    assert "GPU path" in result.stdout
    assert result.returncode in {0, 1}


def test_doctor_reports_trajectory_presence(tmp_path: Path) -> None:
    clip = tmp_path / "demo01"
    (clip / "trajectory").mkdir(parents=True)
    (clip / "trajectory" / "trajectory.npz").write_bytes(b"stub")
    config = tmp_path / "config.yaml"
    config.write_text(
        "runtime: {device: cpu}\n"
        f"paths: {{data_root: {tmp_path}, third_party: third_party, weights: weights}}\n"
        "detection: {min_confidence: 0.75, max_gap: 4, iou_threshold: 0.2}\n"
        "hand: {window: 16, overlap: 8}\n"
        "camera: {window: 200, overlap: 40, resolution: 416, checkpoint: x}\n"
        "stitch: {pixel_stride: 8}\n"
        "refinement: {bone_scale_max_correction: 0.035, wrist_depth_lambda: 0.2}\n"
        "evaluation: {chunk_seconds: 1.0}\n"
        "backends: {mode: mock, timeout_seconds: 10, seed: 0}\n",
        encoding="utf-8",
    )
    by_name = {check.name: check for check in check_data(config, clip="demo01")}
    assert by_name["clip:trajectory"].status == "ok"
    assert by_name["clip:frames"].status == "warn"
    _ = np  # numpy import kept for parity with the other test modules

"""Device resolution, including the no-torch orchestrator on a GPU server."""

from __future__ import annotations

import subprocess

import pytest

from ego3d_action.runtime import device as device_module


class _Completed:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def _fake_driver(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> None:
    monkeypatch.setattr(device_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        device_module.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(stdout, returncode),
    )


def test_no_driver_means_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module.shutil, "which", lambda name: None)
    assert device_module.driver_gpu_names() == []
    assert device_module.resolve_device("auto") == "cpu"


def test_driver_is_consulted_when_torch_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The orchestrator has no torch; 'auto' must still find the server's GPUs."""
    _fake_driver(
        monkeypatch,
        "GPU 0: Tesla P100-PCIE-12GB (UUID: GPU-aaa)\n"
        "GPU 1: Tesla P100-PCIE-12GB (UUID: GPU-bbb)\n",
    )
    names = device_module.driver_gpu_names()
    assert names == [
        "Tesla P100-PCIE-12GB (UUID: GPU-aaa)",
        "Tesla P100-PCIE-12GB (UUID: GPU-bbb)",
    ]
    # Wherever torch is importable the answer still comes from torch, so this
    # assertion only means something on a machine with a real (or faked) driver.
    monkeypatch.setenv("EGO3D_FORCE_CPU", "0")
    if device_module.cuda_available():
        assert device_module.resolve_device("auto") == "cuda:0"


def test_force_cpu_wins_over_the_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_driver(monkeypatch, "GPU 0: Tesla P100-PCIE-12GB (UUID: GPU-aaa)\n")
    monkeypatch.setenv("EGO3D_FORCE_CPU", "1")
    assert device_module.cuda_available() is False
    assert device_module.resolve_device("auto") == "cpu"


def test_driver_failure_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_driver(monkeypatch, "", returncode=9)
    assert device_module.driver_gpu_names() == []

    def boom(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=15)

    monkeypatch.setattr(device_module.subprocess, "run", boom)
    assert device_module.driver_gpu_names() == []

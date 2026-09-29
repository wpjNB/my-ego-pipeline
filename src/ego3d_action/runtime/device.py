"""Device (CPU / CUDA) resolution.

The development machine is CPU-only while the target server has GPUs, so the
device is never hard-coded: every backend adapter asks this module for a device
string derived from the config. ``"auto"`` prefers CUDA when present; an
explicit ``"cuda"`` request on a machine without CUDA raises instead of quietly
falling back to CPU.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

from ..errors import BackendNotAvailableError, ConfigError

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``requested`` (``auto`` / ``cpu`` / ``cuda`` / ``cuda:N``).

    Raises:
        ConfigError: on an unparsable device string.
        BackendNotAvailableError: when CUDA is explicitly requested but
            unavailable (no torch, no driver, or no visible device).
    """
    want = (requested or "auto").strip().lower()
    if want in {"auto", "gpu"}:
        return "cuda:0" if cuda_available() else "cpu"
    if want == "cpu":
        return "cpu"
    if want.startswith("cuda"):
        if not cuda_available():
            raise BackendNotAvailableError(
                "cuda",
                "device was requested as 'cuda' but torch.cuda.is_available() is False. "
                "Set device: cpu in the config, or run on the GPU server.",
            )
        index = want.split(":", 1)[1] if ":" in want else "0"
        if not index.isdigit():
            raise ConfigError(f"invalid CUDA device index in '{requested}'")
        return f"cuda:{index}"
    raise ConfigError(f"unknown device '{requested}' (use auto|cpu|cuda|cuda:N)")


def cuda_available() -> bool:
    """True when *something* can see a CUDA device.

    The orchestrator deliberately has no torch (the model backends bring their
    own environments), so an ``ImportError`` cannot mean "CPU-only" - it means
    "ask the driver instead". Without this fallback the orchestrator resolved
    ``device: auto`` to ``cpu`` on the GPU server and handed ``--device cpu`` to
    backends that then refused to run.
    """
    if os.environ.get("EGO3D_FORCE_CPU", "").strip() in {"1", "true", "yes"}:
        return False
    try:
        import torch  # noqa: PLC0415 - optional heavy dependency
    except ImportError:
        visible = driver_gpu_names()
        logger.debug(
            "torch is not installed; the driver reports %d GPU(s)", len(visible)
        )
        return bool(visible)
    available = bool(torch.cuda.is_available())
    if not available:
        logger.debug("torch %s present but no CUDA device is visible", torch.__version__)
    return available


def driver_gpu_names() -> list[str]:
    """GPU names reported by ``nvidia-smi -L``; ``[]`` when there is no driver.

    Used only to answer "is there a GPU at all?" in environments without torch.
    Actual device selection stays with torch inside the backend.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - driver oddities
        logger.debug("nvidia-smi could not be run: %s", exc)
        return []
    if completed.returncode != 0:
        logger.debug("nvidia-smi -L exited with %d", completed.returncode)
        return []
    return [
        line.split(":", 1)[1].strip()
        for line in completed.stdout.splitlines()
        if line.startswith("GPU ") and ":" in line
    ]


def torch_device(requested: str = "auto") -> "object":
    """Return a ``torch.device`` for ``requested``.

    Raises:
        BackendNotAvailableError: when torch itself is missing.
    """
    try:
        import torch  # noqa: PLC0415 - optional heavy dependency
    except ImportError as exc:
        raise BackendNotAvailableError(
            "torch",
            "torch is not installed in this environment; the model backends "
            "(WiLoR / HaWoR / VGGT-Omega) need their own env - see environment-*.yml",
        ) from exc
    return torch.device(resolve_device(requested))


def torch_dtype(device: str, *, prefer_fp16_on_cuda: bool = True) -> "object":
    """Pick a sensible dtype for a resolved device string."""
    try:
        import torch  # noqa: PLC0415 - optional heavy dependency
    except ImportError as exc:
        raise BackendNotAvailableError("torch", "torch is not installed in this environment") from exc
    if device.startswith("cuda") and prefer_fp16_on_cuda:
        return torch.float16
    return torch.float32


@dataclass(frozen=True)
class EnvironmentReport:
    """A short, loggable description of the execution environment."""

    python: str
    torch: str | None
    cuda_available: bool
    device_names: tuple[str, ...]
    resolved_device: str

    def format(self) -> str:
        torch_version = self.torch or "not installed"
        gpus = ", ".join(self.device_names) if self.device_names else "none"
        return (
            f"python={self.python} torch={torch_version} cuda={self.cuda_available} "
            f"gpus=[{gpus}] device={self.resolved_device}"
        )


def describe_environment(requested: str = "auto") -> EnvironmentReport:
    """Collect a one-line environment summary for logs and ``--dry-run`` output."""
    import platform
    import sys

    torch_version: str | None = None
    names: list[str] = []
    try:
        import torch  # noqa: PLC0415 - optional heavy dependency

        torch_version = torch.__version__
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        torch_version = None
    return EnvironmentReport(
        python=platform.python_version() or sys.version.split()[0],
        torch=torch_version,
        cuda_available=bool(names),
        device_names=tuple(names),
        resolved_device=resolve_device(requested),
    )

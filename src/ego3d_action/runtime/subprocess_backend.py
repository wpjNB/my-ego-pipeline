"""Runner protocol for the external model backends.

Each backend is executed as a subprocess so that the pinned CUDA/torch stacks
stay inside their own environments and the orchestrator never imports them. The
contract is deliberately small:

* the runner is a standalone script that takes CLI arguments,
* it writes its artefacts to a path given on the command line,
* it prints **one** JSON object as the last stdout line
  (``{"status": "ok", ...}`` or ``{"status": "error", "message": ...}``),
* a non-zero exit code or a timeout is an error with the output tail attached.

``mode: mock`` swaps in ``backends/mock_backend.py`` run by the orchestrator's
own interpreter; ``mode: real`` uses the configured interpreter (usually
``conda run -n ego3d_<backend> python``) and the real runner script.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..errors import BackendExecutionError, ConfigError

logger = logging.getLogger(__name__)

MOCK_MODE = "mock"
REAL_MODE = "real"
VALID_MODES = (REAL_MODE, MOCK_MODE)


@dataclass(frozen=True)
class RunnerSpec:
    """Everything needed to launch one backend runner."""

    name: str
    mode: str
    script: Path
    command: tuple[str, ...]
    timeout_seconds: float = 3600.0

    @property
    def display(self) -> str:
        return " ".join((*self.command, str(self.script)))


@dataclass(frozen=True)
class BackendInvocation:
    """Resolved backend configuration shared by the adapters."""

    mode: str
    backends_dir: Path
    python_commands: Mapping[str, Sequence[str]]
    timeout_seconds: float = 3600.0
    seed: int = 0

    @classmethod
    def from_config(cls, config: object) -> "BackendInvocation":
        """Build from an ``ego3d_action.config.Config``."""
        mode = str(config.get("backends.mode", REAL_MODE))  # type: ignore[attr-defined]
        if mode not in VALID_MODES:
            raise ConfigError(f"backends.mode must be one of {VALID_MODES}, got '{mode}'")
        backends_dir = Path(str(config.get("paths.backends", "backends")))  # type: ignore[attr-defined]
        raw = config.get("backends.python", {})  # type: ignore[attr-defined]
        if not isinstance(raw, Mapping):
            raise ConfigError("backends.python must be a mapping of backend -> command list")
        commands: dict[str, tuple[str, ...]] = {}
        for name, value in raw.items():
            if isinstance(value, str):
                commands[str(name)] = tuple(value.split())
            elif isinstance(value, Sequence):
                commands[str(name)] = tuple(str(part) for part in value)
            else:
                raise ConfigError(
                    f"backends.python.{name} must be a list of command parts, got {value!r}"
                )
        return cls(
            mode=mode,
            backends_dir=backends_dir,
            python_commands=commands,
            timeout_seconds=float(config.get("backends.timeout_seconds", 3600.0)),  # type: ignore[attr-defined]
            seed=int(config.get("backends.seed", 0)),  # type: ignore[attr-defined]
        )

    @property
    def is_mock(self) -> bool:
        return self.mode == MOCK_MODE

    def resolve(
        self, name: str, *, mock_subcommand: str | None = None
    ) -> tuple[RunnerSpec, tuple[str, ...]]:
        """Return ``(runner spec, leading CLI arguments)`` for ``name``.

        Args:
            name: logical backend name, e.g. ``"wilor"``.
            mock_subcommand: subcommand of ``mock_backend.py`` used in mock mode
                (``None`` means this backend has no mock implementation).

        Raises:
            ConfigError: in mock mode without a subcommand, or when no
                interpreter is configured for a real run.
        """
        if self.is_mock:
            if mock_subcommand is None:
                raise ConfigError(f"backend '{name}' has no mock implementation")
            script = self.backends_dir / "mock_backend.py"
            if not script.is_file():
                raise ConfigError(f"mock backend runner not found at {script}")
            return (
                RunnerSpec(
                    name=f"mock-{name}",
                    mode=MOCK_MODE,
                    script=script,
                    command=(sys.executable,),
                    timeout_seconds=self.timeout_seconds,
                ),
                (mock_subcommand, "--seed", str(self.seed)),
            )

        script = self.backends_dir / f"{name}_runner.py"
        if not script.is_file():
            raise ConfigError(f"runner script not found at {script}")
        command = self.python_commands.get(name)
        if not command:
            raise ConfigError(
                f"backends.python.{name} is not configured; set it to the interpreter of the "
                f"'ego3d_{name}' environment (e.g. [\"conda\", \"run\", \"-n\", \"ego3d_{name}\", "
                '"python"])'
            )
        return (
            RunnerSpec(
                name=name,
                mode=REAL_MODE,
                script=script,
                command=tuple(command),
                timeout_seconds=self.timeout_seconds,
            ),
            (),
        )


def run_runner(
    spec: RunnerSpec,
    args: Sequence[str],
    *,
    log_path: Path | None = None,
) -> dict[str, object]:
    """Execute a runner and return its JSON summary.

    Raises:
        BackendExecutionError: on a non-zero exit, a timeout, or output that
            does not end with a JSON object.
    """
    command = [*spec.command, str(spec.script), *args]
    logger.info("running backend '%s': %s", spec.name, " ".join(command))
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=spec.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendExecutionError(
            spec.name, f"timeout after {spec.timeout_seconds:.0f}s"
        ) from exc
    except OSError as exc:
        raise BackendExecutionError(spec.name, f"cannot launch runner: {exc}") from exc

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(command)}\n\n[stdout]\n{proc.stdout}\n[stderr]\n{proc.stderr}\n",
            encoding="utf-8",
        )

    payload = parse_summary(proc.stdout)
    if proc.returncode != 0:
        detail = str(payload.get("message")) if payload else tail(proc.stderr)
        raise BackendExecutionError(spec.name, detail, exit_code=proc.returncode)
    if payload is None:
        raise BackendExecutionError(
            spec.name,
            "runner produced no JSON summary; last output: " + tail(proc.stdout or proc.stderr),
            exit_code=proc.returncode,
        )
    if payload.get("status") != "ok":
        raise BackendExecutionError(
            spec.name, str(payload.get("message", payload)), exit_code=proc.returncode
        )
    return payload


def parse_summary(stdout: str) -> dict[str, object] | None:
    """Return the last JSON object printed on stdout, if any."""
    for line in reversed([line.strip() for line in (stdout or "").splitlines() if line.strip()]):
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def tail(text: str, lines: int = 12) -> str:
    """Join the last ``lines`` of ``text`` into one log-friendly string."""
    return " | ".join((text or "").strip().splitlines()[-lines:])

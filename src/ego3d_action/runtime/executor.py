"""Where a stage command runs: the local host or an SSH worker.

The pipeline's stages are already location-transparent in the way that matters -
they communicate only through the artefact contract on disk
(:mod:`ego3d_action.io.artefacts`) and their runner's JSON summary line
(:mod:`ego3d_action.runtime.subprocess_backend`). So a stage can run anywhere
that can see the artefact store, and this module is the only place that has to
know *where*.

Two pieces:

* :class:`HostCapabilities` - a worker declares what it can run (which backend
  environments, whether it has a GPU and how much memory, where the checkouts
  and weights live). Declared rather than probed: the scheduler must be able to
  decide without logging into every machine, and a wrong declaration is a config
  error that surfaces at dispatch time instead of a mysterious OOM hours in.
* :class:`Executor` - runs one command and returns its result. ``local`` is the
  default; ``ssh`` is implemented here. ``slurm`` and ``k8s`` are the documented
  next implementations and plug into :func:`build_executor` without the
  scheduler above noticing.

Environments are never shipped between machines: each host's ``python`` command
already points at its own ``ego3d_*`` environments (conda/CUDA stacks are not
portable, and an env on NFS is unusable). Only frames and artefacts travel.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from ..errors import ConfigError

logger = logging.getLogger(__name__)

#: Executors implemented today. ``slurm``/``k8s`` are the M2 additions.
KNOWN_EXECUTORS = ("local", "ssh")


class CapabilityMismatch(ConfigError):
    """No configured host can satisfy a task's requirements.

    Carries the per-host reasons so the message says *why* every candidate was
    rejected - never a bare "no host found".
    """

    def __init__(self, required: Mapping[str, Any], reasons: Mapping[str, str]) -> None:
        self.required = dict(required)
        self.reasons = dict(reasons)
        detail = "; ".join(f"{name}: {why}" for name, why in sorted(reasons.items())) or "no hosts configured"
        super().__init__(f"no host can run {dict(required)} - {detail}")


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one executed command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    wall_time_seconds: float = 0.0
    command: tuple[str, ...] = ()
    host: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail(self, *, limit: int = 4000) -> str:
        """The end of the combined output, for an error message."""
        combined = f"[stdout]\n{self.stdout}\n[stderr]\n{self.stderr}"
        return combined[-limit:]


@dataclass(frozen=True)
class HostCapabilities:
    """One worker's declared abilities and paths.

    ``backends`` is the set of logical backend names this host can invoke
    (``wilor``/``hawor``/``vggt``). A host without a name simply cannot be
    scheduled for a task that requires it - which is how the two branches of the
    pipeline (hands vs camera) end up on different machines.
    """

    name: str
    backends: tuple[str, ...] = ()
    python: Mapping[str, Sequence[str]] = field(default_factory=dict)
    cuda: str | None = None
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    weights_root: str = "weights"
    third_party_root: str = "third_party"
    backends_root: str = "backends"
    data_root: str = "data"
    repo_root: str = "."
    executor: str = "local"
    ssh_host: str | None = None
    ssh_user: str | None = None
    ssh_port: int | None = None
    max_parallel: int = 1
    #: Interpreter that runs the stage scripts on this host (the ``ego3d_base``
    #: env that has ``ego3d_action`` installed). Resolved on the host itself, so
    #: a remote worker never needs this repository's environment on the caller.
    orchestrator_python: tuple[str, ...] = ("python",)
    extra_env: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("host name must not be empty")
        if self.executor not in KNOWN_EXECUTORS:
            raise ConfigError(
                f"host '{self.name}': executor must be one of {KNOWN_EXECUTORS}, got '{self.executor}'"
            )
        if self.executor == "ssh" and not self.ssh_host:
            raise ConfigError(f"host '{self.name}': executor 'ssh' requires ssh_host")
        if self.gpu_count < 0:
            raise ConfigError(f"host '{self.name}': gpu_count must be >= 0")
        if self.max_parallel < 1:
            raise ConfigError(f"host '{self.name}': max_parallel must be >= 1")
        for backend in self.backends:
            if backend not in self.python:
                raise ConfigError(
                    f"host '{self.name}': backend '{backend}' is declared but has no python command"
                )

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0 and bool(self.cuda)

    def python_command(self, backend: str) -> tuple[str, ...]:
        """The interpreter command for ``backend`` on this host."""
        command = self.python.get(backend)
        if not command:
            raise ConfigError(
                f"host '{self.name}' has no python command for backend '{backend}'"
            )
        return tuple(str(part) for part in command)

    @property
    def stage_python(self) -> tuple[str, ...]:
        """Interpreter used to run the orchestrator's own stage scripts."""
        return tuple(str(part) for part in self.orchestrator_python) or ("python",)

    def explain_shortfall(self, required: Mapping[str, Any]) -> str | None:
        """Why this host cannot run a task, or ``None`` when it can.

        Always returns a concrete reason, so a scheduling failure names the
        missing capability instead of failing opaquely.
        """
        wanted_backends = set(required.get("backends", ()))
        absent = sorted(wanted_backends - set(self.backends))
        if absent:
            return f"missing backend(s) {absent} (has {sorted(self.backends)})"
        if required.get("gpu") and not self.has_gpu:
            return f"needs a GPU but has none (gpu_count={self.gpu_count}, cuda={self.cuda})"
        needed_vram = float(required.get("min_gpu_memory_gb", 0.0) or 0.0)
        if needed_vram > 0 and self.gpu_memory_gb < needed_vram:
            return f"needs >= {needed_vram} GB VRAM but has {self.gpu_memory_gb} GB"
        if required.get("executor") and self.executor != required["executor"]:
            return f"executor '{self.executor}' != required '{required['executor']}'"
        return None

    def matches(self, required: Mapping[str, Any]) -> bool:
        return self.explain_shortfall(required) is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backends": list(self.backends),
            "python": {name: list(cmd) for name, cmd in sorted(self.python.items())},
            "cuda": self.cuda,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory_gb,
            "weights_root": self.weights_root,
            "third_party_root": self.third_party_root,
            "data_root": self.data_root,
            "repo_root": self.repo_root,
            "executor": self.executor,
            "ssh_host": self.ssh_host,
            "ssh_user": self.ssh_user,
            "ssh_port": self.ssh_port,
            "max_parallel": self.max_parallel,
            "orchestrator_python": list(self.orchestrator_python),
        }


def load_hosts(path: str | Path) -> list[HostCapabilities]:
    """Load ``hosts.yaml`` into validated host declarations.

    The file is either ``{"hosts": [...]}`` or a bare list. An empty or missing
    file is an error: a batch run without hosts has nothing to schedule onto.
    """
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"hosts file not found: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source} is not valid YAML: {exc}") from exc

    entries = payload.get("hosts") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ConfigError(f"{source} must define a non-empty 'hosts' list")

    hosts: list[HostCapabilities] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{source}: each host must be a mapping, got {entry!r}")
        unknown = set(entry) - {
            "name",
            "backends",
            "python",
            "cuda",
            "gpu_count",
            "gpu_memory_gb",
            "weights_root",
            "third_party_root",
            "backends_root",
            "data_root",
            "repo_root",
            "executor",
            "ssh_host",
            "ssh_user",
            "ssh_port",
            "max_parallel",
            "orchestrator_python",
            "extra_env",
            "notes",
        }
        if unknown:
            # A typo like 'gpu_mem_gb' would silently disable a constraint.
            raise ConfigError(f"{source}: host has unknown keys {sorted(unknown)}")
        hosts.append(
            HostCapabilities(
                name=str(entry["name"]),
                backends=tuple(str(b) for b in entry.get("backends", ())),
                python={
                    str(name): tuple(str(p) for p in cmd)
                    for name, cmd in dict(entry.get("python", {})).items()
                },
                cuda=None if entry.get("cuda") is None else str(entry["cuda"]),
                gpu_count=int(entry.get("gpu_count", 0)),
                gpu_memory_gb=float(entry.get("gpu_memory_gb", 0.0)),
                weights_root=str(entry.get("weights_root", "weights")),
                third_party_root=str(entry.get("third_party_root", "third_party")),
                backends_root=str(entry.get("backends_root", "backends")),
                data_root=str(entry.get("data_root", "data")),
                repo_root=str(entry.get("repo_root", ".")),
                executor=str(entry.get("executor", "local")),
                ssh_host=None if entry.get("ssh_host") is None else str(entry["ssh_host"]),
                ssh_user=None if entry.get("ssh_user") is None else str(entry["ssh_user"]),
                ssh_port=None if entry.get("ssh_port") is None else int(entry["ssh_port"]),
                max_parallel=int(entry.get("max_parallel", 1)),
                orchestrator_python=tuple(
                    str(part) for part in entry.get("orchestrator_python", ("python",))
                ),
                extra_env={str(k): str(v) for k, v in dict(entry.get("extra_env", {})).items()},
                notes=str(entry.get("notes", "")),
            )
        )
    names = [host.name for host in hosts]
    if len(names) != len(set(names)):
        raise ConfigError(f"{source}: host names must be unique, got {names}")
    return hosts


def select_host(
    hosts: Sequence[HostCapabilities],
    required: Mapping[str, Any],
    *,
    preferred: str | None = None,
) -> HostCapabilities:
    """Pick the least-loaded capable host (or a named one).

    Ties break on ``max_parallel`` then name, so the choice is deterministic
    across runs - a scheduler that picks differently every time makes a scaling
    table impossible to reproduce.

    Raises:
        CapabilityMismatch: when no host qualifies, listing every rejection.
    """
    if preferred:
        for host in hosts:
            if host.name == preferred:
                reason = host.explain_shortfall(required)
                if reason:
                    raise CapabilityMismatch(required, {host.name: reason})
                return host
        raise CapabilityMismatch(required, {preferred: "no such host in hosts.yaml"})

    reasons = {host.name: host.explain_shortfall(required) or "" for host in hosts}
    candidates = [host for host in hosts if not reasons[host.name]]
    if not candidates:
        raise CapabilityMismatch(required, {name: why for name, why in reasons.items() if why})
    candidates.sort(key=lambda host: (-host.max_parallel, host.name))
    return candidates[0]


class Executor(Protocol):
    """Runs one command on one host."""

    host: HostCapabilities

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult: ...

    def describe(self) -> str: ...


def _merged_env(host: HostCapabilities, env: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(host.extra_env)
    merged.update(env or {})
    return merged


@dataclass
class LocalExecutor:
    """Run in this process's machine via ``subprocess``."""

    host: HostCapabilities
    stream_output: bool = False

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        argv = [str(part) for part in command]
        full_env = dict(os.environ)
        full_env.update(_merged_env(self.host, env))
        started = time.monotonic()
        logger.info("[%s] %s", self.host.name, " ".join(argv))
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd or self.host.repo_root,
                env=full_env,
                capture_output=not self.stream_output,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                returncode=124,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr) + f"\ntimeout after {timeout:.0f}s",
                wall_time_seconds=time.monotonic() - started,
                command=tuple(argv),
                host=self.host.name,
            )
        except OSError as exc:
            return ExecResult(
                returncode=127,
                stderr=f"cannot launch command: {exc}",
                wall_time_seconds=time.monotonic() - started,
                command=tuple(argv),
                host=self.host.name,
            )
        return ExecResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            wall_time_seconds=time.monotonic() - started,
            command=tuple(argv),
            host=self.host.name,
        )

    def describe(self) -> str:
        return f"local:{self.host.name}"


@dataclass
class SshExecutor:
    """Run on a remote worker over SSH.

    The remote command is a single quoted string so that the local shell cannot
    interpret it, and SSH runs in batch mode (``BatchMode=yes``) so a missing key
    fails immediately instead of hanging on a password prompt.
    """

    host: HostCapabilities
    connect_timeout: float = 15.0
    extra_options: tuple[str, ...] = ()

    def _target(self) -> str:
        assert self.host.ssh_host is not None  # guaranteed by HostCapabilities
        if self.host.ssh_user:
            return f"{self.host.ssh_user}@{self.host.ssh_host}"
        return self.host.ssh_host

    def build_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Build the ``ssh ...`` argv (exposed for tests, no network needed)."""
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(_merged_env(self.host, env).items())
        )
        remote = " ".join(shlex.quote(str(part)) for part in command)
        if assignments:
            remote = f"env {assignments} {remote}"
        if cwd:
            remote = f"cd {shlex.quote(cwd)} && {remote}"
        argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(self.connect_timeout)}"]
        if self.host.ssh_port:
            argv += ["-p", str(self.host.ssh_port)]
        argv += list(self.extra_options)
        argv += [self._target(), remote]
        return argv

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        argv = self.build_command(command, cwd=cwd, env=env)
        started = time.monotonic()
        logger.info("[%s] ssh %s", self.host.name, " ".join(str(p) for p in command))
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, check=False, timeout=timeout or 3600.0
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                returncode=124,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr) + f"\ntimeout after {timeout or 3600.0:.0f}s",
                wall_time_seconds=time.monotonic() - started,
                command=tuple(str(p) for p in command),
                host=self.host.name,
            )
        except OSError as exc:
            return ExecResult(
                returncode=127,
                stderr=f"cannot launch ssh: {exc}",
                wall_time_seconds=time.monotonic() - started,
                command=tuple(str(p) for p in command),
                host=self.host.name,
            )
        return ExecResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            wall_time_seconds=time.monotonic() - started,
            command=tuple(str(p) for p in command),
            host=self.host.name,
        )

    def describe(self) -> str:
        return f"ssh:{self.host.name}->{self._target()}"


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def build_executor(
    host: HostCapabilities,
    *,
    stream_output: bool = False,
    extra_ssh_options: Sequence[str] = (),
) -> Executor:
    """Create the executor for a host.

    The single seam the M2 executors (``slurm``, ``k8s``) extend: add a branch
    here and the scheduler above is unchanged.
    """
    if host.executor == "local":
        return LocalExecutor(host=host, stream_output=stream_output)
    if host.executor == "ssh":
        return SshExecutor(host=host, extra_options=tuple(extra_ssh_options))
    raise ConfigError(
        f"executor '{host.executor}' is not implemented yet (have {list(KNOWN_EXECUTORS)}); "
        "add it in runtime/executor.py::build_executor"
    )


def with_data_root(host: HostCapabilities, data_root: str) -> HostCapabilities:
    """Return a copy of ``host`` pointed at a different artefact store.

    Used by the batch runner's ``--data-root`` so a scratch run cannot write into
    the shared store by accident.
    """
    return replace(host, data_root=data_root)
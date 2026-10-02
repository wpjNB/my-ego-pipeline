"""Explicit error types.

The project forbids silent failures: every failure path either raises one of
these types or returns a documented error state, and always emits a log record.
"""

from __future__ import annotations


class Ego3DActionError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(Ego3DActionError):
    """The configuration is missing, malformed or internally inconsistent."""


class StageIOError(Ego3DActionError):
    """A stage artefact is missing, malformed or violates the stage contract."""


class InsufficientDataError(Ego3DActionError):
    """Not enough (or degenerate) data to solve a numerical problem."""


class BackendNotAvailableError(Ego3DActionError):
    """An external model backend (WiLoR / HaWoR / VGGT-Omega) cannot be loaded.

    Carries a human readable hint describing what has to be installed.
    """

    def __init__(self, backend: str, hint: str) -> None:
        self.backend = backend
        self.hint = hint
        super().__init__(f"backend '{backend}' is not available: {hint}")


class BackendInvocationNotImplemented(Ego3DActionError):
    """The backend is installed, but its subprocess wrapper is not wired up yet.

    The model backends run inside their own conda environments on the GPU
    server. Until those wrappers exist this error is raised *explicitly* - the
    pipeline must never silently substitute a CPU placeholder for a real model.
    """

    def __init__(self, backend: str, hint: str) -> None:
        self.backend = backend
        self.hint = hint
        super().__init__(f"backend '{backend}' invocation is not wired up: {hint}")


class BackendExecutionError(Ego3DActionError):
    """A backend runner process failed, timed out or broke the runner protocol.

    Carries the runner name, its exit code and the tail of its output so a
    failure on the GPU server is diagnosable from the orchestrator log alone.
    """

    def __init__(self, backend: str, message: str, *, exit_code: int | None = None) -> None:
        self.backend = backend
        self.exit_code = exit_code
        suffix = f" (exit code {exit_code})" if exit_code is not None else ""
        super().__init__(f"backend runner '{backend}' failed{suffix}: {message}")

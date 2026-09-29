"""Content-addressed provenance and idempotency for pipeline stages.

Two things every distributed run needs, and neither existed before this module:

**A stable identity for a unit of work.** A unit is identified by the hash of
everything that can change its output - the parameters, the inputs' *contents*,
and the shard selection. Hashing is always over file bytes, never over mtime or
size: on a farm of machines the clocks and mtimes are not comparable, so
``--skip-existing`` would silently reuse a stale artefact. If the parameters or
an input changed, the hash changes and the unit is recomputed.

**A marker that says a unit finished, and how.** ``.done.json`` records the
params hash, the input hashes, the host, the git revision, the resolved backend
mode and the wall time. ``--skip-existing`` reuses a unit only when the marker's
params hash matches *and* every output it lists still exists - a marker whose
outputs were partly rsynced is treated as incomplete, not as done.

The marker also makes runs comparable across machines: it is exactly the record
a reviewer needs to answer "which commit, which host, which weights produced
this number?".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..errors import StageIOError
from .sharding import WindowSelection

logger = logging.getLogger(__name__)

#: Bookkeeping lives in its own directory so artefact directories stay clean and
#: ``glob("*.npz")`` can never pick a marker up.
PROVENANCE_DIR = ".provenance"
MARKER_SUFFIX = ".done.json"
_HASH_CHUNK = 1 << 20  # 1 MiB


def canonical_bytes(payload: Any) -> bytes:
    """Serialise ``payload`` deterministically for hashing.

    Sorted keys and no whitespace, so two equivalent parameter mappings hash
    identically regardless of insertion order. Paths and NumPy scalars are
    folded to their plain form so callers do not have to normalise first.
    """

    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        item = getattr(value, "item", None)  # numpy scalar
        if callable(item):
            return item()
        raise TypeError(f"cannot hash value of type {type(value).__name__}")

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=default
    ).encode("utf-8")


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's *contents*.

    Raises:
        StageIOError: when the file does not exist. A missing input is an error;
            it is never hashed as the empty string, which would make two
            different failures collide.
    """
    source = Path(path)
    if not source.is_file():
        raise StageIOError(f"cannot hash missing input: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash a set of inputs into ``{name: sha256}`` keyed by file name."""
    return {Path(path).name: hash_file(path) for path in paths}


def params_hash(
    *,
    stage: str,
    params: Mapping[str, Any],
    inputs: Mapping[str, str] | None = None,
    selection: WindowSelection | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """A stable identity for one unit of work.

    Deliberately includes the shard selection: a sharded run owns a different
    subset of the schedule, so it is a different unit even when every parameter
    is identical. Without this, ``--skip-existing`` across a re-shard would reuse
    the wrong result.
    """
    payload = {
        "stage": stage,
        "params": dict(params),
        "inputs": dict(sorted((inputs or {}).items())),
        "selection": (selection or WindowSelection()).to_dict(),
        "extra": dict(sorted((extra or {}).items())),
    }
    return hash_bytes(canonical_bytes(payload))


def git_revision(root: str | Path = ".") -> str | None:
    """Best-effort ``git rev-parse HEAD``, or ``None`` outside a repository."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def host_identity() -> str:
    """A stable ``user@host`` string for the marker's provenance."""
    try:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    except Exception:  # noqa: BLE001 - provenance must never break a run
        user = "unknown"
    return f"{user}@{socket.gethostname()}"


@dataclass(frozen=True)
class CompletionMarker:
    """The record that one unit of work finished."""

    stage: str
    unit: str
    params_hash: str
    outputs: tuple[str, ...] = ()
    inputs: Mapping[str, str] = field(default_factory=dict)
    host: str = ""
    git_revision: str | None = None
    backend_mode: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)
    wall_time_seconds: float | None = None
    platform: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "unit": self.unit,
            "params_hash": self.params_hash,
            "outputs": list(self.outputs),
            "inputs": dict(sorted(self.inputs.items())),
            "host": self.host,
            "git_revision": self.git_revision,
            "backend_mode": self.backend_mode,
            "extra": dict(sorted(self.extra.items())),
            "wall_time_seconds": self.wall_time_seconds,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompletionMarker":
        """Rebuild from JSON, raising on a marker that cannot be trusted."""
        try:
            return cls(
                stage=str(payload["stage"]),
                unit=str(payload["unit"]),
                params_hash=str(payload["params_hash"]),
                outputs=tuple(str(name) for name in payload.get("outputs", ())),
                inputs={str(k): str(v) for k, v in dict(payload.get("inputs", {})).items()},
                host=str(payload.get("host", "")),
                git_revision=payload.get("git_revision"),
                backend_mode=str(payload.get("backend_mode", "")),
                extra=dict(payload.get("extra", {})),
                wall_time_seconds=payload.get("wall_time_seconds"),
                platform=str(payload.get("platform", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StageIOError(f"malformed provenance marker: {exc}") from exc


def marker_path(target_dir: str | Path, unit: str) -> Path:
    """``<target_dir>/.provenance/<unit>.done.json``."""
    return Path(target_dir) / PROVENANCE_DIR / f"{unit}{MARKER_SUFFIX}"


def write_marker(target_dir: str | Path, marker: CompletionMarker) -> Path:
    """Atomically write a completion marker."""
    from ..io.serialization import save_json  # lazy: keeps `runtime` import-light

    path = marker_path(target_dir, marker.unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, marker.to_dict())
    logger.debug("wrote provenance marker %s", path)
    return path


def read_marker(target_dir: str | Path, unit: str) -> CompletionMarker | None:
    """Read a marker, treating an unreadable/corrupt one as "not done".

    A corrupt marker means the unit must be recomputed; it is logged and
    reported as absent rather than crashing a long batch run.
    """
    path = marker_path(target_dir, unit)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring corrupt provenance marker %s: %s", path, exc)
        return None
    try:
        return CompletionMarker.from_dict(payload)
    except StageIOError as exc:
        logger.warning("ignoring malformed provenance marker %s: %s", path, exc)
        return None


def is_complete(
    target_dir: str | Path,
    unit: str,
    *,
    expected_params_hash: str,
    expected_outputs: Sequence[str] | None = None,
) -> bool:
    """Whether ``unit`` can be skipped.

    Requires all three: a marker exists, its params hash matches this run's, and
    every output the marker lists (plus ``expected_outputs``, when given) is
    present on this machine. The last check is what makes a sharded/rsynced run
    safe - a marker that arrived without its artefact is not a finished unit.
    """
    marker = read_marker(target_dir, unit)
    if marker is None:
        return False
    if marker.params_hash != expected_params_hash:
        logger.debug(
            "unit %s has a stale params hash (%s != %s); will recompute",
            unit,
            marker.params_hash[:12],
            expected_params_hash[:12],
        )
        return False
    directory = Path(target_dir)
    required = set(marker.outputs) | set(expected_outputs or ())
    missing = sorted(name for name in required if not (directory / name).exists())
    if missing:
        logger.warning(
            "unit %s is marked complete but its output(s) %s are absent; will recompute",
            unit,
            missing,
        )
        return False
    return True


def resolve_outputs(target_dir: str | Path, outputs: Iterable[str]) -> list[Path]:
    """Map output names to absolute paths, rejecting any that escaped the dir."""
    directory = Path(target_dir).resolve()
    resolved: list[Path] = []
    for name in outputs:
        candidate = (directory / name).resolve()
        if directory not in candidate.parents and candidate != directory:
            raise StageIOError(
                f"refusing to record output '{name}' outside {directory}"
            )
        resolved.append(candidate)
    return resolved


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON atomically without importing the IO package.

    Kept here so provenance works in a minimal interpreter; the format matches
    ``io.serialization.save_json`` (indented, sorted, trailing newline).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def platform_identity() -> str:
    """``<system>-<release>-<machine>`` for cross-machine provenance."""
    return f"{platform.system()}-{platform.release()}-{platform.machine()}"
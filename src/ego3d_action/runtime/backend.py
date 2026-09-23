"""Availability checks for the external model backends.

HaWoR, WiLoR and VGGT-Omega are *not* forked or vendored - they are external
backends with their own environments (see ``environment-*.yml``). This module
answers "is that backend usable right now?" and turns a negative answer into a
typed, actionable error instead of an import traceback halfway through a run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..errors import BackendNotAvailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendSpec:
    """Where a backend lives and how to install it."""

    name: str
    env: str
    install_hint: str
    repository: str


@dataclass(frozen=True)
class BackendStatus:
    """Result of a backend availability probe."""

    spec: BackendSpec
    checkout: Path | None
    weights: Path | None
    missing: tuple[str, ...]

    @property
    def available(self) -> bool:
        return not self.missing

    def format(self) -> str:
        if self.available:
            return f"{self.spec.name}: available (checkout={self.checkout}, weights={self.weights})"
        return f"{self.spec.name}: missing {list(self.missing)}"


def probe_backend(spec: BackendSpec, *, third_party: Path, weights_root: Path) -> BackendStatus:
    """Check that a backend checkout and its weights exist on disk."""
    checkout = Path(third_party) / spec.name
    weights = Path(weights_root) / spec.name.lower()
    missing: list[str] = []
    if not checkout.is_dir():
        missing.append(f"checkout at {checkout}")
    if not weights.is_dir():
        missing.append(f"weights at {weights}")
    status = BackendStatus(
        spec=spec,
        checkout=checkout if checkout.is_dir() else None,
        weights=weights if weights.is_dir() else None,
        missing=tuple(missing),
    )
    logger.debug("%s", status.format())
    return status


def require_backend(spec: BackendSpec, *, third_party: Path, weights_root: Path) -> BackendStatus:
    """Raise :class:`BackendNotAvailableError` unless the backend is ready."""
    status = probe_backend(spec, third_party=third_party, weights_root=weights_root)
    if not status.available:
        raise BackendNotAvailableError(
            spec.name,
            "missing " + "; ".join(status.missing) + f". {spec.install_hint}",
        )
    return status

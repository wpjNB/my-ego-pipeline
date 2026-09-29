"""Loading third-party checkpoints under torch>=2.6.

torch 2.6 changed ``torch.load``'s ``weights_only`` default from ``False`` to
``True``. Every backend in this project ships a checkpoint written by an older
stack, so each one now fails with ``UnpicklingError`` listing a ``GLOBAL`` that
the strict unpickler refuses:

    WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.PoseModel

The documented remedy is to allowlist the classes the checkpoint needs. Doing
that by hand does not scale - the lists run to dozens of entries and differ per
checkpoint - so :func:`allow_trusted_checkpoint_globals` reads torch's own
diagnostic, resolves those names, and retries until the load succeeds.

Two properties are deliberate:

* ``weights_only=True`` stays on. The error message's option (1) suggests
  ``weights_only=False``, which really does unpickle arbitrary code from the
  file; this module never does that.
* A rejected load raises before constructing any object, so building the
  allowlist does not execute anything from the checkpoint. Only genuinely
  importable, already-installed classes are allowlisted.

The checkpoints this is used on are the project's own downloaded weights (see
``scripts/download_weights.sh``), and they are trusted the same way the backend
authors intended - the point here is to keep the *unpickling* safety on while
letting torch load the file at all.
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: How many diagnostic-driven rounds to attempt before giving up. Each round
#: allowlists every name the previous failure named, so this converges quickly
#: (a handful of rounds for the backends in this project) and the cap only
#: guards against a pathological message.
MAX_ROUNDS = 64

_GLOBAL_PATTERN = re.compile(r"GLOBAL ([A-Za-z_][\w\.]*)")


def strict_default_is_on() -> bool:
    """Whether the installed torch defaults ``torch.load`` to ``weights_only=True``."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    try:
        version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    except ValueError:  # a dev/local build such as "2.8.0a0+git..." - assume new
        return True
    return version >= (2, 6)


def allow_trusted_checkpoint_globals(
    checkpoint: str | Path | None, *, label: str = "checkpoint"
) -> list[str]:
    """Allowlist whatever ``torch.load`` needs to read ``checkpoint``.

    Returns the dotted names that were allowlisted (empty when the file loads as
    is, when torch predates the strict default, or when the path does not exist).
    """

    if checkpoint is None or not Path(checkpoint).is_file():
        return []
    if not strict_default_is_on():
        return []
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - callers need torch to load at all
        return []

    allowed: list[str] = []
    seen: set[str] = set()
    for _ in range(MAX_ROUNDS):
        try:
            torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            break
        except Exception as exc:  # noqa: BLE001 - the message is the input we need
            complained = _GLOBAL_PATTERN.findall(str(exc))
        added = False
        for name in complained:
            if name in seen:
                continue
            seen.add(name)
            target = _resolve_global(name)
            if target is None:
                logger.debug("%s: cannot resolve '%s'; leaving it blocked", label, name)
                continue
            torch.serialization.add_safe_globals([target])
            allowed.append(name)
            added = True
        if not added:
            # Nothing new to allowlist: either the file is now loadable (the
            # next round breaks out) or it is failing for another reason.
            break
    if allowed:
        logger.debug("%s: allowlisted %d global(s) for strict loading", label, len(allowed))
    return allowed


def _resolve_global(dotted: str) -> object | None:
    """Import ``module.Class`` without executing anything from the checkpoint.

    Walks upwards from the longest module prefix, because a dotted name may be an
    attribute of a package rather than a module of its own. A bare name (torch
    reports ``GLOBAL getattr`` for a builtin) is looked up in ``builtins``.
    """
    import builtins  # noqa: PLC0415

    if "." not in dotted:
        return getattr(builtins, dotted, None)
    candidate = dotted
    while candidate:
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            candidate = candidate.rpartition(".")[0]
            continue
        target: object = module
        for part in dotted[len(candidate) + 1 :].split("."):
            target = getattr(target, part, None)
            if target is None:
                return None
        return target
    return None


def report(checkpoint: str | Path, allowed: list[str]) -> None:
    """Note an allowlisting on stderr, so a run log shows it happened."""
    if not allowed:
        return
    print(
        f"allowed {len(allowed)} checkpoint global(s) in {Path(checkpoint).name} "
        f"for torch's strict unpickler (torch>=2.6 default): {', '.join(sorted(allowed)[:4])}"
        + (" ..." if len(allowed) > 4 else ""),
        file=sys.stderr,
    )


#: numpy removed these aliases in 1.24; packages written before that still do
#: ``from numpy import bool, int, ...`` at import time. ``unicode`` is the odd
#: one out - it was numpy's alias for ``str`` under Python 2 and has no Python 3
#: builtin, so it maps to ``str`` explicitly.
_LEGACY_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}


def restore_legacy_numpy_aliases() -> list[str]:
    """Put numpy's pre-1.24 scalar aliases back so old packages can import.

    ``chumpy`` - which HaWoR's MANO wrapper imports - starts with
    ``from numpy import bool, int, float, complex, object, unicode, str, nan, inf``.
    numpy 1.24 dropped every name but ``nan``/``inf``, so on any modern numpy the
    import raises before MANO can even be built. The aliases are the builtin
    Python types, which is exactly what numpy shipped them as, so restoring them
    is faithful rather than a workaround.

    Returns the names that were installed. Idempotent.
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:  # pragma: no cover - numpy is a hard requirement
        return []
    restored = [name for name in _LEGACY_NUMPY_ALIASES if not hasattr(np, name)]
    for name in restored:
        setattr(np, name, _LEGACY_NUMPY_ALIASES[name])
        logger.debug("restored numpy.%s (removed in numpy 1.24)", name)
    return restored
"""Work slicing for the batch runner: ``--shard`` and ``--window-range``.

The pipeline is already fan-out friendly because stages only talk through the
on-disk artefact contract (``io/artefacts.py``) and the backends only talk
through ``runtime/subprocess_backend.py``. What was missing was a way to shrink a
stage's work without changing its output. This module is that contract, and it
draws a hard line between the two cases:

**Window-parallel stages** (Phase 2 HaWoR 16/8 windows, Phase 3 VGGT-Omega
200/40 windows) are independent per window, so they may be sliced arbitrarily.
The schedule is always derived from the *global* window/overlap parameters, and
a shard is a pure partition of that schedule: the union of all shards is the
full schedule, and no window is produced twice. Ownership is interleaved
(``ordinal % count == index``) so a shard's load is balanced and - crucially -
**order independent**: a worker that reorders or drops windows still owns the
same set, which makes the partition reproducible across machines.

**Sequence-coupled stages** (Phase 1 detection recovers same-side gaps *across*
frames) must never be sliced by frame, because a frame in isolation has less
information than it does in context. They are sharded at clip granularity only,
and :class:`FrameRange` exists as the documented seam for the M2 detection
slicer: a caller widens the range by ``context`` frames, runs, and then keeps
only the inner range. Keeping that type here (validated, tested, and unused by
detection for now) means the M2 change is a wiring change, not a redesign.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..errors import ConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Shard:
    """One slice of a ``count``-way partition of a stage's work.

    ``count == 1`` (the default) means "this invocation owns everything", which
    is what every existing script already does.
    """

    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ConfigError(f"shard count must be >= 1, got {self.count}")
        if not 0 <= self.index < self.count:
            raise ConfigError(
                f"shard index must be in [0, {self.count}), got {self.index}"
            )

    @property
    def is_whole(self) -> bool:
        """``True`` when this invocation is the only one (no parallelism)."""
        return self.count == 1

    @property
    def label(self) -> str:
        """Human readable form used in logs and provenance."""
        return "whole" if self.is_whole else f"shard {self.index + 1}/{self.count}"

    def owns(self, ordinal: int) -> bool:
        """Whether this shard owns the work item at ``ordinal``.

        Interleaved rather than contiguous, so that consecutive (and therefore
        most similar) work items land on different workers and the load stays
        balanced even when the schedule has a truncated tail.
        """
        return ordinal % self.count == self.index

    def to_dict(self) -> dict[str, int]:
        return {"index": self.index, "count": self.count}

    @classmethod
    def parse(cls, raw: str | None) -> "Shard":
        """Parse ``INDEX/COUNT`` (also accepts ``INDEX:COUNT``).

        Raises:
            ConfigError: on a malformed or out-of-range value. A bad shard is a
                configuration error, never a silent fallback to the whole set.
        """
        if raw is None or str(raw).strip() == "":
            return cls()
        text = str(raw).strip()
        separator = "/" if "/" in text else (":" if ":" in text else None)
        if separator is None:
            raise ConfigError(f"--shard expects INDEX/COUNT (e.g. 0/4), got '{raw}'")
        left, right = text.split(separator, 1)
        try:
            return cls(index=int(left), count=int(right))
        except ValueError as exc:
            raise ConfigError(
                f"--shard expects INDEX/COUNT with integers (e.g. 0/4), got '{raw}'"
            ) from exc


def parse_range(raw: str | None, *, what: str) -> tuple[int, int] | None:
    """Parse an ``A-B`` half-open range into ``(a, b)``.

    Raises:
        ConfigError: on a malformed or empty range.
    """
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    separator = "-" if "-" in text[1:] else (":" if ":" in text[1:] else None)
    if separator is None:
        raise ConfigError(f"{what} expects A-B (e.g. 0-3), got '{raw}'")
    left, right = text.split(separator, 1)
    try:
        start, end = int(left), int(right)
    except ValueError as exc:
        raise ConfigError(f"{what} expects A-B with integers, got '{raw}'") from exc
    if start < 0 or end <= start:
        raise ConfigError(f"{what} must satisfy 0 <= A < B, got '{raw}'")
    return start, end


@dataclass(frozen=True)
class WindowSelection:
    """A sub-selection of a stage's global window schedule.

    Both fields are optional and compose: ``--shard 1/4 --window-range 0-8``
    selects the windows that are *both* in the first eight and owned by shard 1.
    """

    shard: Shard = Shard()
    window_start: int | None = None  # window ordinal, inclusive
    window_end: int | None = None  # window ordinal, exclusive

    def __post_init__(self) -> None:
        if (self.window_start is None) != (self.window_end is None):
            raise ConfigError("--window-range must provide both bounds")
        if self.window_start is not None and self.window_end is not None:
            if self.window_start < 0 or self.window_end <= self.window_start:
                raise ConfigError(
                    f"--window-range must satisfy 0 <= A < B, got "
                    f"{self.window_start}-{self.window_end}"
                )

    @property
    def is_whole(self) -> bool:
        """``True`` when no slicing at all is requested."""
        return self.shard.is_whole and self.window_start is None

    def owns_ordinal(self, ordinal: int) -> bool:
        """Whether the window at ``ordinal`` belongs to this selection."""
        if self.window_start is not None and self.window_end is not None:
            if not self.window_start <= ordinal < self.window_end:
                return False
        return self.shard.owns(ordinal)

    def ordinal_indices(self, total: int) -> list[int]:
        """The ordinals of ``range(total)`` this selection owns."""
        if total < 0:
            raise ConfigError(f"total window count must be >= 0, got {total}")
        return [ordinal for ordinal in range(total) if self.owns_ordinal(ordinal)]

    def select(self, ranges: Sequence[Any]) -> list[Any]:
        """Filter a global schedule down to this selection.

        Accepts anything carrying ``index``/``start``/``end`` (``WindowRange``)
        or a plain ``(start, end)`` pair. The items themselves are returned
        unchanged so callers keep their own types; ``runtime`` deliberately does
        not import ``camera.window`` to avoid a layering inversion.
        """
        selected: list[Any] = []
        for ordinal, item in enumerate(ranges):
            if hasattr(item, "start") and hasattr(item, "end"):
                index = int(getattr(item, "index", ordinal))
                start, end = int(item.start), int(item.end)
            else:
                index, (start, end) = ordinal, item
            if not self.owns_ordinal(index):
                continue
            if end <= start:
                raise ConfigError(f"window [{start}, {end}) is empty or inverted")
            selected.append(item)
        return selected

    def describe(self) -> str:
        """One-line description for logs and provenance."""
        if self.is_whole:
            return "whole"
        parts = []
        if not self.shard.is_whole:
            parts.append(self.shard.label)
        if self.window_start is not None and self.window_end is not None:
            parts.append(f"windows {self.window_start}-{self.window_end - 1}")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard.to_dict(),
            "window_start": self.window_start,
            "window_end": self.window_end,
        }

    @classmethod
    def parse(
        cls,
        *,
        shard: str | None = None,
        window_range: str | None = None,
    ) -> "WindowSelection":
        """Build from raw ``--shard`` / ``--window-range`` strings."""
        parsed = parse_range(window_range, what="--window-range")
        start, end = parsed if parsed is not None else (None, None)
        return cls(shard=Shard.parse(shard), window_start=start, window_end=end)


@dataclass(frozen=True)
class FrameRange:
    """A frame slice plus the context needed to compute it correctly.

    The M2 detection slicer: run over ``expanded()`` (so the gap-recovery rule
    can look at neighbouring frames), then keep only ``inner()``. The padding is
    never written, so a sliced run cannot fabricate a detection.
    """

    start: int
    end: int
    context: int = 0
    num_frames: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ConfigError(f"frame range must satisfy 0 <= start < end, got [{self.start}, {self.end})")
        if self.context < 0:
            raise ConfigError(f"context must be >= 0, got {self.context}")
        if self.num_frames is not None and self.end > self.num_frames:
            raise ConfigError(
                f"frame range [{self.start}, {self.end}) exceeds the clip's {self.num_frames} frames"
            )

    def inner(self) -> tuple[int, int]:
        """The frames this slice actually owns."""
        return self.start, self.end

    def expanded(self) -> tuple[int, int]:
        """``inner()`` widened by ``context`` frames, clamped to the clip."""
        start = max(0, self.start - self.context)
        end = self.end + self.context
        if self.num_frames is not None:
            end = min(end, self.num_frames)
        return start, end

    def to_dict(self) -> dict[str, int | None]:
        return {
            "start": self.start,
            "end": self.end,
            "context": self.context,
            "num_frames": self.num_frames,
        }

    @classmethod
    def parse(
        cls,
        raw: str | None,
        *,
        context: int = 0,
        num_frames: int | None = None,
    ) -> "FrameRange | None":
        parsed = parse_range(raw, what="--frame-range")
        if parsed is None:
            return None
        start, end = parsed
        return cls(start=start, end=end, context=int(context), num_frames=num_frames)


def add_window_selection_arguments(
    parser: argparse.ArgumentParser, *, window_range: bool = True
) -> None:
    """Add the shared sharding flags to a stage parser."""
    parser.add_argument(
        "--shard",
        default=None,
        metavar="INDEX/COUNT",
        help="run only this slice of the stage's windows, e.g. 1/4 (default: all)",
    )
    if window_range:
        parser.add_argument(
            "--window-range",
            default=None,
            metavar="A-B",
            help="restrict to window ordinals [A, B), e.g. 0-3 (default: all)",
        )


def add_skip_existing_argument(parser: argparse.ArgumentParser) -> None:
    """Add the idempotency flag shared by every shardable stage."""
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "reuse units whose output already exists and whose provenance marker matches "
            "(content-hashed parameters and inputs); no computation is redone"
        ),
    )


def selection_from_args(args: argparse.Namespace) -> WindowSelection:
    """Build a :class:`WindowSelection` from a parsed stage namespace."""
    return WindowSelection.parse(
        shard=getattr(args, "shard", None),
        window_range=getattr(args, "window_range", None),
    )


def describe_selection(selection: WindowSelection, chosen: int, total: int) -> str:
    """Log line shared by the sharded stages."""
    return (
        f"{selection.describe()}: {chosen}/{total} window(s)"
        if not selection.is_whole
        else f"whole schedule: {total} window(s)"
    )


def assert_partition(total: int, selections: Iterable[WindowSelection]) -> None:
    """Assert that ``selections`` partition ``range(total)`` exactly.

    Used by the batch runner (and its tests) to prove that dispatching N shards
    cannot drop or duplicate a window. Raises:
        ConfigError: if any ordinal is missing or owned twice.
    """
    seen: dict[int, int] = {ordinal: 0 for ordinal in range(total)}
    for selection in selections:
        for ordinal in selection.ordinal_indices(total):
            seen[ordinal] += 1
    missing = sorted(ordinal for ordinal, count in seen.items() if count == 0)
    duplicated = sorted(ordinal for ordinal, count in seen.items() if count > 1)
    if missing or duplicated:
        raise ConfigError(
            f"shard partition is not a partition of {total} window(s): "
            f"{len(missing)} missing {missing[:8]}, {len(duplicated)} duplicated {duplicated[:8]}"
        )
"""CLI sharding flags shared by the three real backend runners.

The mock runner implements the same flags, so ``--shard`` behaves identically in
``backends.mode: mock`` and ``backends.mode: real`` and the batch scheduler can
be tested without a GPU.

The flags are additive: with no ``--shard``/``--window-range`` the runner writes
exactly the windows it always wrote, so nothing about a single-process run
changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from ego3d_action.errors import StageIOError
from ego3d_action.runtime.provenance import is_complete, params_hash
from ego3d_action.runtime.sharding import WindowSelection, add_skip_existing_argument, add_window_selection_arguments


def add_shard_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--shard``/``--window-range``/``--skip-existing`` to a runner parser."""
    add_window_selection_arguments(parser)
    add_skip_existing_argument(parser)


def selection_from_args(args: argparse.Namespace) -> WindowSelection:
    return WindowSelection.parse(
        shard=getattr(args, "shard", None),
        window_range=getattr(args, "window_range", None),
    )


def select_windows(ranges: Sequence[Any], selection: WindowSelection) -> list[Any]:
    """Apply a selection to a schedule, raising when it drops everything.

    An empty result is an error rather than a successful no-op: a shard that
    matches no window means the shard count and the schedule disagree, and
    silently writing nothing would look like success to the scheduler.
    """
    chosen = selection.select(ranges)
    if not chosen:
        total = len(ranges)
        raise StageIOError(
            f"selection '{selection.describe()}' matched none of the {total} window(s); "
            "check --shard/--window-range against the schedule"
        )
    return chosen


def window_file_names(ranges: Sequence[Any]) -> list[str]:
    """The ``NNNNNN_NNNNNN.npz`` names a schedule writes."""
    names: list[str] = []
    for item in ranges:
        start, end = (int(item.start), int(item.end)) if hasattr(item, "start") else item
        names.append(f"{start:06d}_{end - 1:06d}.npz")
    return names


def partition_is_reusable(
    out_dir: str | Path,
    *,
    stage: str,
    ranges: Sequence[Any],
    params: dict[str, Any],
    selection: WindowSelection,
    inputs: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Whether this shard's windows are already complete and unchanged.

    The unit key must match :meth:`ego3d_action.runtime.batch.BatchRunner`, i.e.
    ``"<stage>/<selection>"``, so the runner's ``--skip-existing`` and the
    scheduler's agree on what "done" means.
    """
    unit = f"{stage}/{selection.describe()}"
    digest = params_hash(
        stage=stage, params=params, inputs=inputs or {}, selection=selection, extra=extra or {}
    )
    directory = Path(out_dir)
    expected = [name for name in window_file_names(ranges) if not (directory / name).is_file()]
    if expected:
        # Missing windows mean the shard is incomplete; do not consult the marker.
        return False
    return is_complete(directory, unit, expected_params_hash=digest, expected_outputs=[])


def record_partition(
    out_dir: str | Path,
    *,
    stage: str,
    ranges: Sequence[Any],
    params: dict[str, Any],
    selection: WindowSelection,
    inputs: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the shard's completion marker (same key as the scheduler's)."""
    from ego3d_action.runtime.provenance import (
        CompletionMarker,
        git_revision,
        host_identity,
        platform_identity,
        write_marker,
    )

    unit = f"{stage}/{selection.describe()}"
    digest = params_hash(
        stage=stage, params=params, inputs=inputs or {}, selection=selection, extra=extra or {}
    )
    return write_marker(
        out_dir,
        CompletionMarker(
            stage=stage,
            unit=unit,
            params_hash=digest,
            outputs=tuple(window_file_names(ranges)),
            inputs=inputs or {},
            host=host_identity(),
            git_revision=git_revision(),
            backend_mode="real",
            extra={"selection": selection.to_dict(), **dict(extra or {})},
            platform=platform_identity(),
        ),
    )
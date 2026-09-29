"""Provenance: a unit is identified by content, and a stale marker never skips.

The load-bearing property is that ``--skip-existing`` must be safe on a farm of
machines: no mtime, no size, no host-specific state may influence whether a unit
is considered done. Each test here pins one of those rules.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.runtime.provenance import (
    CompletionMarker,
    canonical_bytes,
    hash_file,
    hash_files,
    is_complete,
    marker_path,
    params_hash,
    read_marker,
    resolve_outputs,
    write_marker,
)
from ego3d_action.runtime.sharding import WindowSelection


def make_marker(**overrides: object) -> CompletionMarker:
    base = {
        "stage": "camera",
        "unit": "camera/shard 1/2",
        "params_hash": "abc123",
        "outputs": ("000160_000359.npz",),
    }
    base.update(overrides)
    return CompletionMarker(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Hashing is deterministic and content based
# --------------------------------------------------------------------------


def test_canonical_bytes_is_key_order_independent() -> None:
    assert canonical_bytes({"a": 1, "b": 2}) == canonical_bytes({"b": 2, "a": 1})
    assert canonical_bytes({"a": 1}) != canonical_bytes({"a": 2})


def test_params_hash_is_stable_and_sensitive() -> None:
    first = params_hash(stage="hand", params={"window": 16, "overlap": 8})
    second = params_hash(stage="hand", params={"overlap": 8, "window": 16})
    assert first == second
    changed = params_hash(stage="hand", params={"window": 16, "overlap": 4})
    assert changed != first
    other_stage = params_hash(stage="camera", params={"window": 16, "overlap": 8})
    assert other_stage != first


def test_params_hash_includes_the_shard_selection() -> None:
    """A re-shard must be a different unit, or skip-existing would reuse wrongly."""
    whole = params_hash(stage="camera", params={"window": 200}, selection=WindowSelection())
    sharded = params_hash(
        stage="camera",
        params={"window": 200},
        selection=WindowSelection.parse(shard="0/2"),
    )
    assert whole != sharded


def test_params_hash_includes_input_contents(tmp_path: Path) -> None:
    source = tmp_path / "detection.npz"
    source.write_bytes(b"first")
    first = params_hash(stage="hand", params={}, inputs=hash_files([source]))
    source.write_bytes(b"second")
    second = params_hash(stage="hand", params={}, inputs=hash_files([source]))
    assert first != second


def test_hash_file_uses_contents_not_mtime(tmp_path: Path) -> None:
    source = tmp_path / "window.npz"
    source.write_bytes(b"payload")
    before = hash_file(source)
    os.utime(source, (0, 0))  # same bytes, wildly different mtime
    assert hash_file(source) == before


def test_hash_file_refuses_a_missing_input(tmp_path: Path) -> None:
    with pytest.raises(StageIOError):
        hash_file(tmp_path / "absent.npz")


# --------------------------------------------------------------------------
# Markers round-trip and are verified
# --------------------------------------------------------------------------


def test_marker_round_trips_through_disk(tmp_path: Path) -> None:
    marker = make_marker(host="worker-a", git_revision="deadbeef", extra={"mode": "real"})
    write_marker(tmp_path, marker)
    loaded = read_marker(tmp_path, marker.unit)
    assert loaded is not None
    assert loaded == marker
    assert loaded.outputs == ("000160_000359.npz",)


def test_marker_lives_outside_the_artefact_glob(tmp_path: Path) -> None:
    write_marker(tmp_path, make_marker())
    assert not list(tmp_path.glob("*.npz"))
    assert marker_path(tmp_path, "camera/shard 1/2").is_file()


def test_missing_marker_is_not_done(tmp_path: Path) -> None:
    assert read_marker(tmp_path, "camera/whole") is None
    assert not is_complete(tmp_path, "camera/whole", expected_params_hash="abc123")


def test_corrupt_marker_is_treated_as_absent(tmp_path: Path) -> None:
    path = marker_path(tmp_path, "camera/whole")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert read_marker(tmp_path, "camera/whole") is None
    assert not is_complete(tmp_path, "camera/whole", expected_params_hash="abc123")


def test_complete_requires_matching_hash(tmp_path: Path) -> None:
    (tmp_path / "000160_000359.npz").write_bytes(b"window")
    write_marker(tmp_path, make_marker(params_hash="abc123"))
    assert is_complete(tmp_path, "camera/shard 1/2", expected_params_hash="abc123")
    assert not is_complete(tmp_path, "camera/shard 1/2", expected_params_hash="different")


def test_complete_requires_its_outputs_to_exist(tmp_path: Path) -> None:
    """A marker that arrived without its artefact (partial rsync) is not done."""
    write_marker(tmp_path, make_marker(params_hash="abc123"))
    assert not is_complete(tmp_path, "camera/shard 1/2", expected_params_hash="abc123")
    (tmp_path / "000160_000359.npz").write_bytes(b"window")
    assert is_complete(tmp_path, "camera/shard 1/2", expected_params_hash="abc123")


def test_expected_outputs_are_also_checked(tmp_path: Path) -> None:
    write_marker(tmp_path, make_marker(params_hash="abc123", outputs=()))
    assert not is_complete(
        tmp_path,
        "camera/shard 1/2",
        expected_params_hash="abc123",
        expected_outputs=["000000_000199.npz"],
    )
    (tmp_path / "000000_000199.npz").write_bytes(b"window")
    assert is_complete(
        tmp_path,
        "camera/shard 1/2",
        expected_params_hash="abc123",
        expected_outputs=["000000_000199.npz"],
    )


def test_marker_is_written_atomically(tmp_path: Path) -> None:
    """No ``.tmp`` leftovers, and the JSON is complete."""
    write_marker(tmp_path, make_marker())
    leftovers = list(tmp_path.glob("**/.*.tmp"))
    assert leftovers == []
    payload = json.loads(marker_path(tmp_path, "camera/shard 1/2").read_text(encoding="utf-8"))
    assert payload["stage"] == "camera"


def test_resolve_outputs_rejects_escapes(tmp_path: Path) -> None:
    assert resolve_outputs(tmp_path, ["a.npz"]) == [(tmp_path / "a.npz").resolve()]
    with pytest.raises(StageIOError):
        resolve_outputs(tmp_path, ["../outside.npz"])


def test_a_second_run_over_the_same_content_stays_complete(tmp_path: Path) -> None:
    """Idempotency: recomputing the same identity must not invalidate the marker."""
    (tmp_path / "000160_000359.npz").write_bytes(b"window")
    selection = WindowSelection.parse(shard="1/2")
    params = {"window": 200, "overlap": 40}
    marker = make_marker(
        params_hash=params_hash(stage="camera", params=params, selection=selection)
    )
    write_marker(tmp_path, marker)
    time.sleep(0.01)
    again = params_hash(stage="camera", params=params, selection=selection)
    assert is_complete(tmp_path, marker.unit, expected_params_hash=again)
"""Sharding contract: the partition must lose nothing and duplicate nothing.

These tests are the proof behind the batch runner's parallelism claim: whatever
the shard count, the union of the shards' windows is exactly the unsliced
schedule, and no window is written twice.
"""

from __future__ import annotations

import pytest

from ego3d_action.camera.window import make_windows
from ego3d_action.errors import ConfigError
from ego3d_action.hand.hawor import HaworClipRequest
from ego3d_action.runtime.sharding import (
    FrameRange,
    Shard,
    WindowSelection,
    assert_partition,
    parse_range,
)


# --------------------------------------------------------------------------
# Shard parsing
# --------------------------------------------------------------------------


def test_default_shard_is_the_whole_set() -> None:
    shard = Shard.parse(None)
    assert shard.is_whole
    assert shard.owns(0) and shard.owns(7)


def test_shard_parses_index_and_count() -> None:
    shard = Shard.parse("1/4")
    assert (shard.index, shard.count) == (1, 4)
    assert shard.label == "shard 2/4"


def test_shard_accepts_colon_separator() -> None:
    assert Shard.parse("2:5") == Shard(index=2, count=5)


@pytest.mark.parametrize("raw", ["4", "a/4", "1/x", "1/0", "4/4", "-1/2"])
def test_malformed_shard_is_rejected(raw: str) -> None:
    with pytest.raises(ConfigError):
        Shard.parse(raw)


def test_short_range_is_parsed_or_rejected() -> None:
    assert parse_range("0-3", what="--window-range") == (0, 3)
    assert parse_range(None, what="--window-range") is None
    for bad in ("3", "3-3", "5-2", "a-b"):
        with pytest.raises(ConfigError):
            parse_range(bad, what="--window-range")


# --------------------------------------------------------------------------
# Partitioning a camera schedule (Phase 3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_frames", [1, 16, 199, 200, 360, 600, 1001])
def test_camera_shards_partition_the_schedule(num_frames: int) -> None:
    ranges = make_windows(num_frames, window=200, overlap=40)
    total = len(ranges)
    for count in range(1, total + 1):
        selections = [WindowSelection.parse(shard=f"{i}/{count}") for i in range(count)]
        # Union is the full schedule, with no duplicates.
        assert_partition(total, selections)
        owned = [rng.index for selection in selections for rng in selection.select(ranges)]
        assert sorted(owned) == [rng.index for rng in ranges]


def test_camera_shard_selects_only_its_windows() -> None:
    ranges = make_windows(600, window=200, overlap=40)
    assert [rng.start for rng in ranges] == [0, 160, 320, 480]
    # Ownership is interleaved (ordinal % count), so shard 1/2 owns ordinals 1 and 3.
    sel = WindowSelection.parse(shard="1/2")
    assert [(r.start, r.end) for r in sel.select(ranges)] == [(160, 360), (480, 600)]


# --------------------------------------------------------------------------
# Partitioning a HaWoR schedule (Phase 2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_frames", [16, 17, 24, 100, 240, 243])
def test_hand_shards_partition_the_schedule(num_frames: int) -> None:
    request = HaworClipRequest(num_frames=num_frames, frames_dir=".", window=16, overlap=8)
    spans = request.ranges()
    total = len(spans)
    for count in range(1, min(total, 8) + 1):
        selections = [WindowSelection.parse(shard=f"{i}/{count}") for i in range(count)]
        assert_partition(total, selections)


def test_hand_shard_and_window_range_compose() -> None:
    request = HaworClipRequest(num_frames=240, frames_dir=".", window=16, overlap=8)
    spans = request.ranges()
    sel = WindowSelection.parse(shard="0/2", window_range="0-4")
    owned = sel.select(spans)
    # Only even ordinals below 4 survive: 0 and 2.
    assert [span for span in spans if sel.owns_ordinal(spans.index(span))] == owned
    assert [ordinal for ordinal in range(len(spans)) if sel.owns_ordinal(ordinal)] == [0, 2]


def test_window_range_only_accepts_both_bounds() -> None:
    with pytest.raises(ConfigError):
        WindowSelection(shard=Shard(), window_start=1, window_end=None)
    with pytest.raises(ConfigError):
        WindowSelection(shard=Shard(), window_start=4, window_end=2)


def test_assert_partition_detects_a_dropped_window() -> None:
    # Two shards of four would cover 0..7; declaring only the first shard twice
    # must be caught rather than reported as a valid partition.
    with pytest.raises(ConfigError):
        assert_partition(8, [WindowSelection.parse(shard="0/2")])


def test_assert_partition_detects_a_duplicate() -> None:
    with pytest.raises(ConfigError):
        assert_partition(4, [WindowSelection.parse(shard="0/2"), WindowSelection.parse(shard="0/2")])


# --------------------------------------------------------------------------
# Frame coupled stages (the M2 seam)
# --------------------------------------------------------------------------


def test_frame_range_pads_for_context_and_clamps() -> None:
    inner = FrameRange(start=40, end=60, context=4, num_frames=100)
    assert inner.inner() == (40, 60)
    assert inner.expanded() == (36, 64)

    at_start = FrameRange(start=0, end=10, context=4, num_frames=100)
    assert at_start.expanded() == (0, 14)

    at_end = FrameRange(start=90, end=100, context=4, num_frames=100)
    assert at_end.expanded() == (86, 100)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": -1, "end": 10},
        {"start": 10, "end": 10},
        {"start": 20, "end": 10},
        {"start": 0, "end": 10, "context": -1},
        {"start": 0, "end": 10, "num_frames": 5},
    ],
)
def test_frame_range_rejects_invalid_bounds(kwargs: dict[str, int]) -> None:
    with pytest.raises(ConfigError):
        FrameRange(**kwargs)


def test_frame_range_parse_requires_a_range() -> None:
    assert FrameRange.parse(None) is None
    parsed = FrameRange.parse("5-15", context=2, num_frames=50)
    assert parsed is not None
    assert parsed.expanded() == (3, 17)
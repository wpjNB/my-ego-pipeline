"""Phase 3: VGGT-Omega window schedule."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.camera.window import make_windows, window_coverage, window_ranges
from ego3d_action.errors import StageIOError


def test_sixty_frame_schedule_matches_spec() -> None:
    ranges = window_ranges(600)
    assert ranges == [(0, 200), (160, 360), (320, 520), (480, 600)]


def test_full_coverage_of_every_frame() -> None:
    counts = window_coverage(600, make_windows(600))
    assert np.all(counts >= 1)
    assert counts[0] == 1 and counts[-1] == 1
    # Windows are half-open: frame 199 is the last shared frame, frame 200 only
    # belongs to the second window.
    assert counts[199] == 2
    assert counts[200] == 1


def test_overlap_is_forty_frames() -> None:
    windows = make_windows(600)
    for prev, cur in zip(windows, windows[1:], strict=False):
        assert min(prev.end, cur.end) - max(prev.start, cur.start) == 40


def test_short_clip_produces_single_window() -> None:
    windows = make_windows(50)
    assert len(windows) == 1
    assert windows[0].start == 0 and windows[0].end == 50


def test_exact_multiple_of_stride() -> None:
    assert window_ranges(320) == [(0, 200), (160, 320)]


def test_custom_window_and_overlap() -> None:
    assert window_ranges(20, window=16, overlap=8) == [(0, 16), (8, 20)]


def test_invalid_parameters_raise() -> None:
    with pytest.raises(StageIOError):
        make_windows(0)
    with pytest.raises(StageIOError):
        make_windows(100, window=0)
    with pytest.raises(StageIOError):
        make_windows(100, overlap=-1)
    with pytest.raises(StageIOError):
        make_windows(100, window=40, overlap=40)


def test_window_range_helpers() -> None:
    window = make_windows(10, window=4, overlap=2)[0]
    assert window.contains(0) and window.contains(3)
    assert not window.contains(4)
    assert window.frames().tolist() == [0, 1, 2, 3]
    assert window.num_frames == 4

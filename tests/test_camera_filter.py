"""Phase 6.1: 3-frame binomial camera filter."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.refinement.camera_filter import binomial_3_filter, filter_camera_translation


def test_constant_sequence_is_unchanged() -> None:
    signal = np.full((10, 3), 2.5)
    assert np.allclose(binomial_3_filter(signal), signal)


def test_kernel_weights_and_edge_replication() -> None:
    signal = np.zeros((5, 1))
    signal[2] = 1.0
    filtered = binomial_3_filter(signal)
    assert np.allclose(filtered[:, 0], [0.0, 0.25, 0.5, 0.25, 0.0])

    signal = np.arange(3, dtype=np.float64).reshape(-1, 1)
    filtered = binomial_3_filter(signal)
    # Edge replication: t'[0] = 0.25*t0 + 0.5*t0 + 0.25*t1 = 0.25
    assert filtered[0, 0] == pytest.approx(0.25)
    assert filtered[1, 0] == pytest.approx(1.0)
    assert filtered[2, 0] == pytest.approx(1.75)


def test_multiple_passes_widen_the_kernel() -> None:
    signal = np.zeros((7, 1))
    signal[3] = 1.0
    filtered = binomial_3_filter(signal, passes=2)
    assert np.allclose(filtered[:, 0], [0.0, 0.0625, 0.25, 0.375, 0.25, 0.0625, 0.0])


def test_linear_ramp_is_preserved_in_the_interior() -> None:
    ramp = np.arange(20, dtype=np.float64).reshape(-1, 1)
    filtered = binomial_3_filter(ramp)
    assert np.allclose(filtered[1:-1], ramp[1:-1])


def test_invalid_frames_are_not_modified() -> None:
    clean = np.arange(6, dtype=np.float64).reshape(-1, 1)
    valid = np.array([True, True, False, True, True, True])
    spiked = clean.copy()
    spiked[2] = 100.0

    filtered = filter_camera_translation(spiked, valid=valid)
    reference = filter_camera_translation(clean)

    # The invalid frame is preserved verbatim ...
    assert filtered[2, 0] == pytest.approx(100.0)
    # ... and its outlier value does not leak into the valid neighbours.
    assert np.allclose(filtered[valid], reference[valid])


def test_filter_without_mask_matches_plain_filter() -> None:
    signal = np.random.default_rng(0).normal(size=(12, 3))
    assert np.allclose(filter_camera_translation(signal), binomial_3_filter(signal))


def test_all_invalid_returns_input_copy() -> None:
    signal = np.arange(4, dtype=np.float64).reshape(-1, 1)
    out = filter_camera_translation(signal, valid=np.zeros(4, dtype=bool))
    assert np.allclose(out, signal)
    assert out is not signal


def test_invalid_arguments_raise() -> None:
    with pytest.raises(StageIOError):
        binomial_3_filter(np.zeros((0, 3)))
    with pytest.raises(StageIOError):
        binomial_3_filter(np.zeros((4, 3)), passes=0)
    with pytest.raises(StageIOError):
        filter_camera_translation(np.zeros((4, 3)), valid=np.zeros(3, dtype=bool))

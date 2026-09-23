"""Sim(3): exact recovery, robustness to outliers, composition and errors."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.errors import InsufficientDataError
from ego3d_action.geometry.sim3 import Sim3, estimate_sim3, estimate_sim3_robust
from ego3d_action.geometry.umeyama import weighted_umeyama


def test_estimate_sim3_recovers_known_transform(known_sim3: Sim3) -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(64, 3))
    dst = known_sim3.transform(src)

    estimate = estimate_sim3(src, dst)
    assert np.isclose(estimate.scale, known_sim3.scale, atol=1e-9)
    assert np.allclose(estimate.rotation, known_sim3.rotation, atol=1e-9)
    assert np.allclose(estimate.translation, known_sim3.translation, atol=1e-9)


def test_weighted_umeyama_ignores_zero_weight_points(known_sim3: Sim3) -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(32, 3))
    dst = known_sim3.transform(src)
    corrupted = dst.copy()
    corrupted[:8] += 100.0
    weights = np.ones(32)
    weights[:8] = 0.0

    scale, rotation, translation = weighted_umeyama(src, corrupted, weights)
    assert np.isclose(scale, known_sim3.scale, atol=1e-8)
    assert np.allclose(rotation, known_sim3.rotation, atol=1e-8)
    assert np.allclose(translation, known_sim3.translation, atol=1e-8)


def test_estimate_sim3_without_scale(known_sim3: Sim3) -> None:
    rng = np.random.default_rng(2)
    src = rng.normal(size=(40, 3))
    dst = src @ known_sim3.rotation.T + known_sim3.translation
    estimate = estimate_sim3(src, dst, with_scale=False)
    assert np.isclose(estimate.scale, 1.0)
    assert np.allclose(estimate.rotation, known_sim3.rotation, atol=1e-9)


def test_estimate_sim3_robust_rejects_outliers(known_sim3: Sim3) -> None:
    rng = np.random.default_rng(3)
    src = rng.normal(size=(200, 3))
    dst = known_sim3.transform(src)
    outlier_idx = rng.choice(200, size=60, replace=False)
    dst[outlier_idx] += rng.normal(0.0, 1.0, size=(60, 3))

    result = estimate_sim3_robust(src, dst, inlier_threshold=1e-3, random_state=0)
    assert result.inlier_ratio > 0.6
    assert np.isclose(result.sim3.scale, known_sim3.scale, rtol=1e-4)
    assert np.allclose(result.sim3.rotation, known_sim3.rotation, atol=1e-4)
    assert np.allclose(result.sim3.translation, known_sim3.translation, atol=1e-3)
    assert not result.inlier_mask[outlier_idx].any()


def test_estimate_sim3_robust_auto_threshold(known_sim3: Sim3) -> None:
    rng = np.random.default_rng(4)
    src = rng.normal(size=(150, 3))
    dst = known_sim3.transform(src) + rng.normal(0.0, 1e-4, size=(150, 3))
    result = estimate_sim3_robust(src, dst, random_state=0)
    assert result.threshold > 0.0
    assert result.inlier_rmse < 5e-4


def test_sim3_inverse_roundtrip(known_sim3: Sim3) -> None:
    points = np.random.default_rng(5).normal(size=(20, 3))
    forward = known_sim3.transform(points)
    back = known_sim3.inverse().transform(forward)
    assert np.allclose(back, points, atol=1e-9)


def test_sim3_compose_matches_sequential_application(known_sim3: Sim3) -> None:
    other = Sim3(
        scale=0.8,
        rotation=np.linalg.qr(np.random.default_rng(6).normal(size=(3, 3)))[0],
        translation=np.array([-0.5, 0.2, 0.05]),
    )
    if np.linalg.det(other.rotation) < 0:
        other = Sim3(scale=other.scale, rotation=-other.rotation, translation=other.translation)
    points = np.random.default_rng(7).normal(size=(10, 3))
    composed = (known_sim3 @ other).transform(points)
    sequential = known_sim3.transform(other.transform(points))
    assert np.allclose(composed, sequential, atol=1e-9)


def test_sim3_matrix_roundtrip(known_sim3: Sim3) -> None:
    recovered = Sim3.from_matrix(known_sim3.matrix)
    assert np.isclose(recovered.scale, known_sim3.scale)
    assert np.allclose(recovered.rotation, known_sim3.rotation)
    assert np.allclose(recovered.translation, known_sim3.translation)


def test_sim3_transform_poses_matches_point_transform(known_sim3: Sim3) -> None:
    rotation = np.eye(3)
    translation = np.array([[0.1, 0.2, 0.3], [-0.4, 0.0, 0.7]])
    new_rot, new_tr = known_sim3.transform_poses(rotation, translation)
    points = translation + np.array([0.0, 0.0, 0.0])
    expected = known_sim3.transform(points)
    assert np.allclose(new_tr, expected)
    assert np.allclose(new_rot, known_sim3.rotation)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 0.0, "rotation": np.eye(3), "translation": np.zeros(3)},
        {"scale": -1.0, "rotation": np.eye(3), "translation": np.zeros(3)},
        {"scale": 1.0, "rotation": np.eye(2), "translation": np.zeros(3)},
        {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(2)},
    ],
)
def test_sim3_rejects_invalid_fields(kwargs: dict[str, object]) -> None:
    with pytest.raises(InsufficientDataError):
        Sim3(**kwargs)  # type: ignore[arg-type]


def test_estimate_sim3_requires_enough_points() -> None:
    with pytest.raises(InsufficientDataError):
        estimate_sim3(np.zeros((2, 3)), np.zeros((2, 3)))


def test_estimate_sim3_rejects_degenerate_cloud() -> None:
    src = np.zeros((5, 3))
    with pytest.raises(InsufficientDataError):
        estimate_sim3(src, src)


def test_estimate_sim3_robust_rejects_insufficient_inliers(known_sim3: Sim3) -> None:
    rng = np.random.default_rng(8)
    src = rng.normal(size=(40, 3))
    dst = rng.normal(size=(40, 3)) * 5.0
    with pytest.raises(InsufficientDataError):
        estimate_sim3_robust(src, dst, inlier_threshold=1e-6, min_inliers=30, ransac_iterations=16)

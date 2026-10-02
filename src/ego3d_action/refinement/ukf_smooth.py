"""Phase 6: constant-velocity UKF + unscented RTS temporal smoothing (P3).

Port of the reference pipeline's ``smooth_ukf_cam``
(``ego_pipeline/data_cleaning/cleaning_modules/ukf_cam_smoothing.py``),
reduced to the quantities this project's trajectory contract carries: 21
joint positions per hand. The filter core is kept identical - per-channel
constant-velocity state ``[value, velocity]``, a per-channel observation
scale from the MAD of second differences, speed-adaptive observation noise,
and an unscented RTS backward pass. Defaults follow the reference:
``q = 0.6``, ``r = 0.6``, ``beta = 2.0``.

Smoothing runs per hand over the *valid* frames only, with the frame index
as the time axis: a gap between two valid frames is simply a larger ``dt``,
so the constant-velocity model degrades gracefully across holes while the
missing frames themselves are never written (they stay exactly as the gap
fill left them). A hand with fewer than :data:`MIN_VALID` valid frames is
left untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray

#: Smallest valid-frame count a hand needs before the filter runs.
MIN_VALID = 4
_ALPHA = 1.0
_BETA_UT = 2.0
_JITTER = 1e-9
DEFAULT_Q = 0.6
DEFAULT_R = 0.6
DEFAULT_BETA = 2.0
#: Reference parameter limits (``UKF_PARAM_LIMITS`` in the reference).
PARAM_LIMITS = {"q": (0.1, 2.0), "r": (0.1, 2.0), "beta": (0.0, 5.0)}


@dataclass(frozen=True)
class UkfSmoothResult:
    """Smoothed hand trajectory plus a record of what was touched."""

    joints_camera: Array  # [T, 2, 21, 3]
    frames_smoothed: Array  # [T, 2] bool
    hands_smoothed: int


def smooth_hand_joints(
    joints_camera: Array,
    valid: Array,
    *,
    q: float = DEFAULT_Q,
    r: float = DEFAULT_R,
    beta: float = DEFAULT_BETA,
    rts: bool = True,
) -> UkfSmoothResult:
    """Smooth ``[T, 2, 21, 3]`` camera-space joints with a UKF + RTS pass.

    Args:
        joints_camera: camera-space joints (NaN = missing); never modified.
        valid: ``[T, 2]`` frames the filter may read *and* rewrite; frames
            with non-finite joints are excluded even when ``valid`` is set.
        q: process-noise scale (reference default ``0.6``).
        r: observation-noise scale (reference default ``0.6``).
        beta: speed-adaptive observation weight (reference default ``2.0``).
        rts: run the unscented RTS backward pass (reference ``rts >= 0.5``).

    Returns:
        :class:`UkfSmoothResult`; invalid frames keep their input values.

    Raises:
        StageIOError: on shape mismatches or parameters outside
            :data:`PARAM_LIMITS`.
    """
    q, r, beta = float(q), float(r), float(beta)
    for name, value in (("q", q), ("r", r), ("beta", beta)):
        low, high = PARAM_LIMITS[name]
        if not np.isfinite(value) or not low <= value <= high:
            raise StageIOError(f"UKF {name} must be in [{low:g}, {high:g}], got {value}")

    joints = np.asarray(joints_camera, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise StageIOError(f"joints_camera must be [T, 2, 21, 3], got {joints.shape}")
    if mask.shape != joints.shape[:2]:
        raise StageIOError(f"valid must be {joints.shape[:2]}, got {mask.shape}")

    out = joints.copy()
    frames_smoothed = np.zeros(joints.shape[:2], dtype=bool)
    hands_smoothed = 0

    for hand in range(2):
        indices = np.flatnonzero(mask[:, hand] & np.isfinite(joints[:, hand]).all(axis=(1, 2)))
        if indices.size < MIN_VALID:
            logger.info("ukf smooth: hand %d has %d valid frames (< %d), skipped",
                        hand, int(indices.size), MIN_VALID)
            continue
        channels = joints[indices, hand].reshape(indices.size, 21 * 3).T  # [63, N]
        smoothed = _smooth_channels(
            indices.astype(np.float64), channels, q=q, r=r, beta=beta, backward=rts
        )
        out[indices, hand] = smoothed.T.reshape(indices.size, 21, 3)
        frames_smoothed[indices, hand] = True
        hands_smoothed += 1

    logger.info(
        "ukf smooth (q=%.2f, r=%.2f, beta=%.2f, rts=%s): %d hand(s), %d frames rewritten",
        q, r, beta, rts, hands_smoothed, int(frames_smoothed.sum()),
    )
    return UkfSmoothResult(
        joints_camera=out,
        frames_smoothed=frames_smoothed,
        hands_smoothed=hands_smoothed,
    )


def _smooth_channels(
    timestamps: Array,
    values: Array,
    *,
    q: float,
    r: float,
    beta: float,
    backward: bool,
) -> Array:
    """One ``_smooth_sr`` call of the reference: robust sigma, speed weight, filter."""
    sigma = _sigma_meas_batch(values)
    weight = _speed_weight(timestamps, values, beta)
    observation_noise = (r * sigma)[:, None] ** 2 * weight[None, :]
    return _ukf_rts_core(
        timestamps, values, observation_noise, (q * sigma) ** 2, backward=backward
    )


def _sigma_meas_batch(values: Array) -> Array:
    """Per-channel observation scale: MAD of second differences over sqrt(6)."""
    second = values[:, :-2] - 2.0 * values[:, 1:-1] + values[:, 2:]
    median = np.median(second, axis=1, keepdims=True)
    mad = np.median(np.abs(second - median), axis=1)
    return np.maximum(1.4826 * mad / np.sqrt(6.0), 1e-6)


def _sigma_points_batch(mean: Array, covariance: Array, lam: float) -> Array:
    """Batched sigma points ``[C, 2n+1, n]`` for per-channel 2-D states."""
    channels, dims = mean.shape
    matrix = (dims + lam) * covariance
    matrix = 0.5 * (matrix + matrix.transpose(0, 2, 1)) + _JITTER * np.eye(dims)
    chol = np.linalg.cholesky(matrix)
    points = np.empty((channels, 2 * dims + 1, dims))
    points[:, 0] = mean
    for index in range(dims):
        column = chol[:, :, index]
        points[:, 1 + index] = mean + column
        points[:, 1 + dims + index] = mean - column
    return points


def _ukf_rts_core(
    timestamps: Array,
    values: Array,
    observation_noise: Array,
    process_noise: Array,
    *,
    backward: bool = True,
) -> Array:
    """Batched constant-velocity UKF with optional unscented RTS smoothing.

    ``values`` is ``[C, N]``, ``observation_noise`` / ``process_noise`` are
    ``[C]`` (per channel) and ``[C, N]`` (per channel and frame) respectively;
    the return value has the shape of ``values``.
    """
    channels, frames = values.shape
    if frames < MIN_VALID:
        return values.copy()

    dims = 2
    lam = _ALPHA**2 * (dims + (3 - dims)) - dims
    weights_mean = np.full(2 * dims + 1, 1.0 / (2.0 * (dims + lam)))
    weights_cov = weights_mean.copy()
    weights_mean[0] = lam / (dims + lam)
    weights_cov[0] = lam / (dims + lam) + 1.0 - _ALPHA**2 + _BETA_UT
    identity = np.eye(dims)

    filtered_mean = np.zeros((frames, channels, dims))
    predicted_mean = np.zeros((frames, channels, dims))
    predicted_cov = np.zeros((frames, channels, dims, dims))
    cross_cov = np.zeros((frames, channels, dims, dims))

    current_mean = np.zeros((channels, dims))
    current_mean[:, 0] = values[:, 0]
    initial_velocity_var = np.maximum(np.var(np.diff(values, axis=1), axis=1), 1e-6)
    current_cov = np.zeros((channels, dims, dims))
    current_cov[:, 0, 0] = np.maximum(observation_noise[:, 0], 1e-12)
    current_cov[:, 1, 1] = initial_velocity_var
    filtered_mean[0] = current_mean

    for frame in range(1, frames):
        dt = float(timestamps[frame] - timestamps[frame - 1]) or 1.0
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        base_process = np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
        process_cov = process_noise[:, None, None] * base_process

        sigma = _sigma_points_batch(current_mean, current_cov, lam)
        propagated = np.einsum("ij,ckj->cki", transition, sigma)
        mean_pred = np.einsum("k,ckj->cj", weights_mean, propagated)
        delta_pred = propagated - mean_pred[:, None, :]
        cov_pred = np.einsum("k,cki,ckj->cij", weights_cov, delta_pred, delta_pred) + process_cov
        cov_pred = 0.5 * (cov_pred + cov_pred.transpose(0, 2, 1)) + _JITTER * identity
        delta_sigma = sigma - current_mean[:, None, :]
        cross_cov[frame] = np.einsum("k,cki,ckj->cij", weights_cov, delta_sigma, delta_pred)
        predicted_mean[frame] = mean_pred
        predicted_cov[frame] = cov_pred

        update_sigma = _sigma_points_batch(mean_pred, cov_pred, lam)
        observations = update_sigma[:, :, 0]
        observation_mean = np.einsum("k,ck->c", weights_mean, observations)
        delta_observation = observations - observation_mean[:, None]
        innovation = (
            np.einsum("k,ck->c", weights_cov, delta_observation * delta_observation)
            + observation_noise[:, frame]
        )
        state_observation = np.einsum(
            "k,cki,ck->ci", weights_cov, update_sigma - mean_pred[:, None, :], delta_observation
        )
        gain = state_observation / innovation[:, None]
        current_mean = mean_pred + gain * (values[:, frame] - observation_mean)[:, None]
        current_cov = cov_pred - innovation[:, None, None] * np.einsum("ci,cj->cij", gain, gain)
        current_cov = 0.5 * (current_cov + current_cov.transpose(0, 2, 1)) + _JITTER * identity
        filtered_mean[frame] = current_mean

    if not backward:
        return filtered_mean[:, :, 0].T

    smoothed = filtered_mean.copy()
    for frame in range(frames - 2, -1, -1):
        gain = np.linalg.solve(
            predicted_cov[frame + 1].transpose(0, 2, 1),
            cross_cov[frame + 1].transpose(0, 2, 1),
        ).transpose(0, 2, 1)
        smoothed[frame] = filtered_mean[frame] + np.einsum(
            "cij,cj->ci", gain, smoothed[frame + 1] - predicted_mean[frame + 1]
        )
    return smoothed[:, :, 0].T


def _speed_weight(timestamps: Array, values: Array, beta: float) -> Array:
    """Speed-adaptive observation weight over the wrist channels (first 3)."""
    translation = values[:3]
    dt = np.maximum(np.diff(timestamps), 1e-6)
    speed = np.linalg.norm(np.diff(translation, axis=1), axis=0) / dt
    speed = np.concatenate([speed[:1], speed])
    speed = uniform_filter1d(speed, size=5, mode="nearest")
    median = np.median(speed) + 1e-9
    return (1.0 + beta * np.maximum(speed / median - 1.0, 0.0)) ** 2

"""Deterministic synthetic data used by the test-suite and the mock backend.

Keeping the generators in the package (rather than in ``tests/``) means the
mock backend and the unit tests exercise *the same* scene, so an end-to-end run
through the mock can never silently drift away from what the tests assert.
"""

from __future__ import annotations

from .synthetic import (
    build_camera_windows,
    camera_space_hands,
    hand_validity,
    hand_world_trajectory,
    make_camera_trajectory,
    make_intrinsics,
    relative_mismatch,
    render_depth,
    sample_scene_points,
    synthetic_detections,
    to_world0,
    world0_camera_trajectory,
    world_to_camera_points,
)

__all__ = [
    "build_camera_windows",
    "camera_space_hands",
    "hand_validity",
    "hand_world_trajectory",
    "make_camera_trajectory",
    "make_intrinsics",
    "relative_mismatch",
    "render_depth",
    "sample_scene_points",
    "synthetic_detections",
    "to_world0",
    "world0_camera_trajectory",
    "world_to_camera_points",
]

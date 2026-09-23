"""Phase 3-4: VGGT-Omega windows, depth correspondences and Sim(3) stitching."""

from __future__ import annotations

from .depth import (
    Correspondence,
    build_depth_correspondences,
    depth_to_world_points,
    scale_intrinsics,
)
from .stitch import StitchDiagnostics, StitchedCamera, stitch_camera_windows
from .window import CameraWindow, make_windows, window_ranges

__all__ = [
    "CameraWindow",
    "Correspondence",
    "StitchDiagnostics",
    "StitchedCamera",
    "build_depth_correspondences",
    "depth_to_world_points",
    "make_windows",
    "scale_intrinsics",
    "stitch_camera_windows",
    "window_ranges",
]


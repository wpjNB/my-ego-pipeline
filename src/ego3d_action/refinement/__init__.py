"""Phase 6: targeted post-processing.

Wide-window Gaussian smoothing of the hand trajectory is explicitly *not*
implemented: the reference experiments show it lowers acceleration error while
making Action-MPJPE worse. Only the three targeted corrections live here.
"""

from __future__ import annotations

from .bone_scale import BoneScaleResult, correct_bone_scale
from .camera_filter import binomial_3_filter, filter_camera_translation
from .wrist_depth import WristDepthResult, optimize_wrist_depth

__all__ = [
    "BoneScaleResult",
    "WristDepthResult",
    "binomial_3_filter",
    "correct_bone_scale",
    "filter_camera_translation",
    "optimize_wrist_depth",
]


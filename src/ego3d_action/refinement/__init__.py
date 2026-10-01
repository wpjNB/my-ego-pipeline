"""Phase 6: targeted post-processing.

Wide-window Gaussian smoothing of the hand trajectory is explicitly *not*
implemented: the reference experiments show it lowers acceleration error while
making Action-MPJPE worse. Short-gap interpolation plus the three targeted
corrections live here.
"""

from __future__ import annotations

from .bone_scale import BoneScaleResult, correct_bone_scale
from .camera_filter import binomial_3_filter, filter_camera_translation
from .gap_fill import DEFAULT_MAX_GAP, GapFillResult, interpolate_hand_gaps
from .wrist_depth import WristDepthResult, optimize_wrist_depth

__all__ = [
    "DEFAULT_MAX_GAP",
    "BoneScaleResult",
    "GapFillResult",
    "WristDepthResult",
    "binomial_3_filter",
    "correct_bone_scale",
    "filter_camera_translation",
    "interpolate_hand_gaps",
    "optimize_wrist_depth",
]


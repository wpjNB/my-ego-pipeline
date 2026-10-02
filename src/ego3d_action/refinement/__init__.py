"""Phase 6: targeted post-processing.

Wide-window Gaussian smoothing of the hand trajectory is explicitly *not*
implemented: the reference experiments show it lowers acceleration error while
making Action-MPJPE worse. Short-gap interpolation (P2), the three targeted
corrections, and the reference UKF + RTS temporal smoothing (P3) live here.
"""

from __future__ import annotations

from .bone_scale import BoneScaleResult, correct_bone_scale
from .camera_filter import binomial_3_filter, filter_camera_translation
from .gap_fill import DEFAULT_MAX_GAP, GapFillResult, interpolate_hand_gaps
from .ukf_smooth import DEFAULT_BETA, DEFAULT_Q, DEFAULT_R, UkfSmoothResult, smooth_hand_joints
from .wrist_depth import WristDepthResult, optimize_wrist_depth

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_MAX_GAP",
    "DEFAULT_Q",
    "DEFAULT_R",
    "BoneScaleResult",
    "GapFillResult",
    "UkfSmoothResult",
    "WristDepthResult",
    "binomial_3_filter",
    "correct_bone_scale",
    "filter_camera_translation",
    "interpolate_hand_gaps",
    "optimize_wrist_depth",
    "smooth_hand_joints",
]


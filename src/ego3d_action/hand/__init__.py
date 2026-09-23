"""Phase 2: HaWoR temporal reconstruction and MANO bookkeeping."""

from __future__ import annotations

from .mano import (
    FINGERTIP_JOINTS,
    JOINT_NAMES,
    JOINT_PARENTS,
    NUM_JOINTS,
    bone_lengths,
    bone_pairs,
)
from .temporal_blend import (
    HandBlendResult,
    HandWindow,
    blend_hand_windows,
    overlap_alpha,
    overlap_alpha_ramp,
)

__all__ = [
    "FINGERTIP_JOINTS",
    "HandBlendResult",
    "HandWindow",
    "JOINT_NAMES",
    "JOINT_PARENTS",
    "NUM_JOINTS",
    "blend_hand_windows",
    "bone_lengths",
    "bone_pairs",
    "overlap_alpha",
    "overlap_alpha_ramp",
]

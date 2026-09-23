"""MANO joint topology shared by several stages.

The pipeline represents every hand as 21 metric joints in metres. Both HaWoR's
temporal blending (Phase 2) and the bone-scale correction (Phase 6) need the
same kinematic tree, so it lives here once.
"""

from __future__ import annotations

import numpy as np

from ..errors import StageIOError

Array = np.ndarray

NUM_JOINTS = 21

JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

#: Parent index per joint; ``-1`` marks the root (the wrist).
JOINT_PARENTS: tuple[int, ...] = (
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
)

FINGERTIP_JOINTS: tuple[int, ...] = (4, 8, 12, 16, 20)


def bone_pairs() -> list[tuple[int, int]]:
    """Return ``(parent, child)`` pairs for every non-root joint."""
    return [(parent, child) for child, parent in enumerate(JOINT_PARENTS) if parent >= 0]


def joint_children() -> dict[int, list[int]]:
    """Map every joint to its direct children."""
    children: dict[int, list[int]] = {j: [] for j in range(NUM_JOINTS)}
    for parent, child in bone_pairs():
        children[parent].append(child)
    return children


def _check_joints(joints: Array, *, name: str) -> Array:
    arr = np.asarray(joints, dtype=np.float64)
    if arr.shape[-2:] != (NUM_JOINTS, 3):
        raise StageIOError(f"{name} must end with [21, 3], got {arr.shape}")
    return arr


def bone_vectors(joints: Array) -> Array:
    """Return bone vectors ``[..., 20, 3]`` in :func:`bone_pairs` order."""
    arr = _check_joints(joints, name="joints")
    parents = np.array([p for p, _ in bone_pairs()], dtype=np.int64)
    children = np.array([c for _, c in bone_pairs()], dtype=np.int64)
    return arr[..., children, :] - arr[..., parents, :]


def bone_lengths(joints: Array) -> Array:
    """Per-bone lengths ``[..., 20]`` in metres."""
    return np.linalg.norm(bone_vectors(joints), axis=-1)


def hand_span(joints: Array) -> Array:
    """Wrist-to-middle-tip distance ``[...]`` - a cheap hand-scale proxy."""
    arr = _check_joints(joints, name="joints")
    return np.linalg.norm(arr[..., 12, :] - arr[..., 0, :], axis=-1)


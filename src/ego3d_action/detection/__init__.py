"""Phase 1: hand detection (WiLoR) + conservative tracking."""

from __future__ import annotations

from .tracker import (
    HandDetection,
    Track,
    box_iou,
    conservative_track,
    interpolate_box,
)

__all__ = [
    "HandDetection",
    "Track",
    "box_iou",
    "conservative_track",
    "interpolate_box",
]


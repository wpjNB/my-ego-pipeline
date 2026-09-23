"""Per-stage debug visualisation.

Every stage produces a video. Staring at ``.npy`` files is a much worse way to
find a sign error in a camera convention than watching the overlay.
"""

from __future__ import annotations

from .overlay import (
    draw_detections,
    draw_hand_projection,
    plot_world_trajectory,
    write_detection_video,
    write_hand_video,
)

__all__ = [
    "draw_detections",
    "draw_hand_projection",
    "plot_world_trajectory",
    "write_detection_video",
    "write_hand_video",
]


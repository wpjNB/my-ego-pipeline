"""Video / frame / artefact IO for the pipeline."""

from __future__ import annotations

from .artefacts import ClipLayout, load_detection, load_hand, save_detection, save_hand
from .frames import FrameSet, load_frame_set, preprocess_video
from .serialization import load_json, load_npz, save_json, save_npz
from .video import VideoInfo, extract_frames, probe_video

__all__ = [
    "ClipLayout",
    "FrameSet",
    "VideoInfo",
    "extract_frames",
    "load_detection",
    "load_frame_set",
    "load_hand",
    "load_json",
    "load_npz",
    "preprocess_video",
    "probe_video",
    "save_detection",
    "save_hand",
    "save_json",
    "save_npz",
]

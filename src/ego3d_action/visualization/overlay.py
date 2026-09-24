"""Stage overlays and debug videos (Phase 7).

Frame overlays use OpenCV; trajectory plots use matplotlib with the ``Agg``
backend so the module also works headless on a server. Missing optional
dependencies raise a typed error rather than degrading silently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..errors import BackendNotAvailableError, StageIOError
from ..geometry.transforms import project_points
from ..hand.mano import bone_pairs

logger = logging.getLogger(__name__)

Array = np.ndarray

LEFT_COLOUR = (60, 76, 231)  # BGR: left hand
RIGHT_COLOUR = (76, 177, 34)  # BGR: right hand
EDGES = bone_pairs()


def require_cv2() -> object:
    """Import OpenCV, raising an actionable error when it is absent."""
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise BackendNotAvailableError(
            "opencv",
            "install the 'io' extra (pip install 'ego3d-action[io]') or use the base env, "
            "which already ships opencv",
        ) from exc
    return cv2


def draw_detections(
    frame: Array,
    boxes: Array,
    confidence: Array,
    valid: Array,
    *,
    draw_confidence: bool = True,
) -> Array:
    """Draw the two hand boxes with confidence labels on a copy of ``frame``."""
    cv2 = require_cv2()
    canvas = np.asarray(frame).copy()
    boxes = np.asarray(boxes, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if boxes.shape != (2, 4) or valid.shape != (2,):
        raise StageIOError(f"expected boxes [2, 4] and valid [2], got {boxes.shape}, {valid.shape}")

    for hand in range(2):
        if not valid[hand]:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in boxes[hand])
        colour = LEFT_COLOUR if hand == 0 else RIGHT_COLOUR
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label = ("L" if hand == 0 else "R")
        if draw_confidence:
            label += f" {float(confidence[hand]):.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_hand_projection(
    frame: Array,
    joints_camera: Array,
    intrinsics: Array,
    valid: Array,
    *,
    radius: int = 2,
) -> Array:
    """Project camera-space joints into the image and draw the 21-joint skeleton."""
    cv2 = require_cv2()
    canvas = np.asarray(frame).copy()
    joints = np.asarray(joints_camera, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if joints.shape != (2, 21, 3):
        raise StageIOError(f"joints_camera must be [2, 21, 3], got {joints.shape}")
    if valid.shape != (2,):
        raise StageIOError(f"valid must be [2], got {valid.shape}")

    pixels = project_points(np.asarray(intrinsics, dtype=np.float64), joints)
    for hand in range(2):
        if not valid[hand] or not np.isfinite(pixels[hand]).all():
            continue
        colour = LEFT_COLOUR if hand == 0 else RIGHT_COLOUR
        for parent, child in EDGES:
            p0 = tuple(np.round(pixels[hand, parent]).astype(int))
            p1 = tuple(np.round(pixels[hand, child]).astype(int))
            cv2.line(canvas, p0, p1, colour, 2, cv2.LINE_AA)
        for joint in range(joints.shape[1]):
            centre = tuple(np.round(pixels[hand, joint]).astype(int))
            cv2.circle(canvas, centre, radius, colour, -1, cv2.LINE_AA)
    return canvas


def _video_writer(cv2: object, path: Path, fps: float, size: tuple[int, int]) -> object:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise StageIOError(f"cannot open a video writer for {path} (codec mp4v)")
    return writer


def write_detection_video(
    frame_paths: Sequence[Path],
    boxes: Array,
    confidence: Array,
    valid: Array,
    out_path: str | Path,
    *,
    fps: float = 30.0,
    draw_confidence: bool = True,
) -> Path:
    """Write ``01_detection.mp4``: RGB + bboxes + confidence."""
    cv2 = require_cv2()
    frames = list(frame_paths)
    if not frames:
        raise StageIOError("no frames to render")
    if boxes.shape[0] < len(frames):
        raise StageIOError(f"boxes cover {boxes.shape[0]} frames but {len(frames)} were given")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise StageIOError(f"cannot decode {frames[0]}")
    height, width = sample.shape[:2]
    writer = _video_writer(cv2, target, float(fps), (width, height))
    try:
        for index, path in enumerate(frames):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise StageIOError(f"cannot decode {path}")
            writer.write(
                draw_detections(
                    frame,
                    boxes[index],
                    confidence[index],
                    valid[index],
                    draw_confidence=draw_confidence,
                )
            )
    finally:
        writer.release()
    logger.info("wrote %s (%d frames)", target, len(frames))
    return target


def write_hand_video(
    frame_paths: Sequence[Path],
    joints_camera: Array,
    intrinsics: Array,
    valid: Array,
    out_path: str | Path,
    *,
    fps: float = 30.0,
) -> Path:
    """Write ``02_hawor.mp4``: RGB + projected 3D hand skeleton."""
    cv2 = require_cv2()
    frames = list(frame_paths)
    if not frames:
        raise StageIOError("no frames to render")
    joints = np.asarray(joints_camera, dtype=np.float64)
    if joints.shape[0] < len(frames):
        raise StageIOError(f"joints cover {joints.shape[0]} frames but {len(frames)} were given")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise StageIOError(f"cannot decode {frames[0]}")
    height, width = sample.shape[:2]
    writer = _video_writer(cv2, target, float(fps), (width, height))
    try:
        for index, path in enumerate(frames):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise StageIOError(f"cannot decode {path}")
            camera_valid = np.asarray(valid[index], dtype=bool) & np.isfinite(joints[index]).all(axis=(1, 2))
            writer.write(
                draw_hand_projection(frame, joints[index], intrinsics[index], camera_valid)
            )
    finally:
        writer.release()
    logger.info("wrote %s (%d frames)", target, len(frames))
    return target


def plot_world_trajectory(
    joints_world: Array,
    camera_translation: Array | None,
    out_path: str | Path,
    *,
    limit: int | None = 400,
) -> Path:
    """Render a static 3D plot of the world-space hand trajectory."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BackendNotAvailableError(
            "matplotlib", "install the 'viz' extra (pip install 'ego3d-action[viz]')"
        ) from exc

    joints = np.asarray(joints_world, dtype=np.float64)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise StageIOError(f"joints_world must be [T, 2, 21, 3], got {joints.shape}")
    if limit is not None:
        joints = joints[:limit]
        if camera_translation is not None:
            camera_translation = np.asarray(camera_translation)[:limit]

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7, 6))
    axes = figure.add_subplot(111, projection="3d")
    colours = {0: "#e74c3c", 1: "#27ae60"}
    for hand in range(2):
        wrist = joints[:, hand, 0, :]
        finite = np.isfinite(wrist).all(axis=-1)
        if finite.any():
            axes.plot(wrist[finite, 0], wrist[finite, 1], wrist[finite, 2], color=colours[hand], lw=1.5)
    if camera_translation is not None:
        camera = np.asarray(camera_translation, dtype=np.float64)
        axes.plot(camera[:, 0], camera[:, 1], camera[:, 2], color="#2c3e50", lw=1.0, ls="--")
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_zlabel("z [m]")
    axes.set_title("world-space hand trajectory")
    figure.tight_layout()
    figure.savefig(target, dpi=110)
    plt.close(figure)
    logger.info("wrote %s", target)
    return target


def render_overlays(
    frame_paths: Iterable[Path],
    *,
    boxes: Array | None = None,
    confidence: Array | None = None,
    valid: Array | None = None,
    joints_camera: Array | None = None,
    intrinsics: Array | None = None,
    out_dir: str | Path,
) -> list[Path]:
    """Convenience wrapper used by the scripts to emit the debug videos."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if boxes is not None and confidence is not None and valid is not None:
        written.append(
            write_detection_video(frame_paths, boxes, confidence, valid, out / "01_detection.mp4")
        )
    if joints_camera is not None and intrinsics is not None and valid is not None:
        written.append(
            write_hand_video(frame_paths, joints_camera, intrinsics, valid, out / "02_hawor.mp4")
        )
    return written


def _project_wrist(
    point_world: Array, rotation_c2w: Array, translation_c2w: Array, intrinsics: Array
) -> tuple[int, int] | None:
    """Project one world point with one camera; ``None`` when not drawable."""
    point = np.asarray(point_world, dtype=np.float64)
    if not np.isfinite(point).all():
        return None
    camera = np.asarray(rotation_c2w, dtype=np.float64).T @ (point - np.asarray(translation_c2w))
    if camera[2] <= 1e-6:
        return None
    pixel = np.asarray(intrinsics, dtype=np.float64) @ camera
    return int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))


def world_to_camera(
    points_world: Array, rotation_c2w: Array, translation_c2w: Array
) -> Array:
    """``[..., 3]`` world points -> camera space for the same camera.

    This is the transform that every overlay has to apply: the trajectory
    contract keeps ``hand_xyz_world`` in metres, and only ``camera_R_c2w`` /
    ``camera_t_c2w`` know where the camera was. Drawing world points with the
    intrinsics alone (i.e. skipping this step) puts the hands wherever the world
    origin happens to be - the mistake that made an earlier debug figure look
    broken.
    """
    points = np.asarray(points_world, dtype=np.float64)
    rotation = np.asarray(rotation_c2w, dtype=np.float64)
    translation = np.asarray(translation_c2w, dtype=np.float64)
    rotated = np.einsum("ji,...j->...i", rotation, points - translation)
    return rotated


def write_wrist_comparison_video(
    frame_paths: Sequence[Path],
    ground_truth_world: Array,
    rotation_c2w: Array,
    translation_c2w: Array,
    intrinsics: Array,
    valid: Array,
    out_path: str | Path,
    *,
    prediction_world: Array | None = None,
    draw_skeleton: bool = False,
    fps: float = 30.0,
    still_indices: Sequence[int] = (),
    still_dir: str | Path | None = None,
) -> Path:
    """Overlay the reference wrist (and optionally a prediction) on the RGB.

    Both trajectories are projected with the *reference* camera, so this is a
    direct visual check that the ground truth - and the frame conventions the
    whole pipeline uses - line up with the pixels.

    With ``draw_skeleton`` the full 21-joint reference (and prediction) is drawn
    as well, for frames whose joints are finite - i.e. whenever the reference was
    built with MANO. Joints and wrists always go through
    :func:`world_to_camera` first.
    """
    cv2 = require_cv2()
    frames = list(frame_paths)
    if not frames:
        raise StageIOError("no frames to render")
    truth = np.asarray(ground_truth_world, dtype=np.float64)
    if truth.shape[0] < len(frames):
        raise StageIOError(f"ground truth covers {truth.shape[0]} frames but {len(frames)} were given")
    prediction = None if prediction_world is None else np.asarray(prediction_world, dtype=np.float64)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise StageIOError(f"cannot decode {frames[0]}")
    height, width = sample.shape[:2]
    writer = _video_writer(cv2, target, float(fps), (width, height))
    stills = set(int(i) for i in still_indices)
    still_root = Path(still_dir) if still_dir is not None else None
    if stills and still_root is None:
        still_root = target.parent / f"{target.stem}_stills"
    if still_root is not None:
        still_root.mkdir(parents=True, exist_ok=True)

    try:
        for index, path in enumerate(frames):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise StageIOError(f"cannot decode {path}")
            canvas = frame.copy()
            if draw_skeleton:
                # The reference only: the prediction keeps its own orange marker,
                # and the two must stay visually distinguishable.
                joints_camera = world_to_camera(
                    truth[index], rotation_c2w[index], translation_c2w[index]
                )
                drawable = np.asarray(valid[index], dtype=bool) & np.isfinite(
                    joints_camera
                ).all(axis=(1, 2))
                if drawable.any():
                    canvas = draw_hand_projection(
                        canvas, joints_camera, intrinsics[index], drawable, radius=2
                    )
            for hand, colour, label in ((0, LEFT_COLOUR, "L"), (1, RIGHT_COLOUR, "R")):
                if not bool(valid[index, hand]):
                    continue
                gt_pixel = _project_wrist(
                    truth[index, hand, 0, :], rotation_c2w[index], translation_c2w[index], intrinsics[index]
                )
                pred_pixel = (
                    _project_wrist(
                        prediction[index, hand, 0, :],
                        rotation_c2w[index],
                        translation_c2w[index],
                        intrinsics[index],
                    )
                    if prediction is not None
                    else None
                )
                if gt_pixel is not None:
                    cv2.circle(canvas, gt_pixel, 6, colour, 2, cv2.LINE_AA)
                    cv2.drawMarker(canvas, gt_pixel, colour, cv2.MARKER_CROSS, 14, 2)
                    cv2.putText(
                        canvas,
                        f"GT {label}",
                        (gt_pixel[0] + 8, gt_pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        colour,
                        1,
                        cv2.LINE_AA,
                    )
                if pred_pixel is not None:
                    cv2.circle(canvas, pred_pixel, 6, (0, 215, 255), 2, cv2.LINE_AA)
                if gt_pixel is not None and pred_pixel is not None:
                    error_mm = 1000.0 * float(
                        np.linalg.norm(truth[index, hand, 0, :] - prediction[index, hand, 0, :])
                    )
                    cv2.line(canvas, gt_pixel, pred_pixel, (255, 255, 255), 1, cv2.LINE_AA)
                    midpoint = ((gt_pixel[0] + pred_pixel[0]) // 2, (gt_pixel[1] + pred_pixel[1]) // 2)
                    cv2.putText(
                        canvas,
                        f"{error_mm:.0f} mm",
                        midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                canvas,
                f"frame {index}"
                + ("   orange = prediction" if prediction is not None else "")
                + "   coloured = ground truth",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            if index in stills and still_root is not None:
                cv2.imwrite(str(still_root / f"{index:06d}.png"), canvas)
    finally:
        writer.release()
    logger.info("wrote %s (%d frames)", target, len(frames))
    return target

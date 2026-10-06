"""Stage overlays and debug videos (Phase 7).

Frame overlays use OpenCV; trajectory plots use matplotlib with the ``Agg``
backend so the module also works headless on a server. Missing optional
dependencies raise a typed error rather than degrading silently.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
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
SKELETON_COLOUR = (240, 240, 240)  # BGR: near-white, reads on top of the shaded mesh
PREDICTION_COLOUR = (0, 215, 255)  # BGR: orange, the comparison overlay's prediction
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
    colour: Sequence[int] | None = None,
) -> Array:
    """Project camera-space joints into the image and draw the 21-joint skeleton.

    ``colour`` overrides the per-hand left/right colours with one BGR tuple
    for both hands - used when the skeleton is drawn on top of the mesh, where
    the hand colours would blend into it.
    """
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
        hand_colour = tuple(colour) if colour is not None else (LEFT_COLOUR if hand == 0 else RIGHT_COLOUR)
        for parent, child in EDGES:
            p0 = tuple(np.round(pixels[hand, parent]).astype(int))
            p1 = tuple(np.round(pixels[hand, child]).astype(int))
            cv2.line(canvas, p0, p1, hand_colour, 2, cv2.LINE_AA)
        for joint in range(joints.shape[1]):
            centre = tuple(np.round(pixels[hand, joint]).astype(int))
            cv2.circle(canvas, centre, radius, hand_colour, -1, cv2.LINE_AA)
    return canvas


def nudge_hand_overlay_toward_boxes(
    joints_camera: Array,
    vertices_camera: Array,
    boxes: Array,
    intrinsics: Array,
    valid: Array,
    *,
    fraction: float = 0.5,
) -> tuple[Array, Array]:
    """Shift a rendered hand partway toward its tracked box in image space.

    This visualization-only correction addresses WiLoR's residual 2D root
    translation error. It projects the MANO vertices, measures their centroid,
    then applies a fraction of the centroid-to-box-center delta to both the
    mesh and skeleton. The camera-space arrays on disk are never modified.
    """
    joints = np.asarray(joints_camera, dtype=np.float64).copy()
    vertices = np.asarray(vertices_camera, dtype=np.float64).copy()
    box_array = np.asarray(boxes, dtype=np.float64)
    hand_valid = np.asarray(valid, dtype=bool)
    K = np.asarray(intrinsics, dtype=np.float64)

    if joints.shape != (2, 21, 3):
        raise StageIOError(f"joints_camera must be [2, 21, 3], got {joints.shape}")
    if vertices.ndim != 3 or vertices.shape[0] != 2 or vertices.shape[2] != 3:
        raise StageIOError(f"vertices_camera must be [2, V, 3], got {vertices.shape}")
    if box_array.shape != (2, 4):
        raise StageIOError(f"boxes must be [2, 4], got {box_array.shape}")
    if hand_valid.shape != (2,):
        raise StageIOError(f"valid must be [2], got {hand_valid.shape}")
    if K.shape == (3, 3):
        K = np.broadcast_to(K, (2, 3, 3))
    if K.shape != (2, 3, 3):
        raise StageIOError(f"intrinsics must be [3, 3] or [2, 3, 3], got {K.shape}")
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise StageIOError(f"fraction must be in [0, 1], got {fraction}")

    for hand in range(2):
        if not hand_valid[hand] or not np.isfinite(box_array[hand]).all():
            continue
        if box_array[hand, 2] <= box_array[hand, 0] or box_array[hand, 3] <= box_array[hand, 1]:
            continue
        points = vertices[hand]
        pixels = project_points(K[hand], points)
        visible = (
            np.isfinite(pixels).all(axis=1)
            & np.isfinite(points[:, 2])
            & (points[:, 2] > 1e-6)
        )
        if not visible.any():
            continue
        predicted_center = np.mean(pixels[visible], axis=0)
        target_center = np.array(
            [
                (box_array[hand, 0] + box_array[hand, 2]) / 2.0,
                (box_array[hand, 1] + box_array[hand, 3]) / 2.0,
            ],
            dtype=np.float64,
        )
        delta = (target_center - predicted_center) * float(fraction)
        fx, fy = K[hand, 0, 0], K[hand, 1, 1]
        skew = K[hand, 0, 1]
        if not np.isfinite([fx, fy, skew]).all() or fx <= 0.0 or fy <= 0.0:
            raise StageIOError(f"invalid focal matrix for hand {hand}: {K[hand]}")
        shift_y = delta[1] / fy
        shift_x = (delta[0] - skew * shift_y) / fx
        vertices[hand, :, 0] += shift_x * vertices[hand, :, 2]
        vertices[hand, :, 1] += shift_y * vertices[hand, :, 2]
        joints[hand, :, 0] += shift_x * joints[hand, :, 2]
        joints[hand, :, 1] += shift_y * joints[hand, :, 2]

    return joints, vertices


def draw_hand_mesh(
    frame: Array,
    vertices_camera: Array,
    faces: Array,
    intrinsics: Array,
    valid: Array,
    *,
    colours: Sequence[Sequence[int]] = (LEFT_COLOUR, RIGHT_COLOUR),
    ambient: float = 0.30,
) -> Array:
    """Rasterise MANO meshes onto the frame with a depth-sorted painter's algorithm.

    A headlight Lambert shading (face normal dotted with the view direction)
    gives the mesh its shape; no GPU renderer is involved, so this works
    wherever OpenCV does. Faces with any non-finite or behind-camera vertex are
    skipped - holes stay holes, matching the never-fabricate rule.

    Args:
        frame: BGR image.
        vertices_camera: ``[2, V, 3]`` mesh vertices in camera space (metres).
        faces: one ``[F, 3]`` triangle-index array used for both hands, or a
            two-element sequence with per-hand winding (the mirrored left hand
            flips it).
        intrinsics: ``[3, 3]`` or ``[2, 3, 3]`` camera matrix.
        valid: ``[2]`` per-hand validity.
        colours: BGR base colour per hand.
        ambient: floor for the shading term.

    Returns:
        The canvas with both hands drawn.
    """
    cv2 = require_cv2()
    canvas = np.asarray(frame).copy()
    vertices = np.asarray(vertices_camera, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)
    per_hand_faces = [faces_arr, faces_arr] if faces_arr.ndim == 2 else list(faces_arr)
    if vertices.ndim != 3 or vertices.shape[0] != 2:
        raise StageIOError(f"vertices_camera must be [2, V, 3], got {vertices.shape}")
    intrinsics_all = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics_all.ndim == 2:
        intrinsics_all = np.broadcast_to(intrinsics_all, (2, 3, 3))

    for hand in range(2):
        if not np.asarray(valid, dtype=bool)[hand]:
            continue
        faces = per_hand_faces[hand]
        pixels = project_points(intrinsics_all[hand], vertices[hand])
        depth = vertices[hand][:, 2]
        drawable = np.isfinite(pixels).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)
        if drawable.sum() < 3:
            continue
        tri = faces[drawable[faces].all(axis=1)]
        if not len(tri):
            continue
        # Headlight Lambert with a *signed* dot: MANO's faces wind outward, so
        # faces pointing away from the camera go dark (and are overpainted by
        # the front ones anyway) - that is what gives the mesh its 3D read.
        v0, v1, v2 = vertices[hand][tri[:, 0]], vertices[hand][tri[:, 1]], vertices[hand][tri[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        to_camera = -v0 / np.maximum(np.linalg.norm(v0, axis=1, keepdims=True), 1e-12)
        facing = np.einsum("fc,fc->f", normals, to_camera)[..., None] / np.maximum(norm, 1e-12)
        shade = np.clip(ambient + (1.0 - ambient) * np.maximum(facing, 0.0), 0.0, 1.0)
        base = np.asarray(colours[hand], dtype=np.float64)
        order = np.argsort(-np.stack([v0, v1, v2], axis=1).mean(axis=1)[:, 2])  # far first
        polys = np.round(pixels[tri]).astype(np.int32)
        for face_index in order:
            cv2.fillPoly(canvas, [polys[face_index]], tuple(
                int(c) for c in (base * shade[face_index])
            ))
    return canvas
def _video_writer(cv2: object, path: Path, fps: float, size: tuple[int, int]) -> object:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise StageIOError(f"cannot open a video writer for {path} (codec mp4v)")
    return writer


def transcode_to_h264(path: str | Path) -> Path:
    """Re-encode an OpenCV-written mp4 in place as H.264, when ffmpeg exists.

    ``mp4v`` is MPEG-4 Part 2, which browsers and most default players refuse
    to open. ffmpeg's libx264 (yuv420p, +faststart) plays everywhere. Best
    effort: without ffmpeg - or when it fails - the original file is kept and
    the skip is reported, never raised: these are debug artefacts.
    """
    target = Path(path)
    binary = shutil.which("ffmpeg")
    if binary is None or not target.is_file():
        if binary is None:
            logger.warning("%s: kept as mp4v (no ffmpeg on PATH, may not play in browsers)", target)
        return target
    tmp = target.with_name(f".{target.stem}.h264.mp4")
    try:
        subprocess.run(
            [
                binary, "-y", "-loglevel", "error", "-i", str(target),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        logger.warning("%s: H.264 transcode skipped (%s)", target, detail.decode(errors="replace").strip())
        tmp.unlink(missing_ok=True)
        return target
    tmp.replace(target)
    return target


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
    transcode_to_h264(target)
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
    vertices_camera: Array | None = None,
    faces: Sequence[Array] | None = None,
    boxes: Array | None = None,
    box_nudge: float = 0.0,
) -> Path:
    """Write ``02_hawor.mp4``: RGB + projected hands.

    With ``vertices_camera`` (``[T, 2, V, 3]``) and per-hand ``faces`` the full
    MANO mesh is rasterised (the MINT-style overlay) with the 21-joint skeleton
    drawn on top in :data:`SKELETON_COLOUR` - the mesh gives the shape, the
    skeleton the estimated articulation; without vertices the writer falls back
    to the skeleton alone. WiLoR callers may pass boxes and a positive box_nudge
    for an explicitly labeled, image-space-only root adjustment; saved 3D arrays
    remain unchanged.
    """
    cv2 = require_cv2()
    frames = list(frame_paths)
    if not frames:
        raise StageIOError("no frames to render")
    joints = np.asarray(joints_camera, dtype=np.float64)
    if joints.shape[0] < len(frames):
        raise StageIOError(f"joints cover {joints.shape[0]} frames but {len(frames)} were given")
    vertices = (
        None
        if vertices_camera is None or faces is None
        else np.asarray(vertices_camera, dtype=np.float64)
    )
    if vertices is not None and vertices.shape[0] < len(frames):
        raise StageIOError(f"vertices cover {vertices.shape[0]} frames but {len(frames)} were given")
    box_array = None if boxes is None else np.asarray(boxes, dtype=np.float64)
    if box_array is not None and (box_array.ndim != 3 or box_array.shape[1:] != (2, 4)):
        raise StageIOError(f"boxes must be [T, 2, 4], got {box_array.shape}")
    if box_array is not None and box_array.shape[0] < len(frames):
        raise StageIOError(f"boxes cover {box_array.shape[0]} frames but {len(frames)} were given")
    if not np.isfinite(box_nudge) or not 0.0 <= box_nudge <= 1.0:
        raise StageIOError(f"box_nudge must be in [0, 1], got {box_nudge}")
    if box_nudge > 0.0 and box_array is None:
        raise StageIOError("boxes are required when box_nudge is enabled")
    if box_nudge > 0.0 and vertices is None:
        raise StageIOError("vertices_camera are required when box_nudge is enabled")

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
            frame_joints = joints[index]
            frame_vertices = None if vertices is None else vertices[index]
            if frame_vertices is not None and box_array is not None and box_nudge > 0.0:
                frame_joints, frame_vertices = nudge_hand_overlay_toward_boxes(
                    frame_joints,
                    frame_vertices,
                    box_array[index],
                    intrinsics[index],
                    camera_valid,
                    fraction=box_nudge,
                )
            if frame_vertices is not None and faces is not None:
                frame_valid = camera_valid & np.isfinite(frame_vertices).all(axis=(1, 2))
                canvas = draw_hand_mesh(
                    frame, frame_vertices, faces, intrinsics[index], frame_valid
                )
                canvas = draw_hand_projection(
                    canvas, frame_joints, intrinsics[index], camera_valid,
                    colour=SKELETON_COLOUR,
                )
            else:
                canvas = draw_hand_projection(frame, joints[index], intrinsics[index], camera_valid)
            if box_nudge > 0.0:
                cv2.rectangle(canvas, (0, 0), (360, 28), (20, 22, 26), -1)
                cv2.putText(
                    canvas,
                    f"2D box nudge {box_nudge:.0%} (visual only)",
                    (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
            if box_nudge > 0.0:
                cv2.rectangle(canvas, (0, 0), (min(width - 1, 340), 26), (20, 22, 26), -1)
                cv2.putText(
                    canvas,
                    f"2D box nudge {box_nudge:.0%} (visual only)",
                    (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
    finally:
        writer.release()
    transcode_to_h264(target)
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


def _project_pixel(point_camera: Array, intrinsics: Array) -> tuple[int, int] | None:
    """Pixel of an already camera-space point, or None when not projectable."""
    point = np.asarray(point_camera, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if not np.isfinite(point).all() or point[2] <= 1e-6:
        return None
    uv = k @ point
    uv = uv[:2] / uv[2]
    if not (np.isfinite(uv).all()):
        return None
    return int(round(float(uv[0]))), int(round(float(uv[1])))


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
    note: str | None = None,
    prediction_camera: Array | None = None,
) -> Path:
    """Overlay the reference (and optionally a prediction) on the RGB.

    Both trajectories are projected with the *reference* camera, so this is a
    direct visual check that the ground truth - and the frame conventions the
    whole pipeline uses - line up with the pixels.

    With ``draw_skeleton`` both hands are drawn as 21-joint skeletons: the
    ground truth in its per-hand colours, the prediction in
    :data:`PREDICTION_COLOUR`. The wrist markers carry a ``Δ <mm>`` label on
    the connecting line; frames where a hand is GT-valid but the prediction is
    missing say so instead of silently showing one side. ``note`` is echoed in
    the header (e.g. the alignment mode).

    By default the prediction is taken from the world trajectory and projected
    through the reference camera, so the overlay shows the *system* error
    (hands + camera trajectory). Pass ``prediction_camera`` (the prediction's
    own ``hand_xyz_camera``) to render it in its own camera frame instead -
    the residual disagreement with the GT skeleton is then hand-estimation
    error only, with the camera trajectory factored out.
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
                # Ground truth in its per-hand colours, prediction in orange -
                # two full skeletons, not a lone dot next to a skeleton.
                gt_cam = world_to_camera(
                    truth[index], rotation_c2w[index], translation_c2w[index]
                )
                gt_draw = np.asarray(valid[index], dtype=bool) & np.isfinite(
                    gt_cam
                ).all(axis=(1, 2))
                if gt_draw.any():
                    canvas = draw_hand_projection(
                        canvas, gt_cam, intrinsics[index], gt_draw, radius=2
                    )
                pred_cam = None
                if prediction is not None:
                    if prediction_camera is not None:
                        pred_cam = prediction_camera[index]
                    else:
                        pred_cam = world_to_camera(
                            prediction[index], rotation_c2w[index], translation_c2w[index]
                        )
                    pred_draw = np.isfinite(pred_cam).all(axis=(1, 2))
                    if pred_draw.any():
                        canvas = draw_hand_projection(
                            canvas, pred_cam, intrinsics[index], pred_draw,
                            radius=2, colour=PREDICTION_COLOUR,
                        )
            for hand, colour, label in ((0, LEFT_COLOUR, "L"), (1, RIGHT_COLOUR, "R")):
                if not bool(valid[index, hand]):
                    continue
                gt_pixel = _project_wrist(
                    truth[index, hand, 0, :], rotation_c2w[index], translation_c2w[index], intrinsics[index]
                )
                if prediction_camera is not None:
                    pred_pixel = (
                        _project_pixel(prediction_camera[index, hand, 0, :], intrinsics[index])
                        if prediction is not None
                        else None
                    )
                else:
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
                    cv2.circle(canvas, pred_pixel, 6, PREDICTION_COLOUR, 2, cv2.LINE_AA)
                    cv2.putText(
                        canvas,
                        "pred",
                        (pred_pixel[0] + 8, pred_pixel[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        PREDICTION_COLOUR,
                        1,
                        cv2.LINE_AA,
                    )
                if gt_pixel is not None and pred_pixel is not None and prediction_camera is not None:
                    pass  # camera-frame mode: the two skeletons are the message
                elif gt_pixel is not None and pred_pixel is not None and prediction_camera is None:
                    error_mm = 1000.0 * float(
                        np.linalg.norm(truth[index, hand, 0, :] - prediction[index, hand, 0, :])
                    )
                    cv2.line(canvas, gt_pixel, pred_pixel, (255, 255, 255), 1, cv2.LINE_AA)
                    midpoint = ((gt_pixel[0] + pred_pixel[0]) // 2, (gt_pixel[1] + pred_pixel[1]) // 2)
                    # black stroke under the text so it reads on any background
                    cv2.putText(
                        canvas,
                        f"\u0394 {error_mm:.0f} mm",
                        (midpoint[0] + 4, midpoint[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        f"\u0394 {error_mm:.0f} mm",
                        (midpoint[0] + 4, midpoint[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                elif gt_pixel is not None and pred_pixel is None and prediction is not None:
                    cv2.putText(
                        canvas,
                        "no prediction",
                        (gt_pixel[0] + 8, gt_pixel[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (160, 160, 160),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                canvas,
                f"frame {index}"
                + ("   orange = prediction" if prediction is not None else "")
                + "   coloured = ground truth"
                + (f"   [{note}]" if note else ""),
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
    transcode_to_h264(target)
    logger.info("wrote %s (%d frames)", target, len(frames))
    return target

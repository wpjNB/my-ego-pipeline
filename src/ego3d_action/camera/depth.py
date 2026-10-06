"""Depth-derived correspondences for window alignment (Phase 4).

The reference system aligns overlapping VGGT-Omega windows with a
**depth-derived** Sim(3), not with camera centres. For every shared frame and
every sampled pixel we back-project the metric depth into the window's local
world frame; the same ``(frame, v, u)`` in two windows is then a 3D
correspondence between their coordinate systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..errors import StageIOError
from .window import CameraWindow

logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class Correspondence:
    """Correspondences ``src`` (new window) -> ``dst`` (reference window)."""

    points_src: Array  # [N, 3] in the source window frame
    points_dst: Array  # [N, 3] in the destination window frame
    weights: Array  # [N]
    frames: Array  # [N] video frame indices

    def __post_init__(self) -> None:
        src = np.asarray(self.points_src, dtype=np.float64)
        dst = np.asarray(self.points_dst, dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        frames = np.asarray(self.frames, dtype=np.int64).reshape(-1)
        if src.ndim != 2 or src.shape[-1] != 3:
            raise StageIOError(f"points_src must be [N, 3], got {src.shape}")
        if dst.shape != src.shape:
            raise StageIOError(f"points_dst shape {dst.shape} != points_src shape {src.shape}")
        if w.shape[0] != src.shape[0]:
            raise StageIOError(f"weights length {w.shape[0]} != {src.shape[0]}")
        if frames.shape[0] != src.shape[0]:
            raise StageIOError(f"frames length {frames.shape[0]} != {src.shape[0]}")
        object.__setattr__(self, "points_src", src)
        object.__setattr__(self, "points_dst", dst)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "frames", frames)

    @property
    def count(self) -> int:
        return int(self.points_src.shape[0])


def canonical_intrinsics(windows_dir, frame_size: tuple[int, int]):
    """The clip's single intrinsic matrix: per-element median of the windows.

    VGGT estimates intrinsics per camera window and the estimates scatter
    (measured 2026-10-02: first-window vs median up to 7.7 % on the official
    clips, 5.9 % on hot3d_ep000). Reconstruction (the hand stage's focal) and
    rendering (the debug overlays) must resolve the SAME matrix, or the render
    draws through a different camera than the one the prediction was built in
    - a whole-image scale/shift about the principal point that grows with the
    distance from the image centre. The median is robust against the
    occasional outlier window. Returns ``None`` when the clip has no windows.
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    from ..io.serialization import load_npz  # noqa: PLC0415

    matrices = []
    for path in sorted(_Path(windows_dir).glob("*.npz")):
        data = load_npz(path, required=("intrinsics", "depth"))
        depth = np.asarray(data["depth"])
        source = (int(depth.shape[2]), int(depth.shape[1]))
        matrices.append(
            scale_intrinsics(
                np.asarray(data["intrinsics"])[0], source_size=source, target_size=frame_size
            )
        )
    if not matrices:
        return None
    return np.median(np.stack(matrices), axis=0)


def scale_intrinsics(intrinsics: Array, *, source_size: tuple[int, int], target_size: tuple[int, int]) -> Array:
    """Rescale intrinsics from ``source_size`` to ``target_size`` (w, h)."""
    k = np.asarray(intrinsics, dtype=np.float64).copy()
    if k.shape[-2:] != (3, 3):
        raise StageIOError(f"intrinsics must be [..., 3, 3], got {k.shape}")
    sw, sh = source_size
    tw, th = target_size
    if min(sw, sh, tw, th) <= 0:
        raise StageIOError(f"invalid sizes: source={source_size} target={target_size}")
    if (sw, sh) == (tw, th):
        return k
    fx, fy = tw / sw, th / sh
    scale = np.array([[fx, 0.0, 0.0], [0.0, fy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.einsum("ij,...jk->...ik", scale, k)


def input_frame_intrinsics(
    metadata: dict[str, object], target_size: tuple[int, int]
) -> Array | None:
    """Return the calibrated K for the decoded RGB pixels, when recorded.

    This is only the camera model for *drawing 3D predictions onto the source
    frames*. Hand inference continues to use its configured / VGGT-estimated
    focal. ``None`` means the clip has no explicit image calibration and the
    caller should fall back to the canonical VGGT estimate.
    """
    camera = metadata.get("image_camera")
    if camera is None:
        return None
    if not isinstance(camera, dict):
        raise StageIOError("clip metadata image_camera must be an object")
    if camera.get("model") != "pinhole" or camera.get("undistorted") is not True:
        raise StageIOError(
            "preview projection requires image_camera.model='pinhole' and undistorted=true"
        )
    try:
        source_size = (int(camera["width"]), int(camera["height"]))
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise StageIOError(f"invalid image_camera calibration in clip metadata: {exc}") from exc
    if intrinsics.shape != (3, 3):
        raise StageIOError(f"image_camera.intrinsics must be [3, 3], got {intrinsics.shape}")
    if (
        not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
    ):
        raise StageIOError("image_camera.intrinsics must be finite with positive focal lengths")
    return scale_intrinsics(
        intrinsics, source_size=source_size, target_size=target_size
    )


def depth_to_world_points(
    depth: Array,
    intrinsics: Array,
    rotation_c2w: Array,
    translation_c2w: Array,
    *,
    stride: int = 8,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
    min_confidence: float | None = None,
    depth_confidence: Array | None = None,
) -> tuple[Array, Array]:
    """Back-project a depth map into the window-local world frame.

    Returns:
        ``(points [N, 3], pixel_index [N, 3])`` where ``pixel_index`` holds
        ``(row, col, 0)`` so callers can pair identical pixels across windows.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise StageIOError(f"depth must be [H, W], got {d.shape}")
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise StageIOError(f"intrinsics must be [3, 3] for a single frame, got {k.shape}")
    if stride <= 0:
        raise StageIOError(f"stride must be positive, got {stride}")

    height, width = d.shape
    rows = np.arange(0, height, stride, dtype=np.int64)
    cols = np.arange(0, width, stride, dtype=np.int64)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
    grid_r = grid_r.reshape(-1)
    grid_c = grid_c.reshape(-1)
    z = d[grid_r, grid_c]

    mask = np.isfinite(z) & (z > min_depth)
    if max_depth is not None:
        mask &= z < max_depth
    if depth_confidence is not None and min_confidence is not None:
        conf = np.asarray(depth_confidence, dtype=np.float64)
        if conf.shape != d.shape:
            raise StageIOError(f"depth_confidence must be {d.shape}, got {conf.shape}")
        mask &= np.isfinite(conf[grid_r, grid_c]) & (conf[grid_r, grid_c] >= min_confidence)
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    grid_r = grid_r[mask]
    grid_c = grid_c[mask]
    z = z[mask]

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]
    if fx == 0.0 or fy == 0.0:
        raise StageIOError("intrinsics have a zero focal length")

    x = (grid_c - cx) / fx * z
    y = (grid_r - cy) / fy * z
    points_cam = np.stack([x, y, z], axis=-1)

    rot = np.asarray(rotation_c2w, dtype=np.float64)
    tr = np.asarray(translation_c2w, dtype=np.float64)
    if rot.shape != (3, 3) or tr.shape != (3,):
        raise StageIOError(
            f"single-frame rotation must be [3, 3] and translation [3], got {rot.shape}, {tr.shape}"
        )
    points_world = points_cam @ rot.T + tr
    index = np.stack([grid_r, grid_c, np.zeros_like(grid_r)], axis=-1)
    return points_world, index


def _frame_points(
    window: CameraWindow,
    local_frame: int,
    rows: Array,
    cols: Array,
    grid_r: Array,
    grid_c: Array,
    *,
    min_depth: float,
    max_depth: float | None,
    min_confidence: float | None,
) -> tuple[Array, Array, Array | None]:
    """Back-project the shared sampled grid for one frame without per-point objects."""
    z = window.depth[local_frame][np.ix_(rows, cols)].reshape(-1)
    mask = np.isfinite(z) & (z > min_depth)
    if max_depth is not None:
        mask &= z < max_depth

    confidence = None
    if min_confidence is not None and window.depth_confidence is not None:
        confidence = window.depth_confidence[local_frame][np.ix_(rows, cols)].reshape(-1)
        mask &= np.isfinite(confidence) & (confidence >= min_confidence)

    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float64), mask, confidence

    z = z[mask]
    k = window.intrinsics[local_frame]
    fx, fy = k[0, 0], k[1, 1]
    if fx == 0.0 or fy == 0.0:
        raise StageIOError("intrinsics have a zero focal length")
    x = (grid_c[mask] - k[0, 2]) / fx * z
    y = (grid_r[mask] - k[1, 2]) / fy * z
    points_cam = np.stack((x, y, z), axis=-1)
    points_world = points_cam @ window.rotation_c2w[local_frame].T
    points_world += window.translation_c2w[local_frame]
    return points_world, mask, confidence


def build_depth_correspondences(
    window_src: CameraWindow,
    window_dst: CameraWindow,
    *,
    stride: int = 8,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
    min_confidence: float | None = None,
    max_points: int | None = 40000,
) -> Correspondence:
    """Pair depth-derived points of two overlapping windows.

    Args:
        window_src: the newer window (source frame of the Sim(3)).
        window_dst: the reference window (destination frame of the Sim(3)).
        stride: pixel subsampling step.
        min_depth: points closer than this are dropped.
        max_depth: optional far clip.
        min_confidence: optional VGGT depth-confidence floor.
        max_points: deterministic subsample cap.

    Raises:
        StageIOError: when the two windows share no frames.
    """
    if stride <= 0:
        raise StageIOError(f"stride must be positive, got {stride}")
    overlap_start = max(window_src.start, window_dst.start)
    overlap_end = min(window_src.end, window_dst.end)
    if overlap_end <= overlap_start:
        raise StageIOError(
            f"windows {window_src.name} and {window_dst.name} do not overlap"
        )

    # Both windows are sampled on the same pixel coordinates. Restrict the grid
    # to their common image area, then vectorise each frame's validity mask and
    # back-projection instead of building a Python dict entry per depth sample.
    height = min(window_src.depth.shape[1], window_dst.depth.shape[1])
    width = min(window_src.depth.shape[2], window_dst.depth.shape[2])
    rows = np.arange(0, height, stride, dtype=np.int64)
    cols = np.arange(0, width, stride, dtype=np.int64)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
    grid_r = grid_r.reshape(-1)
    grid_c = grid_c.reshape(-1)

    src_points: list[Array] = []
    dst_points: list[Array] = []
    frame_ids: list[Array] = []
    confidence_weights: list[Array] = []
    use_confidence_weights = (
        min_confidence is not None and window_src.depth_confidence is not None
    )
    for frame in range(overlap_start, overlap_end):
        local_src = frame - window_src.start
        local_dst = frame - window_dst.start
        src_frame_points, src_valid, src_conf = _frame_points(
            window_src,
            local_src,
            rows,
            cols,
            grid_r,
            grid_c,
            min_depth=min_depth,
            max_depth=max_depth,
            min_confidence=min_confidence,
        )
        dst_frame_points, dst_valid, _ = _frame_points(
            window_dst,
            local_dst,
            rows,
            cols,
            grid_r,
            grid_c,
            min_depth=min_depth,
            max_depth=max_depth,
            min_confidence=min_confidence,
        )
        common = src_valid & dst_valid
        if not np.any(common):
            continue

        src_points.append(src_frame_points[common[src_valid]])
        dst_points.append(dst_frame_points[common[dst_valid]])
        count = int(np.count_nonzero(common))
        frame_ids.append(np.full(count, frame, dtype=np.int64))
        if use_confidence_weights and src_conf is not None:
            confidence_weights.append(np.clip(src_conf[common], 1e-3, None))

    if not src_points:
        raise StageIOError(
            f"no shared depth samples between windows {window_src.name} and {window_dst.name}"
        )

    points_src = np.concatenate(src_points, axis=0)
    points_dst = np.concatenate(dst_points, axis=0)
    frames = np.concatenate(frame_ids)
    weights = (
        np.concatenate(confidence_weights)
        if use_confidence_weights
        else np.ones(points_src.shape[0], dtype=np.float64)
    )

    if max_points is not None and points_src.shape[0] > max_points:
        step = int(np.ceil(points_src.shape[0] / max_points))
        selection = slice(None, None, step)
        points_src = points_src[selection]
        points_dst = points_dst[selection]
        frames = frames[selection]
        weights = weights[selection]
        logger.info(
            "depth correspondences %s<->%s: subsampled to %d points (driver %d)",
            window_src.name,
            window_dst.name,
            points_src.shape[0],
            max_points,
        )

    logger.info(
        "depth correspondences %s -> %s: %d pairs over %d shared frames",
        window_src.name,
        window_dst.name,
        points_src.shape[0],
        overlap_end - overlap_start,
    )
    return Correspondence(
        points_src=points_src,
        points_dst=points_dst,
        weights=weights,
        frames=frames,
    )

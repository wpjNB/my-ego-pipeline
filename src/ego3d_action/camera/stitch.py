"""Phase 4: depth-derived Sim(3) window stitching.

This is the part the reference blog does **not** publish as reusable glue code,
so it is implemented here from the described algorithm:

1. build depth-derived correspondences on the shared frames of two neighbouring
   windows (same frame, same pixel, back-projected in each window's own frame),
2. solve a robust Sim(3) ``p_dst = s R p_src + t`` with weighted Umeyama +
   RANSAC,
3. chain the transform onto the accumulated ``World-0`` alignment,
4. blend the shared frames linearly (translation) and with SLERP (rotation)
   instead of concatenating windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import StageIOError
from ..geometry.sim3 import Sim3, estimate_sim3_robust
from ..geometry.transforms import rotation_angle, rotation_slerp
from .camera_pose import normalize_to_first_camera
from .depth import build_depth_correspondences
from .window import CameraWindow

logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class StitchDiagnostics:
    """Quality report for one window-to-window alignment."""

    src_window: str
    dst_window: str
    num_correspondences: int
    inlier_ratio: float
    inlier_rmse: float
    threshold: float
    scale: float
    rotation_deg: float
    translation_norm: float
    overlap_frames: int

    def as_dict(self) -> dict[str, object]:
        return {
            "src_window": self.src_window,
            "dst_window": self.dst_window,
            "num_correspondences": self.num_correspondences,
            "inlier_ratio": self.inlier_ratio,
            "inlier_rmse": self.inlier_rmse,
            "threshold": self.threshold,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
            "translation_norm": self.translation_norm,
            "overlap_frames": self.overlap_frames,
        }


@dataclass(frozen=True)
class StitchedCamera:
    """Metric camera trajectory in ``World-0``."""

    rotation_c2w: Array  # [T, 3, 3]
    translation_c2w: Array  # [T, 3]
    valid: Array  # [T] bool
    weight: Array  # [T] number of contributing windows
    intrinsics: Array | None = None  # [T, 3, 3]
    sim3: list[Sim3] = field(default_factory=list)  # local->World0 per window
    diagnostics: list[StitchDiagnostics] = field(default_factory=list)
    num_frames: int = 0

    @property
    def coverage(self) -> float:
        return float(np.mean(self.valid)) if self.num_frames else 0.0

    def to_arrays(self) -> dict[str, Array]:
        return {
            "rotation_c2w": self.rotation_c2w,
            "translation_c2w": self.translation_c2w,
            "valid": self.valid,
            "weight": self.weight,
            **({"intrinsics": self.intrinsics} if self.intrinsics is not None else {}),
        }


def align_window_pair(
    window_src: CameraWindow,
    window_dst: CameraWindow,
    *,
    stride: int = 8,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
    min_confidence: float | None = None,
    inlier_threshold: float | None = None,
    ransac_iterations: int = 128,
    random_state: int | None = 0,
) -> tuple[Sim3, StitchDiagnostics]:
    """Solve the Sim(3) that maps ``window_src`` into ``window_dst``."""
    correspondence = build_depth_correspondences(
        window_src,
        window_dst,
        stride=stride,
        min_depth=min_depth,
        max_depth=max_depth,
        min_confidence=min_confidence,
    )
    result = estimate_sim3_robust(
        correspondence.points_src,
        correspondence.points_dst,
        correspondence.weights,
        with_scale=True,
        inlier_threshold=inlier_threshold,
        ransac_iterations=ransac_iterations,
        random_state=random_state,
    )
    diagnostics = StitchDiagnostics(
        src_window=window_src.name,
        dst_window=window_dst.name,
        num_correspondences=correspondence.count,
        inlier_ratio=result.inlier_ratio,
        inlier_rmse=result.inlier_rmse,
        threshold=result.threshold,
        scale=result.sim3.scale,
        rotation_deg=float(np.degrees(rotation_angle(result.sim3.rotation))),
        translation_norm=float(np.linalg.norm(result.sim3.translation)),
        overlap_frames=int(
            min(window_src.end, window_dst.end) - max(window_src.start, window_dst.start)
        ),
    )
    return result.sim3, diagnostics


def stitch_camera_windows(
    windows: list[CameraWindow] | tuple[CameraWindow, ...],
    *,
    num_frames: int | None = None,
    stride: int = 8,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
    min_confidence: float | None = None,
    inlier_threshold: float | None = None,
    ransac_iterations: int = 128,
    blend: bool = True,
    normalize: bool = True,
    random_state: int | None = 0,
) -> StitchedCamera:
    """Stitch overlapping camera windows into one ``World-0`` trajectory.

    Args:
        windows: window outputs in temporal order (as produced by
            :func:`ego3d_action.camera.window.make_windows`).
        num_frames: clip length; defaults to the last window's end.
        stride: depth subsampling stride used for correspondences.
        min_depth / max_depth / min_confidence: correspondence filters.
        inlier_threshold: absolute Sim(3) inlier threshold in metres; ``None``
            derives it from the residual distribution.
        ransac_iterations: hypotheses per window pair.
        blend: linearly blend the shared frames instead of hard-switching.
        normalize: re-anchor the result on the first valid camera (``World-0``).
        random_state: RANSAC seed.

    Raises:
        StageIOError: on an empty window list or overlapping-but-unordered input.
        InsufficientDataError: when a window pair cannot be aligned.
    """
    if not windows:
        raise StageIOError("stitch_camera_windows requires at least one window")
    ordered = list(windows)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur.start < prev.start:
            raise StageIOError(
                f"windows must be in temporal order: {prev.name} then {cur.name}"
            )

    total = num_frames if num_frames is not None else max(w.end for w in ordered)
    if total < max(w.end for w in ordered):
        raise StageIOError(
            f"num_frames={total} is smaller than the last window end "
            f"{max(w.end for w in ordered)}"
        )

    rotation = np.broadcast_to(np.eye(3, dtype=np.float64), (total, 3, 3)).copy()
    translation = np.zeros((total, 3), dtype=np.float64)
    intrinsics = np.zeros((total, 3, 3), dtype=np.float64)
    assigned = np.zeros(total, dtype=bool)
    weight = np.zeros(total, dtype=np.float64)

    transforms: list[Sim3] = []
    diagnostics: list[StitchDiagnostics] = []

    # First window defines the initial World-0.
    identity = Sim3.identity()
    transforms.append(identity)
    first = ordered[0]
    _write_window(
        rotation,
        translation,
        intrinsics,
        assigned,
        weight,
        first,
        identity,
        np.ones(first.window.num_frames, dtype=bool),
    )

    for index, window in enumerate(ordered[1:], start=1):
        previous = ordered[index - 1]
        local_sim3, diag = align_window_pair(
            window,
            previous,
            stride=stride,
            min_depth=min_depth,
            max_depth=max_depth,
            min_confidence=min_confidence,
            inlier_threshold=inlier_threshold,
            ransac_iterations=ransac_iterations,
            random_state=random_state,
        )
        diagnostics.append(diag)
        # Chain: local -> previous-window frame -> World-0
        chained = transforms[index - 1].compose(local_sim3)
        transforms.append(chained)

        alpha = _overlap_alpha(window, previous, blend=blend)
        _write_window(rotation, translation, intrinsics, assigned, weight, window, chained, alpha)

    if normalize:
        rotation, translation = normalize_to_first_camera(rotation, translation, assigned)

    missing = np.flatnonzero(~assigned)
    if missing.size:
        logger.warning(
            "stitched camera covers %d/%d frames; frames %s are missing",
            int(assigned.sum()),
            total,
            missing[:10].tolist() + (["..."] if missing.size > 10 else []),
        )
    else:
        logger.info("stitched camera covers all %d frames", total)

    return StitchedCamera(
        rotation_c2w=rotation,
        translation_c2w=translation,
        valid=assigned,
        weight=weight,
        intrinsics=intrinsics,
        sim3=transforms,
        diagnostics=diagnostics,
        num_frames=total,
    )


def _overlap_alpha(window: CameraWindow, previous: CameraWindow, *, blend: bool) -> Array:
    """Per-frame blend weight of ``window`` against the accumulated poses."""
    alpha = np.ones(window.window.num_frames, dtype=np.float64)
    if not blend:
        return alpha
    start = max(window.start, previous.start)
    end = min(window.end, previous.end)
    if end <= start:
        return alpha
    count = end - start
    ramp = np.linspace(0.0, 1.0, count) if count > 1 else np.ones(1, dtype=np.float64)
    alpha[start - window.start : end - window.start] = ramp
    return alpha


def _write_window(
    rotation: Array,
    translation: Array,
    intrinsics: Array,
    assigned: Array,
    weight: Array,
    window: CameraWindow,
    transform: Sim3,
    alpha: Array,
) -> None:
    """Place (or blend) one window's poses into the accumulated trajectory."""
    new_rotation, new_translation = transform.transform_poses(
        window.rotation_c2w, window.translation_c2w
    )
    for local in range(window.window.num_frames):
        frame = window.start + local
        a = float(alpha[local])
        if not assigned[frame]:
            rotation[frame] = new_rotation[local]
            translation[frame] = new_translation[local]
            intrinsics[frame] = window.intrinsics[local]
            assigned[frame] = True
            weight[frame] = a
            continue
        if a <= 0.0:
            weight[frame] += 1.0
            continue
        rotation[frame] = rotation_slerp(rotation[frame][None], new_rotation[local][None], a)[0]
        translation[frame] = (1.0 - a) * translation[frame] + a * new_translation[local]
        intrinsics[frame] = (1.0 - a) * intrinsics[frame] + a * window.intrinsics[local]
        weight[frame] += 1.0


def save_stitched_camera(path: str | Path, stitched: StitchedCamera, metadata: dict[str, object] | None = None) -> Path:
    """Persist ``camera/stitched_camera.npz`` (+ optional metadata JSON)."""
    from ..io.serialization import save_json, save_npz

    payload = stitched.to_arrays()
    target = save_npz(path, **payload)
    if metadata is not None:
        save_json(Path(path).with_suffix(".json"), metadata)
    return target


def load_stitched_camera(path: str | Path) -> StitchedCamera:
    """Load a stitched camera trajectory, validating the stage contract."""
    from ..io.serialization import load_npz

    data = load_npz(path, required=("rotation_c2w", "translation_c2w", "valid", "weight"))
    rotation = data["rotation_c2w"]
    return StitchedCamera(
        rotation_c2w=rotation,
        translation_c2w=data["translation_c2w"],
        valid=np.asarray(data["valid"], dtype=bool),
        weight=data["weight"],
        intrinsics=data.get("intrinsics"),
        num_frames=int(rotation.shape[0]),
    )


def save_sim3_transforms(path: str | Path, transforms: list[Sim3]) -> Path:
    """Persist the per-window local->World0 transforms into one ``.npy`` file."""
    if not transforms:
        raise StageIOError("no Sim(3) transforms to save")
    scales = np.array([t.scale for t in transforms], dtype=np.float64)
    rotations = np.stack([t.rotation for t in transforms], axis=0)
    translations = np.stack([t.translation for t in transforms], axis=0)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, scales=scales, rotations=rotations, translations=translations)
    return target


def load_sim3_transforms(path: str | Path) -> list[Sim3]:
    """Inverse of :func:`save_sim3_transforms`."""
    source = Path(path)
    if not source.is_file():
        raise StageIOError(f"Sim(3) transforms not found: {source}")
    with np.load(source, allow_pickle=False) as handle:
        scales = handle["scales"]
        rotations = handle["rotations"]
        translations = handle["translations"]
    return [
        Sim3(scale=float(scale), rotation=rotation, translation=translation)
        for scale, rotation, translation in zip(scales, rotations, translations, strict=False)
    ]

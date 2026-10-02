#!/usr/bin/env python
"""Phase 2: HaWoR 16/8 window reconstruction + temporal blending."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.hand import hawor  # noqa: E402
from ego3d_action.hand.temporal_blend import (  # noqa: E402
    blend_hand_windows,
    smooth_hand_trajectory,
)
from ego3d_action.io.artefacts import clip_metadata, load_detection, save_hand  # noqa: E402
from ego3d_action.io.frames import load_frame_set  # noqa: E402
from ego3d_action.runtime.sharding import (  # noqa: E402
    add_skip_existing_argument,
    add_window_selection_arguments,
    describe_selection,
    selection_from_args,
)
from ego3d_action.runtime.subprocess_backend import BackendInvocation  # noqa: E402
from ego3d_action.visualization.overlay import write_hand_video  # noqa: E402

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "hand reconstruction")
    parser.add_argument("--reuse-windows", action="store_true", help="reuse windows already on disk")
    parser.add_argument(
        "--blend-only",
        action="store_true",
        help="skip the backend and blend the windows already on disk (whole clip)",
    )
    add_window_selection_arguments(parser)
    add_skip_existing_argument(parser)
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        third_party = context.path("paths.third_party")
        weights = context.path("paths.weights")
        invocation = BackendInvocation.from_config(context.config)

        status = hawor.probe(third_party, weights)
        print(f"HaWoR backend: {status.format()}  [mode={invocation.mode}]")

        detection = load_detection(layout)
        num_frames = int(detection["valid"].shape[0])
        window = int(context.config.get("hand.window", 16))
        overlap = int(context.config.get("hand.overlap", 8))
        focal, focal_source = resolve_focal(layout, context.config)
        if focal is None:
            print(
                "WARNING: no focal length available (no camera windows, no reference "
                "camera_K, no hand.focal). HaWoR will fall back to its hard-coded "
                "600 px default, which mis-scales hand depth - do not read the "
                "result as metric. Set --set hand.focal=<px> to fix it."
            )
        else:
            print(f"hand focal: {focal:.2f} px (from {focal_source})")
        request = hawor.HaworClipRequest(
            num_frames=num_frames,
            frames_dir=layout.frames_dir,
            window=window,
            overlap=overlap,
            device=context.device,
            focal=focal,
        )
        spans = request.ranges()
        selection = selection_from_args(args)
        owned = selection.select(spans)
        if not owned:
            return fail(
                f"--shard/--window-range '{selection.describe()}' selects no HaWoR window "
                f"out of {len(spans)}"
            )
        if args.blend_only and not selection.is_whole:
            return fail(
                f"--blend-only blends the whole clip and cannot be combined with a shard "
                f"(got '{selection.describe()}')"
            )

        if args.dry_run:
            if args.blend_only:
                print(f"would blend {len(spans)} HaWoR window(s) from {layout.hand_windows_dir}")
                return 0
            print(describe_selection(selection, len(owned), len(spans)))
            print(
                f"would run HaWoR over {len(owned)} window(s) of {window} frames "
                f"(overlap {overlap}) covering {num_frames} frames [mode={invocation.mode}]"
            )
            return 0

        layout.hand_windows_dir.mkdir(parents=True, exist_ok=True)
        if args.blend_only:
            # The blend is global: it reuses whatever windows the (possibly
            # sharded) backend units wrote. Missing windows are an error - never
            # padded, interpolated or otherwise invented.
            expected = [
                layout.hand_windows_dir / f"{start:06d}_{end - 1:06d}.npz" for start, end in spans
            ]
            missing = [str(path) for path in expected if not path.is_file()]
            if missing:
                return fail(
                    f"{len(missing)} HaWoR window(s) are missing, e.g. {missing[0]}; "
                    "run the hand units (scripts/run_hand.py) first"
                )
            paths = expected
            print(f"blending {len(paths)} HaWoR window(s) from {layout.hand_windows_dir}")
        else:
            expected = [
                layout.hand_windows_dir / f"{start:06d}_{end - 1:06d}.npz" for start, end in owned
            ]
            reuse_all = args.reuse_windows or args.skip_existing
            if reuse_all and all(path.is_file() for path in expected):
                paths = expected
                print(f"reusing {len(paths)} HaWoR window(s) from {layout.hand_windows_dir}")
            else:
                paths = hawor.run_windows(
                    request,
                    layout.hand_windows_dir,
                    invocation=invocation,
                    third_party=third_party,
                    weights_root=weights,
                    detection_path=layout.detection_path,
                    camera_windows=(
                        layout.camera_windows_dir
                        if any(layout.camera_windows_dir.glob("*.npz"))
                        else None
                    ),
                    log_path=layout.hand_dir / "hawor_runner.log",
                    selection=selection,
                    skip_existing=args.skip_existing,
                    # HaWoR has its own knob: fp16 halves its 3 GiB checkpoint but
                    # trips a Double/Float mismatch inside the infiller, so the
                    # VGGT-Omega default (fp16) must not be inherited blindly.
                    precision=context.config.get(
                        "backends.hand_precision", None
                    )
                    or context.config.get("backends.precision", None),
                    crop_size=context.config.get("hand.crop_size", None),
                    box_pad=context.config.get("detection.box_padding", None),
                )
        windows = [hawor.load_hand_window(path) for path in paths]

        if not selection.is_whole:
            # A sharded unit only owns its windows; the whole-clip blend is a
            # separate unit (``--blend-only``) because it needs every window.
            print(f"hand: wrote {len(windows)} window(s) to {layout.hand_windows_dir}")
            return 0

        blended = blend_hand_windows(windows)
        # Damp the per-window reconstruction noise (rapid mesh wobble on the
        # real clips) before the artefact is written; see
        # temporal_blend.smooth_hand_trajectory. 0 disables.
        # Reference recipe: NO hand-joint smoothing (the blog measured every
# variant regressing at its quality level). The host configs opt in.
        smooth_passes = int(context.config.get("hand.smooth_passes", 0))
        joints_smooth, verts_smooth = smooth_hand_trajectory(
            blended.joints_camera,
            blended.valid,
            vertices_camera=blended.vertices_camera,
            passes=smooth_passes,
        )
        hand_arrays = {
            "joints_camera": joints_smooth,
            "valid": blended.valid,
            "confidence": blended.confidence,
            "root_rot": blended.root_rot,
            "betas": np.zeros((blended.joints_camera.shape[0], 2, 10)),
        }
        if verts_smooth is not None:
            hand_arrays["vertices_camera"] = verts_smooth
        save_hand(
            layout,
            hand_arrays,
            metadata={
                "stage": "phase2_hand",
                "backend_mode": invocation.mode,
                "window": window,
                "overlap": overlap,
                "num_windows": len(windows),
                "coverage": float(blended.valid.mean()),
                "world_frame": "camera",
                "units": "meter",
            },
        )
        print(f"hand: coverage {100.0 * blended.valid.mean():.1f}% -> {layout.hand_path}")

        if bool(context.config.get("visualization.enabled", True)):
            frames = load_frame_set(layout.data_root, layout.clip)
            intrinsics = np.broadcast_to(
                _camera_intrinsics(layout), (blended.joints_camera.shape[0], 3, 3)
            ).copy()
            faces = None if blended.vertices_camera is None else _mano_faces(context)
            write_hand_video(
                frames.paths,
                blended.joints_camera,
                intrinsics,
                blended.valid,
                layout.visualization_dir / "02_hawor.mp4",
                fps=float(context.config.get("visualization.fps") or frames.fps),
                vertices_camera=blended.vertices_camera,
                faces=faces,
            )
            print(f"wrote {layout.visualization_dir / '02_hawor.mp4'}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


def _mano_faces(context) -> list[np.ndarray]:
    """MANO triangle indices per hand (left winding mirrored) for mesh drawing."""
    from ego3d_action.hand.mano_model import load_mano_models

    models = load_mano_models(context.path("paths.mano_model"))
    right = np.asarray(models["right"].faces, dtype=np.int64)
    return [np.asarray(models["left"].faces, dtype=np.int64), right]


def _camera_intrinsics(layout) -> np.ndarray:
    """Intrinsics for the debug overlay, at the RGB frame resolution.

    The calibrated reference K wins when it is finite (official HOT3D-Clips:
    f=609 at 1408 while VGGT's estimate is 709, a 16 % stretch that pushed
    off-centre hands off their boxes); WGGT windows are the fallback for
    clips without a reference, then the synthetic default.
    """
    from ego3d_action.camera.depth import scale_intrinsics
    from ego3d_action.io.serialization import load_npz

    metadata = clip_metadata(layout)
    width = int(metadata.get("width", 320))
    height = int(metadata.get("height", 240))
    reference = layout.trajectory_dir / "ground_truth.npz"
    if reference.is_file():
        data = load_npz(reference, required=("camera_K",))
        candidate = np.asarray(data["camera_K"], dtype=np.float64)
        if np.isfinite(candidate).all():
            return candidate[0]
    for path in sorted(layout.camera_windows_dir.glob("*.npz")):
        data = load_npz(path, required=("intrinsics", "depth"))
        depth = np.asarray(data["depth"])
        source = (int(depth.shape[2]), int(depth.shape[1]))  # (width, height)
        return scale_intrinsics(
            np.asarray(data["intrinsics"])[0], source_size=source, target_size=(width, height)
        )
    from ego3d_action.testing.synthetic import make_intrinsics

    return make_intrinsics(width, height)


def resolve_focal(layout, config) -> tuple[float | None, str]:
    """Focal length in *frame* pixels for the hand stage, and where it came from.

    HaWoR reconstructs metric hands by unprojecting image crops with a focal
    length. ``hawor_motion_estimation`` hard-codes **600 px** when it is not told
    otherwise, so a 98-degree egocentric lens (f ~ 221 px at 512 px) produced
    hands at ~2.2x the correct depth and a 662 mm Action-MPJPE on HOT3D. This
    resolves a real value in priority order and reports which it used, so the
    mistake cannot happen silently again.

    Order: ``hand.focal`` in the config -> the reference trajectory's
    calibrated ``camera_K`` (finite values only; official HOT3D-Clips ship
    f=609 at 1408) -> the intrinsics of Phase 3's camera windows (VGGT's
    *estimate*, which measured 709 on the same frames - a 16 % depth bias
    that is exactly the class of error this resolution exists to avoid).
    Returns ``(None, "unavailable")`` when nothing can answer, and the caller
    then warns instead of guessing.
    """
    from ego3d_action.camera.depth import scale_intrinsics
    from ego3d_action.io.serialization import load_npz

    configured = config.get("hand.focal", None)
    if configured:
        return float(configured), "config hand.focal"

    metadata = clip_metadata(layout)
    width = int(metadata.get("width", 320))
    height = int(metadata.get("height", 240))

    reference = layout.trajectory_dir / "ground_truth.npz"
    if reference.is_file():
        data = load_npz(reference, required=("camera_K",))
        intrinsics = np.asarray(data["camera_K"], dtype=np.float64)
        candidate = float(np.median(intrinsics[:, 0, 0]))
        if np.isfinite(candidate) and candidate > 0.0:
            return candidate, "reference camera_K"
        # A reference without intrinsics (NaN camera_K, the HOT3D-Clips
        # mirror) is not an answer - fall through to the estimate.
        logger.debug("reference camera_K holds no finite focal length; ignoring it")

    for path in sorted(layout.camera_windows_dir.glob("*.npz")):
        data = load_npz(path, required=("intrinsics", "depth"))
        depth = np.asarray(data["depth"])
        source = (int(depth.shape[2]), int(depth.shape[1]))
        scaled = scale_intrinsics(
            np.asarray(data["intrinsics"])[0], source_size=source, target_size=(width, height)
        )
        candidate = float(scaled[0, 0])
        if not np.isfinite(candidate) or candidate <= 0.0:
            continue
        return candidate, "Phase 3 camera windows"

    return None, "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())

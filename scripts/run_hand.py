#!/usr/bin/env python
"""Phase 2: HaWoR 16/8 window reconstruction + temporal blending."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.hand import hawor  # noqa: E402
from ego3d_action.hand.temporal_blend import blend_hand_windows  # noqa: E402
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
                )
        windows = [hawor.load_hand_window(path) for path in paths]

        if not selection.is_whole:
            # A sharded unit only owns its windows; the whole-clip blend is a
            # separate unit (``--blend-only``) because it needs every window.
            print(f"hand: wrote {len(windows)} window(s) to {layout.hand_windows_dir}")
            return 0

        blended = blend_hand_windows(windows)
        save_hand(
            layout,
            {
                "joints_camera": blended.joints_camera,
                "valid": blended.valid,
                "confidence": blended.confidence,
                "root_rot": blended.root_rot,
                "betas": np.zeros((blended.joints_camera.shape[0], 2, 10)),
            },
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
            write_hand_video(
                frames.paths,
                blended.joints_camera,
                intrinsics,
                blended.valid,
                layout.visualization_dir / "02_hawor.mp4",
                fps=float(context.config.get("visualization.fps") or frames.fps),
            )
            print(f"wrote {layout.visualization_dir / '02_hawor.mp4'}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


def _camera_intrinsics(layout) -> np.ndarray:
    """Intrinsics for the debug overlay, scaled to the RGB frame size."""
    from ego3d_action.camera.depth import scale_intrinsics
    from ego3d_action.io.serialization import load_npz

    metadata = clip_metadata(layout)
    width = int(metadata.get("width", 320))
    height = int(metadata.get("height", 240))
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
    """Focal length in *frame* pixels for HaWoR, and where it came from.

    HaWoR reconstructs metric hands by unprojecting image crops with a focal
    length. ``hawor_motion_estimation`` hard-codes **600 px** when it is not told
    otherwise, so a 98-degree egocentric lens (f ~ 221 px at 512 px) produced
    hands at ~2.2x the correct depth and a 662 mm Action-MPJPE on HOT3D. This
    resolves a real value in priority order and reports which it used, so the
    mistake cannot happen silently again.

    Order: ``hand.focal`` in the config -> the intrinsics of Phase 3's camera
    windows (scaled to the frame size) -> the reference trajectory's ``camera_K``
    (evaluation clips only). Returns ``(None, "unavailable")`` when nothing can
    answer, and the caller then warns instead of guessing.
    """
    from ego3d_action.camera.depth import scale_intrinsics
    from ego3d_action.io.serialization import load_npz

    configured = config.get("hand.focal", None)
    if configured:
        return float(configured), "config hand.focal"

    metadata = clip_metadata(layout)
    width = int(metadata.get("width", 320))
    height = int(metadata.get("height", 240))
    for path in sorted(layout.camera_windows_dir.glob("*.npz")):
        data = load_npz(path, required=("intrinsics", "depth"))
        depth = np.asarray(data["depth"])
        source = (int(depth.shape[2]), int(depth.shape[1]))
        scaled = scale_intrinsics(
            np.asarray(data["intrinsics"])[0], source_size=source, target_size=(width, height)
        )
        return float(scaled[0, 0]), "Phase 3 camera windows"

    reference = layout.trajectory_dir / "ground_truth.npz"
    if reference.is_file():
        data = load_npz(reference, required=("camera_K",))
        intrinsics = np.asarray(data["camera_K"], dtype=np.float64)
        return float(np.median(intrinsics[:, 0, 0])), "reference camera_K"
    return None, "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())

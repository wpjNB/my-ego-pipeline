#!/usr/bin/env python
"""Phase 2 (alternative): WiLoR per-frame hand reconstruction.

Same contract as ``run_hand.py`` (HaWoR) - writes ``hand/hand_camera.npz`` and
the ``02_*`` overlay video - but reconstructs every tracked hand per frame with
WiLoR instead of HaWoR's temporal windows:

    python scripts/run_hand_wilor.py --config configs/hot3d_p100.yaml \
        --clip hot3d_ep000 --device cuda:0

Why: the 2026-10-01 benchmark table (and our own image-space checks) put WiLoR
well ahead of HaWoR on HOT3D-style egocentric data for hand pose; WiLoR is
per-frame (no temporal model), so temporal stability comes from the downstream
refine stage (UKF). The crop -> camera conversion is calibrated against ep000
(see ``backends/wilor_hand_runner.py``): physical focal + the model's root
offset, depth ratio 0.97 vs GT.

Note: the HOT3D-mirror hand GT is *not* usable for scoring hand placement
(its hands are inconsistent with its own camera poses/images - confirmed
visually on ep000 frame 300, matching the known mirror defect), so judge this
backend by image-space agreement and visual overlays, not by GT MPJPE alone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.hand.temporal_blend import smooth_hand_trajectory  # noqa: E402
from ego3d_action.io.artefacts import load_detection, save_hand  # noqa: E402
from ego3d_action.io.frames import load_frame_set  # noqa: E402
from ego3d_action.io.serialization import load_npz  # noqa: E402
from ego3d_action.runtime.subprocess_backend import (  # noqa: E402
    BackendInvocation,
    RunnerSpec,
    run_runner,
)
from ego3d_action.visualization.overlay import write_hand_video  # noqa: E402
from run_hand import _camera_intrinsics, _mano_faces, resolve_focal  # noqa: E402

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "WiLoR hand reconstruction")
    parser.add_argument("--rescale", type=float, default=None,
                        help="crop padding factor (default: hand.wilor_rescale or 2.0)")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")
        invocation = BackendInvocation.from_config(context.config)
        if invocation.is_mock:
            return fail("the WiLoR hand backend has no mock mode; run with backends.mode=real")

        third_party = context.path("paths.third_party").resolve()
        weights = context.path("paths.weights").resolve()
        checkout = third_party / "WiLoR"
        checkpoint = weights / "wilor" / "wilor_final.ckpt"
        if not checkout.is_dir() or not checkpoint.is_file():
            return fail(
                f"WiLoR backend not available: checkout {checkout} / checkpoint {checkpoint}"
            )

        detection = load_detection(layout)
        num_frames = int(np.asarray(detection["valid"]).shape[0])
        focal, focal_source = resolve_focal(layout, context.config)
        if focal is None:
            # Unlike HaWoR (which silently falls back to 600 px), WiLoR *cannot*
            # produce metric depth without a real focal - refuse rather than
            # write a mis-scaled artefact.
            return fail(
                "no focal length available (config hand.focal / Phase 3 camera "
                "windows / reference camera_K); WiLoR hand depth would not be metric"
            )
        print(f"hand focal: {focal:.2f} px (from {focal_source})")
        rescale = float(
            args.rescale
            if args.rescale is not None
            else context.config.get("hand.wilor_rescale", 2.0)
        )

        command = invocation.python_commands.get("wilor")
        if not command:
            return fail("backends.python.wilor is not configured")
        spec = RunnerSpec(
            name="wilor_hand",
            mode=invocation.mode,
            script=Path(str(invocation.backends_dir)).resolve() / "wilor_hand_runner.py",
            command=tuple(str(part) for part in command),
            timeout_seconds=float(invocation.timeout_seconds),
        )

        if args.dry_run:
            print(
                f"would reconstruct {num_frames} frames with WiLoR "
                f"(rescale {rescale}, focal {focal:.2f} px, device {context.device})"
            )
            return 0

        hand_dir = layout.hand_dir
        hand_dir.mkdir(parents=True, exist_ok=True)
        out_npz = (hand_dir / "wilor_hands.npz").resolve()
        summary = run_runner(
            spec,
            [
                "--frames", str(layout.frames_dir.resolve()),
                "--detection", str(layout.detection_path.resolve()),
                "--out", str(out_npz),
                "--num-frames", str(num_frames),
                "--third-party", str(third_party),
                "--weights", str(weights),
                "--device", str(context.device),
                "--focal", f"{focal:.6f}",
                "--rescale", f"{rescale}",
            ],
            log_path=hand_dir / "wilor_hand_runner.log",
        )
        print(f"WiLoR runner: {summary}")

        data = load_npz(
            out_npz,
            required=("joints_camera", "vertices_camera", "valid", "confidence",
                      "root_rot", "betas"),
        )
        joints = np.asarray(data["joints_camera"], dtype=np.float64)
        vertices = np.asarray(data["vertices_camera"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)

        # Same smoothing as the HaWoR path (host configs opt in via
        # hand.smooth_passes; the reference recipe keeps it off).
        smooth_passes = int(context.config.get("hand.smooth_passes", 0))
        joints_smooth, verts_smooth = smooth_hand_trajectory(
            joints, valid, vertices_camera=vertices, passes=smooth_passes
        )
        hand_arrays: dict[str, np.ndarray] = {
            "joints_camera": joints_smooth,
            "valid": valid,
            "confidence": np.asarray(data["confidence"], dtype=np.float64),
            "root_rot": np.asarray(data["root_rot"], dtype=np.float64),
            "betas": np.asarray(data["betas"], dtype=np.float64),
        }
        if verts_smooth is not None:
            hand_arrays["vertices_camera"] = verts_smooth
        save_hand(
            layout,
            hand_arrays,
            metadata={
                "stage": "phase2_hand",
                "backend": "wilor",
                "backend_mode": invocation.mode,
                "focal": float(focal),
                "focal_source": focal_source,
                "rescale": rescale,
                "smooth_passes": smooth_passes,
                "coverage": float(valid.mean()),
                "world_frame": "camera",
                "units": "meter",
            },
        )
        print(f"hand: coverage {100.0 * valid.mean():.1f}% -> {layout.hand_path}")

        if bool(context.config.get("visualization.enabled", True)):
            frames = load_frame_set(layout.data_root, layout.clip)
            intrinsics = np.broadcast_to(
                _camera_intrinsics(layout), (joints.shape[0], 3, 3)
            ).copy()
            faces = None if verts_smooth is None else _mano_faces(context)
            out_video = layout.visualization_dir / "02_wilor.mp4"
            write_hand_video(
                frames.paths,
                joints_smooth,
                intrinsics,
                valid,
                out_video,
                fps=float(context.config.get("visualization.fps") or frames.fps),
                vertices_camera=verts_smooth,
                faces=faces,
            )
            print(f"wrote {out_video}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

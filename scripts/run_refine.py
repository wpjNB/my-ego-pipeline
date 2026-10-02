#!/usr/bin/env python
"""Phase 6: targeted post-processing (gap fill, camera filter, bone scale, wrist depth, UKF+RTS)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.camera_pose import world_frame_alignment  # noqa: E402
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.fusion.trajectory import camera_joints_to_world, trajectory_metadata  # noqa: E402
from ego3d_action.io.artefacts import clip_metadata  # noqa: E402
from ego3d_action.io.serialization import load_npz, save_json, save_npz  # noqa: E402
from ego3d_action.refinement.bone_scale import correct_bone_scale  # noqa: E402
from ego3d_action.refinement.camera_filter import filter_camera_translation  # noqa: E402
from ego3d_action.refinement.gap_fill import DEFAULT_MAX_GAP, interpolate_hand_gaps  # noqa: E402
from ego3d_action.refinement.ukf_smooth import (  # noqa: E402
    DEFAULT_BETA,
    DEFAULT_Q,
    DEFAULT_R,
    smooth_hand_joints,
)
from ego3d_action.refinement.wrist_depth import optimize_wrist_depth  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "post-processing")
    parser.add_argument("--input", default=None, help="trajectory artefact (default: trajectory_raw.npz)")
    parser.add_argument("--output", default=None, help="write here (default: trajectory.npz)")
    parser.add_argument("--no-camera-filter", action="store_true")
    parser.add_argument("--no-bone-scale", action="store_true")
    parser.add_argument("--no-wrist-depth", action="store_true")
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="keep missing frames missing instead of interpolating short gaps",
    )
    parser.add_argument(
        "--no-ukf-smooth",
        action="store_true",
        help="skip the UKF + RTS temporal smoothing of the hand joints",
    )
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")

        source = Path(args.input) if args.input else layout.trajectory_raw_path
        data = load_npz(source)
        if "hand_xyz_camera" not in data:
            raise StageIOError(f"{source} has no 'hand_xyz_camera' field to refine")

        joints_camera = np.asarray(data["hand_xyz_camera"], dtype=np.float64)
        valid = np.asarray(data["hand_valid"], dtype=bool)
        confidence = np.asarray(data["hand_confidence"], dtype=np.float64)
        rotation = np.asarray(data["camera_R_c2w"], dtype=np.float64)
        translation = np.asarray(data["camera_t_c2w"], dtype=np.float64)
        intrinsics = np.asarray(data["camera_K"], dtype=np.float64)
        fps = float(clip_metadata(layout)["fps"])

        # P2: fill short detection gaps before the other corrections so bone
        # scale and wrist depth see a contiguous trajectory. Filled frames are
        # marked in hand_interpolated and enter hand_valid.
        interpolated = np.zeros(valid.shape, dtype=bool)
        gap_report: dict[str, float] = {}
        if not args.no_gap_fill:
            max_gap = int(context.config.get("refinement.gap_fill_max_frames", DEFAULT_MAX_GAP))
            filled = interpolate_hand_gaps(joints_camera, valid, confidence, max_gap=max_gap)
            joints_camera = filled.joints_camera
            confidence = filled.confidence
            valid = filled.valid
            interpolated = filled.interpolated
            gap_report = {
                "gap_fill_max_frames": float(max_gap),
                "gap_fill_gaps": float(filled.gaps_filled),
                "gap_fill_frames": float(filled.frames_filled),
            }

        refined_translation = translation
        if not args.no_camera_filter:
            refined_translation = filter_camera_translation(
                translation,
                passes=int(context.config.get("refinement.camera_filter_passes", 1)),
            )

        corrected_camera = joints_camera
        bone_report: dict[str, float] = {}
        if not args.no_bone_scale:
            bone = correct_bone_scale(
                joints_camera,
                valid,
                max_correction=float(
                    context.config.get("refinement.bone_scale_max_correction", 0.035)
                ),
                reference=str(context.config.get("refinement.bone_scale_reference", "median")),
            )
            corrected_camera = bone.joints
            bone_report = {
                "bone_max_deviation_before": float(np.nanmax(bone.max_deviation_before)),
                "bone_max_deviation_after": float(np.nanmax(bone.max_deviation_after)),
            }

        wrist_report: dict[str, float] = {}
        if not args.no_wrist_depth:
            lam = float(context.config.get("refinement.wrist_depth_lambda", 0.2))
            wrist = optimize_wrist_depth(
                corrected_camera,
                intrinsics,
                confidence,
                valid,
                lam=lam,
                max_depth_change=context.config.get("refinement.wrist_depth_max_change", None),
            )
            corrected_camera = wrist.joints_camera
            wrist_report = {
                "wrist_depth_lambda": lam,
                "wrist_depth_segments": float(wrist.segments),
            }

        # P3: constant-velocity UKF + unscented RTS over the valid frames of
        # each hand, the final temporal pass of the reference chain. Parameters
        # follow the reference UI's guidance (q up = follows the hand more,
        # r/beta up = smoother); every value is recorded in the metadata so an
        # artefact always says how it was smoothed.
        ukf_report: dict[str, float] = {}
        if not args.no_ukf_smooth:
            ukf_q = float(context.config.get("refinement.ukf_q", DEFAULT_Q))
            ukf_r = float(context.config.get("refinement.ukf_r", DEFAULT_R))
            ukf_beta = float(context.config.get("refinement.ukf_beta", DEFAULT_BETA))
            ukf_rts = bool(context.config.get("refinement.ukf_rts", True))
            smooth = smooth_hand_joints(
                corrected_camera,
                valid,
                q=ukf_q,
                r=ukf_r,
                beta=ukf_beta,
                rts=ukf_rts,
            )
            corrected_camera = smooth.joints_camera
            ukf_report = {
                "ukf_q": ukf_q,
                "ukf_r": ukf_r,
                "ukf_beta": ukf_beta,
                "ukf_rts": float(ukf_rts),
                "ukf_frames_smoothed": float(smooth.frames_smoothed.sum()),
            }

        world = camera_joints_to_world(
            corrected_camera, rotation, refined_translation, hand_valid=valid
        )

        # The 3-frame camera filter moves frame 0 slightly, so the invariant
        # "World-0 is the first frame's camera" is restored with a gauge
        # transform applied to the camera *and* the hands; no camera-relative
        # quantity (and therefore no Action-MPJPE) changes.
        if not args.no_camera_filter and bool(context.config.get("stitch.normalize_world", True)):
            r_g, t_g = world_frame_alignment(rotation, refined_translation)
            rotation = np.einsum("ij,tjk->tik", r_g, rotation)
            refined_translation = np.einsum("ij,tj->ti", r_g, refined_translation - t_g)
            world = np.einsum("ij,thnj->thni", r_g, world - t_g)

        if args.dry_run:
            print(f"would write the refined trajectory to {layout.trajectory_path}")
            return 0

        out = dict(data)
        out.update(
            {
                "hand_xyz_camera": corrected_camera,
                "hand_xyz_world": world,
                "hand_valid": valid,
                "hand_interpolated": interpolated,
                "hand_confidence": confidence,
                "camera_t_c2w": refined_translation,
            }
        )
        output_path = Path(args.output) if args.output else layout.trajectory_path
        save_npz(output_path, **out)
        if output_path == layout.trajectory_path:
            np.save(layout.world_joints_refined_path, world)
        metadata_payload = trajectory_metadata(
            fps=fps,
            world_frame=0,
            num_frames=int(joints_camera.shape[0]),
            extra={
                "stage": "phase6_refinement",
                "clip": layout.clip,
                "variant": output_path.stem,
                "camera_filter_applied": not args.no_camera_filter,
                "bone_scale_applied": not args.no_bone_scale,
                "wrist_depth_applied": not args.no_wrist_depth,
                "gap_fill_applied": not args.no_gap_fill,
                "ukf_smooth_applied": not args.no_ukf_smooth,
                **bone_report,
                **wrist_report,
                **gap_report,
                **ukf_report,
            },
        )
        if output_path == layout.trajectory_path:
            save_json(layout.trajectory_metadata_path, metadata_payload)
        save_json(output_path.with_suffix(".json"), metadata_payload)
        print(f"refined trajectory -> {output_path}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

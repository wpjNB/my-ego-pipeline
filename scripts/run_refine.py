#!/usr/bin/env python
"""Phase 6: P1 outlier rejection -> P2 gap interpolation -> P3 temporal filtering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.camera_pose import world_frame_alignment  # noqa: E402
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.fusion.trajectory import camera_joints_to_world, trajectory_metadata  # noqa: E402
from ego3d_action.io.artefacts import clip_metadata, load_hand  # noqa: E402
from ego3d_action.io.serialization import load_npz, save_json, save_npz  # noqa: E402
from ego3d_action.refinement.bone_scale import correct_bone_scale  # noqa: E402
from ego3d_action.refinement.camera_filter import filter_camera_translation  # noqa: E402
from ego3d_action.refinement.gap_fill import DEFAULT_MAX_GAP, interpolate_hand_gaps  # noqa: E402
from ego3d_action.refinement.mano_preview import build_processed_mano_preview  # noqa: E402
from ego3d_action.refinement.motion_outlier import (  # noqa: E402
    detect_motion_outliers,
    rotation_matrix_valid,
)
from ego3d_action.refinement.temporal_tracks import (  # noqa: E402
    interpolate_rotation_gaps,
    interpolate_vector_gaps,
    smooth_rotation_tracks,
)
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
    parser.add_argument("--no-outlier-reject", action="store_true", help="skip P1 motion outlier rejection")
    parser.add_argument("--no-camera-filter", action="store_true", help="skip camera translation filtering")
    parser.add_argument("--no-rotation-smooth", action="store_true", help="skip SO(3) rotation smoothing")
    parser.add_argument("--no-bone-scale", action="store_true")
    parser.add_argument("--no-wrist-depth", action="store_true")
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="skip P2 short-gap interpolation",
    )
    parser.add_argument(
        "--no-ukf-smooth",
        action="store_true",
        help="skip P3 UKF + RTS smoothing of hand joints",
    )
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")

        source = Path(args.input) if args.input else layout.trajectory_raw_path
        data = load_npz(source)
        required = (
            "hand_xyz_camera", "hand_valid", "hand_confidence", "camera_R_c2w",
            "camera_t_c2w", "camera_K",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise StageIOError(f"{source} is missing fields required by Phase 6: {missing}")

        source_joints = np.asarray(data["hand_xyz_camera"], dtype=np.float64).copy()
        joints_camera = source_joints.copy()
        valid = np.asarray(data["hand_valid"], dtype=bool).copy()
        confidence = np.asarray(data["hand_confidence"], dtype=np.float64).copy()
        rotation = np.asarray(data["camera_R_c2w"], dtype=np.float64).copy()
        translation = np.asarray(data["camera_t_c2w"], dtype=np.float64).copy()
        intrinsics = np.asarray(data["camera_K"], dtype=np.float64)
        source_root_rot = (
            np.asarray(data["mano_root_rot"], dtype=np.float64).copy()
            if "mano_root_rot" in data
            else None
        )
        mano_root_rot = None if source_root_rot is None else source_root_rot.copy()
        fps = float(clip_metadata(layout)["fps"])

        # P1: reject only bounded out-and-back spikes or isolated MAD outliers.
        hand_outlier = np.zeros(valid.shape, dtype=bool)
        hand_block_outlier = np.zeros(valid.shape, dtype=bool)
        hand_hampel_outlier = np.zeros(valid.shape, dtype=bool)
        camera_outlier = np.zeros(valid.shape[0], dtype=bool)
        camera_block_outlier = np.zeros(valid.shape[0], dtype=bool)
        camera_hampel_outlier = np.zeros(valid.shape[0], dtype=bool)
        camera_valid = (
            np.isfinite(translation).all(axis=1)
            & rotation_matrix_valid(rotation)
        )
        outlier_config = {
            "position_step_threshold": float(
                context.config.get("refinement.outlier_position_step_m", 0.15)
            ),
            "rotation_step_threshold_deg": float(
                context.config.get("refinement.outlier_rotation_step_deg", 45.0)
            ),
            "position_recovery": float(
                context.config.get("refinement.outlier_position_recovery_m", 0.15)
            ),
            "rotation_recovery_deg": float(
                context.config.get("refinement.outlier_rotation_recovery_deg", 30.0)
            ),
            "max_block": int(context.config.get("refinement.outlier_max_block", 15)),
            "hampel_k": float(context.config.get("refinement.outlier_hampel_k", 3.0)),
            "position_floor": float(
                context.config.get("refinement.outlier_position_floor_m", 0.10)
            ),
            "rotation_floor_deg": float(
                context.config.get("refinement.outlier_rotation_floor_deg", 30.0)
            ),
        }
        if not args.no_outlier_reject:
            camera_screen = detect_motion_outliers(
                translation,
                camera_valid,
                rotation=rotation,
                **outlier_config,
            )
            camera_outlier = camera_screen.outlier
            camera_block_outlier = camera_screen.block_outlier
            camera_hampel_outlier = camera_screen.hampel_outlier
            camera_valid = camera_screen.valid

        for hand_index in range(2):
            hand_candidate = (
                valid[:, hand_index]
                & np.isfinite(joints_camera[:, hand_index]).all(axis=(1, 2))
            )
            malformed = valid[:, hand_index] & ~hand_candidate
            root_track = None
            if mano_root_rot is not None:
                root_track = mano_root_rot[:, hand_index]
                root_valid = rotation_matrix_valid(root_track)
                malformed |= valid[:, hand_index] & ~root_valid
                hand_candidate &= root_valid
            if args.no_outlier_reject:
                valid[:, hand_index] = hand_candidate
            else:
                screen = detect_motion_outliers(
                    joints_camera[:, hand_index, 0],
                    hand_candidate,
                    rotation=root_track,
                    **outlier_config,
                )
                valid[:, hand_index] = screen.valid
                hand_outlier[:, hand_index] = screen.outlier | malformed
                hand_block_outlier[:, hand_index] = screen.block_outlier
                hand_hampel_outlier[:, hand_index] = screen.hampel_outlier
            rejected = ~valid[:, hand_index]
            joints_camera[rejected, hand_index] = np.nan
            confidence[rejected, hand_index] = 0.0
            if mano_root_rot is not None:
                mano_root_rot[rejected, hand_index] = np.nan

        translation[~camera_valid] = np.nan
        rotation[~camera_valid] = np.nan
        outlier_report: dict[str, object] = {
            "outlier_rejection_applied": not args.no_outlier_reject,
            "outlier_position_step_threshold_m": outlier_config["position_step_threshold"],
            "outlier_rotation_step_threshold_deg": outlier_config["rotation_step_threshold_deg"],
            "outlier_position_recovery_m": outlier_config["position_recovery"],
            "outlier_rotation_recovery_deg": outlier_config["rotation_recovery_deg"],
            "outlier_max_block_frames": outlier_config["max_block"],
            "outlier_hampel_k": outlier_config["hampel_k"],
            "hand_outlier_frames_left": int(hand_outlier[:, 0].sum()),
            "hand_outlier_frames_right": int(hand_outlier[:, 1].sum()),
            "hand_block_outliers_left": int(hand_block_outlier[:, 0].sum()),
            "hand_block_outliers_right": int(hand_block_outlier[:, 1].sum()),
            "hand_hampel_outliers_left": int(hand_hampel_outlier[:, 0].sum()),
            "hand_hampel_outliers_right": int(hand_hampel_outlier[:, 1].sum()),
            "camera_outlier_frames": int(camera_outlier.sum()),
            "camera_block_outliers": int(camera_block_outlier.sum()),
            "camera_hampel_outliers": int(camera_hampel_outlier.sum()),
        }

        # P2: interpolate hand positions linearly and camera/MANO rotations on SO(3).
        interpolated = np.zeros(valid.shape, dtype=bool)
        camera_interpolated = np.zeros(camera_valid.shape, dtype=bool)
        root_interpolated = np.zeros(valid.shape, dtype=bool)
        max_gap = int(context.config.get("refinement.gap_fill_max_frames", DEFAULT_MAX_GAP))
        gap_report: dict[str, object] = {"gap_fill_max_frames": max_gap}
        root_valid = (
            valid & rotation_matrix_valid(mano_root_rot)
            if mano_root_rot is not None
            else np.zeros(valid.shape, dtype=bool)
        )
        if not args.no_gap_fill:
            filled = interpolate_hand_gaps(
                joints_camera, valid, confidence, max_gap=max_gap
            )
            joints_camera = filled.joints_camera
            confidence = filled.confidence
            valid = filled.valid
            interpolated = filled.interpolated
            gap_report.update(
                {
                    "hand_gap_fill_gaps": int(filled.gaps_filled),
                    "hand_gap_fill_frames": int(filled.frames_filled),
                    "hand_gap_fill_left_missing": int(filled.frames_left_missing),
                }
            )
            if mano_root_rot is not None:
                root_fill = interpolate_rotation_gaps(
                    mano_root_rot, root_valid, max_gap=max_gap
                )
                mano_root_rot = root_fill.values
                root_valid = root_fill.valid
                root_interpolated = root_fill.interpolated
                gap_report["mano_root_rotation_gaps"] = int(root_fill.gaps_filled)
                gap_report["mano_root_rotation_frames"] = int(root_fill.frames_filled)

            translation_fill = interpolate_vector_gaps(
                translation, camera_valid, max_gap=max_gap
            )
            rotation_fill = interpolate_rotation_gaps(
                rotation, camera_valid, max_gap=max_gap
            )
            translation = translation_fill.values
            rotation = rotation_fill.values
            camera_valid = translation_fill.valid & rotation_fill.valid
            camera_interpolated = translation_fill.interpolated | rotation_fill.interpolated
            gap_report.update(
                {
                    "camera_translation_gaps": int(translation_fill.gaps_filled),
                    "camera_translation_frames": int(translation_fill.frames_filled),
                    "camera_rotation_gaps": int(rotation_fill.gaps_filled),
                    "camera_rotation_frames": int(rotation_fill.frames_filled),
                }
            )
        else:
            gap_report.update(
                {
                    "hand_gap_fill_gaps": 0,
                    "hand_gap_fill_frames": 0,
                    "camera_translation_gaps": 0,
                    "camera_translation_frames": 0,
                    "camera_rotation_gaps": 0,
                    "camera_rotation_frames": 0,
                }
            )

        # P3 rotation smoothing remains on SO(3). The camera translation uses
        # the project's binomial temporal filter; hand joints use UKF + RTS.
        rotation_report: dict[str, object] = {}
        rotation_strength = float(
            context.config.get("refinement.rotation_smooth_strength", 0.25)
        )
        rotation_passes = int(
            context.config.get("refinement.rotation_smooth_passes", 1)
        )
        if not args.no_rotation_smooth:
            rotation = smooth_rotation_tracks(
                rotation,
                camera_valid,
                strength=rotation_strength,
                passes=rotation_passes,
            )
            if mano_root_rot is not None:
                mano_root_rot = smooth_rotation_tracks(
                    mano_root_rot,
                    root_valid,
                    strength=rotation_strength,
                    passes=rotation_passes,
                )
            rotation_report = {
                "rotation_smoothing_applied": True,
                "rotation_smooth_strength": rotation_strength,
                "rotation_smooth_passes": rotation_passes,
            }
        else:
            rotation_report = {"rotation_smoothing_applied": False}

        refined_translation = translation
        camera_filter_report: dict[str, object] = {}
        if not args.no_camera_filter:
            camera_passes = int(
                context.config.get("refinement.camera_filter_passes", 1)
            )
            refined_translation = filter_camera_translation(
                translation, valid=camera_valid, passes=camera_passes
            )
            camera_filter_report = {
                "camera_filter_applied": True,
                "camera_filter_passes": camera_passes,
            }
        else:
            camera_filter_report = {"camera_filter_applied": False}

        corrected_camera = joints_camera
        bone_report: dict[str, object] = {}
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

        wrist_report: dict[str, object] = {}
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
                "wrist_depth_segments": int(wrist.segments),
            }

        ukf_report: dict[str, object] = {}
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
                "ukf_rts": ukf_rts,
                "ukf_frames_smoothed": int(smooth.frames_smoothed.sum()),
            }

        world_valid = valid & camera_valid[:, None]
        world = camera_joints_to_world(
            corrected_camera, rotation, refined_translation, hand_valid=world_valid
        )

        # Restore World-0 after translation filtering without changing any
        # camera-relative hand coordinate.
        if not args.no_camera_filter and bool(context.config.get("stitch.normalize_world", True)):
            r_g, t_g = world_frame_alignment(rotation, refined_translation)
            rotation = np.einsum("ij,tjk->tik", r_g, rotation)
            refined_translation = np.einsum("ij,tj->ti", r_g, refined_translation - t_g)
            world = np.einsum("ij,thnj->thni", r_g, world - t_g)

        if args.dry_run:
            print(f"would write the cleaned trajectory to {layout.trajectory_path}")
            return 0

        out = dict(data)
        out.update(
            {
                "hand_xyz_camera": corrected_camera,
                "hand_xyz_world": world,
                "hand_valid": world_valid,
                "postprocess_valid": world_valid,
                "hand_interpolated": interpolated,
                "hand_confidence": confidence,
                "hand_outlier": hand_outlier,
                "mano_root_rotation_interpolated": root_interpolated,
                "camera_valid": camera_valid,
                "camera_outlier": camera_outlier,
                "camera_interpolated": camera_interpolated,
                "camera_R_c2w": rotation,
                "camera_t_c2w": refined_translation,
            }
        )
        if mano_root_rot is not None:
            out["mano_root_rot"] = mano_root_rot

        output_path = Path(args.output) if args.output else layout.trajectory_path
        save_npz(output_path, **out)
        if output_path == layout.trajectory_path:
            np.save(layout.world_joints_refined_path, world)

        preview_generated = False
        preview_report: dict[str, object] = {}
        if output_path == layout.trajectory_path:
            source_hand = load_hand(layout)
            if "vertices_camera" in source_hand:
                preview = build_processed_mano_preview(
                    source_hand["vertices_camera"],
                    source_hand["joints_camera"],
                    corrected_camera,
                    source_hand["valid"],
                    valid,
                    hand_outlier,
                    raw_root_rot=source_hand.get("root_rot"),
                    processed_root_rot=mano_root_rot,
                    max_gap=max_gap,
                )
                preview_payload = {
                    "vertices_camera": preview.vertices_camera.astype(np.float32),
                    "joints_camera": corrected_camera.astype(np.float32),
                    "valid": preview.valid,
                    "confidence": confidence.astype(np.float32),
                    "root_rot": (
                        np.broadcast_to(np.eye(3), (len(valid), 2, 3, 3)).copy()
                        if mano_root_rot is None
                        else mano_root_rot.astype(np.float32)
                    ),
                    "betas": np.asarray(source_hand["betas"], dtype=np.float32),
                    "interpolated": preview.interpolated,
                    "root_rotation_interpolated": root_interpolated,
                    "outlier": hand_outlier,
                }
                save_npz(layout.processed_hand_path, **preview_payload)
                preview_generated = True
                preview_report = {
                    "processed_mano_preview_generated": True,
                    "processed_mano_preview_frames": int(preview.valid.sum()),
                    "processed_mano_preview_interpolated_frames": int(
                        preview.interpolated.sum()
                    ),
                    "processed_mano_preview_method": (
                        "short-gap vertex interpolation, cleaned root SO(3) update, "
                        "and wrist re-anchoring; finger articulation is preserved from HaWoR"
                    ),
                }
            else:
                layout.processed_hand_path.unlink(missing_ok=True)
                preview_report = {
                    "processed_mano_preview_generated": False,
                    "processed_mano_preview_reason": "source hand artefact has no vertices_camera",
                }

        metadata_payload = trajectory_metadata(
            fps=fps,
            world_frame=0,
            num_frames=int(joints_camera.shape[0]),
            extra={
                "stage": "phase6_refinement",
                "clip": layout.clip,
                "variant": output_path.stem,
                "pipeline_stages": [
                    "P1 motion outlier rejection",
                    "P2 bounded gap interpolation (linear positions, SLERP rotations)",
                    "P3 temporal filtering (camera binomial, hand UKF+RTS, SO(3) rotation smoothing)",
                ],
                "gap_fill_applied": not args.no_gap_fill,
                "rotation_smoothing_applied": not args.no_rotation_smooth,
                "bone_scale_applied": not args.no_bone_scale,
                "wrist_depth_applied": not args.no_wrist_depth,
                "ukf_smooth_applied": not args.no_ukf_smooth,
                "camera_valid_frames": int(camera_valid.sum()),
                "camera_interpolated_frames": int(camera_interpolated.sum()),
                "hand_interpolated_frames": int(interpolated.sum()),
                **outlier_report,
                **camera_filter_report,
                **rotation_report,
                **bone_report,
                **wrist_report,
                **gap_report,
                **ukf_report,
                **preview_report,
            },
        )
        if output_path == layout.trajectory_path:
            save_json(layout.trajectory_metadata_path, metadata_payload)
        save_json(output_path.with_suffix(".json"), metadata_payload)
        print(f"cleaned trajectory -> {output_path}")
        if preview_generated:
            print(f"processed MANO preview data -> {layout.processed_hand_path}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

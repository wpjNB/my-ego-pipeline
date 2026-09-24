#!/usr/bin/env python
"""Phase 6: targeted post-processing (camera filter, bone scale, wrist depth)."""

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
from ego3d_action.refinement.wrist_depth import optimize_wrist_depth  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "post-processing")
    parser.add_argument("--input", default=None, help="trajectory artefact (default: trajectory_raw.npz)")
    parser.add_argument("--output", default=None, help="write here (default: trajectory.npz)")
    parser.add_argument("--no-camera-filter", action="store_true")
    parser.add_argument("--no-bone-scale", action="store_true")
    parser.add_argument("--no-wrist-depth", action="store_true")
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
                **bone_report,
                **wrist_report,
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

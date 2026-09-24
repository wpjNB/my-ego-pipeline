#!/usr/bin/env python
"""Phase 5: fuse camera-space hands with camera poses into the world frame."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.stitch import load_stitched_camera  # noqa: E402
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.fusion.trajectory import build_trajectory  # noqa: E402
from ego3d_action.io.artefacts import clip_metadata, load_detection, load_hand  # noqa: E402
from ego3d_action.io.serialization import save_json, save_npz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "world fusion")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")

        hand = load_hand(layout)
        stitched = load_stitched_camera(layout.stitched_camera_path)
        detection = load_detection(layout)
        metadata = clip_metadata(layout)
        fps = float(metadata["fps"])

        if stitched.intrinsics is None:
            raise StageIOError(
                "the stitched camera artefact carries no intrinsics; re-run scripts/run_stitch.py "
                "so that camera_K can be written into the trajectory"
            )
        num_frames = int(hand["joints_camera"].shape[0])
        if stitched.num_frames < num_frames:
            raise StageIOError(
                f"stitched camera covers {stitched.num_frames} frames but the hand artefact has "
                f"{num_frames}"
            )

        arrays, traj_metadata = build_trajectory(
            joints_camera=hand["joints_camera"],
            hand_valid=np.asarray(hand["valid"], dtype=bool),
            hand_confidence=hand["confidence"],
            camera_rotation_c2w=stitched.rotation_c2w[:num_frames],
            camera_translation_c2w=stitched.translation_c2w[:num_frames],
            camera_intrinsics=stitched.intrinsics[:num_frames],
            bbox=detection["boxes"][:num_frames],
            track_id=detection["track_id"][:num_frames],
            fps=fps,
            mano_root_rot=hand["root_rot"][:num_frames],
            mano_betas=hand["betas"][:num_frames],
            postprocess_valid=np.asarray(hand["valid"], dtype=bool),
        )
        if args.dry_run:
            print(
                f"would write {num_frames} world-frame hand frames to {layout.trajectory_raw_path}"
            )
            return 0

        save_npz(layout.trajectory_raw_path, **arrays)
        np.save(layout.world_joints_raw_path, arrays["hand_xyz_world"])
        save_json(
            layout.trajectory_dir / "metadata_raw.json",
            {**traj_metadata, "stage": "phase5_fusion", "clip": layout.clip},
        )
        print(
            f"world fusion: T={num_frames}, coverage left="
            f"{100.0 * arrays['hand_valid'][:, 0].mean():.1f}% right="
            f"{100.0 * arrays['hand_valid'][:, 1].mean():.1f}% -> {layout.trajectory_raw_path}"
        )
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

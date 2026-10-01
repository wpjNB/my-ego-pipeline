#!/usr/bin/env python
"""Import HOT3D-Clips (the challenge clips layout) as pipeline clips.

    python scripts/import_hot3d_clips.py \
        --root data/hot3d_clips --data-root data/hot3d \
        --config configs/hot3d_p100.yaml --manifest-out configs/clips.hot3d_bench.yaml

Scans ``--root`` for ``<split>/<sequence>/clip-*/`` directories holding
``clip_info.json`` / ``camera.json`` / ``hands_pose.json`` / ``video.mp4``
(the layout the download script fetches) and imports every clip into the
ordinary clip layout: decoded frames, ``metadata.json`` and the reference
``trajectory/ground_truth.npz`` in the project's trajectory contract, so
``scripts/run_pipeline.py`` and ``scripts/evaluate_hot3d.py`` work unchanged.

What this source provides - and what that means for the reference:

========================================  ==========================================
``camera.json[f]``                        ``camera_R_c2w`` / ``camera_t_c2w``
``T_world_from_camera``                   (already camera-from-world; only the
                                          World-0 re-anchoring is applied)
``hands_pose.json[f][side]``              ``hand_xyz_world[t, h, 0, :]`` (wrist,
``T_world_from_wrist.translation_xyz``    literal reading - see the caveat)
``hands_pose.json[f][side]`` present      ``hand_valid[t, h]``
========================================  ==========================================

**Reference caveat (validated 2026-09-30, clip P0015_e7458eb3/clip-000000):**
the wrist key says ``T_world_from_wrist`` and the values are sequence-world
continuous across clip boundaries, but reading them as world coordinates and
transforming them with ``camera.json`` projects the wrists into the ceiling
(0/162 inside the detected hand boxes), while the pipeline's own HaWoR hands
land on the visible hands. No constant rigid transform relates the stored
wrists to the camera world - a per-clip rigid fit reaches ~13 mm on a static
clip and transfers to 0 % on a fast-rotating one - and the mirror's own
``skeleton.mp4`` renders are correct, so its renderer had access to data this
repack lost (its ``camera.json`` also drifts from VGGT's poses by 7-27 deg
exactly where the head rotates fast). **The wrist reference from this mirror
is therefore imported as-is but must not be scored**; the camera reference is
verified against the images (7 mm median vs VGGT on a static clip) and is the
only valid benchmark target here. Importing the official HOT3D-Clips package
(with ``mano_pose`` + per-frame intrinsics) through the same importer restores
the 21-joint MANO reference.

The mirror of HOT3D-Clips this imports from carries **no camera intrinsics**
and hand poses only in the **UmeTrack** parameterisation (``joint_angles``,
22 values) without MANO thetas or shape betas. The reference therefore ships
as ``hand_joints: "wrist_only"`` (joints 1..20 NaN, which the evaluation
understands) and ``camera_K`` as NaN - the pipeline never reads the
reference's intrinsics (focal resolution prefers VGGT's own), and
``camera_K`` NaN merely disables the ground-truth overlay in
``scripts/render_gt_vs_pred.py``. Re-running this importer against an
official HOT3D-Clips download (which adds ``mano_pose.thetas`` and the clip
``__hand_shapes.json__`` betas) upgrades the reference to 21-joint MANO
forward kinematics via ``ego3d_action.hand.mano_model``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.camera_pose import world_frame_alignment  # noqa: E402
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.datasets.hot3d_gt import Hot3dEpisode, write_episode  # noqa: E402
from ego3d_action.errors import Ego3DActionError, StageIOError  # noqa: E402
from ego3d_action.io.artefacts import ClipLayout  # noqa: E402
from ego3d_action.io.frames import preprocess_video  # noqa: E402
from ego3d_action.io.serialization import save_json  # noqa: E402

Array = np.ndarray


def _discover_clips(root: Path, split: str | None, sequence: str | None) -> list[Path]:
    info_paths = sorted(root.glob("*/*/clip-*/clip_info.json"))
    if split:
        info_paths = [p for p in info_paths if p.parent.parent.parent.name == split]
    if sequence:
        info_paths = [p for p in info_paths if p.parent.parent.name == sequence]
    if not info_paths:
        raise StageIOError(
            f"no clip_info.json found under {root}"
            + (f" (split={split!r})" if split else "")
            + " - is this a HOT3D-Clips download?"
        )
    return info_paths


def _rotation_from_wxyz(quat_wxyz: Array) -> Array:
    """``[T, 4]`` (w, x, y, z, world-from-camera) -> ``[T, 3, 3]`` rotations."""
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64)[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw).as_matrix()


def _fps_from_timestamps(timestamps_ns: Array) -> float:
    diffs = np.diff(timestamps_ns)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise StageIOError("clip timestamps are not increasing")
    fps = float(1e9 / np.median(positive))
    nominal = float(round(fps))
    # HOT3D records at exactly 30 fps; the ns timestamps only jitter around it.
    if nominal > 0 and abs(fps - nominal) / nominal < 0.005:
        return nominal
    return fps


def _read_clip(info_path: Path) -> tuple[dict[str, Array], dict[str, object]]:
    """Convert one clip's JSONs into the trajectory arrays + source metadata."""
    clip_dir = info_path.parent
    info = json.loads(info_path.read_text())
    frames = info["frames"]
    total = int(info.get("clip_length", len(frames)))
    if total != len(frames):
        raise StageIOError(f"{info_path}: clip_length {total} != {len(frames)} frame entries")

    camera = json.loads((clip_dir / "camera.json").read_text())
    hands = json.loads((clip_dir / "hands_pose.json").read_text())

    # camera.json is keyed by frame index and holds T_world_from_camera.
    translation = np.full((total, 3), np.nan)
    quat = np.full((total, 4), np.nan)
    for frame in frames:
        idx = int(frame["frame_idx"])
        entry = camera.get(str(idx))
        if entry is None:
            raise StageIOError(f"{clip_dir}: camera.json has no frame {idx}")
        translation[idx] = entry["translation_xyz"]
        quat[idx] = entry["quaternion_wxyz"]
    if not np.isfinite(translation).all() or not np.isfinite(quat).all():
        raise StageIOError(f"{clip_dir}: camera.json is missing frames")
    rotation_c2w = _rotation_from_wxyz(quat)
    translation_c2w = translation

    wrist_world = np.full((total, 2, 3), np.nan)
    valid = np.zeros((total, 2), dtype=bool)
    for frame in frames:
        idx = int(frame["frame_idx"])
        entry = hands.get(str(idx), {})
        for hand, side in enumerate(("left", "right")):
            pose = entry.get(side)
            if not pose:
                continue
            wrist = pose.get("T_world_from_wrist")
            if not wrist:
                continue
            wrist_world[idx, hand] = wrist["translation_xyz"]
            valid[idx, hand] = True
    # A wrist entry the mocap itself left empty is a missing annotation, not a
    # visible-hand flag: keep it invalid and let the evaluation skip the frame.
    valid &= np.isfinite(wrist_world).all(axis=2)

    timestamps = np.array([frame["timestamp_ns"] for frame in frames], dtype=np.float64)
    fps = _fps_from_timestamps(timestamps)
    return (
        {
            "rotation_c2w": rotation_c2w,
            "translation_c2w": translation_c2w,
            "wrist_world": wrist_world,
            "valid": valid,
            "timestamps": timestamps,
        },
        {
            "sequence": str(info.get("sequence_name", clip_dir.parent.name)),
            "clip_index": int(info.get("clip_index", int(clip_dir.name.split("-")[-1]))),
            "fps": fps,
        },
    )


def _build_episode(
    source: dict[str, Array],
    source_meta: dict[str, object],
    *,
    split: str,
    width: int,
    height: int,
    total: int,
) -> Hot3dEpisode:
    """Anchor to World-0 (frame 0's camera) and assemble the contract arrays."""
    rotation_c2w = source["rotation_c2w"]
    translation_c2w = source["translation_c2w"]
    # Literal reading of the source keys (see the module docstring for why this
    # reference must not be scored): the stored wrists are treated as world
    # positions and the camera-frame arrays are derived from them.
    wrist_world_raw = source["wrist_world"].copy()
    anchor_rotation, anchor_translation = world_frame_alignment(rotation_c2w, translation_c2w)
    hot3d_anchor_rotation = rotation_c2w[0].copy()
    hot3d_anchor_translation = translation_c2w[0].copy()
    rotation_c2w = np.einsum("ij,tjk->tik", anchor_rotation, rotation_c2w)
    translation_c2w = np.einsum("ij,tj->ti", anchor_rotation, translation_c2w - anchor_translation)

    valid = source["valid"]
    joints = np.full((total, 2, 21, 3), np.nan, dtype=np.float64)
    joints[:, :, 0, :] = np.einsum(
        "ij,thj->thi", anchor_rotation, wrist_world_raw - anchor_translation
    )
    joints[~valid, :, :] = np.nan  # re-assert: invalid frames stay NaN everywhere

    root_rot = np.broadcast_to(np.eye(3), (total, 2, 3, 3)).copy()
    hand_pose = np.broadcast_to(np.eye(3), (total, 2, 15, 3, 3)).copy()
    betas = np.zeros((total, 2, 10), dtype=np.float64)

    arrays: dict[str, Array] = {
        "frames": np.arange(total, dtype=np.int64),
        "timestamps": (source["timestamps"] - source["timestamps"][0]) / 1e9,
        "hand_xyz_world": joints,
        "hand_xyz_camera": np.einsum(
            "tji,thkj->thki", rotation_c2w, joints - translation_c2w[:, None, None, :]
        ),
        "hand_valid": valid,
        "hand_confidence": np.where(valid, 1.0, 0.0),
        "hand_interpolated": np.zeros(valid.shape, dtype=bool),
        "camera_R_c2w": rotation_c2w,
        "camera_t_c2w": translation_c2w,
        "camera_K": np.full((total, 3, 3), np.nan, dtype=np.float64),
        "bbox": np.full((total, 2, 4), np.nan, dtype=np.float64),
        "track_id": np.where(valid, 0, -1).astype(np.int64),
        "mano_root_rot": root_rot,
        "mano_hand_pose": hand_pose,
        "mano_betas": betas,
        "postprocess_valid": valid,
    }
    sequence = str(source_meta["sequence"])
    clip_index = int(source_meta["clip_index"])
    metadata: dict[str, object] = {
        "fps": float(source_meta["fps"]),
        "world_frame": 0,
        "hand_representation": "21_joints_metric_xyz",
        "camera_convention": "c2w",
        "units": "meter",
        "num_frames": total,
        "hot3d_world_anchor_rotation": hot3d_anchor_rotation.tolist(),
        "hot3d_world_anchor_translation": hot3d_anchor_translation.tolist(),
        "hot3d_world_anchor_note": (
            "p_hot3d_world = anchor_rotation @ p_world0 + anchor_translation"
        ),
        "source": f"hot3d_clips#{split}/{sequence}/clip-{clip_index:06d}",
        "source_split": split,
        "source_sequence": sequence,
        "source_clip_index": clip_index,
        "task": "HOT3D-Clips egocentric hand-object interaction",
        "hand_joints": "wrist_only",
        "hand_joints_note": (
            "this mirror's hand reference fails projection validation (see module "
            "docstring): T_world_from_wrist is not consistent with its own camera.json, "
            "so joints 1..20 are NaN and the wrist itself must not be scored; the "
            "official HOT3D-Clips package restores a valid reference"
        ),
        "reference_camera_validated": True,
        "reference_wrist_validated": False,
        "camera_K_note": (
            "the source carries no camera intrinsics; camera_K is NaN and focal "
            "resolution uses the Phase 3 camera windows (never the reference)"
        ),
        "bbox_available": False,
        "validity_source": "hands_pose T_world_from_wrist presence",
        "coverage_left": float(np.mean(valid[:, 0])),
        "coverage_right": float(np.mean(valid[:, 1])),
        "width": width,
        "height": height,
    }
    return Hot3dEpisode(
        arrays=arrays,
        metadata=metadata,
        num_frames=total,
        fps=float(source_meta["fps"]),
        width=width,
        height=height,
        video_path=None,
        task=str(metadata["task"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "HOT3D-Clips import")
    parser.add_argument("--root", default="data/hot3d_clips", help="HOT3D-Clips download root")
    parser.add_argument("--split", default=None, help="only import this split (e.g. valid)")
    parser.add_argument("--sequence", default=None, help="only import this source sequence")
    parser.add_argument("--limit", type=int, default=None, help="import at most N clips")
    parser.add_argument("--overwrite", action="store_true", help="re-import existing clips")
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="write a run_batch clip manifest listing every imported clip",
    )
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        data_root = Path(args.data_root) if getattr(args, "data_root", None) else Path(
            context.config.get("paths.data_root", "data/hot3d")
        )
        root = Path(args.root)
        info_paths = _discover_clips(root, args.split, args.sequence)
        if args.limit:
            info_paths = info_paths[: args.limit]
        print(f"{len(info_paths)} clips under {root}")

        imported: list[tuple[str, int]] = []
        for number, info_path in enumerate(info_paths, start=1):
            clip_dir = info_path.parent
            split = clip_dir.parent.parent.name
            sequence = clip_dir.parent.name
            source, source_meta = _read_clip(info_path)
            total = int(source["valid"].shape[0])
            name = f"{sequence}_c{source_meta['clip_index']:06d}"
            layout = ClipLayout(data_root, name)

            if layout.trajectory_dir.joinpath("ground_truth.npz").is_file() and not args.overwrite:
                print(f"[{number}/{len(info_paths)}] {name}: already imported")
                imported.append((name, total))
                continue

            if not args.dry_run:
                frames = preprocess_video(
                    clip_dir / "video.mp4", data_root, clip=name, overwrite=args.overwrite
                )
                if frames.num_frames != total:
                    raise StageIOError(
                        f"{name}: decoded {frames.num_frames} frames but clip_info has {total}"
                    )
                save_json(
                    layout.metadata_path,
                    {
                        "clip": name,
                        "source_video": str(clip_dir / "video.mp4"),
                        "source": f"hot3d_clips#{split}/{sequence}"
                        f"/clip-{source_meta['clip_index']:06d}",
                        "source_split": split,
                        "source_sequence": sequence,
                        "source_clip_index": source_meta["clip_index"],
                        "task": "HOT3D-Clips egocentric hand-object interaction",
                        "fps": source_meta["fps"],
                        "width": frames.width,
                        "height": frames.height,
                        "num_frames": frames.num_frames,
                        "duration": frames.num_frames / float(source_meta["fps"]),
                        "image_format": "jpg",
                        "frame_pattern": "%06d.jpg",
                        "hand_joints": "wrist_only",
                    },
                )

            episode = _build_episode(
                source,
                source_meta,
                split=split,
                width=frames.width if not args.dry_run else 0,
                height=frames.height if not args.dry_run else 0,
                total=total,
            )
            if args.dry_run:
                print(
                    f"[{number}/{len(info_paths)}] {name}: {total} frames, "
                    f"coverage left {100.0 * float(episode.metadata['coverage_left']):.1f}% "
                    f"right {100.0 * float(episode.metadata['coverage_right']):.1f}%"
                )
                imported.append((name, total))
                continue
            write_episode(
                episode,
                layout.trajectory_dir / "ground_truth.npz",
                layout.trajectory_dir / "ground_truth.json",
            )
            print(
                f"[{number}/{len(info_paths)}] {name}: {total} frames @ {episode.fps:.1f} fps, "
                f"{frames.width}x{frames.height}, "
                f"coverage left {100.0 * float(episode.metadata['coverage_left']):.1f}% "
                f"right {100.0 * float(episode.metadata['coverage_right']):.1f}% "
                f"(wrist_only reference)"
            )
            imported.append((name, total))

        if args.manifest_out and not args.dry_run:
            manifest = {
                "clips": [
                    {"clip": name, "num_frames": total, "from_stage": "detection"}
                    for name, total in imported
                ]
            }
            save_json(Path(args.manifest_out), manifest)
            print(f"manifest: {args.manifest_out} ({len(imported)} clips)")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

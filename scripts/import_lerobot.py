#!/usr/bin/env python
"""Import a LeRobot v3 episode as a pipeline clip (frames + HOT3D ground truth).

    python scripts/import_lerobot.py \
        --root data/samples/lerobot_v3 --episode 0 \
        --data-root data/hot3d --clip hot3d_ep000

Produces the usual clip layout plus a reference trajectory:

    data/hot3d/<clip>/frames/                  decoded egocentric RGB
    data/hot3d/<clip>/metadata.json            clip metadata (fps, size, source)
    data/hot3d/<clip>/trajectory/ground_truth.npz
    data/hot3d/<clip>/trajectory/ground_truth.json

See ``ego3d_action.datasets.hot3d_gt`` for exactly which dataset field maps to
which trajectory field. The reference carries the wrist exactly; joints 1..20
come from MANO forward kinematics when ``paths.mano_model`` (or ``--mano-model``)
points at the licence-gated model, and stay NaN without it - ``--no-mano`` forces
the wrist-only mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.datasets.hot3d_gt import convert_episode, write_episode  # noqa: E402
from ego3d_action.datasets.lerobot import LeRobotDataset  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.hand.mano_model import load_mano_models  # noqa: E402
from ego3d_action.io.frames import preprocess_video  # noqa: E402
from ego3d_action.io.serialization import save_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "LeRobot import")
    parser.add_argument("--root", default=None, help="LeRobot v3 dataset root")
    parser.add_argument("--episode", type=int, required=True, help="episode index to import")
    parser.add_argument("--no-frames", action="store_true", help="only write the ground truth")
    parser.add_argument("--overwrite", action="store_true", help="re-decode frames if present")
    parser.add_argument(
        "--mano-model",
        default=None,
        help="MANO model file or directory; enables 21-joint ground truth",
    )
    parser.add_argument(
        "--no-mano",
        action="store_true",
        help="force a wrist-only reference even when a MANO model is configured",
    )
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        if context.layout is None:
            return fail("--clip is required")
        layout = context.layout
        root = Path(args.root or context.config.get("paths.lerobot_root", ""))
        if not root or str(root) in {".", ""}:
            return fail("--root is required (or set paths.lerobot_root in the config)")

        dataset = LeRobotDataset(root)
        mano_path = None if args.no_mano else (
            args.mano_model or context.config.get("paths.mano_model", None)
        )
        mano_models = load_mano_models(mano_path) if mano_path else None
        if mano_models:
            print(
                "MANO models: "
                + ", ".join(
                    f"{hands}={model.source}{' (mirrored)' if model.mirrored else ''}"
                    for hands, model in mano_models.items()
                )
            )
        else:
            if args.no_mano:
                print("--no-mano: wrist-only reference (joints 1..20 stay NaN)")
            else:
                print(
                    "no MANO model configured -> wrist-only reference "
                    "(set paths.mano_model or pass --mano-model)"
                )
        episode = convert_episode(dataset, args.episode, mano_models=mano_models)
        print(
            f"episode {args.episode}: {episode.num_frames} frames, "
            f"{episode.width}x{episode.height} @ {episode.fps:.1f} fps"
        )
        if episode.task:
            print(f"  task: {episode.task}")
        print(
            f"  coverage: left {100.0 * episode.metadata['coverage_left']:.1f}% "
            f"right {100.0 * episode.metadata['coverage_right']:.1f}%"
        )
        print(f"  reference: {episode.metadata['hand_joints']}")

        if args.dry_run:
            print(f"would write {layout.trajectory_dir / 'ground_truth.npz'}")
            return 0

        layout.ensure_dirs()
        if not args.no_frames:
            frames = preprocess_video(
                episode.video_path,
                context.data_root,
                clip=layout.clip,
                overwrite=args.overwrite,
            )
            if frames.num_frames != episode.num_frames:
                return fail(
                    f"decoded {frames.num_frames} frames but the episode has {episode.num_frames}"
                )
            save_json(
                layout.metadata_path,
                {
                    "clip": layout.clip,
                    "source_video": str(episode.video_path),
                    "source": episode.metadata["source"],
                    "dataset_root": str(root),
                    "task": episode.task,
                    "fps": episode.fps,
                    "width": frames.width,
                    "height": frames.height,
                    "num_frames": frames.num_frames,
                    "duration": frames.num_frames / episode.fps,
                    "image_format": "jpg",
                    "frame_pattern": "%06d.jpg",
                    "stage": "phase0_preprocess",
                    "hand_joints": episode.metadata["hand_joints"],
                },
            )

        trajectory, metadata = write_episode(
            episode,
            layout.trajectory_dir / "ground_truth.npz",
            layout.trajectory_dir / "ground_truth.json",
        )
        print(f"ground truth -> {trajectory}")
        print(f"metadata     -> {metadata}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

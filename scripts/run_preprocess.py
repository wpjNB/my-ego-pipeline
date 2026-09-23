#!/usr/bin/env python
"""Phase 0: ``video.mp4 -> frames/ + metadata.json``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.io.frames import preprocess_video  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "preprocess")
    parser.add_argument("video", help="path to the egocentric RGB video")
    parser.add_argument("--overwrite", action="store_true", help="re-decode frames even if present")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required (or derive it from the video name)")
        layout.ensure_dirs()
        if args.dry_run:
            print(f"would decode {args.video} -> {layout.frames_dir}")
            return 0
        frames = preprocess_video(
            args.video,
            context.data_root,
            clip=layout.clip,
            image_format=str(context.config.get("preprocess.image_format", "jpg")),
            overwrite=args.overwrite,
        )
        print(f"{frames.clip}: {frames.num_frames} frames @ {frames.fps:.3f} fps -> {frames.directory}")
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

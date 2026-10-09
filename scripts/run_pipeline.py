#!/usr/bin/env python
"""End-to-end orchestrator.

Stages run as separate subprocesses against the on-disk artefact contract
(``data/<clip>/...``), which is also the boundary that keeps the model backends
in their own environments. The orchestrator itself only needs the base env.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.vggt_omega import probe as probe_vggt  # noqa: E402
from ego3d_action.cli import base_parser, build_context, fail  # noqa: E402
from ego3d_action.detection.wilor import probe as probe_wilor  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.hand.hawor import probe as probe_hawor  # noqa: E402
from ego3d_action.runtime.subprocess_backend import BackendInvocation  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    requires_backend: str | None = None


# The hand stage needs Phase 3's estimated focal and Phase 4's stitched
# World-0 camera. HaWoR runs its infiller against one continuous camera path;
# raw VGGT windows each have their own gauge and jump at window boundaries.
STAGES: tuple[Stage, ...] = (
    Stage("phase0-preprocess", "run_preprocess.py"),
    Stage("phase1-detection", "run_detection.py", "WiLoR"),
    Stage("phase3-camera", "run_camera.py", "VGGT-Omega"),
    Stage("phase4-stitch", "run_stitch.py"),
    Stage("phase2-hand", "run_hand.py", "HaWoR"),
    Stage("phase5-fusion", "run_fusion.py"),
    Stage("phase6-refine", "run_refine.py"),
)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "pipeline")
    parser.add_argument("--video", default=None, help="source video (required unless --from-stage is used)")
    parser.add_argument(
        "--from-stage",
        default=None,
        help="resume from a stage name, e.g. phase4-stitch (skips earlier stages)",
    )
    parser.add_argument("--num-frames", type=int, default=None, help="clip length for camera/stitch stages")
    args = parser.parse_args(argv)

    try:
        context = build_context(args)
        layout = context.layout
        if layout is None:
            return fail("--clip is required")

        print(f"environment: {context.environment.format()}")
        invocation = BackendInvocation.from_config(context.config)
        print(f"backend mode: {invocation.mode}")
        third_party = context.path("paths.third_party")
        weights = context.path("paths.weights")
        statuses = {
            "WiLoR": probe_wilor(third_party, weights),
            "HaWoR": probe_hawor(third_party, weights),
            "VGGT-Omega": probe_vggt(third_party, weights),
        }
        for name, status in statuses.items():
            print(f"  {name}: {'available' if status.available else 'missing ' + str(list(status.missing))}")

        skip = args.from_stage is not None
        for stage in STAGES:
            if skip:
                skip = stage.name != args.from_stage
                if skip:
                    continue
            if args.dry_run:
                blocker = ""
                if (
                    not invocation.is_mock
                    and stage.requires_backend
                    and not statuses[stage.requires_backend].available
                ):
                    blocker = f"  (blocked: {stage.requires_backend} is not installed)"
                print(f"would run {stage.name}: scripts/{stage.script}{blocker}")
                continue
            code = run_stage(stage, args, layout.clip)
            if code != 0:
                return fail(f"stage {stage.name} failed with exit code {code}", code=code)
        return 0
    except Ego3DActionError as exc:
        return fail(str(exc))


def run_stage(stage: Stage, args: object, clip: str) -> int:
    """Execute one stage as a subprocess, forwarding the shared CLI options."""
    command = [sys.executable, str(SCRIPTS_DIR / stage.script)]
    command += ["--config", str(args.config), "--clip", clip]  # type: ignore[attr-defined]
    if getattr(args, "data_root", None):
        command += ["--data-root", args.data_root]  # type: ignore[attr-defined]
    if getattr(args, "device", None):
        command += ["--device", args.device]  # type: ignore[attr-defined]
    if getattr(args, "log_level", None):
        command += ["--log-level", args.log_level]  # type: ignore[attr-defined]
    for override in getattr(args, "overrides", []) or []:
        command += ["--set", override]
    if stage.name == "phase0-preprocess" and getattr(args, "video", None):
        command.append(args.video)
    if stage.name in {"phase3-camera", "phase4-stitch"} and getattr(args, "num_frames", None):
        command += ["--num-frames", str(args.num_frames)]

    print(f"==> {stage.name}: {' '.join(command)}", flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())

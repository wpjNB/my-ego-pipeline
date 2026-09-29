#!/usr/bin/env python
"""Batch pipeline dispatch across heterogeneous hosts (M1).

    # dry run: show the unit plan (no model, no GPU)
    python scripts/run_batch.py --manifest configs/clips.example.yaml \
        --config configs/mock.yaml --shards 2 --dry-run

    # local replay of an already-preprocessed clip, split into 4 window shards
    python scripts/run_batch.py --manifest clips.yaml --config configs/mock.yaml \
        --hosts configs/hosts.local.yaml --shards 4 --max-parallel 2 --skip-existing

The GPU-heavy stages (Phase 2 hand, Phase 3 camera) are sliced into independent
window shards and dispatched onto the hosts whose declared capabilities match
(see ``runtime/executor.py``). Every unit gets a content-addressed provenance
marker, so a second run with ``--skip-existing`` recomputes nothing, and a
failure marks its clip ``degraded`` instead of fabricating output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import configure_logging, fail, parse_overrides  # noqa: E402
from ego3d_action.config import load_config, validate_config  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.runtime.batch import (  # noqa: E402
    SHARDABLE_STAGES,
    BatchRunner,
    build_parser,
    build_plan,
    load_manifest,
)
from ego3d_action.runtime.executor import HostCapabilities, load_hosts  # noqa: E402
from ego3d_action.runtime.sharding import WindowSelection, assert_partition  # noqa: E402


def default_local_host(data_root: str, repo_root: Path) -> HostCapabilities:
    """A single local host running the stage scripts in this interpreter."""
    return HostCapabilities(
        name="local",
        executor="local",
        backends=("wilor", "hawor", "vggt"),
        python={"wilor": ("python",), "hawor": ("python",), "vggt": ("python",)},
        cuda="cuda",
        gpu_count=1,
        gpu_memory_gb=24.0,
        data_root=data_root,
        repo_root=str(repo_root),
        orchestrator_python=(sys.executable,),
        max_parallel=1,
    )


def verify_shards(plan: object, config: dict[str, object]) -> None:
    """Prove that each sharded stage's units partition its global schedule.

    Runs before dispatch so an off-by-one in the scheduler is a hard error, not
    a silently missing window in the final trajectory.
    """
    groups: dict[tuple[str, str, int], list[WindowSelection]] = {}
    for unit in plan.units:  # type: ignore[attr-defined]
        if unit.stage not in SHARDABLE_STAGES or unit.selection.is_whole:
            continue
        key = (unit.clip, unit.stage, unit.num_windows)
        groups.setdefault(key, []).append(unit.selection)
    for (clip, stage, total), selections in sorted(groups.items()):
        assert_partition(total, selections)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configure_logging(args.log_level)
        if args.shards < 1:
            return fail(f"--shards must be >= 1, got {args.shards}")

        config = load_config(args.config)
        if args.overrides:
            config = config.with_overrides(parse_overrides(args.overrides))
            validate_config(config)

        repo_root = Path(__file__).resolve().parents[1]
        data_root = str(args.data_root or config.get("paths.data_root", "data"))
        outputs_root = Path(args.outputs or config.get("paths.outputs", "outputs"))
        report_path = Path(args.report) if args.report else outputs_root / "batch_report.json"

        clips = load_manifest(args.manifest)
        if args.from_stage:
            for clip in clips:
                clip.from_stage = args.from_stage

        stages = None
        if args.stages:
            stages = [part.strip() for part in args.stages.split(",") if part.strip()]

        frames_override: dict[str, int] = {}
        if args.num_frames:
            frames_override = {clip.clip: args.num_frames for clip in clips}

        plan = build_plan(
            clips,
            config=config.data,
            shards=args.shards,
            stages=stages,
            frames_override=frames_override,
        )
        verify_shards(plan, config.data)

        hosts = load_hosts(args.hosts) if args.hosts else [default_local_host(data_root, repo_root)]

        if args.dry_run:
            print(f"plan: {len(plan.units)} unit(s) over {len(clips)} clip(s), {len(hosts)} host(s)")
            for unit in plan.units:
                windows = (
                    f"{len(unit.selection.ordinal_indices(unit.num_windows))}/{unit.num_windows} windows"
                    if unit.stage in SHARDABLE_STAGES and not unit.selection.is_whole
                    else unit.stage
                )
                print(f"  {unit.clip:24s} {unit.stage:10s} {unit.selection.describe():24s} {windows}")
            for stage in SHARDABLE_STAGES:
                totals = sorted(
                    {unit.num_windows for unit in plan.units if unit.stage == stage and not unit.selection.is_whole}
                )
                if totals:
                    print(f"  {stage}: global schedule {totals[0]} window(s), {args.shards} shard(s)")
            return 0

        runner = BatchRunner(
            hosts=hosts,
            config_path=str(args.config),
            data_root=data_root,
            layout_root=Path(data_root),
            max_parallel=args.max_parallel,
            stream_output=args.stream_output,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff,
            timeout_seconds=args.timeout,
            repo_root=str(repo_root),
        )
        results = runner.run_plan(plan, skip_existing=args.skip_existing)
        written = runner.write_report(report_path)

        ok = sum(1 for r in results if r.status == "ok")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = [r for r in results if r.status == "failed"]
        for result in failed:
            print(f"FAILED {result.unit.name} ({result.unit.stage}) on {result.host}: {result.detail}")

        print(
            f"batch: {ok} ok, {skipped} skipped, {len(failed)} failed -> {written}"
        )
        if runner.degraded_clips():
            print(f"degraded clips: {', '.join(runner.degraded_clips())}")
        return 1 if failed else 0
    except Ego3DActionError as exc:
        return fail(str(exc))
    except (ValueError, KeyError) as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
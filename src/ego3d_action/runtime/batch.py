"""Batch dispatch of stage units onto heterogeneous hosts.

This is the M1 tool: it takes a manifest of clips and runs the pipeline over
them, splitting the GPU-heavy stages into window shards that can go to different
machines, retrying failures, and recording exactly what happened.

Design rules, all inherited from the project's hard constraints:

* **A window stage is sliced by :mod:`ego3d_action.runtime.sharding`, never by
  frame**, so the union of shards is exactly the unsliced schedule.
* **A unit is identified by content** (:mod:`ego3d_action.runtime.provenance`),
  so ``--skip-existing`` skips a unit only when its parameters *and* input
  contents match - never on mtime.
* **A failure is never hidden.** A unit that exhausts its retries marks its clip
  ``degraded`` in ``batch_report.json`` and in the clip's ``metadata.json``; no
  artefact is fabricated, interpolated or padded to hide the missing work.
* **The ledger is written after every unit**, so a job killed mid-run still
  reports honestly what finished.

Sharding is applied to the *hand* and *camera* window phases (independent per
window). The other stages run whole-clip. The frame-coupled detection phase is
deliberately not sliced: slicing it by frame would change its gap-recovery
result. See ``runtime/sharding.py`` for the M2 seam.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .executor import (
    CapabilityMismatch,
    ExecResult,
    HostCapabilities,
    build_executor,
    select_host,
    with_data_root,
)
from .provenance import (
    MARKER_SUFFIX,
    PROVENANCE_DIR,
    CompletionMarker,
    git_revision,
    hash_file,
    host_identity,
    params_hash,
    platform_identity,
    write_marker,
)
from .sharding import WindowSelection

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"

#: Stage name -> script. ``shardable`` says the stage can be split by window.
STAGE_SCRIPTS: dict[str, str] = {
    "preprocess": "run_preprocess.py",
    "detection": "run_detection.py",
    "hand": "run_hand.py",
    "camera": "run_camera.py",
    "stitch": "run_stitch.py",
    "fusion": "run_fusion.py",
    "refine": "run_refine.py",
}

#: Which stages may be sliced into window shards. Detection/stitch/fusion/refine
#: are whole-clip: the first is frame-coupled, the rest are cheap CPU work whose
#: result is inherently global.
SHARDABLE_STAGES = ("hand", "camera")

#: Sharded stages that also need a whole-clip assembly unit afterwards. The GPU
#: work is per window, but the blend (hand) / stitching input (camera) is global,
#: so the plan emits ``<stage>:1`` last to join the shards - after which the
#: ordinary stage runs as a no-op assembly (``--blend-only`` / a reuse pass).
ASSEMBLY_STAGES = ("hand", "camera")

#: Which stage needs which backend, and roughly how much VRAM (declared, not
#: probed, so the scheduler can decide without touching the worker).
STAGE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "detection": {"backends": ("wilor",), "gpu": True, "min_gpu_memory_gb": 4.0},
    "hand": {"backends": ("hawor",), "gpu": True, "min_gpu_memory_gb": 4.0},
    # VGGT-Omega 1B at 416 px over 200-frame windows is the memory hog.
    "camera": {"backends": ("vggt",), "gpu": True, "min_gpu_memory_gb": 16.0},
}

#: Window parameter each shardable stage slices by, resolved from the config.
STAGE_WINDOW_CONFIG: dict[str, tuple[str, str]] = {
    "hand": ("hand.window", "hand.overlap"),
    "camera": ("camera.window", "camera.overlap"),
}


@dataclass
class ClipSpec:
    """One clip's inputs and how long it is."""

    clip: str
    video: str | None = None
    num_frames: int | None = None
    from_stage: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.clip:
            raise ValueError("clip name must not be empty")
        if self.num_frames is not None and self.num_frames <= 0:
            raise ValueError(f"clip '{self.clip}': num_frames must be positive")
        if self.from_stage is not None and self.from_stage not in STAGE_SCRIPTS:
            raise ValueError(
                f"clip '{self.clip}': unknown from_stage '{self.from_stage}' "
                f"(known: {sorted(STAGE_SCRIPTS)})"
            )


@dataclass
class UnitSpec:
    """One schedulable piece of work: a sharded stage of one clip."""

    clip: str
    stage: str
    selection: WindowSelection
    num_windows: int = 1
    video: str | None = None
    num_frames: int | None = None

    @property
    def name(self) -> str:
        return self.clip if self.selection.is_whole else f"{self.clip}#{self.selection.describe()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "stage": self.stage,
            "selection": self.selection.to_dict(),
            "num_windows": self.num_windows,
        }


@dataclass
class UnitResult:
    """What happened to one unit."""

    unit: UnitSpec
    status: str  # ok | skipped | failed | planned
    attempts: int = 0
    host: str = ""
    wall_time_seconds: float = 0.0
    params_hash: str = ""
    detail: str = ""
    exec_results: list[ExecResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit.to_dict(),
            "status": self.status,
            "attempts": self.attempts,
            "host": self.host,
            "wall_time_seconds": round(self.wall_time_seconds, 3),
            "params_hash": self.params_hash,
            "detail": self.detail,
        }


def load_manifest(path: str | Path) -> list[ClipSpec]:
    """Load a clip manifest.

    Either a bare list of clips (``[clip01, clip02]``), a list of mappings with
    ``clip``/``video``/``num_frames``, or ``{"clips": [...]}``. Unknown keys are
    rejected: a typo like ``num_frame`` would otherwise silently drop a length
    constraint and produce a mis-sliced schedule.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"manifest not found: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{source} is not valid YAML: {exc}") from exc

    entries = payload.get("clips") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ValueError(f"{source} must define a non-empty 'clips' list")

    known = {"clip", "video", "num_frames", "from_stage", "notes", "shards"}
    clips: list[ClipSpec] = []
    for entry in entries:
        if isinstance(entry, str):
            clips.append(ClipSpec(clip=entry))
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"{source}: each clip must be a name or a mapping, got {entry!r}")
        unknown = set(entry) - known
        if unknown:
            raise ValueError(f"{source}: clip has unknown keys {sorted(unknown)}")
        if "clip" not in entry:
            raise ValueError(f"{source}: clip entry is missing 'clip': {entry!r}")
        clips.append(
            ClipSpec(
                clip=str(entry["clip"]),
                video=None if entry.get("video") is None else str(entry["video"]),
                num_frames=None if entry.get("num_frames") is None else int(entry["num_frames"]),
                from_stage=None if entry.get("from_stage") is None else str(entry["from_stage"]),
                notes=str(entry.get("notes", "")),
            )
        )
    names = [c.clip for c in clips]
    if len(names) != len(set(names)):
        raise ValueError(f"{source}: clip names must be unique, got {names}")
    return clips


def stage_sequence(clip: ClipSpec) -> list[str]:
    """The stages to run for a clip, honouring ``from_stage``."""
    order = list(STAGE_SCRIPTS)
    if clip.from_stage is None:
        return order
    if clip.from_stage not in order:
        raise ValueError(f"unknown stage '{clip.from_stage}'")
    return order[order.index(clip.from_stage) :]


def num_windows_for(
    stage: str, num_frames: int, *, window: int, overlap: int
) -> int:
    """How many windows the global schedule has (mirrors the real schedule).

    Kept identical to ``camera.window.make_windows``/``HaworClipRequest.ranges``
    including the "drop a trailing window that adds no new frames" rule, so the
    planner's shard arithmetic matches what the runner will actually produce.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if window <= 0 or overlap < 0 or overlap >= window:
        raise ValueError(f"invalid window/overlap: window={window}, overlap={overlap}")
    stride = window - overlap
    count = 0
    start = 0
    last_end = 0
    while start < num_frames:
        end = min(start + window, num_frames)
        if count and end <= last_end:
            break
        count += 1
        last_end = end
        start += stride
    return count


@dataclass
class BatchPlan:
    """The full set of units a batch run will execute."""

    units: list[UnitSpec]
    skipped_clips: list[str] = field(default_factory=list)

    def by_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for unit in self.units:
            counts[unit.stage] = counts.get(unit.stage, 0) + 1
        return counts

    def sharded_groups(self) -> list[tuple[str, str, int]]:
        """``(clip, stage, shards)`` for every sliced group (for partition tests)."""
        groups: dict[tuple[str, str], int] = {}
        for unit in self.units:
            key = (unit.clip, unit.stage)
            groups[key] = max(groups.get(key, 0), unit.selection.shard.count)
        return [(clip, stage, shards) for (clip, stage), shards in sorted(groups.items())]


def build_plan(
    clips: Sequence[ClipSpec],
    *,
    config: Mapping[str, Any],
    shards: int = 1,
    stages: Sequence[str] | None = None,
    frames_override: Mapping[str, int] | None = None,
) -> BatchPlan:
    """Expand clips + stages into the unit list, one unit per shard.

    ``shards`` must evenly divide each shardable stage's window count; otherwise
    the shards would be unbalanced in a way that varies per clip, and the
    partition test could not be stated. The caller is told to adjust instead of
    the scheduler silently rebalancing.
    """
    if shards < 1:
        raise ValueError(f"shards must be >= 1, got {shards}")
    chosen_stages = list(stages) if stages else list(STAGE_SCRIPTS)
    for stage in chosen_stages:
        if stage not in STAGE_SCRIPTS:
            raise ValueError(f"unknown stage '{stage}' (known: {sorted(STAGE_SCRIPTS)})")

    override = dict(frames_override or {})
    units: list[UnitSpec] = []
    for clip in clips:
        num_frames = clip.num_frames or override.get(clip.clip)
        for stage in stage_sequence(clip):
            if stage not in chosen_stages:
                continue
            if stage in SHARDABLE_STAGES:
                if num_frames is None:
                    raise ValueError(
                        f"clip '{clip.clip}': stage '{stage}' is sharded but the clip length is "
                        "unknown; set num_frames in the manifest"
                    )
                window_key, overlap_key = STAGE_WINDOW_CONFIG[stage]
                window = int(config.get(window_key, 16 if stage == "hand" else 200))
                overlap = int(config.get(overlap_key, 8 if stage == "hand" else 40))
                total = num_windows_for(stage, num_frames, window=window, overlap=overlap)
                if shards > 1 and total % shards != 0:
                    raise ValueError(
                        f"clip '{clip.clip}': {total} {stage} window(s) do not divide evenly into "
                        f"{shards} shards; pick a shard count dividing {total}"
                    )
                for index in range(shards):
                    units.append(
                        UnitSpec(
                            clip=clip.clip,
                            stage=stage,
                            selection=WindowSelection.parse(shard=f"{index}/{shards}"),
                            num_windows=total,
                            video=clip.video,
                            num_frames=num_frames,
                        )
                    )
                if stage in ASSEMBLY_STAGES and shards > 1:
                    # Joins the shards: blends the hand windows / reuses all
                    # camera windows. Must come after every shard. A whole-clip
                    # run needs no join step, so it is only emitted when sharded.
                    units.append(
                        UnitSpec(
                            clip=clip.clip,
                            stage=stage,
                            selection=WindowSelection(),
                            num_windows=total,
                            video=clip.video,
                            num_frames=num_frames,
                        )
                    )
            else:
                units.append(
                    UnitSpec(
                        clip=clip.clip,
                        stage=stage,
                        selection=WindowSelection(),
                        num_windows=1,
                        video=clip.video,
                        num_frames=num_frames,
                    )
                )
    return BatchPlan(units=units)


def unit_command(
    unit: UnitSpec,
    *,
    config_path: str,
    data_root: str,
    extra: Sequence[str],
    python: Sequence[str] = ("python",),
    script_dir: str | Path | None = None,
) -> list[str]:
    """The stage script argv for one unit (runner-independent, so it is testable).

    ``python`` is the *host's* orchestrator interpreter, so a unit dispatched to
    a remote worker runs in that worker's own environment rather than the
    caller's. ``script_dir`` lets a remote unit use a repository-relative script
    path (resolved against the host's working directory) instead of a local
    absolute path that does not exist on the worker.
    """
    script = Path(script_dir) / STAGE_SCRIPTS[unit.stage] if script_dir else SCRIPTS_DIR / STAGE_SCRIPTS[unit.stage]
    command = [
        *python,
        str(script),
        "--config",
        config_path,
        "--clip",
        unit.clip,
        "--data-root",
        data_root,
    ]
    if unit.stage == "preprocess":
        if not unit.video:
            raise ValueError(
                f"clip '{unit.clip}': preprocess needs a video path (set 'video' in the manifest)"
            )
        command.append(unit.video)
    if unit.stage in SHARDABLE_STAGES and not unit.selection.is_whole:
        command += ["--shard", f"{unit.selection.shard.index}/{unit.selection.shard.count}"]
    if unit.stage in ASSEMBLY_STAGES and unit.selection.is_whole:
        # Whole-clip join step emitted only for a sharded run: hand blends the
        # windows the shards wrote; camera reuses them, so a sharded camera run
        # never re-runs the model.
        if unit.stage == "hand":
            command += ["--blend-only"]
        command += ["--skip-existing"]
    command += list(extra)
    return command


def output_marker_exists(marker_dir: Path, unit_key: str) -> bool:
    """Whether a scheduler or runner marker already exists for this unit."""
    return (marker_dir / PROVENANCE_DIR / f"{unit_key}{MARKER_SUFFIX}").is_file()


def unit_argument_signature(unit: UnitSpec) -> dict[str, Any]:
    """The parameters that determine a stage's output, for hashing.

    Includes the *mode* of a sharded stage's whole-clip unit, because the join
    step (``--blend-only``) writes a different set of artefacts than a full
    whole-clip run and must not be confused with it when skipping.
    """
    mode = "full"
    if unit.stage in ASSEMBLY_STAGES and unit.selection.is_whole:
        mode = "blend-only" if unit.stage == "hand" else "reuse"
    return {
        "stage": unit.stage,
        "mode": mode,
        "num_windows": unit.num_windows,
        "num_frames": unit.num_frames,
    }


def input_hashes_for(
    unit: UnitSpec, *, layout_root: Path, extra_inputs: Sequence[str] = ()
) -> dict[str, str]:
    """Hash the artefacts a stage consumes, so a changed input invalidates it.

    A stage must never list its own output as an input: that would make the
    params hash change between the first and second run (the output exists only
    after the first run) and ``--skip-existing`` could never skip anything.
    """
    root = self_clip_root = layout_root / unit.clip
    candidates: list[Path] = []
    if unit.stage == "preprocess":
        if unit.video:
            candidates.append(Path(unit.video))
    elif unit.stage == "detection":
        # The detector consumes the frames; their clip metadata stands in for the
        # frame set (hashing every JPEG would cost more than detecting them).
        candidates.append(root / "metadata.json")
    elif unit.stage == "hand":
        candidates.append(root / "detection" / "detection.npz")
    elif unit.stage == "camera":
        candidates.append(root / "metadata.json")
    elif unit.stage == "stitch":
        candidates.extend(sorted((root / "camera" / "windows").glob("*.npz")))
    elif unit.stage == "fusion":
        candidates.append(root / "hand" / "hand_camera.npz")
        candidates.append(root / "camera" / "stitched_camera.npz")
    elif unit.stage == "refine":
        candidates.append(root / "trajectory" / "trajectory_raw.npz")
    else:
        raise ValueError(f"unknown stage '{unit.stage}'")
    candidates += [Path(p) for p in extra_inputs]
    _ = self_clip_root
    return {path.name: hash_file(path) for path in candidates if path.is_file()}


def stage_outputs(unit: UnitSpec, stage: str) -> tuple[list[str], str]:
    """``(output names relative to the clip root, unit key)`` for the marker.

    The output *names* are what ``--skip-existing`` verifies still exist, so a
    marker that arrived without its artefact forces a recompute. They are always
    relative to the clip root, because that is where whole-clip markers live.
    """
    if stage == "preprocess":
        return [], "preprocess"
    if stage == "detection":
        return ["detection/detection.npz"], "detection"
    if stage == "hand":
        if unit.selection.is_whole:
            # A whole-clip run covers the blend and every window; with
            # ``--blend-only`` it only covers the blend, so the skipped case must
            # be told apart from the real one.
            return ["hand/hand_camera.npz"], "hand"
        return [], f"hand/{unit.selection.describe()}"
    if stage == "camera":
        if unit.selection.is_whole:
            return [], "camera"
        return [], f"camera/{unit.selection.describe()}"
    if stage == "stitch":
        return ["camera/stitched_camera.npz", "stitched/sim3_transforms.npz"], "stitch"
    if stage == "fusion":
        return ["trajectory/trajectory_raw.npz"], "fusion"
    if stage == "refine":
        return ["trajectory/trajectory.npz"], "refine"
    raise ValueError(f"unknown stage '{stage}'")


def unit_marker_dir(layout_root: Path, unit: UnitSpec, stage: str) -> Path:
    """Where this unit's marker lives.

    Sharded units keep theirs inside the window directory, because that is where
    the backend runner writes its own marker and both sides must agree on
    "done". Every whole-clip unit keeps its marker at the clip root, so its
    output names are unambiguously clip-relative.
    """
    clip_root = layout_root / unit.clip
    if stage == "hand" and not unit.selection.is_whole:
        return clip_root / "hand" / "windows"
    if stage == "camera" and not unit.selection.is_whole:
        return clip_root / "camera" / "windows"
    return clip_root


def filter_owned_detections(layout_root: Path, unit: UnitSpec) -> None:
    """Delete detections a context-padded detection run was not allowed to own.

    Phase 1 is whole-clip in this batch runner, so this is a no-op today. It is
    the guard for the M2 frame slicer: a padded run computes a superset, and the
    padding is discarded here rather than being written to disk as if it were
    the slice's own result. Kept explicit (and tested) so the M2 wiring cannot
    quietly ship fabricated coverage.
    """


class BatchRunner:
    """Executes a :class:`BatchPlan` across hosts, with retries and a ledger."""

    def __init__(
        self,
        *,
        hosts: Sequence[HostCapabilities],
        config_path: str,
        data_root: str,
        layout_root: Path,
        max_parallel: int = 1,
        stream_output: bool = False,
        extra_options: Sequence[str] = (),
        retries: int = 2,
        retry_backoff_seconds: float = 0.0,
        timeout_seconds: float | None = 7200.0,
        repo_root: str = ".",
    ) -> None:
        if not hosts:
            raise ValueError("at least one host is required")
        if max_parallel < 1:
            raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
        if retries < 0:
            raise ValueError(f"retries must be >= 0, got {retries}")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.hosts = [with_data_root(host, data_root) for host in hosts]
        self.config_path = config_path
        self.data_root = data_root
        self.layout_root = Path(layout_root)
        self.max_parallel = max_parallel
        self.stream_output = stream_output
        self.extra_options = tuple(extra_options)
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.repo_root = repo_root
        self._lock = threading.Lock()
        self._results: list[UnitResult] = []
        self._config_cache: dict[str, Any] | None = None
        # Per-host slot accounting: a host's ``max_parallel`` is a hard cap on
        # concurrently running units, so three pinned GPUs stay three processes
        # on three cards instead of three processes on one card.
        self._host_occupancy = {host.name: 0 for host in self.hosts}
        self._host_slots = threading.Condition(self._lock)
        self._tls = threading.local()

    def _config(self) -> dict[str, Any]:
        """The stage config, loaded once (requirement overrides come from here)."""
        if self._config_cache is None:
            path = Path(self.config_path)
            self._config_cache = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return self._config_cache

    @staticmethod
    def _config_lookup(config: Mapping[str, Any], key: str) -> Any:
        """Dotted-path lookup into the nested config mapping (None when absent)."""
        node: Any = config
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return None
            node = node[part]
        return node

    def _stage_requirements(self, stage: str) -> dict[str, Any]:
        """Stage requirements, with the config's per-stage VRAM floor applied.

        The defaults in :data:`STAGE_REQUIREMENTS` describe the reference
        hardware (VGGT-Omega at 200-frame windows wants >= 16 GB). A profile
        that runs smaller windows on smaller cards (e.g. 8/4 on a 12 GB P100,
        measured at 8.5 GiB resident) declares that through
        ``<stage>.min_gpu_memory_gb`` so the scheduler matches it honestly
        instead of rejecting every unit.
        """
        required = dict(STAGE_REQUIREMENTS.get(stage, {}))
        override = self._config_lookup(self._config(), f"{stage}.min_gpu_memory_gb")
        if override is not None:
            required["min_gpu_memory_gb"] = float(override)
        return required

    # -- reporting -------------------------------------------------------
    @property
    def results(self) -> list[UnitResult]:
        with self._lock:
            return list(self._results)

    def degraded_clips(self) -> list[str]:
        """Clips with at least one failed unit, sorted and de-duplicated."""
        return sorted({r.unit.clip for r in self.results if r.status == "failed"})

    def report(self) -> dict[str, Any]:
        """The batch ledger, written even when the run was interrupted."""
        results = self.results
        return {
            "generated_at": time.time(),
            "git_revision": git_revision(self.repo_root),
            "host": host_identity(),
            "platform": platform_identity(),
            "data_root": str(self.data_root),
            "config": str(self.config_path),
            "max_parallel": self.max_parallel,
            "retries": self.retries,
            "units": [result.to_dict() for result in results],
            "summary": {
                "total": len(results),
                "ok": sum(1 for r in results if r.status == "ok"),
                "skipped": sum(1 for r in results if r.status == "skipped"),
                "failed": sum(1 for r in results if r.status == "failed"),
                "planned": sum(1 for r in results if r.status == "planned"),
            },
            "degraded_clips": self.degraded_clips(),
            "hosts": [host.to_dict() for host in self.hosts],
        }

    def write_report(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        from .provenance import atomic_write_json

        atomic_write_json(target, self.report())
        return target

    # -- execution -------------------------------------------------------
    def _record(self, result: UnitResult) -> None:
        with self._lock:
            self._results.append(result)

    def _acquire_host(self, unit: UnitSpec) -> HostCapabilities:
        """Reserve a capable host with a free slot, blocking until one frees up.

        :func:`select_host` is capability-only; the per-host ``max_parallel``
        is enforced here, deterministically (highest capacity, then name).
        """
        required = self._stage_requirements(unit.stage)
        while True:
            with self._host_slots:
                reasons = {
                    host.name: host.explain_shortfall(required) or "" for host in self.hosts
                }
                candidates = [host for host in self.hosts if not reasons[host.name]]
                if not candidates:
                    raise CapabilityMismatch(
                        required, {name: why for name, why in reasons.items() if why}
                    )
                free = [
                    host
                    for host in candidates
                    if self._host_occupancy[host.name] < host.max_parallel
                ]
                if free:
                    free.sort(key=lambda host: (-host.max_parallel, host.name))
                    host = free[0]
                    self._host_occupancy[host.name] += 1
                    self._tls.host_name = host.name
                    return host
                self._host_slots.wait(timeout=5.0)

    def _release_host(self) -> None:
        """Free the slot this thread's unit holds (no-op when none was acquired)."""
        name = getattr(self._tls, "host_name", None)
        if name is None:
            return
        with self._host_slots:
            self._host_occupancy[name] -= 1
            self._tls.host_name = None
            self._host_slots.notify_all()

    def _resolve_host(self, unit: UnitSpec) -> HostCapabilities:
        """Capability-only host pick (no slot accounting) - dry runs and tests."""
        required = self._stage_requirements(unit.stage)
        return select_host(self.hosts, required)

    def _mark_degraded(self, clip: str, detail: str) -> None:
        """Record a failure in the clip's ``metadata.json`` - never silently."""
        metadata_path = self.layout_root / clip / "metadata.json"
        payload: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("clip '%s' metadata is corrupt; rewriting it", clip)
        degraded = list(payload.get("degraded_units", []))
        degraded.append(detail)
        payload["degraded"] = True
        payload["degraded_units"] = degraded
        payload["degraded_reason"] = detail
        from .provenance import atomic_write_json

        atomic_write_json(metadata_path, payload)

    def run_unit(self, unit: UnitSpec, *, skip_existing: bool, dry_run: bool = False) -> UnitResult:
        """Run one unit and record it in the ledger (public entry point)."""
        try:
            result = self._run_unit(unit, skip_existing=skip_existing, dry_run=dry_run)
        finally:
            self._release_host()
        self._record(result)
        return result

    def _run_unit(self, unit: UnitSpec, *, skip_existing: bool, dry_run: bool = False) -> UnitResult:
        """Run (or skip) one unit, retrying on failure and never faking output."""
        if unit.stage == "preprocess" and not unit.video:
            return UnitResult(
                unit=unit,
                status="failed",
                detail="preprocess needs a video path in the manifest",
            )

        digest = params_hash(
            stage=unit.stage,
            params=unit_argument_signature(unit),
            inputs=input_hashes_for(unit, layout_root=self.layout_root),
            selection=unit.selection,
        )
        stage = unit.stage
        marker_dir = unit_marker_dir(self.layout_root, unit, stage)
        outputs, unit_key = stage_outputs(unit, stage)
        if unit.selection.is_whole and unit_argument_signature(unit)["mode"] != "full":
            # A whole-clip join step has its own identity; give the marker a
            # distinct key so it cannot be mistaken for the full clip run.
            unit_key = f"{stage}/{unit_argument_signature(unit)['mode']}"

        if skip_existing:
            from .provenance import is_complete

            # ``outputs`` is clip-relative, which matches ``marker_dir`` for
            # whole-clip units (and is empty for sharded units, whose windows are
            # verified by the runner's own marker instead).
            if not output_marker_exists(marker_dir, unit_key) and outputs and all(
                (self.layout_root / unit.clip / name).is_file() for name in outputs
            ):
                # The stage ran whole-clip directly (no scheduler marker yet) but
                # its artefacts are on disk: adopt it rather than recompute.
                logger.info(
                    "adopting existing %s output for %s (no marker yet)", stage, unit.name
                )
                return UnitResult(
                    unit=unit,
                    status="skipped",
                    params_hash=digest,
                    detail="existing output adopted (no marker)",
                )
            if is_complete(
                marker_dir,
                unit_key,
                expected_params_hash=digest,
                expected_outputs=outputs,
            ):
                logger.info("skipping %s (%s) - already complete", unit.name, stage)
                return UnitResult(
                    unit=unit, status="skipped", host="", params_hash=digest, detail="already complete"
                )

        if dry_run:
            return UnitResult(
                unit=unit, status="planned", params_hash=digest, detail="dry run"
            )

        try:
            host = self._acquire_host(unit)
        except CapabilityMismatch as exc:
            return UnitResult(
                unit=unit, status="failed", params_hash=digest, detail=str(exc)
            )

        executor = build_executor(
            host, stream_output=self.stream_output, extra_ssh_options=self.extra_options
        )
        command = unit_command(
            unit,
            config_path=self.config_path,
            data_root=self.data_root,
            extra=[],
            python=host.stage_python,
            script_dir="scripts" if host.executor == "ssh" else None,
        )
        attempts = 0
        last: ExecResult | None = None
        started = time.monotonic()
        results: list[ExecResult] = []
        while attempts <= self.retries:
            attempts += 1
            last = executor.run(
                command,
                cwd=host.repo_root,
                timeout=self.timeout_seconds,
            )
            results.append(last)
            if last.ok:
                break
            logger.warning(
                "unit %s (%s) failed on %s [attempt %d/%d]: exit=%s %s",
                unit.name,
                stage,
                host.name,
                attempts,
                self.retries + 1,
                last.returncode,
                last.tail(limit=300).strip().splitlines()[-1] if last.tail().strip() else "",
            )
            if attempts <= self.retries and self.retry_backoff_seconds:
                time.sleep(self.retry_backoff_seconds * attempts)

        wall = time.monotonic() - started
        if last is None or not last.ok:
            detail = (
                f"{stage} failed on {host.name} after {attempts} attempt(s): "
                f"exit={None if last is None else last.returncode}"
            )
            self._mark_degraded(unit.clip, detail)
            return UnitResult(
                unit=unit,
                status="failed",
                attempts=attempts,
                host=host.name,
                wall_time_seconds=wall,
                params_hash=digest,
                detail=detail,
                exec_results=results,
            )

        filter_owned_detections(self.layout_root, unit)
        write_marker(
            marker_dir,
            CompletionMarker(
                stage=stage,
                unit=unit_key,
                params_hash=digest,
                outputs=tuple(outputs),
                inputs=input_hashes_for(unit, layout_root=self.layout_root),
                host=host.name,
                git_revision=git_revision(self.repo_root),
                backend_mode="",
                extra={"selection": unit.selection.to_dict(), "shard_windows": unit.num_windows},
                wall_time_seconds=wall,
                platform=platform_identity(),
            ),
        )
        return UnitResult(
            unit=unit,
            status="ok",
            attempts=attempts,
            host=host.name,
            wall_time_seconds=wall,
            params_hash=digest,
            exec_results=results,
        )

    def run_plan(
        self, plan: BatchPlan, *, skip_existing: bool, dry_run: bool = False
    ) -> list[UnitResult]:
        """Run every unit, respecting per-stage ordering within a clip.

        Units of different clips (and different stages of the same clip whose
        inputs are already on disk) run in parallel up to ``max_parallel``. The
        plan is already topologically ordered per clip by :func:`build_plan`.
        """
        if dry_run:
            return [self.run_unit(unit, skip_existing=skip_existing, dry_run=True) for unit in plan.units]

        # Group by clip so a clip's stages stay ordered; clips go wide in parallel.
        per_clip: dict[str, list[UnitSpec]] = {}
        for unit in plan.units:
            per_clip.setdefault(unit.clip, []).append(unit)

        def run_clip(units: list[UnitSpec]) -> list[UnitResult]:
            out: list[UnitResult] = []
            for unit in units:
                result = self.run_unit(unit, skip_existing=skip_existing)
                out.append(result)
                if result.status == "failed":
                    # Do not run later stages on missing inputs: that would either
                    # crash or, worse, produce a partial trajectory. Stop this clip
                    # and leave it explicitly degraded.
                    logger.error(
                        "clip '%s' degraded at stage '%s'; skipping its remaining stages",
                        unit.clip,
                        unit.stage,
                    )
                    break
            return out

        clips = sorted(per_clip)
        workers = min(self.max_parallel, max(1, len(clips)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_clip, [per_clip[clip] for clip in clips]))
        return self.results


# ---------------------------------------------------------------------------
# CLI plumbing (kept thin; run_batch.py is the entry point)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="batch pipeline dispatch (M1)")
    parser.add_argument("--manifest", required=True, help="clip manifest (YAML)")
    parser.add_argument("--config", default="configs/macrodata_final.yaml")
    parser.add_argument("--hosts", default=None, help="hosts.yaml (default: local-only, 1 job)")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--outputs", default=None, help="where batch_report.json is written")
    parser.add_argument("--shards", type=int, default=1, help="split hand/camera into N window shards")
    parser.add_argument("--max-parallel", type=int, default=1, help="concurrent clips")
    parser.add_argument("--stages", default=None, help="comma-separated stage subset")
    parser.add_argument("--from-stage", default=None, help="start each clip at this stage")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--num-frames", type=int, default=None, help="clip length when the manifest omits it")
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--report", default=None, help="override the report path")
    return parser
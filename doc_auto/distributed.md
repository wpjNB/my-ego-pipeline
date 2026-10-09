# Distributed / heterogeneous batch execution (M1)

Last modified: 2026-10-06 16:07 (+08:00)

The pipeline is already location-transparent where it matters: stages talk only
through the on-disk artefact contract (`io/artefacts.py`) and the three model
backends only through `runtime/subprocess_backend.py` (subprocess + one JSON
summary line). What M1 adds is the missing trio - **scheduling, idempotency and
declared heterogeneity** - without changing a single stage's output.

Everything here is CPU-testable with `backends.mode: mock`.

## What M1 ships

| Piece | Module | Purpose |
| --- | --- | --- |
| Work slicing | `runtime/sharding.py` | `--shard i/N` / `--window-range a-b`; a partition of the *global* schedule |
| Content-addressed identity | `runtime/provenance.py` | `params_hash` over parameters + input *contents*; `.provenance/<unit>.done.json` |
| Where a stage runs | `runtime/executor.py` | declared host capabilities + `local`/`ssh` executors (`slurm`/`k8s` are the documented next branch) |
| Batch dispatch | `runtime/batch.py`, `scripts/run_batch.py` | clip manifest -> unit plan -> parallel dispatch, retries, degraded marking, ledger |

Run it:

```bash
make dry-batch SHARDS=4                 # print the unit plan
make batch SHARDS=4 PARALLEL=2          # local, mock backend by default
# or directly:
python scripts/run_batch.py --manifest configs/clips.example.yaml \
    --config configs/macrodata_final.yaml --hosts configs/hosts.example.yaml \
    --shards 4 --max-parallel 2 --skip-existing
```

## Which stages may be sliced

The choice is a data-dependency decision, not a convenience one.

Each clip runs detection, camera-window inference, camera stitching, hand
reconstruction, fusion and refinement in that order. HaWoR consumes the
estimated focal from Phase 3 and the stitched World-0 trajectory from Phase 4.
Its hand windows can be parallelised after those clip-wide camera inputs exist;
camera and hand phases are not peers.

| Stage | Sliced by | Why |
| --- | --- | --- |
| Phase 2 hand (HaWoR 16/8) | window | Windows are independent; `HaworClipRequest.ranges()` owns the schedule |
| Phase 3 camera (VGGT 200/40) | window | Windows are independent (VGGT never sees across a window) |
| Phase 1 detection | **not sliced** | The tracker recovers a same-side gap *across* frames, so a frame slice has less context and could change the result. `backends/wilor_runner.py` rejects `--shard`/`--frame-range`/`--window-range` with that reason instead of ignoring them |
| Phases 4-7 (stitch/fusion/refine) | whole clip | Cheap CPU work whose result is inherently global |

A sharded `hand`/`camera` group therefore gets **one extra whole-clip unit after
the shards**: `hand` runs `--blend-only` (blend the windows the shards wrote),
`camera` reuses its windows. The scheduler never runs a later stage on a clip
whose earlier stage failed.

## The partition guarantee

Shards are derived from the **global** window/overlap parameters and ownership is
interleaved (`ordinal % count == index`), which makes the partition
order-independent and balanced. `scripts/run_batch.py` calls
`sharding.assert_partition` before dispatching, so a scheduler bug is a hard
error rather than a missing window in the final trajectory.

`tests/test_batch_e2e.py` proves the stronger statement on real numpy content: a
2-shard mock run produces window files that are **array-for-array identical** to
an unsliced run, and the blend assembled from the shards equals the whole-clip
blend.

## Idempotency and provenance

A unit's identity is

```
params_hash = sha256(stage, parameters, input contents, shard selection)
```

* hashing is over **file bytes**, never mtime or size (clocks and mtimes are not
  comparable across machines);
* the shard selection is part of the identity, so re-sharding cannot reuse a
  wrong result;
* a stage never hashes its own output as an input;
* HaWoR inference hashes detection, clip metadata, all camera windows and the
  stitched camera; a blend-only unit hashes the input hand windows. Changed
  camera poses therefore invalidate stale MANO output;
* `.provenance/<unit>.done.json` records `{params_hash, outputs, inputs, host,
  git_revision, backend_mode, wall_time, platform}`;
* `--skip-existing` reuses a unit only when the params hash matches **and** every
  recorded output still exists - a marker that arrived without its artefact
  (partial rsync) forces a recompute.

A marker found on disk but not yet written by the scheduler (e.g. an ordinary
whole-clip `scripts/run_hand.py` run) is *adopted*: logs say so, and nothing is
recomputed.

## Hosts and capability matching

`configs/hosts.example.yaml` shows the intended shape: the orchestrator is
CPU-only; Phase 1/2/3 land on three different workers because each declares the
backend env it has and the VRAM it can spare. `runtime/executor.py` matches a
stage's requirements (`backends`, `gpu`, `min_gpu_memory_gb`) against the
declarations *without logging in*. When nothing matches, the error names every
host's shortfall.

Rules that stay true for the later milestones:

* conda environments are **not** shipped between machines (an env on NFS is
  unusable) - each host's `python` points at its own env;
* ~11 GB of weights stay resident per host; only frames and artefacts travel;
* never wrap CUDA calls in a thread pool or `multiprocessing` (a forked CUDA
  context breaks); cross-machine work goes through processes/services;
* `local` and `ssh` only. `slurm`/`k8s` plug into
  `executor.build_executor` without the scheduler above noticing.

## Failure semantics (project hard constraint)

* a failed unit is retried `retries + 1` times, then its clip is marked
  `degraded` in `outputs/batch_report.json` **and** in the clip's
  `metadata.json`;
* later stages of that clip are skipped;
* **no artefact is fabricated, interpolated or padded** to hide the missing
  work - verified by `tests/test_batch.py::test_failed_unit_writes_no_marker_or_artefact`;
* the ledger is written after the run with every unit's status, attempt count,
  host, params hash and wall time, so a reviewer can tell which commit, which
  host and which weights produced a number.

## Tests

| File | What it pins |
| --- | --- |
| `tests/test_sharding.py` | partition/union/no-duplicate for both schedules; malformed flags rejected |
| `tests/test_provenance.py` | content-not-mtime hashing; stale/corrupt/incomplete markers never skip |
| `tests/test_batch.py` | capability matching; retry counts; degraded marking; zero computation on re-run; report contents |
| `tests/test_batch_e2e.py` | sharded run == unsliced run (arrays), idempotent second run, ssh command construction |

## Not in M1 (next milestones)

* **M2** - `slurm`/`k8s` executors, `rsync`/S3 transport, a detection frame
  slicer with `FrameRange` context padding (the type exists and is tested; only
  the wiring is missing), per-host placement policies.
* **M3** - resident inference services (`backends/serve_*.py`) so the HaWoR 3 GB /
  VGGT 4.3 GB checkpoints load once per worker instead of once per clip, plus a
  1/2/4-GPU scaling table.
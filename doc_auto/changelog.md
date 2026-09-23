# Changelog

## 2026-09-23 17:22 (+08:00) - runnable end to end on CPU (runner protocol + mock backend)

* `runtime/subprocess_backend.py`: the runner protocol
  (`BackendInvocation`, `RunnerSpec`, `run_runner`) with typed
  `BackendExecutionError` for non-zero exits, timeouts, launch failures and
  malformed output; JSON summaries are parsed from the last stdout line.
* `backends/`: `mock_backend.py` (deterministic stand-in, four subcommands:
  `wilor`, `hawor`, `vggt`, `truth`) and the three real runners
  (`wilor_runner.py`, `hawor_runner.py`, `vggt_runner.py`) with full argument
  handling, artefact writing and a GPU-free `--check` mode. Only `run_model()`
  remains to be written against the backend APIs.
* Adapters now actually invoke their backend: `wilor.detect_clip`,
  `hawor.run_windows` (one process for the whole clip, model loaded once),
  `vggt_omega.run_window` (same), each validating the artefacts it receives.
* `configs/mock.yaml` + `backends.mode` config plumbing; `validate_config`
  rejects an unknown mode and a real run without interpreters.
* `testing/synthetic.py`: deterministic scene, camera, hands, detections and
  windows shared by the mock backend and the test-suite.
* `scripts/demo_mock_pipeline.sh` + `make demo`: Phases 0-7 in ~10 s on CPU.
  Latest run: 300 frames, pipeline wall time 5.65 s -> 53.10 FPS,
  Action-MPJPE 24.64 mm (raw) / 25.90 mm (refined), coverage 90.67 %,
  wrist error 22.92 -> 16.06 mm. See `doc_auto/ablation.md`.
* Two gauge bugs found and fixed while building this: the mock reference
  trajectory was expressed in the raw world frame instead of `World-0`, and the
  camera translation filter moved frame 0 away from the origin; both are now
  documented in `doc_auto/architecture.md`.
* Test suite: **193 passed** (~13 s), including a full end-to-end mock run.

## 2026-09-23 17:12 (+08:00) - project bootstrap

* Created the `ego3d_base` conda environment (Python 3.11, numpy/scipy/opencv/
  pytest/matplotlib) and verified it on the CPU-only laptop.
* Added the `src/ego3d_action` package: io, detection, hand, camera, geometry,
  fusion, refinement, evaluation, visualization, runtime, config, cli.
* Implemented Phases 0, 4, 5, 6 and 7 plus the model-free half of Phase 1.
* Added `scripts/` entry points for every phase, the orchestrator, and the
  `environment-*.yml` backend specs for the GPU server.
* Added the unit/integration test suite; it passes end to end on CPU.
* Documented the on-disk contract, coordinate conventions and device handling.

## 2026-09-23 17:06 (+08:00) - verification and GPU-free demo

* Test suite: **166 passed** on CPU (`conda run -n ego3d_base python -m pytest -q`,
  ~6 s), covering geometry, tracking, blending, stitching on a synthetic scene,
  the trajectory contract, ffmpeg IO, visualisation and the Action-MPJPE
  protocol.
* Phase 0 smoke test on a synthesised clip: 60 frames decoded, numbered from
  `000000.jpg` (`scripts/smoke_test.sh`).
* Added `scripts/demo_synthetic.py`: renders two overlapping windows whose local
  frames differ by a known Sim(3) (scale 1.35, 29.2 deg, |t| 1.197 m) and runs
  the real stitcher. Recovered scale 0.7407 (= 1/1.35), 100 % inliers, rmse
  0.000 mm; the stitched trajectory matches ground truth to < 0.001 mm and
  < 0.0001 deg. Outputs live in `outputs/demo/`.

## 2026-09-23 17:05 (+08:00) - git repository note

The workspace root ships a read-only `.git` mount point, so `git init` cannot
write into it. Repository metadata therefore lives in `.gitstore/` (excluded by
`.gitignore`); use `GIT_DIR=.gitstore GIT_WORK_TREE=$PWD git ...`, or move the
project to a normal directory. Work happens on branch
`feature/pipeline-bootstrap`; nothing is pushed to a protected branch.

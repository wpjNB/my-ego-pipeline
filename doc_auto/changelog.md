# Changelog

## 2026-09-23 18:40 (+08:00) - one script downloads every weight

* `weights.manifest.yaml`: declares every asset (id, backend, destination,
  mirror list, size floor, expected container format, sha256 slot, optional /
  manual flags, source page and follow-up command).
* `runtime/weights.py` + `scripts/download_weights.py`: resumable
  (`.part` + HTTP `Range`), verified (size, sha256, format sniffing that does
  not need torch) and atomic (a file only appears after it passes; corrupt
  downloads are quarantined as `*.part.bad`). Supports `https://`, `hf://` and
  `file://` (air-gapped mirrors), `--dry-run`, `--only`, `--verify-only`,
  `--force`, `--url-override` and `--json`.
* Honest by construction: MANO (licence-gated) and any entry whose URL is still
  a `<placeholder>` are never fetched - the report prints the page, the exact
  filename, the destination and the next command. The manifest header records
  that its URLs could not be verified from this machine.
* `doctor.py` fix hints now point at `download_weights.py --only <backend>`.
* Test suite: **268 passed** (19 new: manifest validation, format sniffing,
  verification, resume from a partial file, checksum quarantining, mirror
  fallback, unreachable/unsupported sources and the CLI on a `file://` mirror).

## 2026-09-23 18:25 (+08:00) - setup + runbook, and an asset audit tool

* `runtime/doctor.py` + `scripts/doctor.py`: audits host tools, python
  dependencies, config validity, the three backends (checkout + weights, and
  optionally each runner's `--check` executed through its configured
  interpreter) and the data on disk. Never raises; every failing check carries
  the fix; `--json`/`--strict` for CI.
* Asset audit result on this machine: `ego3d_base` complete, **all model
  checkpoints and checkouts absent**, MANO absent (only `mano_mean_params.npz`
  exists in the HaWoR/VITRA clones). CPU path ready, GPU path needs the
  downloads.
* `doc_auto/setup.md`: prerequisites, base env, the three backend envs, checkout
  commands, the weights table (what/where/source/needed-for), MANO conversion,
  environment variables and known constraints.
* `doc_auto/runbook.md`: Path A (CPU/mock) and Path B (GPU/real) step by step
  with the exact commands, expected artefacts per stage, what "done" looks like,
  a troubleshooting table and the API drift to expect on the first real run.
* README leads with the honest status table and links both documents.
* Test suite: **249 passed**.

## 2026-09-23 18:05 (+08:00) - MANO forward kinematics, backend runners written

* `hand/mano_model.py`: numpy MANO forward kinematics (shape blend shapes, pose
  blend shapes, kinematic tree, linear blend skinning) plus the 21-landmark
  mapping (16 MANO joints + 5 standard fingertip vertices) into this project's
  joint convention, model loading from `.npz`/`.pkl`, right-to-left mirroring
  (mesh *and* pose conjugation) and a topology sanity check. Tested against a
  synthetic model with the same structure: 13 tests pin the mapping slot by
  slot, per-finger isolation of local rotations, root placement, shape scaling
  and mirroring.
* `datasets/hot3d_gt.py` gained an optional MANO model: with one, the HOT3D
  reference becomes a full 21-joint reference (`hand_joints: mano_fk`), with the
  wrist still placed exactly where the dataset put it; without one it stays
  wrist-only. Verified on episode 0: 21 joints per valid hand-frame, exact wrist,
  stable bone lengths, landmarks projected onto the real hands.
* `scripts/convert_mano.py` (official pickle -> `.npz`, needs chumpy in the HaWoR
  env) and `scripts/make_synthetic_mano.py` (clearly-labelled stand-in for
  plumbing runs; `--mano-model` wiring in `scripts/import_lerobot.py` and
  `scripts/demo_hot3d_sample.sh`).
* Backend runners are now written against the real APIs, with every conversion
  moved into the package and unit-tested:
  `detection/wilor.py::detections_from_predictions` / `build_raw_detection_arrays`,
  `hand/hawor.py::hawor_tracks_from_detection` / `hand_windows_from_joints`,
  `camera/vggt_omega.py::camera_window_from_output`, plus VGGT pose-encoding
  decoding with intrinsics rescaled to the depth grid.
* Two load-bearing findings recorded in the docs: our conservative tracker now
  drives HaWoR through its `model_tracks.npy` seam (replacing `thresh=0.2`), and
  `hawor_infiller` hard-depends on HaWoR's own SLAM output (hands come back in
  its world frame and are converted to camera space by the runner).
* Test suite: **239 passed** (~12 s).

## 2026-09-23 17:36 (+08:00) - HOT3D sample bridge, real-data viewer, wrist-level evaluation

* `datasets/lerobot.py`: a LeRobot v3 reader (info/tasks/episode metadata,
  per-frame labels as `[T, n]` arrays, episode video lookup) with the schema
  cross-checked against `meta/info.json`.
* `datasets/hot3d_gt.py` + `scripts/import_lerobot.py`: convert a sample episode
  into the project's trajectory contract - camera poses (inverted
  `extrinsics_w2c`), intrinsics, wrist position, MANO root/hand rotations and
  shape, validity - **re-anchored to World-0** with the HOT3D frame kept in
  `hot3d_world_anchor_*`.
* Reference is wrist-only: the sample carries a wrist pose plus 15 joint
  rotations, and deriving 21 joint positions needs the licence-gated MANO mesh
  model, which is not present. Joints 1..20 are written as `NaN` rather than
  invented.
* `evaluation/action_mpjpe.py` now masks **per joint** (`safe_nanmean`), so a
  wrist-only reference yields a wrist-level Action-MPJPE plus a
  `referenced joints: x %` line and an explicit note in the report.
* `visualization/overlay.py::write_wrist_comparison_video` +
  `scripts/render_gt_vs_pred.py`: project the reference wrist (and optionally a
  prediction) on the RGB with the reference camera, writing a video and stills.
  Verified visually: the imported reference lands exactly on both hands in
  `hot3d_ep000`.
* `scripts/demo_hot3d_sample.sh` + `make sample`: import the bundled episode and
  render the viewer; `WITH_MOCK=1` additionally runs Phases 1-6 on the real
  512x512 footage as a plumbing check (3 camera windows, both Sim(3) alignments
  100 % inliers, 0.0000 m rmse, 100 % stitched coverage).
* Environment: added `pyarrow`/`pandas` to `environment-base.yml` for parquet.
* Test suite: **212 passed** (~12 s).

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

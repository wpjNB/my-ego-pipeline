# Changelog

## 2026-09-24 22:30 (+08:00) - the "broken hands" figure was a bad render, not the reference

A debug still (`outputs/real_mano_frame150.png`, produced by a throw-away snippet
in an earlier session, not by any script in this repository) showed the hands
floating over the bowl. The reference itself is fine - the *figure* was wrong.

Diagnosis, reproduced rather than guessed: projecting `hand_xyz_world` with
`camera_K` and **without** `camera_R_c2w` / `camera_t_c2w` reproduces that image
to a mean |delta| of 0.63/255 per pixel (i.e. bbox-exact). The overlay had
skipped the world -> camera transform, so the joints stayed in the World-0 frame
and the hands landed wherever that origin projects to. The pipeline's own
visualisers (`write_wrist_comparison_video`, `write_hand_video`) always apply it:
with the transform, the `GT L` / `GT R` markers sit on the wrists and the
projected MANO mesh covers the real hands.

Two related traps, both now documented rather than rediscovered:

* the `01_detection.mp4` / `02_hawor.mp4` in `data/hot3d/hot3d_ep000/` were
  written on 2026-09-23 with `backends.mode: mock` - their "hands" are the
  deterministic stand-in, so any judgement of quality from them is void;
* a 21-joint stick figure drawn over a *grasping* hand fans its fingers out
  because the fingertips are curled behind the hand - a mesh or the wrist
  marker is the honest thing to look at.

Also, while auditing, a real data-level finding: the sample stores each hand
pose twice (`observation.state` axis-angle vs the `*_hand_pose` 135-value matrix
column) and the two disagree by up to ~33 deg on a few joints (index 6/9/10/12
for this episode) - ~4 cm at the fingertips. The importer uses the matrix column
plus `*_orient_world`, which is the encoding declared as `hand_frame: world`.
That inconsistency is in the sample, not in the conversion, and it is now
recorded instead of silently averaged away.

Fixes: added `visualization.overlay.world_to_camera()` (with a test that fails if
someone projects world points with the intrinsics alone), a `--skeleton` flag on
`scripts/render_gt_vs_pred.py`, and `scripts/demo_hot3d_sample.sh` now renders the
reference overlay *with* the skeleton. The four misleading ad-hoc PNGs were
deleted and replaced by `outputs/reference_overlay_frame{150,375}.png` from the
real script.

Test suite: **288 passed**.

## 2026-09-24 22:12 (+08:00) - MANO is wired into both hands and both real configs

`weights/mano` now holds both official models, so the 21-joint reference uses
`MANO_RIGHT.npz` **and** `MANO_LEFT.npz` verbatim - `mano_mirrored` is `False`
for both sides in the fresh sample import (it used to mirror the left hand from
the right model). Two follow-ups so nothing stays stale:

* `configs/macrodata_final.yaml` gained `paths.mano_model: weights/mano`; the
  reference blog config pointed nowhere, so `scripts/doctor.py` reported
  "MANO model not configured -> references are wrist-only" even with the model
  on disk. `hot3d.yaml` already had it.
* `scripts/verify_weights.sh` audits `weights/mano/MANO_LEFT.pkl` too (it checked
  only the right one, plus HaWoR's two copies and WiLoR's), and the
  missing-MANO note counts five locations instead of four.

Also: every `scripts/*.py`, `scripts/*.sh` and `backends/*.py` is now
executable, so the `./scripts/...` invocations in the README and runbook work
as written (they were `-rw-rw-r--`).

Verification on the real sample clip: importing `hot3d_ep000` with the
configured MANO writes 21 finite joints in **434/450** left frames and
**450/450** right frames, with median bone lengths that are anatomically right
(wrist->index MCP 32.4 mm, index MCP->PIP 21.5 mm, PIP->DIP 23.7 mm,
thumb chain 33.0/37.8/30.0 mm). `scripts/doctor.py` now ends with "CPU path:
ready" and a single GPU-path gap (the backend environments).

Test suite: **286 passed**.

## 2026-09-24 21:57 (+08:00) - MANO installed; read the official pickle without chumpy

The author downloaded `mano_v1_2.zip` from the official site (the download is
behind a login, so no script can fetch it). `scripts/install_mano.sh --from
mano_v1_2.zip` unpacked it and installed both models into the places that need
them: HaWoR's `_DATA/data/mano` and `_DATA/data_left/mano_left`, `weights/mano`
and WiLoR's `mano_data` (the last one only matters for WiLoR's own 3D model).

The interesting part was avoiding chumpy: chumpy 0.70 needs ``numpy<1.24`` *and*
Python <= 3.10 (it calls ``inspect.getargspec``), which cannot coexist with the
orchestrator's numpy 2.x. Inspecting the archive showed only ``shapedirs`` is
chumpy-wrapped - a ``reordering.Select`` over a plain ``(778, 3, 20)`` array with
23340 indices and a ``preferred_shape``. ``hand/mano_model.py::read_mano_pickle``
now unpickles with a stand-in class, materialises ``Select``/``Ch``/csc_matrix and
normalises the archive's ``J_regressor`` spelling, so:

* the official pickles load directly - no chumpy, no extra environment;
* ``scripts/convert_mano.py`` writes ``.npz`` in the base env (it used to demand
  chumpy);
* ``configs/hot3d.yaml`` points at ``weights/mano``, so HOT3D references are now
  **21-joint** (`hand_joints: mano_fk`);
* ``--no-mano`` on the importer gives the wrist-only reference on demand.

Verification on the real model: 778 vertices / 10 betas / (16,778) regressor /
(778,16) weights / (778,3,135) posedirs / (1538,3) faces; rest-pose landmark
chains monotone with middle 175 mm, index 170, ring 165, pinky 142, thumb 129 -
anatomically right. The 21-joint skeleton now wraps the real hands in the sample
footage (it previously fanned straight out because the shape came from the
synthetic stand-in).

Also fixed: the topology check was running per frame and warning on real curled
hands - it is only meaningful in the rest pose, so it now runs once per model via
``validate_landmark_mapping`` (rest-pose monotonicity + tip-is-farthest).

Test suite: **281 passed**.

## 2026-09-24 21:38 (+08:00) - backend runners rewritten against the real sources

With the checkouts finally on disk, every call was re-derived from the code
instead of from documentation, and three of my assumptions were wrong:

* **Phase 1 needs WiLoR's detector, not its model.** WiLoR's demo splits
  `YOLO('detector.pt')` (boxes + handedness) from `load_wilor(wilor_final.ckpt)`
  (3D, needs MANO); HaWoR's `detect_track` uses the same YOLO arrangement. Since
  this pipeline takes tracking from WiLoR and reconstruction from HaWoR,
  `backends/wilor_runner.py` is now a detector runner (ultralytics, low `--conf`
  0.1 so gap-recovery candidates survive the 0.75 anchor threshold, HaWoR's
  `external/detector.pt` accepted as a stand-in). `wilor_final.ckpt` and WiLoR's
  MANO copy are documented as *not needed*.
* **VGGT-Omega's package is `vggt_omega`**: `VGGTOmega().eval()` +
  `load_state_dict(torch.load(...))`, `encoding_to_camera(pose_enc, image_size)`
  (not `pose_encoding_to_extri_intri`), `load_and_preprocess_images(..., image_resolution=416)`,
  and CUDA is mandatory (the upstream demo raises without it) - so Phase 3 has no
  CPU fallback and says so.
* **HaWoR's call sequence was right**: `hawor_motion_estimation` ->
  `hawor_slam` -> `hawor_infiller` -> `run_mano`/`run_mano_left`, with
  `load_slam_cam` at `lib/eval_utils/custom_utils.py:129` and our conservative
  tracking writing `model_tracks.npy` in place of `detect_track(thresh=0.2)`.
  MANO is required there (right mandatory, left recommended).

Also: `detection.detector_confidence` config key (default 0.1) wired through
`run_detection.py`; MANO messaging in `install_mano.sh` and `verify_weights.sh`
updated to say who really needs it.

Test suite: **276 passed**.

## 2026-09-24 21:33 (+08:00) - weights downloaded for real; MANO is needed in four places

First run with real network access (the sandbox blocks DNS, so the download was
run unsandboxed at the author's request):

| asset | size | source |
| --- | --- | --- |
| `wilor_final.ckpt` | 2.39 GiB | hf-mirror (resumed from a 407 MB partial file) |
| `model_config.yaml` / `detector.pt` | 2 KiB / 51.1 MiB | hf-mirror |
| `hawor.ckpt` / `infiller.pt` / `model_config.yaml` | 3.04 GiB / 399 MiB / 3 KiB | hf-mirror |
| `vggt_omega_1b_416_reproduce.pt` (+ `configuration.json`, `LICENSE.txt`) | 4.26 GiB / 64 B / 12 KiB | ModelScope CDN |

`./scripts/verify_weights.sh` now reports *all required weights present and
readable* (exit 0). The three backend checkouts are cloned too
(`WiLoR` 23 MB, `HaWoR` 162 MB, `VGGT-Omega` 88 MB).

Findings and fixes from doing it for real:

* **huggingface.co is unreachable from this network, hf-mirror.com works.**
  `download_weights.sh` now orders sources by `HF_ENDPOINT` (already set on this
  machine), then the official host, then `hf-mirror`; `fetch()` takes several
  URLs and tries them in order, so a mirror needs no code change.
* **The real file names and sizes** replace the earlier estimates, and every
  size floor is now measured + env-overridable (`WILOR_MIN_BYTES`,
  `HAWOR_MIN_BYTES`, `INFILLER_MIN_BYTES`, `VGGT_MIN_BYTES`).
* **VGGT-Omega's package is `vggt_omega`, not `vggt`**; its decoder is
  `encoding_to_camera` and the loader is
  `vggt_omega.utils.load_fn.load_and_preprocess_images`. Its `reproduction.md`
  confirms the 416 reproduction checkpoint is the recommended one for
  benchmarking and that inference must use `image_resolution=416` - the project
  configuration was already right.
* **MANO is needed in four independent places** (HaWoR right, HaWoR left, WiLoR's
  `mano_data/`, our own FK). The checkouts ship only `.gitkeep`, so
  `scripts/install_mano.sh` installs a licensed copy into all four (copy, or
  `--link`), and both backend runners now report the missing file in `--check`.
* git clone robustness: a flaky GnuTLS failure (mid-pack) no longer aborts the
  run - clones are retried with HTTP/1.1 + `--depth 1`, then a `GITHUB_MIRROR`
  if set; the VGGT-Omega repository URL is the real one
  (`facebookresearch/vggt-omega`).
* `doctor.py` now diagnoses the remaining gap precisely: with the checkouts and
  weights in place it reports `backend:WiLoR/HaWoR/VGGT-Omega: ok` and, for the
  runners, `backend environment 'ego3d_wilor' does not exist yet` plus the exact
  `conda env create -f environment-wilor.yml` command, instead of conda's raw
  error.
* Test suite: **276 passed**.

## 2026-09-24 10:20 (+08:00) - real VGGT-Omega source (ModelScope) wired in

The VGGT-Omega checkpoint URL was a placeholder; the author pointed the project
at the actual hub. Read from the ModelScope files page (revision ``master``):

| file | size |
| --- | --- |
| ``vggt_omega_1b_416_reproduce.pt`` | 4.58 GB (the reference configuration's checkpoint) |
| ``vggt_omega_1b_512.pt`` | 4.58 GB |
| ``vggt_omega_1b_256_text.pt`` | 5.40 GB |
| ``configuration.json`` / ``LICENSE.txt`` / ``README.md`` | 64 B / 11.72 KB / 2.34 KB |

Licence: FAIR Noncommercial Research License; repo updated 2026-09-09.

* ``scripts/download_weights.sh`` now defaults to
  ``VGGT_MODEL_ID=facebook/VGGT-Omega`` and ``VGGT_FILE=vggt_omega_1b_416_reproduce.pt``,
  with the 512 file and the ModelScope API endpoint as ordered fallbacks and
  ``modelscope``/``hf`` CLIs after that; ``configuration.json`` and
  ``LICENSE.txt`` are fetched as optional provenance. ``fetch`` now takes
  *several* URLs and tries them in order, so a moved host needs no code change.
* ``scripts/verify_weights.sh`` accepts any published VGGT-Omega file name
  (glob), raises its floor to 4 GB so a truncated 4.58 GB download cannot pass,
  and can be tuned with ``VGGT_MIN_BYTES`` for small mirrors/tests.
* ``camera/vggt_omega.py`` gained ``CHECKPOINT_FILENAMES`` and
  ``resolve_checkpoint()``: the requested checkpoint is resolved to the actual
  file, and a substitution (e.g. only the 512 file on disk while 416 was asked
  for) is reported in the runner warning, the runner JSON summary and
  ``camera/vggt_run.json`` - never silently.
* Test suite: **267 passed**.

## 2026-09-23 22:35 (+08:00) - bash weight scripts replace the Python downloader

At the author's request the manifest-driven Python downloader was dropped
(``weights.manifest.yaml``, ``runtime/weights.py``, ``scripts/download_weights.py``
and their tests are gone). Weights are now fetched by two dependency-light bash
scripts that work on a bare server:

* ``scripts/download_weights.sh`` - ``wget -c``/``curl -C -`` into
  ``<name>.part`` (resume), size check, then move into place; ``--only``,
  ``--dest``, ``--dry-run`` and ``--with-repos`` (clones WiLoR/HaWoR recursively
  and VGGT-Omega, per upstream); it runs the verifier automatically at the end.
* ``scripts/verify_weights.sh`` - the suggested verify script, hardened: size
  floor **and** container magic sniffing (zip for modern ``torch.save``, pickle
  for legacy, the 8-byte JSON header for ``safetensors``, text for yaml), so a
  truncated or HTML-error download is caught before ``torch.load`` sees it.
  ``--quiet``/``--strict`` supported, non-zero exit when something required is
  missing or unusable.
* Weight paths are now ``weights/wilor/``, ``weights/hawor/checkpoints/``,
  ``weights/vggt-omega/``, ``weights/mano/``. Because the flat layouts people
  end up with (``weights/hawor/hawor.ckpt``, ``weights/vggt/...``) are easy to
  produce by hand, ``hand/hawor.py::find_weights_files`` and
  ``camera/vggt_omega.py::find_checkpoint`` now accept both, and
  ``hawor_runner``/``vggt_runner`` use them.
* MANO stays manual and is reported as such (licence-gated).
* Test suite: **266 passed** (14 new for the bash scripts, 4 for the weight-path
  resolvers).

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

# Ego3D Hand Action Pipeline

An orchestration layer that turns a monocular **egocentric RGB video** into a
**world-frame, metric, 21-joint 3D hand-action trajectory**.

This is not a fork of HaWoR and not a wrapper that just "runs WiLoR +
VGGT-Omega". It implements the whole reference system itself - the stages, the
intermediate artefacts, the coordinate conventions, the window stitching, the
post-processing and the evaluation - and treats WiLoR, HaWoR and VGGT-Omega as
external backends that run in their own environments.

## Reference configuration

| Module | Configuration |
| --- | --- |
| Hand tracking | WiLoR detector + conservative tracking (conf >= 0.75, same-side gap <= 4, IoU >= 0.20) |
| Hand reconstruction | HaWoR, 16-frame windows / 8-frame overlap |
| Camera reconstruction | VGGT-Omega, 416 px / 200-frame windows / 40-frame overlap |
| Window alignment | depth-derived Sim(3) (weighted Umeyama + RANSAC) |
| Overlap fusion | linear blending |
| Camera trajectory filter | 3-frame binomial translation filter |
| Bone/depth correction | <= 3.5 % |
| Wrist depth optimisation | ray-constrained, lambda = 0.2, confidence weighting |
| Output | `[T, 2, 21, 3]` metric world-frame joints |

Everything is pinned in [`configs/macrodata_final.yaml`](configs/macrodata_final.yaml),
the single authoritative parameter table of the project.

## Pipeline

```
Egocentric RGB
      |
      +---------------------------+
      v                           v
 WiLoR detection            VGGT-Omega camera
 conservative tracking      200-frame windows / 40 overlap
      |                           |
      v                           v
 HaWoR 16/8 windows         depth-derived Sim(3)
 temporal blending          window stitching + blending
      |                           |
      +------------+--------------+
                   v
        world fusion  p_w = R_c2w p_c + t_c2w
                   v
        post-processing: camera filter / bone scale / wrist depth
                   v
        21-joint metric world-frame hand trajectory
```

## Quick start (CPU laptop)

```bash
make env              # or: conda env create -f environment-base.yml
make install          # editable install into ego3d_base
make test             # unit + integration tests, no GPU required
make demo             # full Phases 0-7 on a synthetic clip, ~10 s, no GPU
make sample           # import the bundled HOT3D sample and render its viewer

# Phase 0 on any video you have:
conda run -n ego3d_base python scripts/run_preprocess.py \
    --config configs/macrodata_final.yaml --clip demo01 --data-root data demo01.mp4
```

The orchestrator environment has no torch dependency: everything in Phases
0, 1-tracking, 4, 5, 6 and 7 is pure numpy/scipy and runs on CPU.

## The bundled HOT3D sample

`data/samples/lerobot_v3` is a LeRobot v3 dataset with eight 15-second HOT3D
egocentric clips carrying per-frame ground truth (camera extrinsics +
intrinsics, wrist pose, 15 joint rotations, MANO shape, validity).

```bash
make sample                                   # episode 0: import + viewer
EPISODE=3 make sample
WITH_MOCK=1 bash scripts/demo_hot3d_sample.sh  # also run Phases 1-6 on real footage

# or step by step
conda run -n ego3d_base python scripts/import_lerobot.py \
    --config configs/hot3d.yaml --clip hot3d_ep000 --data-root data/hot3d \
    --root data/samples/lerobot_v3 --episode 0
conda run -n ego3d_base python scripts/render_gt_vs_pred.py \
    --config configs/hot3d.yaml --clip hot3d_ep000 --data-root data/hot3d \
    --prediction data/hot3d/hot3d_ep000/trajectory/trajectory.npz \
    --ground-truth data/hot3d/hot3d_ep000/trajectory/ground_truth.npz
```

The importer decodes the episode video, writes `frames/` + `metadata.json`, and
converts the labels into `trajectory/ground_truth.npz` in **the same contract the
pipeline emits** - camera poses, intrinsics, wrist position, MANO parameters and
validity - re-anchored to `World-0` so a prediction is directly comparable. The
HOT3D world frame stays recoverable through `hot3d_world_anchor_*` in the
metadata.

The viewer projects the reference wrist (and a prediction, if given) on the RGB
with the reference camera:

![ground truth on the sample](/home/wpj/ego/my-ego-pipeline/data/hot3d/hot3d_ep000/visualization/gt_vs_pred_stills/000150.png)

### Wrist-only by default, 21 joints with MANO

The sample stores a wrist pose plus 15 joint rotations, not 21 joint positions.
Turning those into fingertips needs the MANO mesh model
(`v_template`/`shapedirs`/`J_regressor`/`weights`), which is licence-gated and
not bundled. Two honest modes:

* **default - wrist-only.** Joints 1..20 are written as `NaN`, the metadata says
  `hand_joints: "wrist_only"`, and the evaluation reports
  `referenced joints: 4.7 %` plus a note. You get a wrist-level Action-MPJPE
  instead of a silently fabricated average.
* **21 joints - with MANO.** Convert your licensed pickle once
  (`python scripts/convert_mano.py --input MANO_RIGHT.pkl ...`), then:

  ```bash
  MANO_MODEL=weights/mano make sample
  ```

  `hand_xyz_world` then carries all 21 joints from numpy forward kinematics
  (`hand/mano_model.py`: shape blend shapes, pose blend shapes, linear blend
  skinning, the standard 21-landmark mapping), the wrist stays exactly where the
  dataset put it, and missing frames stay missing. A right-hand model is enough:
  the left is mirrored and `mano_mirrored` is recorded.

  No MANO model at hand? `python scripts/make_synthetic_mano.py` fabricates a
  structural stand-in so the same code path can be exercised end to end - its
  output lives under `*_synthetic`, and any metric computed from it is
  meaningless by construction.

`data/*` is git-ignored: the sample dataset and everything derived from it stay
on disk and out of the repository.

## Backends, and the mock mode

Model backends are never imported by the orchestrator. Each one is a standalone
runner script that takes CLI arguments, writes its artefacts and prints one JSON
summary line as its last stdout line:

```
backends/wilor_runner.py   WiLoR detection      -> boxes/confidence/right_score/left_score/count
backends/hawor_runner.py   HaWoR 16/8 windows   -> hand/windows/*.npz (camera-space joints)
backends/vggt_runner.py    VGGT-Omega windows   -> camera/windows/*.npz (poses + metric depth)
```

`backends.python.<name>` says which interpreter runs each one - normally
`conda run -n ego3d_<name> python`, i.e. the backend's own pinned CUDA stack.
Every runner also supports `--check`, which reports availability as JSON without
needing a GPU, and any non-zero exit, timeout or malformed output surfaces as a
typed `BackendExecutionError` carrying the output tail.

`backends.mode: mock` swaps the three runners for
[`backends/mock_backend.py`](backends/mock_backend.py), a deterministic stand-in
built from `ego3d_action.testing.synthetic`. It is *not* a model: it exists so
the whole orchestration, the artefact contract, the stitcher and the evaluation
can be run and tested on a laptop. Every artefact it produces is marked
`backend_mode: mock` in its metadata.

The real runners are written against each backend's own API. The part that can
be verified without a GPU - turning model output into pipeline artefacts - lives
in the package and is unit-tested:

| Runner | Calls | Conversion (tested) |
| --- | --- | --- |
| `wilor_runner.py` | `wilor.models.load_wilor` + detector | `detection/wilor.py::detections_from_predictions`, `build_raw_detection_arrays` |
| `hawor_runner.py` | `hawor_motion_estimation` -> `hawor_slam` -> `hawor_infiller` -> `run_mano` | `hand/hawor.py::hawor_tracks_from_detection`, `hand_windows_from_joints` |
| `vggt_runner.py` | `VGGT.from_pretrained` + `pose_encoding_to_extri_intri` | `camera/vggt_omega.py::camera_window_from_output`, `scale_intrinsics` |

Two findings from reading HaWoR's source, both load-bearing:

* **Our tracker drives HaWoR.** HaWoR's demo starts at
  `detect_track(imgfiles, thresh=0.2)`; this project replaces exactly that with
  the conservative tracker of Phase 1 by writing its decision into the
  `model_tracks.npy` structure `hawor_motion_estimation` reads. HaWoR then
  reconstructs only the frames we kept.
* **HaWoR's infiller needs its own SLAM.** `hawor_infiller` reads
  `SLAM/hawor_slam_w_scale_*.npz` and emits hands in *HaWoR's SLAM world frame*,
  so the runner calls `hawor_slam` as a coordinate carrier and converts the
  hands back into **camera space** before Phase 3. VGGT-Omega still owns the
  metric world trajectory - the win is a metric camera, not necessarily a
  cheaper Phase 2.

What remains for a real run is the backend call itself: it needs the checkout,
the weights and a GPU. Every runner reports `--check` as JSON and raises with
the exact missing piece instead of writing an empty artefact.

## GPU server

```bash
conda env create -f environment-hawor.yml   # python 3.10 / torch 1.13 / cuda 11.7
conda env create -f environment-vggt.yml    # VGGT-Omega
conda env create -f environment-wilor.yml   # WiLoR

# device: auto resolves to cuda:0 when a GPU is visible, and to cpu otherwise.
conda run -n ego3d_base python scripts/run_pipeline.py \
    --config configs/macrodata_final.yaml --clip demo01 --video /data/demo01.mp4
```

`runtime.device` accepts `auto`, `cpu`, `cuda`, `cuda:N`. Asking for `cuda` on a
machine without CUDA is an error rather than a silent downgrade, so a run that
was supposed to use the GPU can never quietly fall back to the CPU. See
[`environment-hawor.yml`](environment-hawor.yml) and
[`environment-vggt.yml`](environment-vggt.yml).

## Stages

| Phase | Script | Produces | Runs on CPU? |
| --- | --- | --- | --- |
| 0 preprocess | `scripts/run_preprocess.py` | `frames/`, `metadata.json` | yes |
| 1 detection | `scripts/run_detection.py` | `detection/detection.npz` | tracker yes, WiLoR no |
| 2 hand | `scripts/run_hand.py` | `hand/hand_camera.npz` | backend needed |
| 3 camera | `scripts/run_camera.py` | `camera/windows/*.npz` | backend needed |
| 4 stitch | `scripts/run_stitch.py` | `camera/stitched_camera.npz`, `stitched/sim3_transforms.npz` | yes |
| 5 fusion | `scripts/run_fusion.py` | `trajectory/trajectory_raw.npz` | yes |
| 6 refine | `scripts/run_refine.py` | `trajectory/trajectory.npz` | yes |
| 7 evaluate | `scripts/evaluate_hot3d.py` | Action-MPJPE / Coverage / FPS report | yes |

`scripts/run_detection.py --raw-detections FILE.npz` runs the conservative
tracker on detections exported by the WiLoR environment, so the model-free half
of Phase 1 is usable without a GPU.

## On-disk contract

Nothing is passed between stages as a live Python object; every stage writes
under `data/<clip>/` and the next stage reads it back. This is what makes it
possible to answer "was this 3D hand raw or post-processed?", "which window
produced this Sim(3)?" and "is this error from the hand or the camera?".

```text
data/<clip>/
|-- frames/            000000.jpg ...
|-- detection/         detection.npz (+ boxes.npy confidence.npy valid.npy track_id.npy)
|-- hand/              hand_camera.npz
|-- camera/
|   |-- windows/       000000_000199.npz ...
|   `-- stitched_camera.npz
|-- stitched/          sim3_transforms.npz
|-- trajectory/        trajectory_raw.npz  trajectory.npz  metadata.json
`-- visualization/     01_detection.mp4  02_hawor.mp4 ...
```

The final `trajectory.npz` follows one fixed schema (`frames`, `timestamps`,
`hand_xyz_world`, `hand_xyz_camera`, `hand_valid`, `hand_confidence`,
`camera_R_c2w`, `camera_t_c2w`, `camera_K`, `bbox`, `track_id`, `mano_root_rot`,
`mano_hand_pose`, `mano_betas`, `postprocess_valid`) and is validated on write,
so a schema drift fails at the producer instead of in training data later.

## Evaluation

The benchmark is **Action-MPJPE**, not per-frame MPJPE: the trajectory is cut
into 1-second chunks, each chunk is expressed in the camera frame at its first
frame, and neither prediction nor ground truth is re-translated, re-rotated or
re-scaled before comparing.

```bash
conda run -n ego3d_base python scripts/evaluate_hot3d.py \
    --prediction data/demo01/trajectory/trajectory.npz \
    --ground-truth data/hot3d/demo01/trajectory.npz \
    --pipeline-seconds 38.6
```

| Pipeline | MPJPE | Coverage | FPS |
| --- | --- | --- | --- |
| HaWoR original | | | |
| + VGGT | | | |
| + Sim(3) | | | |
| + 40 overlap | | | |
| + camera filter | | | |
| + bone scale | | | |
| + wrist depth | | | |
| **Final** | | | |

## Testing

```bash
make test        # whole suite
make test-fast   # skips the synthetic stitching scene
make smoke       # synthesises a video and runs Phase 0 end to end
conda run -n ego3d_base python scripts/demo_synthetic.py   # GPU-free Phase-4 demo
```

The suite covers Sim(3)/Umeyama, transforms, hand blending, wrist depth, bone
scale, camera filter, conservative tracking, window scheduling, stitching on a
synthetic scene with a known Sim(3), the trajectory contract, ffmpeg frame IO
and the Action-MPJPE protocol.

## Repository note

This workspace ships a read-only `.git` mount point, so the repository metadata
lives in `.gitstore/` instead. Use `GIT_DIR=.gitstore GIT_WORK_TREE=$PWD git ...`
for git commands, or move the project to a normal directory for a conventional
checkout.

## Current status

Implemented and tested now: every phase, the on-disk contract, config
validation, device selection, the backend availability probes, the runner
protocol and the mock backend. `make demo` runs Phases 0-7 end to end on CPU
(300 frames in ~6 s of pipeline time) and `make test` covers it with 193 tests.

What remains is one function per real backend: `run_model()` in
`backends/{wilor,hawor,vggt}_runner.py`, calling the installed model's inference
API. Everything around it - loading frames, owning the window schedule, writing
and validating the artefacts, reporting JSON and failures - is already there and
exercised by the mock, so the GPU-server step is a small, well-bounded edit that
cannot quietly produce an empty artefact.

```text
make demo  ->
  phases 0-6            : 300 frames, 5.65 s wall time (53.10 FPS on CPU)
  Action MPJPE (raw)    : 24.6393 mm   coverage 90.67 %
  Action MPJPE (refined): 25.8964 mm   wrist error 22.92 -> 16.06 mm
```

## Design rules

* No model is forked; backends are adapters.
* No artefact is passed in memory between stages.
* Missing 3D pose stays missing - it is never interpolated into existence.
* Wide-window Gaussian smoothing of the hand trajectory is deliberately absent:
  it lowers acceleration error but makes Action-MPJPE worse.
* Every failure path logs, re-raises or returns an explicit error state.
* New logic ships with tests.

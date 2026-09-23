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

# Phase 0 on any video you have:
conda run -n ego3d_base python scripts/run_preprocess.py \
    --config configs/macrodata_final.yaml --clip demo01 --data-root data demo01.mp4
```

The orchestrator environment has no torch dependency: everything in Phases
0, 1-tracking, 4, 5, 6 and 7 is pure numpy/scipy and runs on CPU.

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

Implemented and tested now: Phases 0, 4, 5, 6 and 7, the model-free half of
Phase 1, the on-disk contract, config validation, device selection and the
backend availability probes.

Blocked on the GPU server (explicitly, never silently stubbed): the WiLoR,
HaWoR and VGGT-Omega *invocations*. Those adapters raise
`BackendInvocationNotImplemented` with the exact environment, checkpoint and
missing-path hints instead of pretending to run a model.

## Design rules

* No model is forked; backends are adapters.
* No artefact is passed in memory between stages.
* Missing 3D pose stays missing - it is never interpolated into existence.
* Wide-window Gaussian smoothing of the hand trajectory is deliberately absent:
  it lowers acceleration error but makes Action-MPJPE worse.
* Every failure path logs, re-raises or returns an explicit error state.
* New logic ships with tests.

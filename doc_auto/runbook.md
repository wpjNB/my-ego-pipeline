# Runbook: how to run this project

Last modified: 2026-10-06 19:10 (+08:00)

Two paths. Path A needs no weights and no GPU; Path B is the real pipeline.
Start every session with the audit:

```bash
conda run -n ego3d_base python scripts/doctor.py --config configs/macrodata_final.yaml
```

## Path A - CPU, no weights (mock backends)

Purpose: verify the orchestration, the artefact contract, the stitcher, the
post-processing and the evaluation. The three models are replaced by a
deterministic stand-in (`backends/mock_backend.py`, `backends.mode: mock`), so
**scores from this path are meaningless as accuracy results** and every artefact
records `backend_mode: mock`.

```bash
make env && make install      # once
make test                     # 286 tests, ~20 s
make demo                     # Phases 0-7 on a synthetic 300-frame clip, ~10 s
make sample                   # import HOT3D sample episode 0 + ground-truth viewer
WITH_MOCK=1 bash scripts/demo_hot3d_sample.sh   # Phases 1-6 on real 512x512 footage
```

`make sample` now writes a **21-joint** reference (`hand_joints: mano_fk`,
`mano_mirrored: {left: false, right: false}`) because `weights/mano` holds both
official models and `configs/hot3d.yaml` points at it.

### Judging the reference (and why a debug figure once looked broken)

```bash
python scripts/render_gt_vs_pred.py --config configs/hot3d.yaml --clip hot3d_ep000 \
    --data-root data/hot3d --skeleton \
    --prediction data/hot3d/hot3d_ep000/trajectory/ground_truth.npz \
    --ground-truth data/hot3d/hot3d_ep000/trajectory/ground_truth.npz
```

Both the wrist marker and `--skeleton` go through
`visualization.overlay.world_to_camera()`. Skipping that transform (projecting
`hand_xyz_world` with `camera_K` alone) is the mistake that produced a
now-deleted debug PNG where the hands floated over the bowl - it is not a
pipeline bug, and the test `test_world_to_camera_is_the_inverse_of_the_stored_pose`
exists to keep it that way.

Interpretation rules: `01_detection.mp4` / `02_hawor.mp4` written with
`backends.mode: mock` show the deterministic stand-in, not a model; and a
21-joint stick figure on a *grasping* hand looks fanned out because the
fingertips curl behind the palm - look at the wrist marker or the projected mesh
instead.

Hand focal resolution and EGO preview projection use the same validated, undistorted `metadata.json:image_camera` when available. Focal priority is `hand.focal` override -> input-frame calibration -> canonical VGGT median; legacy clips without image calibration use VGGT. Prediction stages never read ground-truth camera labels. WiLoR previews may apply the configured, labeled 2D box nudge to rendered pixels only; saved 3D arrays remain unchanged.

What `make demo` prints on this machine (CPU, mock):

```
phases 0-6            : 300 frames, 5.65 s wall time (53.10 FPS)
Sim(3)                : scale 0.7543 recovered, 100% inliers, 0.0000 m rmse
Action MPJPE (raw)    : 24.6393 mm   coverage 90.67 %
Action MPJPE (refined): 25.8964 mm   wrist error 22.92 -> 16.06 mm
```

Per-clip command sequence (equivalent to the demo):

```bash
python scripts/run_preprocess.py    --config configs/mock.yaml --clip clip01 --data-root data/mock --video clip01.mp4
python scripts/run_detection.py     --config configs/mock.yaml --clip clip01 --data-root data/mock
python scripts/run_camera.py        --config configs/mock.yaml --clip clip01 --data-root data/mock
python scripts/run_stitch.py        --config configs/mock.yaml --clip clip01 --data-root data/mock
python scripts/run_hand.py          --config configs/mock.yaml --clip clip01 --data-root data/mock
python scripts/run_fusion.py        --config configs/mock.yaml --clip clip01 --data-root data/mock
python scripts/run_refine.py        --config configs/mock.yaml --clip clip01 --data-root data/mock
python backends/mock_backend.py truth --out data/mock/clip01/trajectory/truth.npz --num-frames 300
python scripts/evaluate_hot3d.py    --config configs/mock.yaml --clip clip01 \
    --prediction data/mock/clip01/trajectory/trajectory.npz \
    --ground-truth data/mock/clip01/trajectory/truth.npz
```

(All via `conda run -n ego3d_base`.) The same sequence is what
`scripts/run_pipeline.py` executes as subprocesses; `--dry-run` prints it
without running anything, `--from-stage phase4-stitch` resumes mid-way.

## Path B - real backends (GPU server)

Prerequisites: `doc_auto/setup.md` sections 2-4 (envs, checkouts, weights). The
weights come from one command - `./scripts/download_weights.sh` (add
`--with-repos` to clone the checkouts as well), then `./scripts/verify_weights.sh`
- which prints anything that still needs a manual step. Gate
each step with `--check` before spending GPU time:

```bash
python backends/wilor_runner.py --check --third-party third_party --weights weights
python backends/hawor_runner.py --check --third-party third_party --weights weights
python backends/vggt_runner.py  --check --third-party third_party --weights weights
```

Each prints one JSON line: `{"backend": "...", "available": true|false, "detail": ...}`.

### Step by step (one clip)

```bash
CFG=configs/macrodata_final.yaml
CLIP=demo01
RUN="conda run -n ego3d_base python"

# 0 - decode
$RUN scripts/run_preprocess.py --config $CFG --clip $CLIP --video /data/$CLIP.mp4
#    -> data/$CLIP/frames/000000.jpg ... , data/$CLIP/metadata.json

# 1 - WiLoR + conservative tracking (GPU env ego3d_wilor)
$RUN scripts/run_detection.py --config $CFG --clip $CLIP
#    -> detection/detection.npz (+ 01_detection.mp4)
#    prints: coverage left=xx% right=xx%

# 2 - VGGT-Omega windows (GPU env ego3d_vggt; one process for the whole clip)
$RUN scripts/run_camera.py --config $CFG --clip $CLIP
#    -> camera/windows/000000_000199.npz ...

# 3 - depth-derived Sim(3) camera stitching (CPU)
$RUN scripts/run_stitch.py --config $CFG --clip $CLIP
#    -> camera/stitched_camera.npz, stitched/sim3_transforms.npz

# 4 - HaWoR 16/8 (GPU env ego3d_hawor; consumes the stitched camera path)
#    focal priority: hand.focal -> validated input image_camera -> VGGT canonical median
$RUN scripts/run_hand.py --config $CFG --clip $CLIP
#    -> hand/hand_camera.npz, hand/windows/*.npz (+ 02_hawor.mp4)
#    prints per-pair: scale, rmse, inlier%, rotation

# 5 - world fusion (CPU)
$RUN scripts/run_fusion.py --config $CFG --clip $CLIP
#    -> trajectory/trajectory_raw.npz

# 6 - post-processing (CPU)
$RUN scripts/run_refine.py --config $CFG --clip $CLIP
#    -> trajectory/trajectory.npz + trajectory/metadata.json
```

Or `scripts/run_pipeline.py --config $CFG --clip $CLIP --video /data/$CLIP.mp4`
for the whole chain in one command.

### Evaluation (HOT3D sample)

```bash
$RUN scripts/import_lerobot.py --config configs/hot3d.yaml --clip $CLIP --data-root data/hot3d \
     --root data/samples/lerobot_v3 --episode 0   # MANO comes from paths.mano_model
     # (weights/mano, already configured); add --no-mano for a wrist-only reference
$RUN scripts/evaluate_hot3d.py --config configs/hot3d.yaml --clip $CLIP \
     --prediction data/hot3d/$CLIP/trajectory/trajectory.npz \
     --ground-truth data/hot3d/$CLIP/trajectory/ground_truth.npz \
     --pipeline-seconds <measured>
$RUN scripts/render_gt_vs_pred.py --config configs/hot3d.yaml --clip $CLIP --data-root data/hot3d \
     --prediction data/hot3d/$CLIP/trajectory/trajectory.npz \
     --ground-truth data/hot3d/$CLIP/trajectory/ground_truth.npz
```

The report prints Action-MPJPE / Coverage / FPS plus camera/wrist/depth errors,
`referenced joints: x %` and, for a wrist-only reference, an explicit note.

### What "done" looks like for a stage

| Stage | Artefact | Debug video | Sanity signal |
| --- | --- | --- | --- |
| 0 | `frames/`, `metadata.json` | - | frame count matches ffprobe |
| 1 | `detection/detection.npz` | `01_detection.mp4` | boxes on both hands, gaps stay gaps |
| 2 | `hand/hand_camera.npz` | `02_hawor.mp4` | 3D hand projects onto the RGB hand |
| 3 | `camera/windows/*.npz` | - | finite depth, intrinsics match the depth grid |
| 4 | `camera/stitched_camera.npz`, `stitched/sim3_transforms.npz` | - | inlier% high, rmse small, scale stable |
| 5 | `trajectory/trajectory_raw.npz` | - | `validate_trajectory` clean |
| 6 | `trajectory/trajectory.npz` | `trajectory.png` | wrist error drops, camera smooth |
| 7 | report + `gt_vs_pred.mp4` | `gt_vs_pred.mp4` | reference lands on the hands |

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `backend runner 'X' failed ... no JSON summary` | the backend printed to stdout before/without its summary, or crashed | run the runner's `--check`, then the failing stage with `-v`; look at `data/<clip>/<stage>/*_runner.log` |
| `BackendNotAvailableError: checkout at third_party/X` | checkout missing | `git clone ...` (setup.md §3) |
| `BackendNotAvailableError: weights at ...` | weights missing | download them (setup.md §4) |
| `ModuleNotFoundError: torch` in a stage log | runner executed with the wrong interpreter | fix `backends.python.<name>` |
| `cuda requested but unavailable` | `runtime.device: cuda` on a CPU host | use `auto`/`cpu` (`EGO3D_FORCE_CPU=1` forces it) |
| `no shared depth samples between windows` | depth all-NaN or windows do not overlap | check Phase 3 output; verify the window schedule |
| `robust Sim(3) rejected: only N inliers` | overlap has too little depth structure | raise `stitch.pixel_stride` quality (lower value), check `min_depth`/`max_depth` |
| `RuntimeWarning: Mean of empty slice` | old evaluation code | gone: `safe_nanmean` returns NaN quietly |
| `referenced joints: 4.6 %` | wrist-only HOT3D reference | expected without MANO (setup.md §4) |
| matplotlib config error | read-only `$HOME` | `export MPLCONFIGDIR=/tmp/mpl-cache` |
| `git` cannot write | workspace root has a read-only `.git` mount | see `doc_auto/environments.md` |

## Expected API drift on first real run

The runners are written against each backend's documented API but have never
executed here (no checkouts, no weights, no GPU). Expect small fixes on the
first real run, all localised to the runner that fails:

* WiLoR output field names (`detection/wilor.py::detections_from_predictions`
  accepts several spellings and raises listing what it saw);
* VGGT's pose-decoding helper import path (`pose_encoding_to_extri_intri`);
* HaWoR's checkpoint/infiller paths inside `third_party/HaWoR`.

Every one of them fails loudly with the available keys/paths, never silently.

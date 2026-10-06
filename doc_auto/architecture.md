# Architecture

Last modified: 2026-10-06 16:07 (+08:00)

## Dataset bridge (LeRobot v3 / HOT3D)

`datasets/lerobot.py` reads a LeRobot v3 dataset (`meta/info.json`, one parquet
holding every episode's labels, one mp4 per episode) and `datasets/hot3d_gt.py`
converts one episode into the project's trajectory contract:

| dataset field | trajectory field |
| --- | --- |
| `extrinsics_w2c` (16) | `camera_R_c2w` / `camera_t_c2w` (inverted; `p_cam = R p_world + t`) |
| `intrinsics` (9) | `camera_K` |
| `left/right_transl_world` (3) | `hand_xyz_world[t, h, 0, :]` (wrist) |
| `left/right_orient_world` (9) | `mano_root_rot[t, h]` |
| `left/right_hand_pose` (135) | `mano_hand_pose[t, h]` (15 rotations) |
| `observation.state` (61 per hand) | `mano_betas[t, h]` (layout inferred, recorded) |
| `state_mask` & `*_kept` | `hand_valid[t, h]` |

Two decisions worth knowing:

* the reference is re-anchored to ``World-0`` (`camera_pose.world_frame_alignment`)
  so it is directly comparable with a prediction; the original HOT3D frame is
  preserved in `hot3d_world_anchor_rotation/translation`;
* joints 1..20 stay ``NaN`` because no MANO mesh model is available, and the
  evaluation masks **per joint**, so a wrist-only reference still yields a
  meaningful number plus an explicit `referenced joints: x %` line instead of a
  silently fabricated average.

## MANO forward kinematics

``hand/mano_model.py`` implements the standard SMPL-family pipeline in numpy:
shape blend shapes, pose blend shapes, the kinematic tree, linear blend
skinning, and the 21-landmark mapping (16 MANO joints + the five standard
fingertip vertices) into this project's joint convention. Given a MANO model,
``datasets/hot3d_gt.py`` produces a full 21-joint reference; the wrist is placed
exactly where the dataset says it is, so the 21-joint reference stays consistent
with the wrist-only one it replaces.

The asset itself is licence-gated and absent here, so the module is exercised
against a synthetic model with the same structure
(``testing/synthetic.py::make_synthetic_mano_model``): identity pose reproduces
the template, a local rotation moves only its own finger, shape parameters scale
the hand, the root rotation rotates everything, and the landmark mapping is
pinned slot by slot. ``scripts/convert_mano.py`` converts the official pickle to
``.npz`` (needs ``chumpy``, i.e. the HaWoR env); ``scripts/make_synthetic_mano.py``
fabricates the stand-in for plumbing runs.

## Backend seam

| Step | Where | Verified |
| --- | --- | --- |
| our tracking -> HaWoR ``model_tracks.npy`` | ``hand/hawor.py::hawor_tracks_from_detection`` | yes (unit tests) |
| model output -> detection artefact | ``detection/wilor.py`` | yes |
| model output -> camera window | ``camera/vggt_omega.py::camera_window_from_output`` | yes |
| HaWoR 21 joints -> 16/8 window files | ``hand/hawor.py::hand_windows_from_joints`` | yes |
| the backend call itself | ``backends/*_runner.py::run_model`` | needs the GPU server |

### What each phase actually requires (from reading the sources)

| Phase | Backend code | Weights | MANO | GPU |
| --- | --- | --- | --- | --- |
| 1 detection | `ultralytics` YOLO | `detector.pt` (51 MiB) | no | optional |
| 2 hand | HaWoR checkout | `hawor.ckpt`, `infiller.pt` | **yes** (right; left recommended) | yes |
| 3 camera | `vggt_omega` package | `vggt_omega_1b_416_reproduce.pt` | no | **required** (VGGT-Omega raises without CUDA) |
| 4-7 | this repository | - | only for 21-joint references | no |

WiLoR's `wilor_final.ckpt` and its MANO copy are therefore *not* needed by this
pipeline: Phase 1 takes tracking from WiLoR and reconstruction from HaWoR.

HaWoR's infiller hard-depends on its own SLAM output and emits hands in HaWoR's
SLAM world frame; the runner therefore converts them back to camera space with
HaWoR's own poses, keeping VGGT-Omega the sole authority on the metric world
trajectory.

## Backend invocation

The orchestrator never imports a model. Each backend is a standalone runner
script executed as a subprocess (`runtime/subprocess_backend.py`) with a small
protocol: CLI arguments in, artefacts on disk, one JSON summary line on stdout,
non-zero exit on failure. `backends.python.<name>` selects the interpreter (the
backend's own conda env), `backends.mode` selects `real` (the runners in
`backends/*_runner.py`) or `mock` (`backends/mock_backend.py`, a deterministic
stand-in used for CPU-only runs and tests).

## Stage map

Execution dependencies put camera-window inference and camera stitching before
HaWoR hand reconstruction. HaWoR needs the estimated focal from the windows and
a continuous World-0 camera path for its infiller; each raw VGGT window has its
own local world gauge. The single-clip pipeline, batch planner and viewer
pipeline all follow this order.

| Phase | Module | Artefact | Notes |
| --- | --- | --- | --- |
| 0 preprocess | `io/frames.py`, `io/video.py` | `frames/`, `metadata.json` | ffmpeg/ffprobe via subprocess, resumable |
| 1 detection | `detection/wilor.py`, `detection/tracker.py` | `detection/detection.npz` | conservative tracking is model-free |
| 2 hand | `hand/hawor.py`, `hand/temporal_blend.py` | `hand/hand_camera.npz` | 16/8 windows, linear + SLERP blend |
| 3 camera | `camera/vggt_omega.py`, `camera/window.py` | `camera/windows/*.npz` | 416 px / 200 frames / 40 overlap |
| 4 stitch | `camera/depth.py`, `camera/stitch.py`, `geometry/*` | `camera/stitched_camera.npz`, `stitched/sim3_transforms.npz` | depth-derived Sim(3) + linear blending |
| 5 fusion | `fusion/trajectory.py` | `trajectory/trajectory_raw.npz` | `p_w = R_c2w p_c + t_c2w` |
| 6 refine | `refinement/*` | `trajectory/trajectory.npz` | short-gap interpolation (P2), camera filter, bone scale, wrist depth |
| 7 evaluate | `evaluation/*` | report | Action-MPJPE / Coverage / FPS |

## Batch execution (offline)

`scripts/run_batch.py` runs a clip manifest by dispatching each stage as a unit
onto a capable host (see [`distributed.md`](distributed.md)). Phase 2 and Phase 3
units are sliced by **window** (`--shard i/N`), which is a pure partition of the
global schedule: the union of the shards equals the unsliced run, verified
array-for-array in `tests/test_batch_e2e.py`. Phase 1 is deliberately *not*
sliced (its tracker recovers gaps across frames) and refuses the flag with a
reason. A sharded `hand`/`camera` group ends with one whole-clip join unit
(`--blend-only` / window reuse) so the downstream stages still see the whole clip.

Every unit writes a content-addressed provenance marker
(`.provenance/<unit>.done.json`), so `--skip-existing` recomputes nothing while
input contents and parameters are unchanged, and a failure marks its clip
`degraded` in `batch_report.json` and `metadata.json` instead of fabricating
output.

## Coordinate conventions

* Extrinsics are **c2w**: `p_w = R_c2w @ p_c + t_c2w`.
* Intrinsics are OpenCV-style; depth is the metric z-coordinate in metres.
* Hands are `[T, 2, 21, 3]`: index 0 = left, index 1 = right.
* `World-0` is the camera of the first frame; after stitching the trajectory is
  re-anchored with `camera_pose.normalize_to_first_camera`.
* MANO topology lives once in `hand/mano.py` (`JOINT_PARENTS`, `bone_pairs()`).

## Why depth-derived Sim(3) and not camera centres

Camera centres give a weak, often degenerate constraint for the scale between
two windows. Back-projecting depth on the shared frames yields thousands of
correspondences, and the reference experiments report this formulation performs
best. `camera/depth.py` therefore never touches camera centres: it unprojects
identical `(frame, row, col)` samples in both windows and hands the point pairs
to `geometry/sim3.py`.

## Why every stage writes to disk

Otherwise three questions become unanswerable while debugging: whether a 3D hand
is raw or post-processed, which window produced a given Sim(3), and whether an
error comes from the hand or the camera. `io/artefacts.py` owns the layout so no
two stages can disagree about a path.

## Deliberate omissions

* No wide-window Gaussian smoothing of hand trajectories
  (`refinement/__init__.py` documents why).
* No pose interpolation beyond `refinement/gap_fill.py`: missing runs of at
  most `refinement.gap_fill_max_frames` (default 12) hand-frames between two
  valid anchors are filled with the per-joint linear blend of the anchors and
  marked in `hand_interpolated`; leading/trailing frames and longer holes stay
  ``NaN``, and nothing is ever extrapolated. The evaluator reports the
  predicted-only numbers next to the interpolated ones.
* No `DROID-SLAM` / `Metric3D` in the HaWoR environment.

## World-0 gauge

`World-0` is the first frame's camera, and two code paths keep that invariant
honest:

* Phase 4 re-anchors the stitched trajectory with
  `camera_pose.normalize_to_first_camera`;
* Phase 6 applies a gauge transform (`camera_pose.world_frame_alignment`) after
  the camera translation filter, because the 3-frame binomial filter moves frame
  0 slightly. The same transform is applied to the camera *and* the hands, so
  every camera-relative quantity - and therefore Action-MPJPE - is unchanged.

Forgetting that gauge is a classic way to produce a constant offset in the
evaluation: the mock reference trajectory, for example, has to be expressed in
`World-0` (`testing.synthetic.to_world0`) rather than in the raw synthetic world
frame.

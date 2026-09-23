# Architecture

Last modified: 2026-09-23 17:22 (+08:00)

## Backend invocation

The orchestrator never imports a model. Each backend is a standalone runner
script executed as a subprocess (`runtime/subprocess_backend.py`) with a small
protocol: CLI arguments in, artefacts on disk, one JSON summary line on stdout,
non-zero exit on failure. `backends.python.<name>` selects the interpreter (the
backend's own conda env), `backends.mode` selects `real` (the runners in
`backends/*_runner.py`) or `mock` (`backends/mock_backend.py`, a deterministic
stand-in used for CPU-only runs and tests).

## Stage map

| Phase | Module | Artefact | Notes |
| --- | --- | --- | --- |
| 0 preprocess | `io/frames.py`, `io/video.py` | `frames/`, `metadata.json` | ffmpeg/ffprobe via subprocess, resumable |
| 1 detection | `detection/wilor.py`, `detection/tracker.py` | `detection/detection.npz` | conservative tracking is model-free |
| 2 hand | `hand/hawor.py`, `hand/temporal_blend.py` | `hand/hand_camera.npz` | 16/8 windows, linear + SLERP blend |
| 3 camera | `camera/vggt_omega.py`, `camera/window.py` | `camera/windows/*.npz` | 416 px / 200 frames / 40 overlap |
| 4 stitch | `camera/depth.py`, `camera/stitch.py`, `geometry/*` | `camera/stitched_camera.npz`, `stitched/sim3_transforms.npz` | depth-derived Sim(3) + linear blending |
| 5 fusion | `fusion/trajectory.py` | `trajectory/trajectory_raw.npz` | `p_w = R_c2w p_c + t_c2w` |
| 6 refine | `refinement/*` | `trajectory/trajectory.npz` | camera filter, bone scale, wrist depth |
| 7 evaluate | `evaluation/*` | report | Action-MPJPE / Coverage / FPS |

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
* No pose interpolation for missing frames: `NaN` marks missing, everywhere.
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

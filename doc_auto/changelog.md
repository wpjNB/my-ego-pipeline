# Changelog

## 2026-10-09 12:02 (+08:00) - re-anchor HOT3D MANO root rotations with the camera

The LeRobot importer moved camera poses and wrist positions into World-0 before
MANO forward kinematics, but left `mano_root_rot` in the original HOT3D world
basis. The wrist therefore projected correctly while the fingers rotated around
it in the wrong direction. The converter now left-multiplies each root rotation
by the same anchor rotation; local finger rotations stay local.

On sample_ep001 frame 175, corrected GT wrist positions are unchanged while the
non-wrist joints move by a mean 17.5 cm (left) / 13.1 cm (right). The corrected
GT re-import changes WiLoR Action-MPJPE from 199.34 to 91.74 mm and HaWoR from
175.47 to 77.44 mm. Camera error remains 153.50 mm; the same frame's predicted
camera differs from the reference by 137.9 mm and 4.66 degrees. The default `gt_vs_pred` view includes that drift; camera-space wrist medians
remain 45.9/50.5 mm, so hand estimation also has residual error.
Regression: `tests/test_hot3d_gt.py`. Full suite: 405 passed, 15 skipped in 80.39 s.


## 2026-10-06 21:47 (+08:00) - keep HOT3D hand inference RGB-only

Hand focal and EGO preview K come from VGGT's RGB-derived camera windows
(canonical per-element median), with explicit `hand.focal` retained as an
experiment override. Importer `image_camera` metadata and `ground_truth.npz`
camera labels are not consumed by prediction code. On P0002_clip001971 the
VGGT focal was 623.47 px; the RGB-only WiLoR output measured wrist reprojection
medians of 84/28 px. A separate 608.54 px calibrated-focal run reached 70/19 px,
but is an oracle calibration ablation and is not reported as an RGB-only score.
The 0.5 box nudge remains labeled and preview-only. Sharded hand/camera joins
record their mode to preserve command and provenance correctness. Full suite on
the clean PR branch: 404 passed, 15 skipped (79.48 s).

## 2026-09-30 15:05 (+08:00) - combined EGO | WORLD viewer, after the Wuji reference

Studied the reference ecosystem's own visualisation stack before building:
Macrodata's open-source repo is `macrodata-labs/refiner` (an Apache-2.0
dataset-processing library - no viewer in it), and Wuji Technology's
`wuji-hand-teleop` (ROS2 teleop) visualises through RViz + a Qt monitor, with
the robot hand as open URDFs. The blog's three-panel screenshot is therefore
not a single open tool; the panel styles were reproduced here instead:

* `scripts/render_viewer.py` - one synchronised video, two panels per frame:
  EGO VIEW (RGB + MANO mesh reprojected + left/right validity badges) beside
  WORLD SPACE (trails growing to the current frame, camera path, hero hands).
  Reuses `render_world_space`'s drawing and the mesh rasteriser; H.264 via the
  standard transcoder. Outputs `visualization/viewer.mp4` per clip
  (`--gt` overlays reference trails in the world panel).
* The third reference panel (ROBOT HAND) is now a documented follow-up, not a
  hard wall: `wuji-technology/wuji-retargeting` is MIT-licensed pure Python
  (DexRetargeting-based, NLOPT), the Wuji Hand URDFs ship in the teleop repo,
  and MuJoCo 3.11 is already installed in the `ego3d` env. Path: MANO world
  joints -> retargeting configs -> joint angles -> MuJoCo render.

Also fixed while wiring the viewer: cv2.VideoWriter now initialises from the
first composed frame's true size (the ego panel's scaled width is not round;
a fixed declared size silently dropped every frame and produced a 257-byte
file).

## 2026-09-30 13:25 (+08:00) - WORLD SPACE panel: camera + both hands in the stitched world

Implemented the reference system's middle panel (the blog's "WORLD SPACE"):
`scripts/render_world_space.py` renders the stitched world frame with both
MANO wrist trails (cyan left / yellow right), ghost skeletons along the
trails, the camera path, dashed drop lines to the floor grid, the +X/+Y/+Z
triad at World-0, per-wrist height labels, a 10 cm scale bar and a top-down
view - dark theme, centimetres, 3D + XY double view. `--video` animates the
trails growing with the current hands and camera pose per frame; `--gt`
overlays the reference trails (the height labels then directly show the
predicted-vs-reference wrist heights, e.g. L 28.2 vs 32.4 cm on ep000).
Pure matplotlib/Agg + the existing H.264 transcoder; no GPU. Outputs:
`visualization/world_space.png` + `world_space_time.mp4` per clip, embedded
in `outputs/visual_gallery.html`.

Follow-up after first review: the hand fans are now dense (skeleton every 10
frames, thicker current hand with joint dots) so the hands dominate the
panel, and the title carries the measured wrist |PRED-GT| statistics
(median 8.9 cm on ep000, 12.6 cm / p90 19.0 cm on ep003). The gap is real
error, not a plotting artefact: both world frames are normalised to their
frame-0 camera (cameras at the origin, wrist error ~2 cm at frame 0) and the
distance grows with the stitched camera drift - the same camera/depth error
budget the evaluation reports. The panel shows raw world frames (no
per-chunk re-anchoring), which is why the drift is visible here while the
chunk-anchored metric hides it. Also fixed in passing: the title statistic
initially reported the first frame's error instead of the median.

## 2026-09-30 12:45 (+08:00) - reference-recipe audit against the Macrodata blog

Full section-by-section comparison of the repo against the reference blog
(`macrodata.co/blog/turning-egocentric-video-into-3d-hand-actions`), now in
`doc_auto/blog_comparison.md`. Verdict: faithful at the module level; the
52.04 mm reference value is unreachable on P100s (their own sweep puts the
camera window as the dominant lever: 60 frames -> 62.14 mm, 200 -> 55.95; we
run 8). Two recipe divergences found and fixed:

* **Bone-scale reference `median` -> `mean`** in every config (the recipe's
  ablation measured mean better, median regressed; our own two-episode A/B
  splits within noise: 186.75/187.92 and 189.22/188.49).
* **Hand-joint smoothing code default 1 -> 0** (the recipe disables it; every
  variant regressed at reference quality). The host configs keep an explicit
  `smooth_passes: 1` opt-in - our operating point is far noisier and one pass
  measured MPJPE-neutral while damping the mesh wobble.

Open conformance gap, documented not yet run: the learned `hawor_infiller` is
still in the runner while the recipe measured it worse than benchmark gap
filling (+1.59 mm) and disabled it - the experiment is a runner change plus
~5 GPU-minutes on both clips.

Test suite: **410 passed**, 1 skipped.

## 2026-09-30 12:00 (+08:00) - hand-trajectory smoothing: the mesh wobble damped where it can be

User-visible defect: rapid mesh jitter in `02_hawor.mp4`. Measured, not
guessed - three separate components:

* **White reconstruction noise** (per-frame, uncorrelated): wrist step median
  17-20 mm/frame with ~8 mm high-frequency residual. Now damped by one
  `[1, 2, 1] / 4` binomial pass over the blended camera-space hands
  (`temporal_blend.smooth_hand_trajectory`, config `hand.smooth_passes`, one
  by default) - per hand, only inside valid runs, edges and gaps untouched,
  vertices smoothed with the same taps. Wrist step drops to 11 mm/frame;
  MPJPE is unchanged-to-slightly-better (the GT punishes noise).
* **Mid-frequency oscillation** (~2 deg/frame palm-direction residual,
  p90 4-6 deg; 4-5 px/frame per-vertex): HaWoR's own temporal model. Two and
  three smoothing passes leave it untouched (measured) while smearing real
  motion - not fixable post-hoc; that is a model-quality limit and the honest
  answer to "why does it still wobble a little".
* Ruled out: tracked-box jitter (2-3 px high-frequency residual on the box
  centres), valid-mask flicker (median valid runs 37-96 frames), intrinsics
  drift (the sample camera K is constant), and render-side sort instability
  (the geometry jitter dominates anything the painter's algorithm adds).

`hand.smooth_passes: 1` shipped in `hot3d_p100.yaml` / `unified.yaml` with the
measurement note. Regression test: spike damping + gap/edge preservation +
vertex consistency. Test suite: **410 passed**, 1 skipped.

## 2026-09-30 11:30 (+08:00) - phantom hands fixed: slot-exclusive, continuity-first tracking

The mesh-overlay follow-up exposed what the previous tracker fix had only
hidden: with anchors admitted at confidence 0.5 and no mutual exclusion
between slots, HaWoR reconstructed *phantom* hands. hot3d_ep003 windows
144-175 / 352-375 diverged by up to 980 px because (a) a slot anchored on a
forearm / frame-edge fragment after the real hand left the view (34 % of
ep003's boxes touch the border; the phantoms all sat in the 0.5-0.75
confidence band), and (b) when only one hand was visible BOTH slots tracked
the same detection and the backend fit a second hand to its crop.

`detection/tracker.py` is restructured around a joint two-slot selector
(`select_candidates_joint`):

* one detection feeds at most one slot (mutual exclusion);
* a slot with no same-label candidate may adopt an unused detection only when
  it overlaps the slot's own previous box (IoU >= 0.10, within the recovery
  window) - swapped labels at crossings are corrected by the continuity
  upgrade, including the both-slots-hold-each-other's-hand case (straight
  trade) and steals from the other slot when it has no continuity claim;
* anchors require confidence >= 0.75 again (config: `detection.min_confidence`
  back to the spec value with the rationale inline); the 0.5-0.75 band only
  feeds IoU-gated gap recovery;
* adoption stops after `max_gap` frames without a candidate.

Measured (ep000 / ep003, predicted-only MPJPE): **187.8 / 188.8 mm** (was
190.2 / 191.8 with phantoms at 95 % coverage, 185.4 / 189.5 strict), hand
coverage 63.6 % / 69.3 % (honest gaps instead of phantom hands), ep003
worst-window image error 980 px -> 439 px, slot consistency 11.5 px median.
Visual acceptance on the previously divergent regions: mesh wrapped on the
cube-holding hand at frame 150, both meshes on their hands at frame ~420.
Regressions: crossing-with-swapped-labels, single-hand mutual exclusion,
distant-detection non-adoption. Test suite: **409 passed**, 1 skipped.

## 2026-09-30 10:40 (+08:00) - continuity-first tracking: the mesh no longer jumps hands

User-visible defect: the MANO mesh overlay lagged and periodically sat on the
wrong hand entirely. Diagnosis against the MANO ground-truth projection:

* **Slot swap at hand crossings** (the big one): the tracker trusted WiLoR's
  per-detection handedness label every frame, and the labels swap when both
  hands are in frame - on hot3d_ep000 around frame 240 each slot reconstructed
  the OTHER hand (left-model output sitting in the bowl, box on the right
  hand). `conservative_track` now selects each frame's candidate by IoU
  continuation of its own previous box (within the recovery window, floor
  0.10) and only lets the label decide when continuity is uninformative -
  track start or the stale horizon after a long gap. Regression-tested with a
  synthetic crossing whose labels swap mid-way.
* **No systematic temporal lag**: regressing box-vs-GT-projection error on
  hand velocity gives ±0.15 frames on the right hand (~2 frames on the left,
  x only). The remaining "not exactly on the hand" is the WiLoR box centre
  sitting 43 px (right) / 63 px (left) from the GT joint centroid - a
  detector-box-vs-joint-centroid systematic that HaWoR inherits from the crop,
  not a tracking bug.

Effect (ep000 / ep003): detection coverage 92.2/99.6 % and 91.6/98.7 %, hand
coverage **95.8 % / 95.1 %** (was 78.1 / 87.4 %), predicted-only MPJPE
190.2 / 191.8 mm (was 185.4 / 189.5 - the recovered frames are the hard ones,
same trade as the threshold change). Slot consistency after the fix:
reconstruction 14 px (median) from its own box, 21/413 frames with any slot
nearer the other box. Full chain re-run on both clips, videos re-rendered.

Test suite: **406 passed**, 1 skipped.

## 2026-09-29 23:59 (+08:00) - MANO mesh overlay + detection thresholds adopted; stitch knobs rejected

Two upgrades from the "what would improve this" list, one negative result:

* **Mesh overlay (`02_hawor.mp4` now draws the MANO mesh, not a skeleton).**
  HaWoR's `run_mano` already returns per-frame vertices alongside joints, so
  `hawor_runner` passes them through the same world->camera transform
  (`to_camera_space(..., vertices=...)`, `vertices_camera` in the window and
  clip artefacts), `blend_hand_windows` blends them with the joint weights,
  and `overlay.draw_hand_mesh` rasterises them with a depth-sorted
  painter's algorithm + headlight Lambert shading - no pytorch3d needed (the
  P100 host has no nvcc). Missing frames stay missing; per-hand winding comes
  from `MANO_RIGHT.npz` with `mirror_to_left`'s flip. Without vertices the
  writer falls back to the skeleton. Tests: to_camera_space vertices, blend
  round-trip, rasteriser occlusion/hole cases.
* **Detection thresholds adopted on this host** (`hot3d_p100.yaml`,
  `unified.yaml`): `min_confidence` 0.5 / `max_gap` 8. Measured trade on
  ep000: hand coverage 54.8 % -> 78.1 % (left detection 22.7 % -> 58.7 %),
  wrist error improves, predicted-only MPJPE 183.1 -> 185.4 mm (the recovered
  frames are the hard ones). ep003 under the same policy: hand coverage 87.4 %,
  detection 85.8 % / 89.1 %. The reference profiles stay strict.
* **Stitch knobs rejected**: `pixel_stride` 4 + `ransac_iterations` 512 leave
  predicted-only MPJPE unchanged (185.4 mm) and nudge the camera error up
  (80.6 -> 83.4 mm) - the stitcher is not correspondence-limited; the window
  size is the binding constraint. Defaults stay 8/128
  (`doc_auto/ablation.md`).

Note: the working tree switched to `main` mid-session (reflog: checkout
during the ep003 re-run); all work lives on `feature/pipeline-bootstrap`,
where it was committed as `8c2ca23`. Test suite: **404 passed**, 1 skipped.

## 2026-09-29 23:45 (+08:00) - P2: short-gap interpolation of missing hand poses

The reference pipeline's post-processing stage P2 (interpolate missing poses)
is now implemented as `refinement/gap_fill.py` and runs by default in
`scripts/run_refine.py` (`--no-gap-fill` restores the old behaviour;
`refinement.gap_fill_max_frames`, default 12, caps the interpolated hole
length at 0.4 s / 30 fps). A hand-frame is an anchor only when `hand_valid`
is set *and* every joint is finite; each missing run of at most `max_gap`
frames between two anchors is filled per joint with the linear blend of its
anchors (the trajectory contract carries joint positions, not rotation
parameters, so the figure's SLERP reduces to linear position blending - the
same convention `hand/temporal_blend.py` already uses). Confidence follows the
same blend so the wrist-depth stage treats filled frames like their
neighbours. Leading/trailing missing frames and holes longer than `max_gap`
stay missing.

Every filled frame is recorded in a new required trajectory-contract field
`hand_interpolated` (`[T, 2]` bool; fusion and the HOT3D reference emit
all-`False`), and `evaluate_hot3d.py` reports the predicted-only numbers
next to the headline ones whenever the mask is non-empty, so interpolated
and predicted coverage can never be confused. Long holes stay missing: on
`hot3d_ep000` the left hand's long blind stretches are untouched by design.

Measured on the current artefacts (the on-disk `trajectory_raw.npz` valid
rates - 78.1 % on `hot3d_ep000`, 87.4 % on `hot3d_ep003` - postdate the
detection-coverage numbers quoted above; both baselines re-measured here):
gap fill interpolates 40 hand-frames on `hot3d_ep000` (16 holes) and 33 on
`hot3d_ep003` (9 holes), raising evaluator coverage 77.9 -> 82.1 % and
86.6 -> 89.2 % at +6.3 mm / +1.8 mm aggregate MPJPE (the fabricated frames
are, as expected, less accurate than predicted ones; the predicted-only
numbers are unchanged by construction). On the mock scene, where the planted
10-frame hole sits on smooth motion, P2 improves both columns:
25.90 -> 25.81 mm at 90.7 -> 92.0 % coverage. Full tables in
`doc_auto/ablation.md`; reports in `outputs/hot3d_ep0{00,03}_trajectory{,_gapfill}_report.json`.
Suite: 404 passed.

## 2026-09-29 20:50 (+08:00) - focal fix validated on MANO ground truth; first scored real run

`data/hot3d` turned out to carry the same partial-sync damage as the meta
files: in `hot3d_ep000` 371 of 450 frames were 0 bytes and every artefact npz
(`ground_truth`, `hand_camera`, `trajectory*`) was a 262 144-byte truncated
stub. `hot3d_ep000` was re-imported from `data/samples/lerobot_v3`
(`scripts/import_lerobot.py --overwrite`; 450 frames + MANO 21-joint reference,
coverage left 96.4 % / right 100 %). `hot3d_ep003` is damaged the same way and
still needs a re-import; `hot3d_real000`, `real01`, `real24` are intact.

The full real chain then ran on `hot3d_ep000` on this host (aius-01, 3x P100):
WiLoR detection 24 s -> 112 VGGT windows of 8 frames (`camera.window` raised
4/2 -> 8/4 now that the cards are idle; 8 is still the sm_60 ceiling - no
flash attention) -> Sim(3) stitch with 81-98 % inliers -> HaWoR 16/8 in 2 min
(`camera_source: vggt`, focal 227.48 px) -> fusion -> refine. Scored against
the MANO reference:

| Hand focal | Action MPJPE | Wrist | Depth |
| --- | --- | --- | --- |
| 600 px (HaWoR's silent default) | 664.72 mm | 528.46 mm | 585.14 mm |
| 227.48 px (resolved from Phase 3 windows) | **183.08 mm** | **57.35 mm** | **147.83 mm** |

Identical camera trajectory in both rows, so the 3.6x drop is the focal length
alone. The refine stages move aggregate MPJPE by < 1 mm on real data (wrist
58.35 -> 57.35 mm still comes from the wrist-depth stage); full tables and the
caveats (left-hand detection 22.7 %, window-8 VGGT) are in
`doc_auto/ablation.md`, raw reports in `outputs/hot3d_ep000_{ablation,focal_fix,focal600}.json`.

One more defect surfaced while producing the 600 px control row, in code this
changeset had already staged: `hawor_runner.run_model` wrote our tracking into
`tracks_0_450/` *before* `invalidate_stale_hand_cache` ran, so a re-run with a
changed focal - the exact scenario the invalidation exists for - deleted the
tracks it had just written and HaWoR found no input. The invalidation now runs
first. The host deviations live in a new `configs/hot3d_p100.yaml` (shared
`ego3d` env, fp16 for VGGT, 8/4 windows) while `configs/hot3d.yaml` is
restored to the hardware-independent reference (200/40, per-backend envs) that
`test_shipped_configs_are_valid` pins; the test now covers both files.

Test suite: **392 passed**.

Visual refresh for the real run: `02_hawor.mp4` re-blended from the restored
227 px windows (the 600 px control run had overwritten it), `gt_vs_pred.mp4`
+ 6 stills re-rendered against the MANO reference (the old file was another
0-byte stub), and the VGGT reconstruction set (point cloud turntable, depth
overlay) produced from the real 112 windows. `render_vggt_reconstruction.py`
needed a NumPy-2 fix (`ndarray.ptp()` -> `np.ptp`). The script's "camera
height 6 cm -> SUSPICIOUS depth scale" warning is a false alarm here: it
assumes a floor-anchored world, but the stitched world is normalized to
World-0 (frame 0's camera at the origin), so the camera sits at ~0 by
construction.

Every debug video used to come out as OpenCV's `mp4v` (MPEG-4 Part 2), which
browsers and most default players refuse to open. `overlay.py` now gains
`transcode_to_h264`: after OpenCV releases its writer, ffmpeg re-encodes the
file in place to H.264/yuv420p/+faststart (best effort - without ffmpeg the
mp4v file is kept and the skip logged, never raised). All writer call sites
(the two overlay videos, the wrist comparison - missed on the first pass,
caught because ep003's fresh `gt_vs_pred.mp4` still probed as mpeg4 - and both
VGGT renders) go through it, and the five existing hot3d_ep000 videos were
transcoded in place (frame counts verified).

`hot3d_ep003` - the clip whose 0-byte `gt_vs_pred.mp4` prompted all this - was
then rebuilt for real: re-imported with a MANO reference and run through the
full chain (focal 232.82 px, hand coverage 67.2 %): **Action-MPJPE 187.35 mm**
(wrist 81.36 mm), within 4.3 mm of ep000 despite a very different detection
profile (left 68.2 % / right 66.2 % vs 22.7 % / 86.9 %) - the ~180 mm level is
the pipeline's operating point at window 8, not one episode's fluke. To watch
the results without fighting players, `outputs/visual_gallery.html` embeds
both clips' videos and stills, served at
`http://localhost:8899/outputs/visual_gallery.html` by
`python -m http.server 8899 --bind 127.0.0.1` on aius-01 (VS Code Remote-SSH
forwards the port automatically).

Also ran the detection-coverage trade-off on ep000 (results in
`doc_auto/ablation.md`, raw `outputs/hot3d_ep000_det_ablation.json`):
`min_confidence` 0.75 -> 0.5 + `max_gap` 4 -> 8 lifts hand coverage
54.8 % -> 78.1 % (left detection 22.7 % -> 58.7 %) and slightly improves the
wrist error, while MPJPE stays ~flat (183.1 -> 185.3 mm - the recovered frames
are the hard ones). The shipped configs keep the strict baseline; flipping the
two knobs is a runtime `--set`. The error budget at the operating point is
camera/depth dominated (80-118 + 83-153 mm), not hand-dominated (wrist
57-81 mm) - which is what makes the VGGT window size the real lever.

## 2026-09-28 12:50 (+08:00) - first real end-to-end run on the aius server

`aius` (3x Tesla P100-12GB, driver 580) now runs the pipeline with the **real**
WiLoR + HaWoR + VGGT-Omega backends. Getting there turned up seven genuine
defects, none of which the mock path could see:

1. **`device: auto` resolved to CPU on a GPU host.** The orchestrator has no
   torch by design, so `cuda_available()` said `False` and every backend received
   `--device cpu` (VGGT then refuses outright). `runtime/device.py` now falls back
   to `nvidia-smi -L` when torch is missing; `EGO3D_FORCE_CPU=1` still wins.
2. **HaWoR's checkpoint was being restored onto the GPU.** `HAWOR.load_from_
   checkpoint` hands Lightning's own `_default_map_location` to `torch.load`,
   which picks CUDA: the 3.35 GB state dict was materialised on the card *and*
   then copied again by Lightning's `model.to(device)`. That - not inference,
   crops, window size or dtype - is why every attempt OOMed at the identical
   4.32 GB. The runner now steers `map_location` to CPU (an explicit
   `str`/`torch.device` from a caller still wins).
3. **The renderer stub returned the wrong mask rank.** PyTorch3D is absent (no
   nvcc/gcc), so a stub stands in for `lib.vis.renderer`; it returned a
   `(1, H, W)` mask while HaWoR accumulates `model_masks[frame] += mask` into a
   `(T, H, W)` array - "non-broadcastable output operand".
4. **The synthesised SLAM npz was float64.** `hawor_infiller` feeds it straight
   into `torch.einsum` beside float32 model outputs ("expected scalar type Double
   but found Float"). DROID-SLAM's own file is float32; ours is now too.
5. **The landmarks array was hand-major.** `(2, T, 21, 3)` cannot reach a
   `"tji,thnj->thni"` einsum, and the validity mask is frame-major. The conversion
   is now a tested helper (`backends/hawor_runner.py::to_camera_space`) that
   validates shapes and keeps NaN wherever either validity source fails.
6. **Relative paths followed the chdir.** HaWoR must run from inside its checkout,
   so `--camera-windows data/<clip>/camera/windows` resolved into the checkout and
   reported "holds no *.npz" while 39 windows sat there. `absolutize_paths()`
   resolves frames/out-dir/detection/camera-windows up front.
7. **One precision knob for two very different backends.** VGGT-Omega halves its
   aggregator (2.87 GB resident instead of 4.3), which is what fits next to
   another job on a 12 GB card; HaWoR does not survive fp16 (see 4). `unified.yaml`
   now separates `backends.precision` (VGGT) from `backends.hand_precision`
   (HaWoR), and both runners gained `--precision` and `--crop-size`.

Also: runner failures now print a full traceback (the one-line JSON summary is
unchanged); the server's `data/samples/lerobot_v3/meta/{info.json,tasks.parquet}`
were 0 bytes from a partial copy and had to be re-synced; and every command needs
the env's `bin/` on PATH or `ffprobe` is reported missing.

What the real run produced on `real01` (79 frames, 512x512, 30 fps):

| Stage | Result |
| --- | --- |
| Phase 1 WiLoR | real detector, coverage left 99 %+ / right 93 %+ |
| Phase 3 VGGT-Omega | 39 windows (4-frame schedule), fp16 aggregator on a shared P100 |
| Phase 4 Sim(3) stitch | 38 alignments, scale 1.005-1.027/pair, 86-98 % inliers, 9-42 mm rmse, < 2 deg |
| Phase 2 HaWoR | camera-space hands, 62.7 % coverage, driven by the VGGT trajectory (no DROID-SLAM) |
| Phase 5/6 fusion + refine | left 31.6 % / right 93.7 % valid; bone-scale 18.8 -> 16.0 %; wrist depth mean 9.2 mm |

The reference windows (camera 200/40, hands 16/8) still need an idle GPU: with
another job holding ~7 GB of each P100 only ~4.7 GB is free, which is why
`unified.yaml` pins 4/2 and reports that degradation instead of hiding it.

Test suite: **390 passed**.

## 2026-09-26 22:35 (+08:00) - M1: window sharding, idempotent re-runs, batch dispatch

Added the scheduler-agnostic half of the distributed pipeline (design in
[`distributed.md`](distributed.md)). No stage's *output* changed; what changed is
that a stage can now be run in slices, re-run cheaply, and dispatched to a host
that declares it can run it.

New modules:

* `runtime/sharding.py` - `--shard i/N` and `--window-range a-b` as a pure
  partition of the **global** window schedule (interleaved ownership, so the
  partition is order-independent). `FrameRange` is the tested seam for the M2
  detection slicer.
* `runtime/provenance.py` - `params_hash` over parameters + input **contents** +
  the shard selection, and `.provenance/<unit>.done.json`. Hashing never looks at
  mtime, because mtimes are not comparable across machines.
* `runtime/executor.py` - declared `HostCapabilities` (backends, CUDA, VRAM,
  paths, concurrency) matched without logging into a worker, plus `local` and
  `ssh` executors. `slurm`/`k8s` extend `build_executor` without touching the
  scheduler.
* `runtime/batch.py` + `scripts/run_batch.py` - clip manifest -> unit plan ->
  parallel dispatch, bounded retries, degraded marking, and a ledger.

Changed:

* `scripts/run_hand.py` and `scripts/run_camera.py` accept `--shard`,
  `--window-range`, `--skip-existing`; `run_hand.py` also gained `--blend-only`
  (the whole-clip join step after sharded hand units).
* `backends/{vggt,hawor}_runner.py` implement the same flags (validated against
  the schedule; a selection that matches nothing is an error, never a silent
  no-op). `backends/wilor_runner.py` **rejects** frame-level slicing with the
  reason, because the tracker's gap recovery is frame-coupled.
* `backends/_shard_cli.py` is the shared runner-side contract, so mock and real
  runners agree on what a shard is and write the same markers.
* `configs/{clips.example,hosts.example,hosts.local}.yaml` and
  `make batch` / `make dry-batch`.

Proof, all on CPU with the mock backend:

* `tests/test_batch_e2e.py` - a 2-shard run is **array-for-array identical** to an
  unsliced run, the blend assembled from the shards equals the whole-clip blend,
  and a second `--skip-existing` run performs zero computation.
* `tests/test_batch.py` - a failing unit is retried `retries + 1` times, marks its
  clip `degraded` in both the ledger and `metadata.json`, skips the clip's later
  stages, and writes **no** artefact or marker.
* `tests/test_sharding.py`, `tests/test_provenance.py` - partition/union/
  no-duplicate over both real schedules; stale, corrupt and incomplete markers
  never skip.

Also repaired in the working tree while establishing a baseline: the `100755`
bits on 26 tracked `scripts/*.py|sh` and `backends/*_runner.py` files had been
lost (git recorded `100755`, the checkout was `100644`), which made 26 tests fail
with `PermissionError`. `chmod +x` restored them; no file content changed.

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
real script (`outputs/wrong_world_projection_150.png` keeps the reproduction of
the broken one for comparison). The stale stills from the 2026-09-23 runs in
`data/hot3d/hot3d_ep000/visualization/gt_vs_pred_stills/` were regenerated too -
their filenames depend on the still-index selection, so the README now points at
the explicitly named `outputs/` copies instead of a still path.

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

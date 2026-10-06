# Ablation table

Last modified: 2026-10-06 19:15 (+08:00)

## Mock backend (CPU, 300 frames, deterministic - plumbing validation only)

Produced by `bash scripts/demo_mock_pipeline.sh`, then
`scripts/run_refine.py --output ... <flags>` + `scripts/evaluate_hot3d.py`
against `backends/mock_backend.py truth`. Re-measured 2026-10-01 from one run
of the current code (the Phase-2 binomial pass landed 09-30, so the older rows
shifted by a few tenths of a millimetre).

| Pipeline | Action MPJPE | Coverage | Wrist error |
| --- | --- | --- | --- |
| raw (no post-processing) | 24.6393 mm | 90.67 % | 22.92 mm |
| + camera filter (3-frame binomial) + wrist depth | 26.7479 mm | 90.67 % | 16.06 mm |
| + camera filter + bone scale (<= 3.5 %) | 24.0959 mm | 90.67 % | 22.92 mm |
| + gap fill (P2, `max_gap` 12) | 25.9840 mm | 92.00 % | 16.03 mm |
| **Final** (+ UKF + RTS, P3) | **12.4935 mm** | **92.00 %** | **10.86 mm** |

Pipeline wall time 17.92 s for 300 frames -> 16.74 FPS on CPU (an earlier
quiet-machine run measured 5.65 s / 53.10 FPS; the stage composition is what
matters here). Gap fill is the only stage that moves coverage: the planted
10-frame right-hand hole sits on smooth motion, so the linear blend beats the
mock's per-frame depth noise and both columns improve. P3 then halves the
error because the mock's corruption (per-frame white depth noise + bone
wobble) is exactly the component an RTS-smoothed UKF removes - on real HaWoR
output the same stage moves aggregate MPJPE by < 0.5 mm (see below).

### How to read this

The mock injects its own corruption model: independent per-joint depth noise
(sigma = 30 mm, i.e. what a monocular hand reconstruction actually suffers from)
plus a per-frame whole-hand bone wobble of +/- 12 %. That is deliberately
*different* from the reference system's real HaWoR artefacts, so these numbers
validate the machinery - stage plumbing, post-processing effects, the evaluation
protocol - and must not be read as an algorithmic result. The directly
attributable effect is visible: the ray-constrained wrist-depth stage cuts the
wrist error from 22.92 mm to 16.06 mm, exactly what it is for.

## Reference numbers (from the source blog, for orientation only)

| System | Action MPJPE | Coverage | FPS |
| --- | --- | --- | --- |
| Macrodata final | 52.0435 mm | 81.23 % | 15.53 |
| HaWoR reference | 59.1198 mm | 87.11 % | 3.34 |

These were measured on ten HOT3D episodes chosen by the blog author and are not a
promise for this implementation.

## Real data - hot3d_ep000 (real WiLoR + HaWoR + VGGT-Omega, aius-01 P100s)

Produced 2026-09-29 on `aius-01` (3x Tesla P100-12GB) with
`configs/hot3d_p100.yaml` (`camera.window` 8, overlap 4 — the P100 maximum, see
the note below; `configs/hot3d.yaml` stays the hardware-independent reference
at 200/40). 450 frames, 512x512 @ 30 fps; the reference is MANO forward
kinematics with 21 joints (coverage left 96.4 % / right 100 %); real WiLoR
detection covers left 22.7 % / right 86.9 % of frames, so 54.8 % of
hand-frames are valid and the rest are kept missing. All numbers from
`scripts/evaluate_hot3d.py`; the raw reports live in
`outputs/hot3d_ep000_ablation.json`, `outputs/hot3d_ep000_focal_fix.json`,
`outputs/hot3d_ep000_focal600.json`.

### Focal length (the dominant factor)

HaWoR silently falls back to a hard-coded 600 px focal when none is passed,
which unprojects every crop at the wrong depth. `scripts/run_hand.py` now
resolves the real focal (here 227.48 px from Phase 3's camera windows). Same
clip, same detections, same camera trajectory — only the focal differs:

| Hand focal | Action MPJPE | Wrist error | Depth error | Camera error |
| --- | --- | --- | --- | --- |
| 600 px (HaWoR's silent default, pre-fix behaviour) | 664.72 mm | 528.46 mm | 585.14 mm | 82.47 |
| **227.48 px (resolved from VGGT windows)** | **183.08 mm** | **57.35 mm** | **147.83 mm** | 82.47 |

The camera error is byte-identical between the rows (one shared Phase 3 run),
so the 3.6x drop is attributable to the focal length alone.

### Second episode: hot3d_ep003 (re-imported 2026-09-29, previously 0-byte stubs)

Same chain and config; this episode's detection is balanced across hands
(left 68.2 % / right 66.2 %, hand coverage 67.2 %, focal 232.82 px):

| Episode | Action MPJPE | Wrist error | Coverage | Hand frames |
| --- | --- | --- | --- | --- |
| hot3d_ep000 | 183.0813 mm | 57.35 mm | 54.78 % | 493 |
| hot3d_ep003 | 187.3548 mm | 81.36 mm | 67.22 % | 605 |

Two episodes with very different detection profiles land within 4.3 mm of each
other - the ~180 mm level is the pipeline's current operating point at
window 8, not an artefact of one episode's hand mix. Per-episode reports:
`outputs/hot3d_ep000_ablation.json`, `outputs/hot3d_ep003_real.json`; browse
all videos at `outputs/visual_gallery.html` (served by
`python -m http.server 8899 --bind 127.0.0.1`).

### Detection coverage vs accuracy (ep000, 2026-09-29)

Three stacked changes, each measured separately:

1. **Relaxed tracker thresholds** (`detection.min_confidence` 0.75 -> 0.5,
   `detection.max_gap` 4 -> 8; WiLoR itself still detects at conf 0.1) - a lot
   of continuity for ~nothing:

| Detection | Coverage L/R | Hand coverage | Action MPJPE | Wrist |
| --- | --- | --- | --- | --- |
| min_conf 0.75 / gap 4 (reference profile) | 22.7 % / 86.9 % | 54.8 % | 183.08 mm | 57.35 mm |
| min_conf 0.5 / gap 8 (**adopted in `hot3d_p100.yaml` / `unified.yaml`**) | 58.7 % / 97.8 % | 78.1 % | 185.32 mm | 56.77 mm |

2. **Continuity-first, mutually-exclusive tracker** (fixes the mesh "jumping to
   the other hand" and the phantom hands). Two defects shared one root: the
   tracker trusted WiLoR's per-detection handedness label every frame.
   - Label swaps at crossings: hot3d_ep000 frame ~240 - each slot
     reconstructed the OTHER hand.
   - Phantom hands: hot3d_ep003 windows 144-175 and 352-375 - when the real
     hand left the view (or only one hand was visible), a slot anchored on a
     forearm / frame-edge fragment (34 % of ep003's boxes touch the frame
     border) or both slots tracked the SAME single-hand detection, and HaWoR
     reconstructed a second hand from that crop. Divergence up to 980 px.
   The shipped tracker (`select_candidates_joint`) therefore: assigns each
   detection to at most one slot; lets a slot adopt an unused detection only
   when it overlaps the slot's own previous box (IoU >= 0.10, within the
   recovery window); trades label-matched boxes for continuity when they
   disagree; and **anchors only at confidence >= 0.75** - the 0.5-0.75 band on
   this lens is mostly forearm/edge fragments and only feeds IoU-gated gap
   recovery. Measured (predicted-only MPJPE):

| Tracker state | Detection L/R (ep000 / ep003) | Hand coverage | MPJPE | Worst-window image error (ep003) |
| --- | --- | --- | --- | --- |
| label-only, strict | 22.7/86.9 %, 68.2/66.2 % | 54.8 % / 87.4 %* | 183.1 / 189.5 mm | 910-982 px |
| + relaxed anchors 0.5 | 58.7/97.8 %, 85.8/89.1 % | 78.1 % / 87.4 % | 185.4 / 189.5 mm | 910-982 px |
| + continuity (no exclusion) | 92.2/99.6 %, 91.6/98.7 % | 95.8 % / 95.1 % | 190.2 / 191.8 mm | 910-982 px |
| **+ exclusion & 0.75 anchors (shipped)** | 34.9/92.2 %, 70.7/68.0 % | 63.6 % / 69.3 % | **187.8 / 188.8 mm** | **423-439 px** |

   *the 87.4 % row predates the divergence analysis. Slot consistency after
   the shipped state: reconstruction 11.5 px (median) from its own box,
   3/154 frames with any slot ambiguity; the previously catastrophic regions
   now reconstruct correctly (frame 150: mesh wrapped on the cube-holding
   hand; frame ~420: both meshes on their hands). MPJPE improves over the
   phantom-inflated states because garbage frames no longer dilute the
   average; ep000's left-hand coverage drops to 34.9 % because most of its
   true detections live in the unusable 0.5-0.75 band - honest gaps instead
   of phantom hands.

3. What remains as "the mesh is not exactly on the hand": the WiLoR box
   centre sits 43 px (right) / 63 px (left, p90 167 px) from the GT-projected
   joint centroid - a detector-box vs joint-centroid systematic, with no
   systematic temporal lag (right hand: ±0.15 frames). HaWoR reconstructs
   where the crop is, so it inherits that bias.

The error budget at the operating point is dominated by world placement, not
hands: camera error 80-118 mm + depth error 83-156 mm per episode vs wrist
error 57-88 mm - i.e. VGGT at window 8 with 111 Sim(3) stitches per clip (the
sm_60/P100 ceiling) and the scale drift they accumulate (bone-scale measured
17-20 % hand-size wobble before correction).

### Stitch knobs: denser correspondences do not help (negative result)

`stitch.pixel_stride` 8 -> 4 + `ransac_iterations` 128 -> 512 on ep000:
camera error 80.6 -> 83.4 mm, predicted-only MPJPE unchanged (185.4 mm). The
stitcher is not correspondence-limited; the residual error lives in the
8-frame windows themselves (sm_60 has no flash-attention). Defaults stay
8/128; the only real camera-side lever is bigger windows on better hardware.

### Finger-depth bias: the dominant HaWoR failure and the box-padding fix

Why the mesh "looks bad" while the wrist is fine (2026-09-30 diagnosis, all
measured on ep000/ep003 camera-space hands vs the GT camera frame):

* **Wrist placement is good** (29-57 mm) and **xy is near-perfect** (5-13 mm
  mean over all 21 joints). The failure is **depth**: the per-joint z error
  grows monotonically from the wrist (+1 cm) to the fingertips
  (**+22-27 cm**), biased away from the camera, in HaWoR's *raw
  motion-estimation output* - i.e. it is not the infiller, not the world
  round trip, not blending/smoothing, and not tracked-box jitter (2-3 px).
* **Cause (measured)**: the tracked boxes are ~27 % narrower than the hand
  (85 px vs 117 px expected), so the 256 px crops cut the fingers; HaWoR
  then guesses the depth of what it cannot see. Widening the boxes before
  HaWoR (`detection.box_padding: 1.5`, new runner flag `--box-pad`) removed
  ~20 % of the finger-z bias and brought Action-MPJPE on ep000 from
  **187.9 -> 161.6 mm** (wrist trades up slightly, 38 -> 57 mm). ep003:
  188.5 -> 185.6 mm.
* **Cache correctness fix**: HaWoR short-circuits to its cached
  reconstruction whenever `est_focal.txt` matches - it never saw the padded
  boxes on the first attempt. The invalidation marker now records
  `{focal, box_pad}` and drops the cache when either changes.
* **Hard floor**: even padded, the residual z bias (~10 cm at fingertips) is
  bound by input quality - the sample footage is 512x512, 2.75x below the
  Aria sensor's native 1408x1408, with egocentric motion blur. The reference
  system's 32.35 mm camera-space error was measured on its own episodes at
  full quality on an H100.

### Post-processing stages (all at the resolved focal)

| Pipeline | MPJPE | Wrist error | Coverage |
| --- | --- | --- | --- |
| raw (fusion output, no post-processing) | 183.2169 mm | 58.35 mm | 54.78 % |
| + camera filter + wrist depth | 182.3550 mm | 57.35 mm | 54.78 % |
| + camera filter + bone scale | 183.8787 mm | 58.13 mm | 54.78 % |
| **Final** (all three) | **183.0813 mm** | **57.35 mm** | 54.78 % |

Unlike on the mock scene, the refine stages move aggregate MPJPE by less than
1 mm here; the wrist-depth stage still improves the wrist error (58.35 ->
57.35 mm) and remains worthwhile, while bone scale is neutral-to-slightly-
negative on real HaWoR output. This is expected: the mock's corruption model
(synthetic depth noise + bone wobble) is far larger than HaWoR's real per-frame
artefacts.

### Short-gap interpolation (P2, added 2026-09-29 23:45)

`refinement/gap_fill.py` fills missing runs of at most
`refinement.gap_fill_max_frames` (default 12) hand-frames between two valid
anchors with the per-joint linear blend of the anchors and marks every filled
frame in the new `hand_interpolated` contract field; `evaluate_hot3d.py`
reports the predicted-only numbers next to the headline ones whenever the
mask is non-empty. Measured on the current artefacts - the on-disk
`trajectory_raw.npz` valid rates (78.1 % on ep000, 87.4 % on ep003) postdate
the detection-coverage numbers quoted above, so the no-fill baselines here
differ from the older tables:

| Episode / variant | MPJPE | pred-only MPJPE | Coverage | interpolated |
| --- | --- | --- | --- | --- |
| hot3d_ep000, no fill | 185.32 mm | = | 77.89 % | - |
| hot3d_ep000 + P2 | 191.62 mm | 185.39 mm | 82.11 % | 4.44 % |
| hot3d_ep003, no fill | 189.45 mm | = | 86.56 % | - |
| hot3d_ep003 + P2 | 191.25 mm | 189.46 mm | 89.22 % | 3.67 % |

Reading: P2 trades a little aggregate MPJPE for coverage. The fabricated
frames are, as expected, less accurate than predicted ones (+6.3 mm / +1.8 mm
overall), while the predicted frames themselves are untouched - the pred-only
column matches the no-fill baseline, with the small residual coming from
wrist-depth now optimising longer merged segments. This is the same trade the
reference system makes when it reports 81 % coverage. The left hand's long
blind stretches on ep000 stay missing: holes longer than `max_gap` are never
extrapolated. The baselines in this table predate the 2026-10-01 box-padding
re-run (their old `outputs/*.json` reports were cleared with the rest of
`outputs/`); the same-source table under P3 below supersedes them and its
reports live next to the artefacts in `data/hot3d/<clip>/trajectory/`.

### UKF + RTS temporal smoothing (P3, added 2026-10-01)

`refinement/ukf_smooth.py` ports the reference pipeline's `smooth_ukf_cam`
(their P3): a per-channel constant-velocity UKF over the valid frames of each
hand, observation scale from the MAD of second differences, speed-adaptive
observation noise, and an unscented RTS backward pass. The filter code keeps
the reference's defaults (`q = r = 0.6`, `beta = 2.0`); the shipped configs
carry the reference *UI*'s default instead - `q 0.7, r 0.5, beta 0.3`
("lighter smoothing"; recommended q 0.4-1.0, r 0.3-1.0, beta 0.2-3.0, safe
q/r 0.1-2.0, beta 0-5, matching `PARAM_LIMITS`). `refinement.ukf_q/r/beta/rts`
override, `--no-ukf-smooth` disables, and the resolved values are recorded in
`trajectory/metadata.json`. It runs in camera space after wrist depth and
before the world transform; missing frames are neither read nor written, and a
hand with fewer than 4 valid frames is left untouched.

Same-source comparison on the 2026-10-01 artefacts (ep000/ep003 re-run with
box padding; all three variants produced from the same `trajectory_raw.npz`):

| Episode / variant | MPJPE | pred-only MPJPE | Coverage | Wrist | Wrist accel (L/R) |
| --- | --- | --- | --- | --- | --- |
| ep000 baseline (no P2/P3) | 161.59 mm | = | 63.56 % | 67.37 mm | 6.7 / 8.8 mm |
| ep000 + P2 | 162.77 mm | 161.56 mm | 65.33 % | 67.19 mm | 6.5 / 8.8 mm |
| ep000 + P2 + P3 (shipped 0.7/0.5/0.3) | 162.61 mm | 161.40 mm | 65.33 % | 66.96 mm | **3.2 / 4.1 mm** |
| ep003 baseline | 185.66 mm | = | 69.33 % | 84.10 mm | 13.0 / 13.7 mm |
| ep003 + P2 | 185.65 mm | 185.64 mm | 71.78 % | 84.59 mm | 12.7 / 13.7 mm |
| ep003 + P2 + P3 (shipped 0.7/0.5/0.3) | 185.43 mm | 185.41 mm | 71.78 % | 84.25 mm | **5.2 / 5.8 mm** |

"Wrist accel" is the median over valid frames of the mean per-joint
acceleration magnitude (second difference; frames where three consecutive
hand-frames are valid). Reading: P3 is a jitter killer, not an MPJPE mover -
on real HaWoR output the aggregate moves < 0.5 mm while the frame-to-frame
acceleration drops 52 % (ep000) and 60 % (ep003) at the shipped parameters.
The Phase-2 binomial pass (2026-09-30 changelog) only damps the white-noise
component; P3 removes the rest of the high-frequency wobble, which is what the
eye sees in the mesh videos. On the mock, where the corruption is per-frame
white noise, the same stage halves the error (25.98 -> 12.49 mm; the mock
config keeps the library defaults). Reports:
`data/hot3d/<clip>/trajectory/eval_{v_base,v_p2,p2p3}.json`.

**Parameter sweep** (2026-10-01, same two episodes; each row is a full refine
run with only `refinement.ukf_*` changed). Every set keeps MPJPE inside
0.6 mm - the knob buys smoothness, not accuracy - so the choice is
"smoother vs more follow-through"; the guidance from the reference UI is
q up = follows the hand more, r/beta up = smoother:

| UKF set (q/r/beta) | ep000 MPJPE · accel L/R | ep003 MPJPE · accel L/R |
| --- | --- | --- |
| baseline (no P3) | 161.59 mm · 6.7 / 8.8 mm | 185.66 mm · 13.0 / 13.7 mm |
| follow 1.0 / 0.3 / 0.2 | 162.69 mm · 4.8 / 6.5 mm | 185.54 mm · 9.1 / 10.0 mm |
| **shipped 0.7 / 0.5 / 0.3** (reference UI default) | 162.61 mm · 3.2 / 4.1 mm | 185.43 mm · 5.2 / 5.8 mm |
| library 0.6 / 0.6 / 2.0 | 162.57 mm · 2.8 / 3.6 mm | 185.29 mm · 4.1 / 4.5 mm |
| heavy 0.5 / 1.0 / 3.0 | 162.49 mm · 2.7 / 3.0 mm | 185.11 mm · 3.4 / 3.6 mm |

On these two episodes the heavier sets measure weakly better on all three
columns, so the sub-millimetre spread is what a default is choosing between;
the shipped set follows the reference product's default, and the heavier
library set is one config edit away. Sweep artefacts:
`data/hot3d/_p3_sweep/`.

### Cross-participant subset of HOT3D-Clips (2026-10-01, generalisation check)

The repo's mirror of the **HOT3D-Clips** challenge set (135 clips across three
participants, 1408×1408, 81 frames each, imported by
`scripts/import_hot3d_clips.py`) is a different capture campaign from the three
512×512 episodes every stage above was tuned on. A 15-clip subset (5 per
participant, indices spread) ran the full real chain on the 3 P100s
(`configs/clips.cross_subset.yaml`; ~25 min wall). What can be scored here is
constrained by the mirror's reference quality (see the importer docstring):
**the camera GT is valid** (verified 7 mm median vs VGGT on a static clip),
**the hand/wrist GT is not scoreable** - it projects into the ceiling, and the
spot-check render makes it obvious (predicted wrists land on the real wrists,
the GT wrist markers float on the whiteboard).

| clip | camera err (med) | det cov L/R | hand cov (interp) |
| --- | --- | --- | --- |
| P0015_c000000 | **0.42° / 7 mm** | 100/100 % | 100 % |
| P0015_c000009 | **1.94° / 24 mm** | 100/100 % | 100 % |
| P0015_c000018 | 11.10° / 108 mm | 100/88 % | 99 % |
| P0002_c000016 | 15.52° / 33 mm | 100/100 % | 100 % |
| P0001_c000000 | 19.56° / 131 mm | 80/100 % | 90 % |
| P0002_c000000 | 22.09° / 154 mm | 100/100 % | 100 % |
| P0002_c000032 | 23.52° / 56 mm | 100/80 % | 90 % |
| P0002_c000008 | 34.76° / 318 mm | 100/100 % | 100 % |
| P0002_c000024 | 37.52° / 281 mm | 0/100 % | 50 % |
| P0001_c000036 | 37.67° / 149 mm | 53/63 % | 59 % |
| P0001_c000018 | 45.77° / 148 mm | 53/75 % | 65 % |
| P0015_c000027 | 48.49° / 155 mm | 51/65 % | 58 % |
| P0015_c000036 | 52.87° / 239 mm | 100/98 % | 99 % |
| P0001_c000009 | 51.09° / 112 mm | 0/73 % | 46 % |
| P0001_c000027 | 69.42° / 201 mm | 59/78 % | 93 % |
| **median (15)** | **34.8° / 148 mm** | | |

Reading, in three parts:

1. **The chain generalises.** Hand detection on the full-res 1408² clips is far
   stronger than on the 512² episodes (10/15 clips ≥ 80 % both hands vs
   ep000's 35/92 %), the focal resolution, stitch, fusion and P2/P3 stages ran
   unmodified, and the visual spot-checks show wrists on wrists.
2. **The camera numbers are a weak reference, not a leaderboard.** The mirror's
   `camera.json` itself drifts from VGGT by 7-27° during fast head rotation
   (documented in the importer), so mid-range errors are ambiguous; the two
   near-static clips at 0.42°/1.94° are the honest end-to-end validation.
   One-sided detection clips (P0001_c000009, P0002_c000024: one hand at 0 %)
   previously crashed HaWoR - now handled (below).
3. **Two batch defects surfaced and were fixed** (2026-10-01):
   `build_command` added `--blend-only` to *every* whole-clip hand unit, so any
   batch run without `--shards` failed with "HaWoR window(s) are missing"
   (the join is now an explicit unit `mode`, regression-pinned in
   `tests/test_batch.py`); and HaWoR crashed on clips where one hand is never
   detected (`hawor_tracks_from_detection` now drops the empty hand, and the
   runner short-circuits an all-invalid result). `render_gt_vs_pred.py` falls
   back to Phase 3 window intrinsics when the reference `camera_K` is NaN.
   Reports: `data/hot3d/cross_subset_summary.json`,
   `data/hot3d/batch_cross_subset.json`.

### How to read the absolute numbers

183 mm is far from the reference system's 52-59 mm, and three causes are
known, none of them in the refine stages:

1. **Left-hand detection at 22.7 %** — WiLoR loses the left hand for most of
   this episode, so left-hand errors are mostly absent from the average rather
   than small. The reference blog's episodes were selected so detection works.
2. **`camera.window` 8, not 200** — Tesla P100 (sm_60) has no flash-attention
   kernel, so VGGT-Omega runs on the memory-hungry math-attention path and
   OOMs past ~8 frames on 12 GB. Shorter windows mean more Sim(3) stitches and
   more accumulated depth drift (depth error 147.8 mm is the largest error
   component).
3. The remaining gap is algorithmic and unchanged from the reference
   comparison.

## Real-data plumbing check (not an accuracy result)

`WITH_MOCK=1 bash scripts/demo_hot3d_sample.sh` runs Phases 1-6 on the real
`hot3d_ep000` footage (450 frames, 512x512, 30 fps) with the mock backend and
evaluates against the real HOT3D reference:

| Quantity | Value |
| --- | --- |
| camera windows / Sim(3) alignments | 3 windows, 2 alignments, 100 % inliers, 0.0000 m rmse |
| stitched camera coverage | 100 % |
| Action-MPJPE (wrist-level) | 1109.19 mm |
| coverage | 96.78 % |
| referenced joints | 4.61 % |

The 1.1 m error is expected and meaningless: the mock backend substitutes
synthetic hands and a synthetic camera trajectory, so this run only proves that
every stage, the artefact contract and the evaluation survive real-resolution,
real-length, real-motion input.

## HOT3D-Clip P0002: EGO overlay camera-path isolation (2026-10-06)

To check whether the remaining HaWoR mesh offset came from VGGT camera poses,
HaWoR was run twice on `P0002_clip001971` with the same frames, detections,
focal (`623.473 px`), crop padding (`1.5`) and weights. The first run used
`camera_source=vggt-stitched`; the control omitted camera windows and reported
`camera_source=constant`.

| Comparison | Result |
| --- | --- |
| valid mask | identical, 220 hand-frames |
| maximum joint camera-space difference | `1.32e-7 m` |
| maximum vertex camera-space difference | `1.79e-7 m` |
| calibrated-K pixel-error median, left wrist / all joints | `71.6 / 67.1 px` in both runs |
| calibrated-K pixel-error median, right wrist / all joints | `38.6 / 23.9 px` in both runs |

For this clip and backend, the VGGT pose input does not cause the residual in
the EGO hand overlay: HaWoR's camera-space prediction is invariant to the
camera-path choice up to floating-point noise. VGGT's world trajectory remains
a separate issue; the official camera error for this clip is `162.1 mm`, which
affects world-space accuracy. Diagnostic report:
`data/hot3d/P0002_clip001971/trajectory/eval_mano_alignment.json`.

## P0002 calibrated focal rerun: WiLoR and official score (2026-10-06)

The undistorted source-frame calibration is `f=608.544 px`; VGGT estimated
`623.473 px`. WiLoR was rerun with `hand.focal` resolved from
`metadata.json:image_camera`, holding detections, crop rescale (`2.5`) and
smoothing (one pass) fixed. The camera-space 3D wrist error stayed nearly
unchanged, while projected wrist agreement improved on shared hand frames:

| WiLoR focal | left wrist px | right wrist px | left wrist 3D | right wrist 3D |
| --- | ---: | ---: | ---: | ---: |
| VGGT estimate 623.473 px | 84.2 px | 28.0 px | 49.6 mm | 34.7 mm |
| input K 608.544 px | 70.4 px | 19.0 px | 49.1 mm | 35.6 mm |

With the display-only 0.5 detector-box nudge, opening frames 0-40 improve from
31.3 to 23.2 px median wrist error on the lower/left hand; all-joint medians
are 42.3 to 39.2 px. The saved hand arrays now use the input focal, and Phase
5-7 were rerun. Official results changed from Action-MPJPE `120.93` to
`115.8324 mm`, wrist `122.1` to `91.6857 mm`, depth `58.8` to `46.3373 mm`;
the camera error remains `162.14 mm` because the VGGT camera trajectory was
unchanged. The full report is
`data/hot3d/P0002_clip001971/trajectory/eval_official.json`.

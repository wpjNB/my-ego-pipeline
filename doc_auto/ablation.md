# Ablation table

Last modified: 2026-10-09 13:30 (+08:00)

## Mock backend (CPU, 300 frames, deterministic - plumbing validation only)

Produced by `bash scripts/demo_mock_pipeline.sh`, then
`scripts/run_refine.py --output ... <flags>` + `scripts/evaluate_hot3d.py`
against `backends/mock_backend.py truth`.

| Pipeline | Action MPJPE | Coverage | Wrist error |
| --- | --- | --- | --- |
| raw (no post-processing) | 24.6393 mm | 90.67 % | 22.92 mm |
| + camera filter (3-frame binomial) + wrist depth | 26.7479 mm | 90.67 % | 16.06 mm |
| + camera filter + bone scale (<= 3.5 %) | 23.9437 mm | 90.67 % | 22.92 mm |
| + gap fill (P2, `max_gap` 12) on the three stages | 25.8118 mm | 92.00 % | 16.03 mm |
| **Final** (camera filter + wrist depth + bone scale + gap fill) | **25.8118 mm** | **92.00 %** | 16.03 mm |

Pipeline wall time 17.92 s for 300 frames -> 16.74 FPS on CPU (an earlier
quiet-machine run measured 5.65 s / 53.10 FPS; the stage composition is what
matters here). Gap fill is the only stage that moves coverage: the planted
10-frame right-hand hole sits on smooth motion, so the linear blend beats the
mock's per-frame depth noise and both columns improve.

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
extrapolated. Reports:
`outputs/hot3d_ep0{00,03}_trajectory{,_gapfill}_report.json`.

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


## VGGT dense-depth prior for HaWoR (P0001/P0002/P0003, 2026-10-09) - negative pilot

HaWoR's upstream `model.inference` accepts RGB crops, boxes, `img_focal`,
`img_center`, and handedness; it has no dense-depth input. This project already
passes the VGGT-derived focal and stitched camera trajectory. To probe the
remaining depth channel without changing HaWoR weights, a wrapper-side adapter
was tested on the existing camera-space MANO outputs: project vertices using
per-window VGGT K, take the front-most projected vertex per depth pixel, sample
a 3x3 VGGT-depth median inside the tracked box, and translate each hand by the
median depth residual while preserving the wrist pixel. Ground truth was used
only after this adjustment for scoring.

| Clip | Wrist 3D median L/R, baseline -> depth prior | All-joint frame-mean 3D median L/R, baseline -> depth prior | VGGT depth vs GT wrist Z, absolute median |
| --- | --- | --- | ---: |
| P0001_clip001849 | 14.8/22.9 -> 54.4/47.2 mm | 31.7/18.3 -> 52.0/40.1 mm | 14.5 mm |
| P0002_clip001971 | 28.6/13.3 -> 161.7/311.1 mm | 54.0/21.9 -> 181.3/304.5 mm | 186.3 mm |
| P0003_clip002059 | 34.2/55.6 -> 182.6/171.2 mm | 40.7/66.6 -> 181.4/162.0 mm | 129.5 mm |

The per-frame root-Z shift was -93/-213 mm (L/R) on P0002 and about -179 mm
for both hands on P0003. The wrist 2D error was held fixed by construction,
but other joints expanded away from their image locations. Results are worse
on all three clips; the dense VGGT maps are not a reliable direct hand-root-Z
measurement at these pixels, even when their camera intrinsics are used. This
was an output-level geometric pilot, not a retrained HaWoR depth-conditioned
network, so it is not enabled in the pipeline.

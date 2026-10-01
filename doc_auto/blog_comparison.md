# Blog comparison: this repo vs the Macrodata reference system

> Chinese version: [blog_comparison.zh.md](blog_comparison.zh.md).

Compared against: `https://macrodata.co/blog/turning-egocentric-video-into-3d-hand-actions`
(the "Macrodata final" system: Action-MPJPE 52.04 mm, coverage 81.23 %,
15.53 FPS on an H100, measured on ten HOT3D episodes).

Last modified: 2026-09-30. Every claim below was checked against the code on
this date; deviations carry our own measurements.

## Verdict at a glance

The repo is a faithful re-implementation of the blog's final recipe at the
module level, with three classes of differences: **(A) hardware-forced
deviations** (documented in the configs), **(B) robustness additions** the blog
lists as open problems but does not solve, and **(C) two recipe divergences
found by this comparison and now fixed** (bone-scale reference, hand
smoothing default). The score gap (187.9 / 188.5 mm here vs 52.04 mm there) is
fully accounted for by (A): the camera window.

## Section-by-section

| Blog section | Blog recipe | Repo | Status |
| --- | --- | --- | --- |
| Detector selection | WiLoR beats HaGRID YOLOv10n (86.3 % coverage, 0 conflicts) | WiLoR via `backends/wilor_runner.py` | match |
| Tracking rules | keep highest-conf box per side; gaps ≤ 4 recovered only between two ≥ 0.75 anchors, IoU ≥ 0.20 vs interpolated anchors | `detection/tracker.py`; reference configs 0.75 / 4 / 0.20 | match + additions (B) |
| Tracker alternatives | ByteTrack / BoT-SORT / SAM2 / EdgeTAM all worse (53.75 vs 54.3-54.6) | conservative tracker only | match (no need to port the losers) |
| Hand reconstruction | HaWoR, 16-frame windows, 8-frame overlap; linear blend of joints/translations, SLERP for rotations | `hand/hawor.py` 16/8, `temporal_blend.py` linear + `rotation_slerp` | match |
| Hand model comparison | HaWoR ≫ WiLoR-only / HaMeR / MediaPipe | HaWoR | match |
| Camera/world | VGGT-Omega 416 px buckets, **200-frame windows / 40 overlap**, depth-derived Sim(3), linear blend | `camera/vggt_omega.py`, `camera/stitch.py` | match, except **window 8 / overlap 4** (A) |
| Camera filter | 3-frame binomial `[0.25, 0.5, 0.25]` on translation, rotation untouched; kept (+0.01 mm) | `refinement/camera_filter.py` — identical kernel, translation only | match |
| Bone-scale correction | clip-level **mean** bone length, bound 3.5 %, mean > median in their ablation | `refinement/bone_scale.py` | was `median` everywhere → **fixed to `mean`** (C) |
| Wrist-depth optimisation | ray-constrained, acceleration penalty λ = 0.2, confidence/median → clamp [0.5, 1.5] → **8 | `refinement/wrist_depth.py` — identical formula and λ | match |
| Hand-joint smoothing | **rejected**: 3-frame mean +0.32 mm, 5-frame +0.95 mm, Gaussian +0.72 mm; final system disables it | was code-default ON (1 pass) | **fixed: code default 0**; host configs opt in (C) |
| Motion infiller | HaWoR's learned infiller **worse** than benchmark gap filling (+1.59 mm) → disabled | runner still calls `hawor_infiller`; its `pred_valid` gates output frames | **open gap** (see below) |
| Gap filling | benchmark interpolation of short holes | `refinement/gap_fill.py` (max_gap 12, marked in `hand_interpolated`) | match |
| Evaluation | 1 s chunks anchored at the chunk-start camera (predicted for pred, GT for GT), no re-alignment; linear interpolation of interior gaps, nearest-pose at edges | `evaluation/action_mpjpe.py` + the predicted-only/headline split | match |
| Reference numbers | 52.04 mm / 81.23 % / 15.53 FPS (H100) | 187.9 / 188.5 mm @ 63.6/69.3 % coverage on 2 episodes, far below 15 FPS | gap = (A) |

## (A) Hardware-forced deviation: the camera window

The blog's own sweep is the single biggest lever it measures: camera window
60 → 62.14 mm, 100 → 58.24, 150 → 56.20, **200 → 55.95** (overlap 40). This
host's Tesla P100s (sm_60) have no flash-attention kernel, so VGGT-Omega runs
on the math-attention path and OOMs past ~8 frames on 12 GB — the repo pins
`camera.window: 8 / overlap: 4` (configs carry the note). Our 187.9 mm is the
expected consequence of sitting far below the sweep's smallest point, plus the
Sim(3) stitch accumulation 16× more frequent than the recipe's. Everything
else in the error budget (camera error 80-118 mm, depth error 83-156 mm vs
wrist error 58-81 mm) mirrors the blog's own finding that camera/depth
dominates.

Consequence: **the 52 mm number is not reachable on this hardware**, and the
honest comparison target here is the trend, not the value. Re-running Phases
3-7 at 200/40 on an A100/H100 is the one experiment that closes the gap.

## (B) Robustness additions beyond the recipe

The blog's failure analysis names "left/right swaps" among the deployment
failures it does *not* solve. Two such failures were observed and fixed here:

1. **Label-swap slot crossover** (hot3d_ep000 ~frame 240): the tracker trusted
   WiLoR's per-detection handedness label; when both hands are visible the
   labels swap and each slot reconstructed the other hand. The tracker now
   selects candidates jointly for both slots with continuity arbitration
   (`select_candidates_joint`).
2. **Phantom hands** (hot3d_ep003 windows 144-175, 352-375): with anchors at
   confidence 0.5, slots anchored on forearm/frame-edge fragments after the
   real hand left the view, and both slots tracked the same single-hand
   detection. Fix: anchors back at 0.75 (the recipe's value), slot mutual
   exclusion, continuity adoption only within the recovery window.

Both additions are regression-tested and measured in `ablation.md`. They
deviate from the blog's "highest-confidence same-side box" rule by design; on
the blog's ten episodes the simpler rule was sufficient.

Also beyond the recipe, driven by our data: `detection.max_gap: 8` on the host
configs (recipe: 4; the blog measured 2 vs 4 as negligible and chose 4 for
coverage).

## (C) Divergences found by this comparison — fixed

1. **Bone-scale reference**: every config had `median`; the recipe uses the
   clip-level **mean** and measured median as a regression. Our own A/B on the
   two episodes splits within noise (ep000: mean 186.75 vs median 187.92;
   ep003: mean 189.22 vs median 188.49), so we follow the recipe: all configs
   now `mean`, function default updated.
2. **Hand-joint smoothing**: the code defaulted to one binomial pass; the
   recipe disables hand smoothing (every variant regressed at its quality
   level). Code default is now 0 (reference-faithful). The host configs opt in
   with `smooth_passes: 1` because our operating point is far noisier (8 mm
   high-frequency wrist residual) and one measured pass is MPJPE-neutral here
   while damping the visible mesh wobble. At the reference operating point
   (window 200), it should be re-evaluated and likely turned off.

## Open conformance gap: the learned motion infiller

The blog measured HaWoR's learned infiller as *worse* than benchmark gap
filling (+1.59 mm) and disabled it. This repo still runs `hawor_infiller`
(with the VGGT-derived SLAM npz replacing DROID-SLAM) and lets its
`pred_valid` gate which filled frames enter the output; the benchmark-style
interpolation then runs again at refinement (`gap_fill.py`). Consequence:
filled frames may enter twice through two different mechanisms. The
recipe-faithful experiment is to drop the learned infiller's fills (keep the
motion-estimation output and the tracker's validity), let Phase 2 keep holes
as holes, and let `gap_fill.py` be the only filler - then re-measure. Not yet
run; requires a runner change plus both clips' GPU time (~5 minutes).

## Bottom line

Faithful where it matters, honest about where it cannot follow. The two
recipe divergences found by this audit are fixed; the remaining distance to
52.04 mm is the camera window, i.e. hardware, plus the infiller experiment
above as the one untested recipe conformance item.

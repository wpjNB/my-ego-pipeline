# Ablation table

Last modified: 2026-09-29 20:45 (+08:00)

## Mock backend (CPU, 300 frames, deterministic - plumbing validation only)

Produced by `bash scripts/demo_mock_pipeline.sh`, then
`scripts/run_refine.py --output ... <flags>` + `scripts/evaluate_hot3d.py`
against `backends/mock_backend.py truth`.

| Pipeline | Action MPJPE | Coverage | Wrist error |
| --- | --- | --- | --- |
| raw (no post-processing) | 24.6393 mm | 90.67 % | 22.92 mm |
| + camera filter (3-frame binomial) + wrist depth | 26.7479 mm | 90.67 % | 16.06 mm |
| + camera filter + bone scale (<= 3.5 %) | 23.9437 mm | 90.67 % | 22.92 mm |
| **Final** (all three) | 25.8964 mm | 90.67 % | 16.06 mm |

Pipeline wall time 5.65 s for 300 frames -> 53.10 FPS on CPU.

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

Relaxing the tracker (`detection.min_confidence` 0.75 -> 0.5,
`detection.max_gap` 4 -> 8; WiLoR itself still detects at conf 0.1) buys a lot
of continuity and costs nothing meaningful in accuracy:

| Detection | Coverage L/R | Hand coverage | Action MPJPE | Wrist |
| --- | --- | --- | --- | --- |
| min_conf 0.75 / gap 4 (baseline) | 22.7 % / 86.9 % | 54.8 % | 183.08 mm | 57.35 mm |
| min_conf 0.5 / gap 8 | 58.7 % / 97.8 % | 78.1 % | 185.32 mm | 56.77 mm |

The recovered frames are exactly the ones WiLoR was unsure about, so their
errors - previously excused as "missing" - now enter the average and cancel
the gain; the wrist error still improves. Read: for visual continuity and
downstream use, relax the thresholds (one `--set` away); for a leaderboard
number, the thresholds are not what is holding the score back. The error
budget at the operating point is dominated by world placement, not hands:
camera error 80-118 mm + depth error 83-153 mm per episode vs wrist error
57-81 mm - i.e. VGGT at window 8 with 111 Sim(3) stitches per clip (the
sm_60/P100 ceiling) and the scale drift they accumulate (bone-scale measured
17-20 % hand-size wobble before correction).

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

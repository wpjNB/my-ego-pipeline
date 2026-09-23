# Ablation table

Last modified: 2026-09-23 17:22 (+08:00)

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

## Real-data table (to fill on the GPU server)

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

# Implementation status

Last modified: 2026-10-06 16:16 (+08:00)

Test suite: **438 passed**, 1 skipped in ~86 s on a CPU-only interpreter
(`conda run -n ego3d_base python -m pytest -q`). 119 of those are newer than
the M0 suite: the sharding/provenance/executor/batch modules, the sharded E2E
comparison, and the HaWoR focal-resolution + cache-invalidation tests. The
newest additions are the post-processing stages P2 (`tests/test_gap_fill.py`)
and P3 (`tests/test_ukf_smooth.py`).

HaWoR now runs after Phase 4 stitches the VGGT camera windows. Batch provenance hashes those camera inputs, so a changed camera invalidates stale hand output.

The `-vsync` wart is gone: `io/video.py` probes ffmpeg and picks
`-fps_mode passthrough` (5.1+) over `-vsync 0`, so any ffmpeg works.

Asset state on this checkout (aius-01, 3x Tesla P100-12GB): all three
backends' weights are downloaded and verified (10.8 GB), the three
repositories are cloned, MANO is installed, and the backends run in **one
shared conda env `ego3d`** (torch 2.8.0+cu126) instead of the three
per-backend envs - the torch-1.13 pin only existed for DROID-SLAM + Metric3D,
which VGGT-Omega replaces (`configs/unified.yaml` / `configs/hot3d.yaml`
record the decision). All three runners pass `--check` here and have run for
real (see below).

Data caveats from the partial sync that produced this checkout:
`data/hot3d/hot3d_ep000` and `hot3d_ep003` had 0-byte frames and truncated
artefact stubs; both were re-imported from `data/samples/lerobot_v3` with MANO
references and re-run through the real chain on 2026-09-29 (`hot3d_real000`,
`real01`, `real24` are intact; `real24` still needs its Phase 3-6). All debug
videos are transcoded to H.264 after writing (browsers play them);
`outputs/visual_gallery.html` + `python -m http.server 8899 --bind 127.0.0.1`
serves every clip's videos and stills in one page.

## Complete and tested (CPU)

| Area | Module | Tests |
| --- | --- | --- |
| Sim(3) / weighted Umeyama / RANSAC | `geometry/sim3.py`, `geometry/umeyama.py` | `tests/test_sim3.py` |
| Transforms, SLERP, projection | `geometry/transforms.py` | `tests/test_transforms.py` |
| Conservative tracking | `detection/tracker.py` | `tests/test_tracker.py` |
| Handedness handling | `detection/handedness.py` | `tests/test_tracker.py` |
| HaWoR window blending | `hand/temporal_blend.py` | `tests/test_blending.py` |
| Window scheduling | `camera/window.py` | `tests/test_window.py` |
| Depth correspondences + stitching | `camera/depth.py`, `camera/stitch.py` | `tests/test_stitch.py` |
| World fusion + trajectory contract | `fusion/trajectory.py`, `io/serialization.py` | `tests/test_fusion.py`, `tests/test_serialization.py` |
| Camera filter / bone scale / wrist depth | `refinement/*` | `tests/test_camera_filter.py`, `tests/test_bone_scale.py`, `tests/test_wrist_depth.py` |
| Short-gap interpolation (P2) | `refinement/gap_fill.py` | `tests/test_gap_fill.py` |
| UKF + RTS smoothing (P3) | `refinement/ukf_smooth.py` | `tests/test_ukf_smooth.py` |
| Action-MPJPE / coverage / report | `evaluation/*` | `tests/test_action_mpjpe.py` |
| Phase 0 video IO | `io/video.py`, `io/frames.py` | `tests/test_frames_io.py` |
| Synthetic scene / mock data | `testing/synthetic.py` | `tests/test_mock_pipeline.py` |
| Runner protocol | `runtime/subprocess_backend.py` | `tests/test_runner_protocol.py` |
| Work slicing / shard partition | `runtime/sharding.py` | `tests/test_sharding.py` |
| Content-addressed idempotency | `runtime/provenance.py` | `tests/test_provenance.py` |
| Host capabilities + local/ssh executors | `runtime/executor.py` | `tests/test_batch.py` |
| Batch dispatch / retries / ledger | `runtime/batch.py`, `scripts/run_batch.py` | `tests/test_batch.py`, `tests/test_batch_e2e.py` |
| Full pipeline (mock backend) | `scripts/run_pipeline.py` + all stages | `tests/test_mock_pipeline.py` |
| LeRobot v3 reader | `datasets/lerobot.py` | `tests/test_lerobot.py` |
| HOT3D reference conversion | `datasets/hot3d_gt.py` | `tests/test_lerobot.py` |
| Reference-vs-prediction viewer | `visualization/overlay.py` | `tests/test_visualization.py` |
| Wrist-level evaluation | `evaluation/action_mpjpe.py` | `tests/test_hot3d_evaluation.py` |
| MANO forward kinematics | `hand/mano_model.py` | `tests/test_mano_model.py` |
| Model-output conversions | `detection/wilor.py`, `hand/hawor.py`, `camera/vggt_omega.py` | `tests/test_backend_conversions.py` |

## Interfaces in place; real backends run on this host

| Backend | Adapter + runner | Status here |
| --- | --- | --- |
| WiLoR | `detection/wilor.py` + `backends/wilor_runner.py` | Runs for real (hot3d_ep000: 450 frames in 24 s, coverage left 22.7 % / right 86.9 %) |
| HaWoR | `hand/hawor.py` + `backends/hawor_runner.py` | Runs for real, driven by the VGGT trajectory (`camera_source: vggt`, no DROID-SLAM); takes a resolved `--focal` and drops stale caches when the focal changes |
| VGGT-Omega | `camera/vggt_omega.py` + `backends/vggt_runner.py` | Runs for real (112 windows of 8 frames, fp16 aggregator, 8.5 GiB resident on a P100) |

The first real end-to-end run happened 2026-09-28 on `real01` (see changelog);
on 2026-09-29 the full chain ran on `hot3d_ep000` and was scored against the
MANO reference: **Action-MPJPE 183.08 mm** with the resolved focal vs
**664.72 mm** at HaWoR's silent 600 px default (`doc_auto/ablation.md` has the
breakdown). The 09-28 session also fixed seven real-run defects, including
`device: auto` resolving to CPU without torch (now falls back to
`nvidia-smi -L`) and HaWoR's checkpoint being restored onto the GPU.

Every step before the model call is unit-tested, and every failure path names
the missing checkout, weights, checkpoint or device - a half-configured server
still cannot emit an artefact that looks like a successful run.

Running the whole pipeline today works through `backends.mode: mock`
(`configs/mock.yaml`), which substitutes a deterministic stand-in for the three
models, **and** through `backends.mode: real` on this host's P100s
(`configs/unified.yaml`; `configs/hot3d_p100.yaml` for the HOT3D benchmark,
whose reference profile stays in `configs/hot3d.yaml`).

## Next steps, in order

1. **Detection is still the weakest stage**, though 2026-09-30/10-01 work
   (continuity-first exclusive tracker, `box_padding: 1.5` before HaWoR)
   moved ep000 to 34.9 % / 92.2 % and 161.6 mm. Left-hand recall on this
   lens remains the cap on every aggregate number; the honest option left is
   a detector better calibrated for this resolution, or selecting episodes.
2. **P1 (outlier screening) is the one reference post-processing stage still
   missing.** The reference's `block_outlier.block_spike` rejects temporal and
   motion outliers before interpolation; here `hand_valid` carries that
   information instead. A MAD/block screen is only worth adding together with
   a coverage-policy decision (drop vs down-weight), since dropping frames
   lowers the coverage number.
3. **Full ablation table.** The focal, detection, stitch, finger-depth and
   P2/P3 rows are filled (`doc_auto/ablation.md`); the upstream rows (HaWoR
   without VGGT, +40 overlap, HaWoR-original pipeline) still need dedicated
   runs - `scripts/run_hawor_standalone.py` provides the native-path control.
4. **VGGT window size on better hardware.** 200/40 is what the reference
   configuration wants; the P100 tops out at 8. On an A100/H100, re-run Phase
   3-7 at 200/40 and update the table.
5. HaWoR's `eigen` submodule fetch (`git -C third_party/HaWoR submodule update
   --init --recursive`) - only needed if the DROID-SLAM path is ever revived;
   the pipeline no longer uses it.

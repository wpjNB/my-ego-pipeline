# Implementation status

Last modified: 2026-09-23 18:05 (+08:00)

Test suite: **239 passed in ~12 s** on the CPU-only laptop
(`conda run -n ego3d_base python -m pytest -q`).

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
| Action-MPJPE / coverage / report | `evaluation/*` | `tests/test_action_mpjpe.py` |
| Phase 0 video IO | `io/video.py`, `io/frames.py` | `tests/test_frames_io.py` |
| Synthetic scene / mock data | `testing/synthetic.py` | `tests/test_mock_pipeline.py` |
| Runner protocol | `runtime/subprocess_backend.py` | `tests/test_runner_protocol.py` |
| Full pipeline (mock backend) | `scripts/run_pipeline.py` + all stages | `tests/test_mock_pipeline.py` |
| LeRobot v3 reader | `datasets/lerobot.py` | `tests/test_lerobot.py` |
| HOT3D reference conversion | `datasets/hot3d_gt.py` | `tests/test_lerobot.py` |
| Reference-vs-prediction viewer | `visualization/overlay.py` | `tests/test_visualization.py` |
| Wrist-level evaluation | `evaluation/action_mpjpe.py` | `tests/test_hot3d_evaluation.py` |
| MANO forward kinematics | `hand/mano_model.py` | `tests/test_mano_model.py` |
| Model-output conversions | `detection/wilor.py`, `hand/hawor.py`, `camera/vggt_omega.py` | `tests/test_backend_conversions.py` |

## Interfaces in place; only the model call is blocked on the GPU server

| Backend | Adapter + runner | Behaviour today |
| --- | --- | --- |
| WiLoR | `detection/wilor.py` + `backends/wilor_runner.py` | `load_detector` (checkpoint/config discovery) and the output->artefact conversion are written and the conversion is tested; the detector call needs the checkout + GPU |
| HaWoR | `hand/hawor.py` + `backends/hawor_runner.py` | our tracking is written into HaWoR's `model_tracks.npy`, the call sequence (motion -> slam -> infiller -> run_mano) and the camera-space conversion are written; running them needs the checkpoint + GPU |
| VGGT-Omega | `camera/vggt_omega.py` + `backends/vggt_runner.py` | model loading by checkpoint directory, `pose_encoding_to_extri_intri` decoding, intrinsics rescaling to the depth grid and window validation are written; inference needs the checkpoint + GPU |

Every step before the model call is unit-tested, and every failure path names
the missing checkout, weights, checkpoint or device - a half-configured server
still cannot emit an artefact that looks like a successful run.

Running the whole pipeline today works through `backends.mode: mock`
(`configs/mock.yaml`), which substitutes a deterministic stand-in for the three
models and marks every artefact with `backend_mode: mock`.

## Next steps, in order

1. Run Phases 1-3 on one GPU clip, then check the debug videos and `--check`
   output; expect small API drift (WiLoR's output fields, VGGT's pose decoding
   helper) and fix it against the installed versions.
2. Convert a licensed MANO model (`scripts/convert_mano.py`) so the reference is
   21-joint instead of wrist-only.
3. Fill the real-data ablation table (`doc_auto/ablation.md`) from
   `scripts/evaluate_hot3d.py` outputs.
4. Decide whether HaWoR's infiller SLAM step can be skipped (it currently costs
   a full SLAM run per clip just to carry coordinates for Phase 2).

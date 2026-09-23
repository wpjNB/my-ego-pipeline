# Implementation status

Last modified: 2026-09-23 17:36 (+08:00)

Test suite: **212 passed in ~12 s** on the CPU-only laptop
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

## Interfaces in place; only the model call is blocked on the GPU server

| Backend | Adapter + runner | Behaviour today |
| --- | --- | --- |
| WiLoR | `detection/wilor.py` + `backends/wilor_runner.py` | availability probe, frame discovery, argument handling, artefact format and `--check` are done; `run_model()` must call WiLoR's detector |
| HaWoR | `hand/hawor.py` + `backends/hawor_runner.py` | window schedule, detection plumbing, artefact format and `--check` are done; `run_model()` must call HaWoR |
| VGGT-Omega | `camera/vggt_omega.py` + `backends/vggt_runner.py` | window schedule, checkpoint allow-list, artefact format and `--check` are done; `run_model()` must call VGGT-Omega |

Each `run_model()` raises `NotImplementedError` naming the checkout, weights,
checkpoint and device, so a half-configured server can never emit an empty
artefact that looks like a successful run.

Running the whole pipeline today works through `backends.mode: mock`
(`configs/mock.yaml`), which substitutes a deterministic stand-in for the three
models and marks every artefact with `backend_mode: mock`.

## Next steps, in order

1. Implement the three `run_model()` bodies against the backend APIs.
2. Run Phases 1-3 on one GPU clip and check the debug videos.
3. Fill the real-data ablation table (`doc_auto/ablation.md`) from
   `scripts/evaluate_hot3d.py` outputs.
4. If 21-joint references are wanted, obtain the MANO model and add forward
   kinematics to `datasets/hot3d_gt.py` (the MANO parameters already travel
   through the contract).

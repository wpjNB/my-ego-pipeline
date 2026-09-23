# Implementation status

Last modified: 2026-09-23 17:12 (+08:00)

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

## Interfaces in place, invocation blocked on the GPU server

| Backend | Adapter | Behaviour today |
| --- | --- | --- |
| WiLoR | `detection/wilor.py` | availability probe + tracker path fully working; the detector call raises `BackendInvocationNotImplemented` |
| HaWoR | `hand/hawor.py` | availability probe + window request/response types; the reconstruction call raises `BackendInvocationNotImplemented` |
| VGGT-Omega | `camera/vggt_omega.py` | availability probe + window request type + checkpoint allow-list; inference raises `BackendInvocationNotImplemented` |

This is deliberate: no CPU placeholder ever stands in for a real model, and the
error message names the environment, the checkpoint and the missing paths.

## Next steps, in order

1. Wire the three subprocess wrappers (export detections / window poses + depth).
2. Run Phases 1-3 on one GPU clip and check the debug videos.
3. Fill the ablation table from `scripts/evaluate_hot3d.py` outputs.
4. Add the HOT3D episode loader once the sequence list is fixed.


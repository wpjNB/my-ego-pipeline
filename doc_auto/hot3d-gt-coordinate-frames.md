# HOT3D GT MANO coordinate frames

Last modified: 2026-10-11 20:47 (+08:00)

## Reference conversion

LeRobot `left_orient_world` and `right_orient_world` store each MANO wrist's root orientation in the original HOT3D world frame. This pipeline re-anchors camera poses and wrist translations to `World-0`, the first camera frame, before producing the trajectory. MANO root orientations must use that same rotation before forward kinematics:

```text
R_hot3d_from_world0 = metadata.hot3d_world_anchor_rotation
R_world0_from_hot3d = transpose(R_hot3d_from_world0)
R_root_world0 = R_world0_from_hot3d @ R_root_hot3d
```

The 15 finger-joint rotations are local MANO rotations and stay unchanged. The wrist is placed from the re-anchored `*_transl_world` translation. If only the wrist is transformed and the root rotation stays in HOT3D's original basis, the wrist marker remains in place while the rendered fingers rotate in the wrong world axes.

## Refreshing references and overlays

After changing the conversion, regenerate the reference without re-decoding the video, then rerender the skeleton overlay with the existing prediction:

```bash
python scripts/import_lerobot.py --config configs/hot3d_p100.yaml \
  --data-root data/hot3d --clip sample_ep000 \
  --root data/samples/lerobot_v3 --episode 0 --no-frames \
  --mano-model weights/mano
python scripts/render_gt_vs_pred.py --config configs/hot3d_p100.yaml \
  --data-root data/hot3d --clip sample_ep000 \
  --prediction data/hot3d/sample_ep000/trajectory/trajectory.npz \
  --ground-truth data/hot3d/sample_ep000/trajectory/ground_truth.npz --skeleton
```

`tests/test_lerobot.py::test_mano_root_orientations_are_reanchored_to_world_zero` checks the basis transform for both hands.

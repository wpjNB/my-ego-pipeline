#!/usr/bin/env python
"""Import an official HOT3D-Clips tar (bop-benchmark/hot3d) into the clip layout.

    python scripts/import_bop_hot3d.py \
        --tar data/hot3d_official/train_aria/clip-001991.tar \
        --data-root data/hot3d --clip P0002_clip001991

Produces the ordinary clip layout the whole pipeline consumes:

* ``frames/NNNNNN.jpg`` - the 214-1 RGB stream, **undistorted** from its
  FISHEYE624 projection to a pinhole camera (f = W/2 at 90 deg HFOV, principal
  point at the centre), with ``metadata.json`` describing it;
* ``trajectory/ground_truth.npz`` - camera + hand reference in the project's
  contract, World-0 anchored (first frame's camera = origin, same gauge the
  pipeline applies):
    - ``camera_R_c2w`` / ``camera_t_c2w`` from ``cameras.json`` (214-1);
    - ``hand_xyz_world`` / ``hand_xyz_camera`` ``[T, 2, 21, 3]`` from the
      official **MANO** annotation (``mano_pose.thetas`` are 15 PCA
      coefficients: decoded with ``smplx`` ``use_pca=True, num_pca_comps=15``,
      the same configuration the official toolkit uses; ``wrist_xform[:3]`` is
      the global orientation (axis-angle) and ``[3:]`` the translation in
      metres), forward-kinematicked into this project's 21-landmark order via
      ``hand/mano_model.py::MANO_TO_LANDMARK``;
    - ``camera_K`` - the pinhole intrinsics of the undistorted frames (finally
      a *valid* reference K);
    - ``bbox`` - the official amodal 214-1 hand boxes, warped fisheye ->
      pinhole (informational; the pipeline takes boxes from its own detector).

Why MANO and not UmeTrack: the official docs call UmeTrack the primary
annotation, but its landmark set is 20 joints in UmeTrack's own convention, so
mapping it into this project's 21-joint contract needs a guessed permutation.
The MANO annotation sits in the same file and decodes directly into the
contract's semantics; the official toolkit itself notes MANO is solved from
UmeTrack and "slightly worse", which is an acceptable trade for a v1 reference
(the UmeTrack path can be added later without touching the contract).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.camera.camera_pose import world_frame_alignment  # noqa: E402
from ego3d_action.errors import StageIOError  # noqa: E402
from ego3d_action.hand.mano_model import MANO_TO_LANDMARK  # noqa: E402
from ego3d_action.io.serialization import save_json, save_npz  # noqa: E402

MANO_PCA_COMPS = 15
FPS_FALLBACK = 30.0


def _undistort_maps(calibration: dict, width: int, height: int, focal: float):
    """Inverse-warp maps fisheye(214-1) -> pinhole, from the stream's calibration."""
    from hand_tracking_toolkit.camera import PinholePlaneCameraModel  # noqa: PLC0415

    model = calibration["model"]
    ys, xs = np.mgrid[0:height, 0:width]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float64)
    pinhole = PinholePlaneCameraModel(
        width=width, height=height, f=float(focal), c=(width / 2.0, height / 2.0),
        distort_coeffs=[],
    )
    rays = np.asarray(pinhole.window_to_eye(pixels), dtype=np.float64)
    src = np.asarray(model.eye_to_window(rays), dtype=np.float64)
    map_x = src[:, 0].reshape(height, width).astype(np.float32)
    map_y = src[:, 1].reshape(height, width).astype(np.float32)
    return map_x, map_y


def _warp_point(model, target, uv: np.ndarray) -> np.ndarray:
    """Fisheye pixel -> pinhole pixel through the same chain as the image warp."""
    rays = np.asarray(model.window_to_eye(np.asarray(uv, dtype=np.float64)))
    return np.asarray(target.eye_to_window(rays), dtype=np.float64)


def _mano_joints_world(mano_layer, theta: np.ndarray, betas: np.ndarray,
                       wrist_xform: np.ndarray) -> np.ndarray:
    """Official MANO annotation -> this project's 21 landmarks in world metres."""
    import torch  # noqa: PLC0415

    out = mano_layer(
        betas=torch.tensor(betas, dtype=torch.float32),
        global_orient=torch.tensor(wrist_xform[:, :3], dtype=torch.float32),
        hand_pose=torch.tensor(theta, dtype=torch.float32),
        transl=torch.tensor(wrist_xform[:, 3:], dtype=torch.float32),
        return_verts=True,
    )
    verts = np.asarray(out.vertices.detach(), dtype=np.float64)
    joints = np.asarray(out.joints.detach(), dtype=np.float64)
    return np.stack(
        [joints[:, i, :] if source == "joint" else verts[:, i, :]
         for source, i in MANO_TO_LANDMARK],
        axis=1,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True, help="clip-XXXXXX.tar from train_aria/train_quest3")
    parser.add_argument("--data-root", default="data/hot3d")
    parser.add_argument("--clip", default=None, help="clip name (default: <tar stem>)")
    parser.add_argument("--mano-dir", default="weights/mano")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    tar_path = Path(args.tar)
    if not tar_path.is_file():
        print(f"tar not found: {tar_path}", file=sys.stderr)
        return 1
    clip = args.clip or tar_path.stem
    clip_dir = Path(args.data_root) / clip
    frames_dir = clip_dir / "frames"
    if frames_dir.exists() and not args.overwrite and any(frames_dir.glob("*.jpg")):
        print(f"{clip_dir} already imported (pass --overwrite to redo)")
        return 1
    if args.overwrite:
        # Re-importing replaces the frames, so every artefact derived from
        # them is stale: detection, hands, camera windows (their VGGT focal
        # feeds the hand stage!), stitching, trajectories and renders. The
        # calibrated ground_truth.npz is rewritten by this script anyway.
        import shutil  # noqa: PLC0415

        for stale in ("frames", "detection", "hand", "camera", "stitched",
                      "trajectory", "visualization"):
            target = clip_dir / stale
            if target.is_dir():
                shutil.rmtree(target)

    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import smplx  # noqa: PLC0415

    from hand_tracking_toolkit import dataset as D  # noqa: PLC0415

    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        frame_ids = sorted({int(n.split(".")[0]) for n in names if n[:6].isdigit()})
        total = len(frame_ids)
        read_json = lambda name: json.loads(tf.extractfile(name).read())  # noqa: E731
        info0 = read_json(f"{frame_ids[0]:06d}.info.json")
        shapes = read_json("__hand_shapes.json__")
        cameras0 = read_json(f"{frame_ids[0]:06d}.cameras.json")
        calib = D.decode_cam_params(cameras0)
        stream = "214-1"
        if stream not in calib:
            raise StageIOError(f"{tar_path}: no {stream} stream (Quest3 clips are not supported yet)")
        model = calib[stream]

        first_img = cv2.imdecode(
            np.frombuffer(tf.extractfile(f"{frame_ids[0]:06d}.image_{stream}.jpg").read(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        height, width = first_img.shape[:2]
        # Official convention: clip_util.convert_to_pinhole_camera(focal_scale=1.0)
        # keeps the fisheye focal (f ~= 609 at 1408), i.e. ~98 deg FOV - wider
        # than a 90 deg pinhole and the value the reference visualizer uses.
        # NOTE: content beyond ~49 deg off-axis is still cropped by any sane
        # pinhole; HOT3D-Clips do contain hands raised right next to the lens
        # (measured up to 64 deg off-axis in clip-001991), which this
        # preparation cannot represent - the challenge handles those with
        # per-hand crops instead. The 3D GT reference is unaffected.
        focal = float(getattr(model, "f", [width / 2.0])[0])
        map_x, map_y = _undistort_maps({"model": model}, width, height, focal)
        pinhole_spec = {
            "width": width, "height": height,
            "f": focal, "c": [width / 2.0, height / 2.0],
        }
        from hand_tracking_toolkit.camera import PinholePlaneCameraModel  # noqa: PLC0415

        target = PinholePlaneCameraModel(
            width=width, height=height, f=focal, c=(width / 2.0, height / 2.0),
            distort_coeffs=[],
        )

        # ---------------- frames (undistorted) ----------------
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, fid in enumerate(frame_ids):
            raw = cv2.imdecode(
                np.frombuffer(tf.extractfile(f"{fid:06d}.image_{stream}.jpg").read(), np.uint8),
                cv2.IMREAD_COLOR,
            )
            warped = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ---------------- camera + hand reference ----------------
        rotation = np.full((total, 3, 3), np.nan)
        translation = np.full((total, 3), np.nan)
        timestamps = np.zeros(total, dtype=np.float64)
        theta = {s: np.full((total, 15), np.nan) for s in ("left", "right")}
        wrist = {s: np.full((total, 6), np.nan) for s in ("left", "right")}
        bbox_fisheye = {s: np.full((total, 4), np.nan) for s in ("left", "right")}
        for i, fid in enumerate(frame_ids):
            cams = cameras0 if fid == frame_ids[0] else read_json(f"{fid:06d}.cameras.json")
            entry = cams[stream]["T_world_from_camera"]
            w, x, y, z = entry["quaternion_wxyz"]
            rotation[i] = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])
            translation[i] = entry["translation_xyz"]
            frame_info = info0 if fid == frame_ids[0] else read_json(f"{fid:06d}.info.json")
            timestamps[i] = frame_info["ref_timestamp_ns"]
            hands = read_json(f"{fid:06d}.hands.json")
            for side in ("left", "right"):
                pose = hands.get(side)
                if not pose:
                    continue
                bbox_fisheye[side][i] = np.asarray(pose["boxes_amodal"][stream], dtype=np.float64)
                if "mano_pose" not in pose:
                    continue
                theta[side][i] = np.asarray(pose["mano_pose"]["thetas"], dtype=np.float64)
                wrist[side][i] = np.asarray(pose["mano_pose"]["wrist_xform"], dtype=np.float64)

    betas = np.asarray(shapes["mano"], dtype=np.float64)
    dt = np.diff(timestamps) / 1e9
    fps = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else FPS_FALLBACK

    # MANO FK per hand, batched over the valid frames only.
    joints_world = np.full((total, 2, 21, 3), np.nan)
    valid = np.zeros((total, 2), dtype=bool)
    for hand, side in enumerate(("left", "right")):
        mask = np.isfinite(theta[side]).all(axis=1) & np.isfinite(wrist[side]).all(axis=1)
        if not mask.any():
            continue
        layer = smplx.create(
            str(Path(args.mano_dir) / f"MANO_{side.upper()}.pkl"), "mano",
            use_pca=True, is_rhand=(side == "right"), num_pca_comps=MANO_PCA_COMPS,
        )
        jm = _mano_joints_world(
            layer,
            theta[side][mask],
            np.broadcast_to(betas, (int(mask.sum()), 10)),
            wrist[side][mask],
        )
        joints_world[mask, hand] = jm
        valid[mask, hand] = True

    # World-0 anchoring (same gauge the pipeline and the mirror importer use).
    anchor_rotation, anchor_translation = world_frame_alignment(rotation, translation)
    hot3d_anchor_rotation = rotation[0].copy()
    hot3d_anchor_translation = translation[0].copy()
    rotation = np.einsum("ij,tjk->tik", anchor_rotation, rotation)
    translation = np.einsum("ij,tj->ti", anchor_rotation, translation - anchor_translation)
    joints_world = np.einsum("ij,t...j->t...i", anchor_rotation, joints_world - anchor_translation)
    joints_world[~valid] = np.nan

    # Warped boxes (fisheye -> pinhole), corners through the same chain.
    bbox = np.full((total, 2, 4), np.nan)
    for hand, side in enumerate(("left", "right")):
        for i in range(total):
            box = bbox_fisheye[side][i]
            if not np.isfinite(box).all():
                continue
            corners = np.array([[box[0], box[1]], [box[2], box[1]],
                                [box[2], box[3]], [box[0], box[3]]])
            try:
                warped = _warp_point(model, target, corners)
            except Exception:  # noqa: BLE001 - a corner off the fisheye model
                continue
            if np.isfinite(warped).all():
                bbox[i, hand] = [warped[:, 0].min(), warped[:, 1].min(),
                                 warped[:, 0].max(), warped[:, 1].max()]

    camera_frame = np.einsum(
        "tji,thkj->thki", rotation, joints_world - translation[:, None, None, :]
    )
    camera_K = np.broadcast_to(
        np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]),
        (total, 3, 3),
    ).copy()

    trajectory_dir = clip_dir / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    save_npz(
        trajectory_dir / "ground_truth.npz",
        frames=np.arange(total, dtype=np.int64),
        timestamps=(timestamps - timestamps[0]) / 1e9,
        hand_xyz_world=joints_world,
        hand_xyz_camera=camera_frame,
        hand_valid=valid,
        hand_confidence=np.where(valid, 1.0, 0.0),
        hand_interpolated=np.zeros(valid.shape, dtype=bool),
        camera_R_c2w=rotation,
        camera_t_c2w=translation,
        camera_K=camera_K,
        bbox=bbox,
        track_id=np.where(valid, 0, -1).astype(np.int64),
        mano_root_rot=np.full((total, 2, 3, 3), np.nan),
        mano_hand_pose=np.full((total, 2, 15, 3, 3), np.nan),
        mano_betas=np.broadcast_to(betas, (total, 2, 10)).copy(),
        postprocess_valid=valid,
    )
    save_json(
        trajectory_dir / "metadata.json",
        {
            "stage": "official_import",
            "world_frame": 0,
            "hand_representation": "21_joints_metric_xyz",
            "camera_convention": "c2w",
            "units": "meter",
            "num_frames": total,
            "fps": fps,
            "source": f"bop-benchmark/hot3d#{tar_path.name}",
            "source_stream": stream,
            "source_clip": int(tar_path.stem.split("-")[-1]),
            "hand_joints": "mano_fk",
            "hand_joints_note": (
                "official MANO annotation (15 PCA coeffs + wrist_xform) via smplx "
                "use_pca=True num_pca_comps=15, mapped to this project's 21-landmark "
                "order; UmeTrack (the primary official annotation, 20 landmarks in "
                "its own convention) can be added later"
            ),
            "reference_camera_validated": True,
            "reference_wrist_validated": True,
            "undistortion": {
                "from": "FISHEYE624 (214-1)", "to": "pinhole",
                "pinhole": pinhole_spec, "note": "GT 3D is undistortion-invariant; K above matches the frames",
            },
            "hot3d_world_anchor_rotation": hot3d_anchor_rotation.tolist(),
            "hot3d_world_anchor_translation": hot3d_anchor_translation.tolist(),
        },
    )
    save_json(
        clip_dir / "metadata.json",
        {
            "clip": clip,
            "dataset_root": str(Path(args.data_root)),
            "fps": fps,
            "num_frames": total,
            "width": width,
            "height": height,
            "image_format": "jpg",
            "frame_pattern": "%06d.jpg",
            "duration": total / fps,
            "hand_joints": "mano_fk",
            "stage": "official_import",
            "source": f"bop-benchmark/hot3d#{tar_path.name}",
            "source_stream": stream,
            "task": "HOT3D-Clips egocentric hand-object interaction (official)",
            "reference_note": "official camera + MANO hand GT; validated against the images",
        },
    )
    print(
        f"imported {tar_path.name} -> {clip_dir} | {total} frames @ {fps:.1f} fps | "
        f"hand valid: left {valid[:, 0].mean() * 100:.0f}% right {valid[:, 1].mean() * 100:.0f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

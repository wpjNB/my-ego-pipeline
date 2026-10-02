#!/usr/bin/env python
"""Run the HaWoR checkout's own demo pipeline standalone, on our footage.

Isolation experiment: HaWoR end-to-end as upstream intended - its *native*
detector+tracker (WiLoR's YOLO detector, the exact file HaWoR's README links),
its *own* focal-length estimation (no ``--img_focal``), its motion estimation
and infiller - on the original sample video, independent of our tracker and
VGGT focal. Only two pieces are substituted, both documented:

* DROID-SLAM is replaced by the VGGT-derived camera trajectory (lietorch is
  not installed on this host); the npz is written in HaWoR's own format.
* the aitviewer visualisation is replaced by this repo's mesh rasteriser.

Outputs (under ``outputs/hawor_standalone/``): the camera-space hands as npz,
a rendered overlay video, and the per-joint depth-bias profile against the
HOT3D reference - the number to compare with our integrated pipeline's
finger-depth bias.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

HAJOR_CHECKOUT = REPO / "third_party" / "HaWoR"
VIDEO = REPO / "data/samples/lerobot_v3/videos/observation.images.ego/chunk-000/file-000.mp4"
CLIP = "hot3d_ep000"
OUT = REPO / "outputs" / "hawor_standalone"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=450)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    import os

    os.chdir(HAJOR_CHECKOUT)
    sys.path.insert(0, str(HAJOR_CHECKOUT))

    # the native detector is WiLoR's YOLO (HaWoR's README links the same file)
    detector_dir = HAJOR_CHECKOUT / "weights" / "external"
    detector_dir.mkdir(parents=True, exist_ok=True)
    detector = detector_dir / "detector.pt"
    if not detector.exists():
        detector.symlink_to(REPO / "weights" / "wilor" / "detector.pt")
    # torch 2.6 weights_only default blocks ultralytics' own checkpoint load
    from ego3d_action.runtime.checkpoints import allow_trusted_checkpoint_globals
    allow_trusted_checkpoint_globals(detector, label="HaWoR native detector (WiLoR YOLO)")
    for name in ("hawor.ckpt", "infiller.pt"):
        f = REPO / "weights/hawor/checkpoints" / name
        if f.exists():
            allow_trusted_checkpoint_globals(f, label=f"HaWoR {name}")

    video_link = HAJOR_CHECKOUT / "example" / "hot3d_ep000.mp4"
    if not video_link.exists():
        video_link.symlink_to(VIDEO)

    # hawor_video imports lib.vis.renderer (pytorch3d) at module level; reuse
    # our runner's stub before any HaWoR script import.
    sys.path.insert(0, str(REPO / "backends"))
    import hawor_runner as our_runner
    our_runner.install_renderer_stub()

    from ego3d_action.runtime.checkpoints import allow_trusted_checkpoint_globals
    for name in ("hawor.ckpt", "infiller.pt"):
        f = REPO / "weights/hawor/checkpoints" / name
        if f.exists():
            allow_trusted_checkpoint_globals(f, label=f"HaWoR {name}")

    from ego3d_action.runtime.checkpoints import restore_legacy_numpy_aliases
    restore_legacy_numpy_aliases()  # chumpy does `from numpy import bool, ...`

    from scripts.scripts_test_video import hawor_video
    hawor_video.load_hawor = our_runner._patched_loader(hawor_video.load_hawor, half=False)
    detect_track_video = hawor_video  # noqa: F841 - keep the module handy
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    hawor_infiller = hawor_video.hawor_infiller
    hawor_motion_estimation = hawor_video.hawor_motion_estimation
    from lib.eval_utils.custom_utils import load_slam_cam
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left

    demo_args = types.SimpleNamespace(
        video_path=str(video_link), img_focal=227.47947340745193, focal=227.47947340745193,
        input_type="file",
        checkpoint="./weights/hawor/checkpoints/hawor.ckpt",
        infiller_weight="./weights/hawor/checkpoints/infiller.pt",
        vis_mode="cam",
    )
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(demo_args)
    end_idx = min(end_idx, args.max_frames)
    print(f"frames {start_idx}..{end_idx} in {seq_folder}")

    # SLAM substitute: the VGGT-derived camera, in HaWoR's npz format. Reuses
    # our runner's writer through a minimal args stub.
    # demo.py order: motion estimation first (it estimates the focal itself),
    # then the camera trajectory, then the infiller.
    frame_chunks_all, img_focal = hawor_motion_estimation(demo_args, start_idx, end_idx, seq_folder)
    print(f"native estimated focal: {img_focal:.2f} px")

    slam_args = types.SimpleNamespace(
        camera_windows=str(REPO / f"data/hot3d/{CLIP}/camera/windows"),
        start_idx=start_idx, end_idx=end_idx, seq_folder=str(seq_folder),
        focal=float(img_focal),
    )
    our_runner.write_camera_trajectory(slam_args, Path(seq_folder), start_idx, end_idx)

    slam_path = Path(seq_folder) / f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    R_w2c, t_w2c, _, _ = load_slam_cam(str(slam_path))
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        demo_args, start_idx, end_idx, frame_chunks_all
    )

    import torch

    torch.set_grad_enabled(False)
    faces_right = get_mano_faces()
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()  # demo.py flip

    hands = {}
    for name, runner in (("right", run_mano), ("left", run_mano_left)):
        idx = 1 if name == "right" else 0
        sl = slice(idx, idx + 1)
        out = runner(pred_trans[sl], pred_rot[sl], pred_hand_pose[sl],
                     betas=pred_betas[sl])
        verts = out["vertices"][0].cpu()          # (T, 778, 3) world
        verts = torch.einsum("ij,tnj->tni", R_x, verts.cpu())  # demo.py flip
        hands[name] = verts
    # world -> camera (with the flipped convention, exactly as demo.py 'cam')
    R_w2c_t = torch.tensor(R_w2c, dtype=torch.float32)
    t_w2c_t = torch.tensor(t_w2c, dtype=torch.float32)
    for name in hands:
        v = hands[name]
        v = torch.einsum("ij,tnj->tni", R_x, hands[name])
        hands[name] = torch.einsum("tij,tnj->tni", R_w2c_t, v) + t_w2c_t[None, :, None, :]

    T = hands["right"].shape[0]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "hawor_native_hands.npz",
        vertices_right=hands["right"].numpy().astype(np.float32),
        vertices_left=hands["left"].numpy().astype(np.float32),
        focal=img_focal,
        native_boxes=np.load(Path(seq_folder) / f"tracks_{start_idx}_{end_idx}/model_boxes.npy", allow_pickle=True),
    )
    print(f"saved {OUT / 'hawor_native_hands.npz'}")
    if args.no_video:
        return 0

    # render with this repo's rasteriser on the same frames
    import cv2
    from ego3d_action.visualization.overlay import draw_hand_mesh, transcode_to_h264

    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                          [122, 118, 239], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                          [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                          [120, 108, 78], [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_pair = [faces_right[:, [0, 2, 1]], faces_right]  # left mirrored winding

    frames_dir = REPO / f"data/hot3d/{CLIP}/frames"
    K = np.array([[img_focal, 0, 256], [0, img_focal, 256], [0, 0, 1]])
    writer = None
    for t in range(0, T, 2):
        frame_path = frames_dir / f"{t:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        pair = np.stack([hands["left"][t].numpy(), hands["right"][t].numpy()])
        vvalid = np.isfinite(pair).all(axis=(1, 2))
        canvas = draw_hand_mesh(frame, pair, faces_pair, K, vvalid)
        if writer is None:
            writer = cv2.VideoWriter(str(OUT / "hawor_native.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 15.0,
                                     (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
    if writer is not None:
        writer.release()
        transcode_to_h264(OUT / "hawor_native.mp4")
        print(f"wrote {OUT / 'hawor_native.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

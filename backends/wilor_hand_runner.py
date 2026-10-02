#!/usr/bin/env python
"""WiLoR hand reconstruction runner (Phase 2 alternative to HaWoR).

    python backends/wilor_hand_runner.py \
        --frames data/<clip>/frames --detection data/<clip>/detection/detection.npz \
        --out data/<clip>/hand/wilor_hands.npz --num-frames 450 --focal 227.48 \
        --third-party third_party --weights weights --device cuda:0

Reconstructs the tracked hands per frame with WiLoR (``wilor_final.ckpt``) and
writes camera-space metric hands in this project's contract:

* ``joints_camera`` ``[T, 2, 21, 3]`` - this project's 21 landmarks
  (``hand/mano_model.py::MANO_TO_LANDMARK``), metres, NaNs where invalid;
* ``vertices_camera`` ``[T, 2, 778, 3]``; ``root_rot``, ``betas``,
  ``valid``/``confidence`` from the tracked detection.

The crop -> camera conversion is the calibrated core (2026-10-01, measured
against ep000): the crop weak-perspective translation converts to a full-image
perspective translation with the *physical* camera focal (``--focal``, px at
the frame resolution) via WiLoR's own ``cam_crop_to_full`` formula, and the
per-joint camera positions are ``landmarks + translation`` (the model's root is
NOT the wrist - it sits ~96 mm away; dropping that offset was worth 8 cm of
wrist error). Depth-ratio check against ep000: predicted/GT 0.97 (median).

The runner is deliberately per-frame: WiLoR has no temporal model, so the
downstream blend/refine stages (UKF smoothing) carry the temporal stability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ego3d_action.io.serialization import load_npz, save_npz  # noqa: E402
from ego3d_action.runtime.checkpoints import (  # noqa: E402
    allow_trusted_checkpoint_globals,
    restore_legacy_numpy_aliases,
)

CHECKPOINT_CANDIDATES = ("wilor/wilor_final.ckpt",)
CONFIG_CANDIDATES = ("wilor/model_config.yaml",)
CHECKOUT_DIRNAME = "WiLoR"

#: WiLoR's training camera convention (config: EXTRA.FOCAL_LENGTH / IMAGE_SIZE).
#: It only enters the model's *internal* weak-perspective translation; the
#: metric conversion below always uses the physical focal passed in via
#: ``--focal`` (see the calibration note in the module docstring).
NUM_HANDS = 2
NUM_LANDMARKS = 21
NUM_VERTICES = 778


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def find_checkpoint(weights_root: str | Path) -> Path | None:
    root = Path(weights_root)
    for relative in CHECKPOINT_CANDIDATES:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def find_config(weights_root: str | Path) -> Path | None:
    root = Path(weights_root)
    for relative in CONFIG_CANDIDATES:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def install_pyrender_stub() -> None:
    """Make ``wilor.utils`` importable without pyrender.

    WiLoR's renderer modules (pulled in by ``wilor.models``) import pyrender at
    module level, but every pyrender call site lives inside Renderer methods -
    and the model is constructed with ``init_renderer=False``, so those methods
    are never reached. pyrender needs OpenGL/EGL that a headless box does not
    have; the stub keeps the import graph intact without it. Mirrors the
    pytorch3d stub the HaWoR runner installs for the same reason.
    """
    if "pyrender" in sys.modules:
        return
    try:
        import pyrender  # noqa: F401, PLC0415
        return
    except ImportError:
        pass

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return _Dummy

    stub = types.ModuleType("pyrender")
    for attr in ("Node", "DirectionalLight", "OffscreenRenderer", "MetallicRoughnessMaterial",
                 "Mesh", "Scene", "IntrinsicsCamera", "RenderFlags", "Viewer", "Camera",
                 "PerspectiveCamera", "Light"):
        setattr(stub, attr, type(attr, (_Dummy,), {}))
    sys.modules["pyrender"] = stub


def cam_crop_to_full(cam_bbox: np.ndarray, box_center: np.ndarray, box_size: np.ndarray,
                     img_size: np.ndarray, focal_length: float) -> np.ndarray:
    """Faithful port of ``wilor.utils.renderer.cam_crop_to_full`` (pure numpy).

    ``cam_bbox`` is ``[N, 3]`` weak-perspective ``(s, tx, ty)`` (left hands
    already un-mirrored), ``box_size`` the crop square edge in *original image*
    pixels, ``focal_length`` the physical focal in the same pixel units.
    """
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return np.stack([tx, ty, tz], axis=-1)


def undo_openpose_remap(joints_openpose: np.ndarray, joint_map: np.ndarray) -> np.ndarray:
    """Recover the raw ``[N, 21, 3]`` MANO joints+tips array from the wrapper's
    openpose-remapped output: ``openpose[j] = full[joint_map[j]]``."""
    out = np.empty_like(joints_openpose)
    for j, src in enumerate(joint_map):
        out[:, src, :] = joints_openpose[:, j, :]
    return out


def our_landmarks(vertices: np.ndarray, joints_full: np.ndarray) -> np.ndarray:
    """This project's 21 landmarks from MANO 16 joints + 5 fingertip vertices."""
    from ego3d_action.hand.mano_model import MANO_TO_LANDMARK  # noqa: PLC0415

    return np.stack(
        [joints_full[:, i, :] if source == "joint" else vertices[:, i, :]
         for source, i in MANO_TO_LANDMARK],
        axis=1,
    )


def run_model(args: argparse.Namespace) -> dict[str, np.ndarray]:
    restore_legacy_numpy_aliases()
    install_pyrender_stub()

    # Absolutise everything that is read below *before* chdir-ing into the
    # checkout: relative sys.path entries and file arguments would otherwise
    # resolve against the checkout instead of the caller's working directory
    # (the same trap hawor_runner.absolutize_paths exists for).
    checkout = (Path(args.third_party) / CHECKOUT_DIRNAME).resolve()
    weights = Path(args.weights).resolve()
    frames_dir = Path(args.frames).resolve()
    detection_path = Path(args.detection).resolve()

    checkpoint = find_checkpoint(weights)
    if checkpoint is None:
        raise FileNotFoundError(
            f"wilor_final.ckpt not found under {args.weights} (looked for "
            f"{list(CHECKPOINT_CANDIDATES)}) - run ./scripts/download_weights.sh --only wilor"
        )
    config_path = find_config(weights)
    if config_path is None:
        raise FileNotFoundError(f"model_config.yaml not found under {args.weights}")
    if args.focal is None or not np.isfinite(args.focal) or args.focal <= 0:
        # The HaWoR default-focal incident (2.2x-too-deep hands) is exactly what
        # refusing here prevents: a WiLoR run without a real focal is not metric.
        raise ValueError("--focal must be a positive pixel focal length (px at frame resolution)")

    allow_trusted_checkpoint_globals(checkpoint, label="WiLoR")

    if not checkout.is_dir():
        raise FileNotFoundError(f"WiLoR checkout missing: {checkout}")
    # load_wilor rewrites MANO paths to ./mano_data/ - everything below runs
    # from inside the checkout, like the HaWoR runner does.
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))

    import torch  # noqa: PLC0415
    from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: PLC0415
    from wilor.models import load_wilor  # noqa: PLC0415

    model, model_cfg = load_wilor(str(checkpoint), str(config_path))
    model = model.to(torch.device(args.device)).eval()
    joint_map = model.mano.joint_map.detach().cpu().numpy()
    print(
        f"WiLoR loaded | IMAGE_SIZE {model_cfg.MODEL.IMAGE_SIZE} | device {args.device} | "
        f"focal {args.focal:.2f} px",
        file=sys.stderr,
    )

    detection = load_npz(detection_path, required=("boxes", "confidence", "valid"))
    boxes = np.asarray(detection["boxes"], dtype=np.float64)
    confidence = np.asarray(detection["confidence"], dtype=np.float64)
    valid = np.asarray(detection["valid"], dtype=bool)
    total = int(args.num_frames or boxes.shape[0])
    if boxes.shape[0] < total:
        raise ValueError(f"detection covers {boxes.shape[0]} frames, expected {total}")
    frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) < total:
        raise FileNotFoundError(f"found {len(frames)} frames for {total} detection rows")

    import cv2  # noqa: PLC0415

    joints = np.full((total, NUM_HANDS, NUM_LANDMARKS, 3), np.nan, dtype=np.float64)
    vertices = np.full((total, NUM_HANDS, NUM_VERTICES, 3), np.nan, dtype=np.float64)
    root_rot = np.full((total, NUM_HANDS, 3, 3), np.nan, dtype=np.float64)
    betas = np.full((total, NUM_HANDS, 10), np.nan, dtype=np.float64)
    out_valid = np.zeros((total, NUM_HANDS), dtype=bool)

    for frame_id, path in enumerate(frames[:total]):
        slots = np.where(valid[frame_id])[0]
        if len(slots) == 0:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode {path}")
        img_h, img_w = image.shape[:2]
        bx = boxes[frame_id][slots].astype(np.float32)
        right = (slots == 1).astype(np.float32)
        dataset = ViTDetDataset(model_cfg, image, bx, right, rescale_factor=args.rescale)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)
        for batch in loader:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                out = model(batch)
                mano_params = out["pred_mano_params"]
                batch_size = batch["right"].shape[0]
                mano_output = model.mano(
                    global_orient=mano_params["global_orient"].reshape(batch_size, -1, 3, 3),
                    hand_pose=mano_params["hand_pose"].reshape(batch_size, -1, 3, 3),
                    betas=mano_params["betas"].reshape(batch_size, -1),
                    pose2rot=False,
                )
            mult = (2 * batch["right"] - 1).detach().cpu().numpy()  # +1 right, -1 left
            pred_cam = out["pred_cam"].detach().cpu().numpy().copy()
            pred_cam[:, 1] = mult * pred_cam[:, 1]
            center = batch["box_center"].detach().cpu().numpy().astype(np.float64)
            box_size = batch["box_size"].detach().cpu().numpy().astype(np.float64).reshape(-1)
            img_size = batch["img_size"].detach().cpu().numpy().astype(np.float64)
            if img_size.ndim == 1:
                img_size = np.broadcast_to(img_size, (len(box_size), 2))
            translation = cam_crop_to_full(pred_cam, center, box_size, img_size, float(args.focal))

            raw_verts = mano_output.vertices.detach().cpu().numpy().astype(np.float64)
            openpose = mano_output.joints.detach().cpu().numpy().astype(np.float64)
            full_joints = undo_openpose_remap(openpose, joint_map)
            land = our_landmarks(raw_verts, full_joints)
            # Un-mirror left hands (network always sees a right hand: crops are
            # flipped for left) - vertices/joints by x-flip, the root rotation
            # by conjugating with diag(-1, 1, 1) so it stays a proper rotation.
            land[:, :, 0] *= mult[:, None]
            raw_verts[:, :, 0] *= mult[:, None]
            rot = mano_params["global_orient"].detach().cpu().numpy().reshape(batch_size, 3, 3)
            flip_diag = np.diag([-1.0, 1.0, 1.0])
            rot = np.where(mult[:, None, None] > 0, rot,
                           flip_diag @ rot @ flip_diag)
            beta = mano_params["betas"].detach().cpu().numpy().reshape(batch_size, -1)

            for i, slot in enumerate(slots):
                joints[frame_id, slot] = land[i] + translation[i]
                vertices[frame_id, slot] = raw_verts[i] + translation[i]
                root_rot[frame_id, slot] = rot[i]
                betas[frame_id, slot] = beta[i]
                out_valid[frame_id, slot] = True

    return {
        "joints_camera": joints,
        "vertices_camera": vertices,
        "root_rot": root_rot,
        "betas": betas,
        "valid": out_valid,
        "confidence": np.where(out_valid, confidence[:total], 0.0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WiLoR hand reconstruction runner")
    parser.add_argument("--check", action="store_true", help="report availability and exit")
    parser.add_argument("--frames", default=None)
    parser.add_argument("--detection", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--third-party", default="third_party")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--focal", type=float, default=None,
                        help="physical focal in px at the frame resolution (required)")
    parser.add_argument("--rescale", type=float, default=2.0,
                        help="crop padding factor, WiLoR demo default 2.0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = find_checkpoint(args.weights)
    if args.check:
        available = checkpoint is not None and (Path(args.third_party) / CHECKOUT_DIRNAME).is_dir()
        emit({
            "status": "ok" if available else "error",
            "backend": "wilor_hand",
            "available": available,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "device": args.device,
        })
        return 0 if available else 1
    try:
        if args.frames is None or args.detection is None or args.out is None:
            raise ValueError("--frames, --detection and --out are required")
        # resolve before run_model chdir-s into the checkout
        args.out = str(Path(args.out).resolve())
        arrays = run_model(args)
        save_npz(args.out, **arrays)
        emit({
            "status": "ok",
            "backend": "wilor_hand",
            "frames": int(arrays["valid"].shape[0]),
            "hands": int(arrays["valid"].sum()),
            "focal": float(args.focal),
            "rescale": float(args.rescale),
            "output": str(args.out),
            "device": args.device,
        })
        return 0
    except Exception as exc:  # noqa: BLE001 - reported through the runner protocol
        emit({"status": "error", "backend": "wilor_hand", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

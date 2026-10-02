#!/usr/bin/env python
"""Run VGGT-Omega on the bundled ``third_party/VGGT-Omega/examples`` videos and
turn each reconstruction into things you can actually look at.

This is a thin, self-contained demo harness: it reuses the *installed* backend
(``vggt_omega``) and the upstream ``visual_util`` GLB builder, but drives them
directly instead of through the Gradio UI, so it can run headless on a GPU box:

    conda run -n ego3d python scripts/run_vggt_examples.py \
        --checkpoint weights/vggt-omega/vggt_omega_1b_416_reproduce.pt \
        --out-dir outputs/vggt_examples --num-frames 24 --resolution 416

For every example video it writes, under ``out_dir/<clip>/``:

* ``frames/``                 - the sampled RGB frames actually fed to the model
* ``predictions.npz``         - raw depth / depth_conf / extrinsics / intrinsics
* ``scene.glb``               - point cloud + camera frusta (trimesh, GLB)
* ``rgb_depth_strip.png``     - RGB | depth-heat | RGB*depth overlay, one per frame
* ``depth_overlay.mp4``       - the same strip as an H.264 video
* ``turntable.mp4``           - rotating 3D point cloud with the camera trajectory
* ``summary.json``            - per-clip shapes, depth stats and a scale sanity check

Nothing is synthesised: every number comes from the model output. The 416
checkpoint is the one this project pins (see ``doc_auto/setup.md``); pass
``--precision fp16`` to halve the aggregator if GPU memory is tight.

On scale: the recovered depth is **up to scale**, not metric - the same scene can
come back at 0.7 m or 20 m of apparent depth depending on the clip (this is
exactly why the pipeline stitches windows with Sim(3) rather than trusting the
absolute numbers; see ``camera/stitch.py``). Treat the numbers in
``summary.json`` as relative, and the ground-plane fit as a consistency check.

Note on memory: ``--num-frames`` is limited by the *quadratic* inter-frame
attention, which on a P100 (sm_60, no flash kernel) is materialised densely.
Measured on a 12 GiB P100 at 416px: 8 frames fit, 12 do not; at 256px, 24 frames
fit (the tokens per frame drop ~7x). Use ``--resolution 256 --num-frames 24`` for
a longer trajectory, or ``--precision fp16`` to halve the aggregator weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ``third_party/VGGT-Omega`` provides the ``vggt_omega`` package and visual_util;
# the project's own overlay helpers live under ``src``.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "third_party" / "VGGT-Omega"))
sys.path.insert(0, str(_REPO / "src"))

# Running ``python`` by absolute path (e.g. ``conda run`` or a bare
# ``/envs/ego3d/bin/python``) does not put the env's ``bin`` on PATH, so ffmpeg -
# which lives next to the interpreter - is invisible to ``shutil.which`` and to
# matplotlib's FFMpegWriter. Prepend it so H.264 transcodes and the turntable
# render both work.
_INTERPRETER_BIN = Path(sys.executable).parent
if (_INTERPRETER_BIN / "ffmpeg").is_file() and str(_INTERPRETER_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_INTERPRETER_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
# On a 12 GiB P100 the aggregator attention is the peak; expandable segments keep
# the allocator from stranding free blocks between the per-view attention passes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ego3d_action.visualization.overlay import transcode_to_h264  # noqa: E402

#: Rows kept when back-projecting for the 3D turntable: dropping the top of the
#: frame (sky) leaves the ground/road plane, which is where the scale check is
#: meaningful.
_GROUND_ROWS = (0.55, 1.0)
#: Depth band (model units) kept in the point cloud. Depth is up to scale, so this
#: is a generous window, not a metric bound.
_DEPTH_RANGE = (0.1, 120.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", default=str(_REPO / "third_party" / "VGGT-Omega" / "examples"))
    parser.add_argument("--checkpoint", default=str(_REPO / "weights" / "vggt-omega" / "vggt_omega_1b_416_reproduce.pt"))
    parser.add_argument("--out-dir", default=str(_REPO / "outputs" / "vggt_examples"))
    parser.add_argument("--clips", nargs="*", default=None, help="subset of clip names (default: all)")
    # VGGT-Omega's inter-frame attention is global over all frames, so its memory
    # grows with (frames * tokens_per_frame)^2. On an A100 that stays small thanks
    # to flash attention, but the P100 this pipeline runs on is sm_60 - no flash
    # kernel - so scaled_dot_product_attention materialises the full (heads, N, N)
    # scores. Measured at 416px on a 12 GiB P100: 8 frames fit, 12 already OOM.
    parser.add_argument("--num-frames", type=int, default=8, help="frames sampled per clip (P100-safe default)")
    parser.add_argument("--resolution", type=int, default=416, help="VGGT image_resolution")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["auto", "fp16"], default="auto")
    parser.add_argument("--max-points", type=int, default=150_000, help="points in the turntable render")
    parser.add_argument("--frame-stride", type=int, default=6, help="turntable: render every Nth frame")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_frames(video: Path, out_dir: Path, num_frames: int) -> list[Path]:
    """Extract ``num_frames`` evenly spaced frames as PNGs; returns their paths."""
    import cv2  # noqa: PLC0415 - backend import

    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total - 1, min(num_frames, total)).round().astype(int)
    idxs = sorted(set(int(i) for i in idxs))
    paths: list[Path] = []
    for out_idx, frame_idx in enumerate(idxs):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok:
            continue
        path = out_dir / f"{out_idx:04d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    capture.release()
    if not paths:
        raise RuntimeError(f"no frames decoded from {video}")
    return paths


def load_model(checkpoint: Path, device: str, precision: str) -> object:
    import torch  # noqa: PLC0415
    from vggt_omega.models import VGGTOmega  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega requires CUDA; torch.cuda.is_available() is False")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    model = VGGTOmega().eval()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state, strict=True)
    # ``fp16`` halves the aggregator only; VGGTOmega runs the heads inside
    # autocast(enabled=False), so casting them to half would break the heads.
    if precision == "fp16":
        model.aggregator = model.aggregator.half()
    model = model.to(device)
    print(f"[model] {checkpoint.name} on {device} (precision={precision})", file=sys.stderr)
    return model


def infer(model: object, frame_paths: list[Path], resolution: int, device: str) -> dict[str, np.ndarray]:
    """Run one VGGT-Omega forward pass and decode to this project's convention."""
    import torch  # noqa: PLC0415
    from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: PLC0415
    from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: PLC0415

    images = load_and_preprocess_images([str(p) for p in frame_paths], image_resolution=resolution).to(device)
    with torch.inference_mode():
        predictions = model(images)
    height, width = predictions["images"].shape[-2:]
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], (height, width))

    out: dict[str, np.ndarray] = {
        "images": predictions["images"].detach().float().cpu().numpy()[0],
        "extrinsics": extrinsics.detach().float().cpu().numpy()[0],
        "intrinsics": intrinsics.detach().float().cpu().numpy()[0],
        "depth": predictions["depth"].detach().float().cpu().numpy()[0],
        "depth_conf": predictions["depth_conf"].detach().float().cpu().numpy()[0],
    }
    if out["depth"].ndim == 4:  # (views, H, W, 1) -> (views, H, W)
        out["depth"] = out["depth"][..., 0]
    if out["depth_conf"].ndim == 4:
        out["depth_conf"] = out["depth_conf"][..., 0]
    # Depth/intrinsics may live on a coarser grid than the RGB input.
    out["intrinsics"] = scale_to_depth(out["intrinsics"], out["images"].shape[-2:], out["depth"].shape[-2:])
    return out


def scale_to_depth(intrinsics: np.ndarray, image_hw: tuple[int, int], depth_hw: tuple[int, int]) -> np.ndarray:
    """Rescale pixel intrinsics from the RGB grid onto the (possibly coarser) depth grid."""
    if image_hw == depth_hw:
        return intrinsics
    sy = depth_hw[0] / image_hw[0]
    sx = depth_hw[1] / image_hw[1]
    scale = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    return np.einsum("ij,vjk->vik", scale, intrinsics)


def backproject(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, *, stride: int, rows: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth grid into world space; returns (points, depth)."""
    h, w = depth.shape
    r0 = int(round(rows[0] * h))
    r1 = max(r0 + 1, int(round(rows[1] * h)))
    sub = depth[r0:r1:stride, ::stride]
    vs, us = np.meshgrid(np.arange(r0, r1, stride), np.arange(0, w, stride), indexing="ij")
    pixels = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], axis=1)
    cam = (pixels @ np.linalg.inv(intrinsics).T) * sub.ravel()[:, None]
    world = cam @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    return world, sub.ravel()


def ground_plane_check(points: np.ndarray) -> dict[str, float] | None:
    """Fit a plane to the lower half of the points and report its geometry.

    The recovered depth is up to scale, so an absolute camera height is not
    meaningful. What *is* useful is the plane's orientation: on a road/desert/
    forest clip the ground should be a horizontal plane roughly one normal away
    from the first camera, so a wildly tilted normal or a plane that passes
    through the camera signals a bad reconstruction. SVD fit of the lowest 40 %
    of points (in world coordinates, which are anchored at the first camera).
    """
    if len(points) < 500:
        return None
    low = points[points[:, 1] <= np.percentile(points[:, 1], 40)]
    if len(low) < 100:
        return None
    centroid = low.mean(axis=0)
    _, _, vh = np.linalg.svd(low - centroid, full_matrices=False)
    normal = vh[-1]
    return {
        "ground_normal": normal.tolist(),
        "n_points_used": float(len(low)),
        "plane_offset": float(normal @ centroid),
    }


def render_rgb_depth_strip(frames: np.ndarray, depth: np.ndarray, out_path: Path) -> np.ndarray | None:
    """RGB | depth heatmap | RGB*depth overlay, stacked into one image per frame."""
    import cv2  # noqa: PLC0415

    strips = []
    for rgb, dep in zip(frames, depth, strict=True):
        rgb_u8 = np.transpose(np.clip(rgb, 0, 1) * 255, (1, 2, 0)).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        valid = dep[np.isfinite(dep) & (dep > 0)]
        hi = float(np.percentile(valid, 95)) if valid.size else float(dep.max() or 1.0)
        norm = np.clip(dep / max(hi, 1e-6), 0, 1)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat[~np.isfinite(dep)] = 0
        heat = cv2.resize(heat, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(rgb_bgr, 0.5, heat, 0.5, 0)
        strips.append(np.concatenate([rgb_bgr, heat, overlay], axis=1))
    if not strips:
        return None
    video = np.stack(strips)
    cv2.imwrite(str(out_path), strips[len(strips) // 2])
    return video


def write_video(frames: np.ndarray, out_path: Path, fps: float) -> Path:
    import cv2  # noqa: PLC0415

    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return transcode_to_h264(out_path)


def render_turntable(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, *, out_path: Path, max_points: int, stride: int, frame_stride: int, seed: int) -> None:
    """Rotating 3D point cloud (world space) with the camera trajectory drawn on it."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.animation  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    points, depths = [], []
    for view in range(depth.shape[0]):
        world, dep = backproject(depth[view], intrinsics[view], extrinsics[view], stride=stride, rows=(0.0, 1.0))
        keep = np.isfinite(world).all(axis=1) & (dep > _DEPTH_RANGE[0]) & (dep < _DEPTH_RANGE[1])
        points.append(world[keep])
        depths.append(dep[keep])
    points = np.concatenate(points)
    depths = np.concatenate(depths)
    if len(points) > max_points:
        sel = rng.choice(len(points), max_points, replace=False)
        points, depths = points[sel], depths[sel]

    centres = -np.einsum("vij,vj->vi", np.transpose(extrinsics[:, :3, :3], (0, 2, 1)), extrinsics[:, :3, 3])
    marker_stride = max(1, frame_stride)
    #: A full 360-degree sweep regardless of frame count, so the rotation reads
    #: the same whether the clip has 8 or 200 views.
    angles = np.linspace(-90, 270, 36)

    figure = plt.figure(figsize=(9, 7), dpi=90)
    axes = figure.add_subplot(111, projection="3d")

    def draw(angle: float):
        axes.clear()
        axes.scatter(points[::2, 0], points[::2, 1], points[::2, 2], c=depths[::2], cmap="turbo", s=0.4, alpha=0.5, linewidths=0)
        axes.plot(centres[:, 0], centres[:, 1], centres[:, 2], color="#ff4d4d", lw=1.4)
        axes.scatter(centres[::marker_stride, 0], centres[::marker_stride, 1], centres[::marker_stride, 2], color="#ff4d4d", s=6)
        axes.view_init(elev=22, azim=angle)
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_zlabel("z (m)")
        axes.set_title(f"VGGT-Omega point cloud + trajectory ({len(points):,} pts)")

    animation = matplotlib.animation.FuncAnimation(
        figure, draw, frames=angles, interval=80, blit=False
    )
    animation.save(str(out_path), writer=matplotlib.animation.FFMpegWriter(fps=12, bitrate=2400))
    plt.close(figure)


def process_clip(clip: Path, model: object, args: argparse.Namespace, root: Path) -> dict[str, object]:
    clip_dir = root / clip.stem
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    clip_dir.mkdir(parents=True)
    print(f"\n=== {clip.stem} ===", file=sys.stderr)

    frames = sample_frames(clip, clip_dir / "frames", args.num_frames)
    print(f"[frames] {len(frames)} sampled", file=sys.stderr)

    started = time.time()
    out = infer(model, frames, args.resolution, args.device)
    elapsed = time.time() - started
    np.savez_compressed(clip_dir / "predictions.npz", **out)
    print(f"[infer] {elapsed:.1f}s  depth{out['depth'].shape} images{out['images'].shape}", file=sys.stderr)

    # Upstream GLB point cloud + camera frusta.
    from visual_util import predictions_to_glb  # noqa: PLC0415

    glb_pred = {
        "images": out["images"],
        "extrinsic": out["extrinsics"],
        "depth": out["depth"][..., None],
        "depth_conf": out["depth_conf"],
        "world_points_from_depth": _world_points(out["depth"], out["extrinsics"], out["intrinsics"]),
    }
    scene = predictions_to_glb(glb_pred, conf_thres=20.0, show_cam=True, max_points=300_000)
    scene.export(file_obj=str(clip_dir / "scene.glb"))
    print("[glb] scene.glb written", file=sys.stderr)

    strip = render_rgb_depth_strip(out["images"], out["depth"], clip_dir / "rgb_depth_strip.png")
    depth_video = clip_dir / "depth_overlay.mp4"
    if strip is not None:
        write_video(strip, depth_video, fps=6.0)

    turntable = clip_dir / "turntable.mp4"
    render_turntable(
        out["depth"], out["intrinsics"], out["extrinsics"],
        out_path=turntable, max_points=args.max_points, stride=4,
        frame_stride=args.frame_stride, seed=args.seed,
    )

    ground_points, _ = backproject(out["depth"][0], out["intrinsics"][0], out["extrinsics"][0], stride=3, rows=_GROUND_ROWS)
    summary = {
        "clip": clip.stem,
        "source": str(clip),
        "num_frames": len(frames),
        "resolution": args.resolution,
        "infer_seconds": round(elapsed, 2),
        "images_shape": list(out["images"].shape),
        "depth_shape": list(out["depth"].shape),
        "depth_stats": {
            "min": float(np.nanmin(out["depth"])),
            "median": float(np.nanmedian(out["depth"])),
            "max": float(np.nanmax(out["depth"])),
        },
        "focal_px": [float(out["intrinsics"][0, 0, 0]), float(out["intrinsics"][0, 1, 1])],
        "scale_check": ground_plane_check(ground_points),
        "artefacts": sorted(p.name for p in clip_dir.iterdir() if p.is_file()),
    }
    (clip_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {clip.stem} -> {clip_dir}", file=sys.stderr)
    return summary


def _world_points(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """(views, H, W, 3) world points, matching the upstream demo helper."""
    num, height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xs = np.broadcast_to(xs[None], (num, height, width))
    ys = np.broadcast_to(ys[None], (num, height, width))
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]
    cam = np.stack([(xs - cx) / fx * depth, (ys - cy) / fy * depth, depth], axis=-1)
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)), cam - translation[:, None, None, :])


def write_gallery(summaries: list[dict[str, object]], root: Path) -> Path:
    rows = []
    for item in summaries:
        clip = item["clip"]
        scale = item.get("scale_check") or {}
        rows.append(
            f"""
  <section>
    <h2>{clip}</h2>
    <p><small>{item['num_frames']} frames @ {item['resolution']}px · {item['infer_seconds']}s ·
    depth {item['depth_shape'][1]}×{item['depth_shape'][2]} · focal {item['focal_px'][0]:.0f}px ·
    depth median {item['depth_stats']['median']:.1f} m</small></p>
    <div class="row">
      <figure><img src="{clip}/rgb_depth_strip.png"><figcaption>RGB | depth | overlay</figcaption></figure>
      <div>
        <video controls loop src="{clip}/depth_overlay.mp4"></video><figcaption>depth overlay</figcaption>
        <video controls loop src="{clip}/turntable.mp4"></video><figcaption>point cloud + trajectory</figcaption>
      </div>
    </div>
    <p>GLB: <code>{clip}/scene.glb</code> (drag into <a href="https://gltf-viewer.donmccurdy.com/">gltf-viewer</a>) ·
    predictions: <code>{clip}/predictions.npz</code>
    {'· ground normal ' + str([round(v, 2) for v in scale['ground_normal']]) if scale else ''}</p>
  </section>"""
        )
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>VGGT-Omega 示例可视化</title>
<style>
  body {{ background:#111418; color:#d8dee6; font-family:system-ui,sans-serif; margin:24px auto; max-width:1150px; }}
  h1 {{ font-size:20px; }} h2 {{ font-size:16px; color:#7ec8ff; margin-top:8px; }}
  video {{ width:100%; max-width:520px; display:block; margin:4px 0; background:#000; }}
  img {{ max-width:520px; border:1px solid #333; }}
  .row {{ display:flex; flex-wrap:wrap; gap:20px; }}
  figure {{ margin:0; }} figcaption {{ font-size:12px; color:#8a97a6; }}
  section {{ border-top:1px solid #222; padding-top:16px; }}
  code {{ color:#9fd48a; }} p,li {{ font-size:13px; line-height:1.6; }}
</style></head>
<body>
<h1>VGGT-Omega 示例视频重建（{len(summaries)} clips）</h1>
<p>用 <code>ego3d</code> 环境 + <code>VGGT-Omega-1B-416-Reproduction</code> 检查点，
脚本 <code>scripts/run_vggt_examples.py</code>。上排为 RGB | 深度热力 | 叠加，下排为深度视频与旋转点云（含相机轨迹）。</p>
<p><small>注意：VGGT-Omega 恢复的深度是<b>相对尺度</b>（同一场景可能回来 0.7 m 也可能 20 m），
所以点云/轨迹的单位只表示相对关系；这正是流水线用 Sim(3) 拼接窗口而非直接采信绝对数值的原因。
P100（sm_60 无 flash attention）12 GiB 下 416px 上限约 8 帧，256px 约 24 帧。</small></p>
{''.join(rows)}
</body></html>
"""
    path = root / "index.html"
    path.write_text(html)
    return path


def main() -> int:
    args = parse_args()
    examples = Path(args.examples)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    clips = sorted(examples.glob("*.mp4"))
    if args.clips:
        wanted = set(args.clips)
        clips = [c for c in clips if c.stem in wanted]
    if not clips:
        raise SystemExit(f"no example videos found under {examples}")

    model = load_model(Path(args.checkpoint), args.device, args.precision)
    summaries = [process_clip(clip, model, args, root) for clip in clips]
    gallery = write_gallery(summaries, root)
    print(f"\n[gallery] {gallery}", file=sys.stderr)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""HaWoR runner: executed **inside** the HaWoR environment (``ego3d_hawor``).

    python backends/hawor_runner.py --check
    python backends/hawor_runner.py --detection detection.npz --out-dir hand/windows \
        --frames data/clip/frames --num-frames 450 --window 16 --overlap 8 \
        --third-party third_party --weights weights --device cuda

How this differs from HaWoR's own demo
--------------------------------------

``demo.py`` runs ``detect_track_video`` (WiLoR + HaWoR's own ``thresh=0.2``
rule set), then ``hawor_motion_estimation`` -> ``hawor_slam`` ->
``hawor_infiller``. This runner replaces the first step: the tracking decision
comes from **our** conservative tracker via ``detection.npz``, written into the
``model_tracks.npy`` structure ``hawor_motion_estimation`` reads. HaWoR then
reconstructs exactly the frames Phase 1 kept.

Two facts discovered by reading HaWoR's source, both load-bearing here:

1. ``hawor_infiller`` hard-depends on ``SLAM/hawor_slam_w_scale_*.npz`` and
   produces hands in **HaWoR's world frame** (the comment in its source says
   "camera space to world space"). Upstream fills that file with DROID-SLAM.
   This project replaces DROID-SLAM with VGGT-Omega (see
   ``environment-hawor.yml``), so the runner **synthesises the same npz from the
   VGGT camera windows** (:func:`write_camera_trajectory`) and then converts the
   hands back into **camera space** with those poses - which is what Phase 2 must
   hand to fusion. VGGT owns the metric world trajectory; the npz is only a
   coordinate carrier. With ``--camera-windows`` the summary reports
   ``camera_source: vggt``; without it the camera is constant and the summary
   says ``constant``, so a degraded run can never look like a real one.
2. ``run_mano``/``run_mano_left`` are what turn ``(trans, rot, hand_pose,
   betas)`` into 21 landmarks, and they need the MANO model inside the HaWoR
   checkout.

``hawor_motion_estimation`` also builds a hand mask with HaWoR's pytorch3d
renderer. That mask is read only by DROID-SLAM, so :func:`install_renderer_stub`
substitutes an inert renderer when pytorch3d is missing, which removes the
nvcc/gcc build requirement without changing the reconstruction.

The conversion of 21 camera-space joints into our 16/8 window artefacts lives in
``ego3d_action.hand.hawor`` and is unit-tested. What cannot be verified without
the checkpoint and a GPU is the HaWoR call sequence itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ego3d_action.hand.hawor import (  # noqa: E402
    HAWOR_FILE_CANDIDATES,
    find_weights_files,
    hawor_tracks_from_detection,
    hand_windows_from_joints,
    save_hawor_tracks,
)
from ego3d_action.io.serialization import load_npz  # noqa: E402
from ego3d_action.runtime.checkpoints import (  # noqa: E402
    allow_trusted_checkpoint_globals,
    restore_legacy_numpy_aliases,
)
from ego3d_action.runtime.checkpoints import report as report_checkpoint_globals  # noqa: E402

from _shard_cli import (  # noqa: E402
    add_shard_arguments,
    partition_is_reusable,
    record_partition,
    select_windows,
    selection_from_args,
)

BACKEND_MODULE = "hawor"
#: Which camera the hands were reconstructed against, so the summary can report
#: ``vggt`` (real) vs ``constant`` (degraded) instead of hiding the difference.
_LAST_TRAJECTORY: dict[str, object] = {"source": None}
#: MANO is licence-gated and is *not* part of the checkout. HaWoR loads the right
#: hand from _DATA/data/mano and the left from _DATA/data_left/mano_left (see
#: hawor/utils/process.py); the left model is optional because run_mano_left has
#: a fix_shapedirs workaround, the right one is required.
MANO_RIGHT_CANDIDATES = ("_DATA/data/mano/MANO_RIGHT.pkl", "_DATA/data/mano/MANO_RIGHT.npz")
MANO_LEFT_CANDIDATES = (
    "_DATA/data_left/mano_left/MANO_LEFT.pkl",
    "_DATA/data_left/mano_left/MANO_LEFT.npz",
)


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def install_renderer_stub() -> None:
    """Make ``lib.vis.renderer`` importable without pytorch3d.

    ``hawor_video`` imports HaWoR's ``Renderer`` at module level, and that module
    needs ``pytorch3d`` - which in turn needs nvcc/gcc to build. The only thing
    the renderer is used for inside ``hawor_motion_estimation`` is rasterising
    the hand mesh into ``model_masks.npy``, and that mask is read by exactly one
    consumer: DROID-SLAM. This project takes the camera from VGGT-Omega instead
    (``environment-hawor.yml`` says so explicitly), so the mask is never read and
    a stub that produces an empty mask is behaviourally identical.

    Installed only if pytorch3d is genuinely absent, so a fully-provisioned
    machine still gets HaWoR's real renderer.
    """
    import importlib.util
    import types

    # The stub is needed whenever pytorch3d cannot be imported - which is the
    # case on any host without nvcc/gcc. ``find_spec`` on the renderer itself
    # would succeed (the .py is present), so the only correct guard is probing
    # pytorch3d, the thing that actually fails.
    try:
        import pytorch3d  # noqa: F401, PLC0415
    except ImportError:
        pass
    else:
        return  # the real renderer imports fine; leave HaWoR untouched

    class _StubRenderer:
        def __init__(self, width: int, height: int, *args: object, **kwargs: object) -> None:
            self.width = width
            self.height = height

        def create_camera_from_cv(self, rotation: object, translation: object) -> tuple[None, None]:
            return None, None

        def render_multiple(self, *args: object, **kwargs: object):
            import torch  # noqa: PLC0415

            # an all-False mask: nothing is masked out for a SLAM run we skip
            import numpy as _np  # noqa: PLC0415

            # Upstream's renderer returns a per-frame mask with *no* batch
            # dimension - HaWoR accumulates it as ``model_masks[frame] += mask``
            # into an (T, H, W) numpy array, so a (1, H, W) mask raises
            # "non-broadcastable output operand". Return one frame's worth.
            empty = _np.zeros((self.height, self.width), dtype=bool)
            return torch.from_numpy(empty[None]), empty

    package = types.ModuleType("lib.vis")
    module = types.ModuleType("lib.vis.renderer")
    module.Renderer = _StubRenderer
    package.renderer = module
    sys.modules["lib.vis"] = package
    sys.modules["lib.vis.renderer"] = module


def write_camera_trajectory(
    args: argparse.Namespace, seq_folder: Path, start_idx: int, end_idx: int
) -> Path:
    """Write the ``SLAM/hawor_slam_w_scale_*.npz`` that ``hawor_infiller`` reads.

    HaWoR's infiller needs a per-frame world->camera pose set; upstream gets it
    from DROID-SLAM. This project replaces DROID-SLAM with VGGT-Omega, so the
    poses come from the VGGT camera windows when they are available. The hands
    then land in a frame consistent with the camera Phase 4 stitches, which is
    what fusion needs.

    Without ``--camera-windows`` the trajectory degrades to a constant camera
    (identity rotation, zero translation): the hands are still reconstructed,
    but camera motion is not compensated, so the result is only usable for a
    static/lightly-moving clip. That degradation is reported in the JSON summary
    as ``camera_source: constant`` so it can never pass for a real run.

    Layout (mirrors ``hawor_slam.py``, which is the upstream producer):
        ``traj``     (N, 7)  = [tx, ty, tz, qx, qy, qz, qw]  (see
                     :func:`_matrix_to_quaternion` for the quaternion order)
        ``scale``    scalar  metric scale applied to ``traj[:, :3]``
        ``tstamp``   (N,)    frame indices
    """
    from ego3d_action.camera.window import load_camera_window  # noqa: PLC0415

    count = end_idx - start_idx
    rotations = np.broadcast_to(np.eye(3), (count, 3, 3)).astype(np.float64).copy()
    translations = np.zeros((count, 3), dtype=np.float64)
    source = "constant"

    if args.camera_windows:
        windows_dir = Path(args.camera_windows)
        candidates = sorted(windows_dir.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"--camera-windows {windows_dir} holds no *.npz; run Phase 3 first "
                "(scripts/run_camera.py) or omit the flag to use a constant camera"
            )
        coverage = np.zeros(count, dtype=bool)
        # Prefer the *stitched* trajectory: each raw VGGT window is normalised
        # to its own first frame, so adjacent windows disagree on shared frames
        # (~4 deg / ~2 cm, measured on hot3d_ep000) and the naive per-frame
        # assembly produced a pose stream that JUMPED at every window boundary -
        # the infiller's temporal model then operated on discontinuous data.
        # Phase 4 (run_stitch) aligns all windows into one World-0 frame; use it
        # whenever it exists.
        stitched_path = windows_dir.parent / "stitched_camera.npz"
        if stitched_path.is_file():
            from ego3d_action.camera.stitch import load_stitched_camera  # noqa: PLC0415

            stitched = load_stitched_camera(stitched_path)
            if stitched.num_frames < end_idx:
                raise FileNotFoundError(
                    f"{stitched_path} covers {stitched.num_frames} frames but the clip "
                    f"needs {end_idx}; re-run scripts/run_stitch.py"
                )
            rotations[:] = stitched.rotation_c2w[start_idx:end_idx]
            translations[:] = stitched.translation_c2w[start_idx:end_idx]
            coverage[:] = np.asarray(stitched.valid, dtype=bool)[start_idx:end_idx]
            if not coverage.all():
                missing = int((~coverage).sum())
                print(
                    f"WARNING: the stitched camera marks {missing} of {count} frames "
                    f"([{start_idx}, {end_idx})) invalid; those fall back to the "
                    "identity camera.",
                    file=sys.stderr,
                )
            source = "vggt-stitched"
        else:
            print(
                f"WARNING: {stitched_path} is missing, so the poses come from the raw "
                "VGGT windows. Raw windows have per-window world frames, so this pose "
                "stream jumps at window boundaries; run Phase 4 (scripts/run_stitch.py) "
                "first for a consistent world frame.",
                file=sys.stderr,
            )
            for path in candidates:
                window = load_camera_window(path)
                first, last = window.window.start, window.window.end
                for frame in range(max(first, start_idx), min(last, end_idx)):
                    local = frame - first
                    slot = frame - start_idx
                    rotations[slot] = window.rotation_c2w[local]
                    translations[slot] = window.translation_c2w[local]
                    coverage[slot] = True
            if not coverage.all():
                missing = int((~coverage).sum())
                print(
                    f"WARNING: the VGGT windows do not cover {missing} of {count} frames "
                    f"([{start_idx}, {end_idx})); those fall back to the identity camera. "
                    "Check camera.window/camera.overlap against this clip's frame count.",
                    file=sys.stderr,
                )
            source = "vggt-raw-windows"

    # VGGT outputs metric depth, so no extra scale correction is needed.
    quaternions = _matrix_to_quaternion(rotations)
    trajectory = np.concatenate([translations, quaternions], axis=1)
    slam_dir = seq_folder / "SLAM"
    slam_dir.mkdir(parents=True, exist_ok=True)
    save_path = slam_dir / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    # float32, not float64: ``hawor_infiller`` feeds this straight into
    # ``torch.einsum`` next to the model's float32 outputs, and DROID-SLAM's own
    # npz is float32. Writing doubles raised "expected scalar type Double but
    # found Float" inside cam2world_convert.
    np.savez(
        save_path,
        tstamp=np.arange(start_idx, end_idx, dtype=np.int64),
        traj=trajectory.astype(np.float32),
        img_focal=np.float32(float(args.focal) if args.focal else 0.0),
        img_center=np.zeros(2, dtype=np.float32),
        scale=np.float32(1.0),
    )
    _LAST_TRAJECTORY["source"] = source
    return save_path


def _matrix_to_quaternion(rotations: np.ndarray) -> np.ndarray:
    """Rotation matrices -> ``qx, qy, qz, qw``.

    The order matters: ``load_slam_cam`` reads ``traj[:, 3:]`` and reindexes it
    with ``[[3, 0, 1, 2]]`` before handing it to a ``(w, x, y, z)`` decoder, so
    the stored order is DROID-SLAM's imaginary-first ``(x, y, z, w)`` - which is
    exactly what ``scipy``'s ``as_quat()`` returns.

    scipy is used rather than a hand-rolled trace formula because the closed-form
    sqrt version loses precision for near-identity rotations: an egocentric rig
    barely rotates between frames (here ``~2e-4`` rad), and that formula returns
    an absolute error of the same magnitude for exactly those matrices.
    """
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None]
    return Rotation.from_matrix(matrices).as_quat()


def backend_available(third_party: Path) -> tuple[bool, str]:
    """Report whether HaWoR (and its MANO assets) can be imported."""
    checkout = third_party / "HaWoR"
    if not checkout.is_dir():
        return False, f"HaWoR checkout not found at {checkout}"
    right = next((checkout / rel for rel in MANO_RIGHT_CANDIDATES if (checkout / rel).exists()), None)
    left = next((checkout / rel for rel in MANO_LEFT_CANDIDATES if (checkout / rel).exists()), None)
    if right is None:
        return False, (
            "MANO_RIGHT.pkl is missing - HaWoR's run_mano cannot build joints. "
            "Get it from https://mano.is.tue.mpg.de/ and run "
            "scripts/install_mano.sh --from <mano dir>"
        )
    sys.path.insert(0, str(checkout))
    try:
        __import__(BACKEND_MODULE)
    except ImportError as exc:
        return False, f"cannot import '{BACKEND_MODULE}' from {checkout}: {exc}"
    detail = f"'{BACKEND_MODULE}' importable from {checkout}, MANO right at {right}"
    if left is None:
        detail += " (left model absent - run_mano_left will use fix_shapedirs)"
    return True, detail


def build_hawor_args(
    args: argparse.Namespace,
    seq_folder: Path,
    frames_dir: Path,
    checkout: Path | None = None,
) -> object:
    """Assemble the namespace HaWoR's functions expect.

    HaWoR derives ``seq_folder`` from ``video_path`` (``<dir>/<stem>``) and reads
    the frames from ``<seq_folder>/extracted_images``. We point it at our own
    Phase-0 frames instead of letting it re-extract them.

    ``video_path`` is deliberately absolute: the caller ``chdir``s into the
    checkout (HaWoR resolves ``./_DATA`` against the CWD), and a relative
    ``<dir>/<stem>`` would then point at the checkout instead of our clip.

    ``checkout`` is optional: when given it is checked up front so a wrong
    ``--third-party`` fails with the path instead of an import error from deep
    inside HaWoR.
    """
    if checkout is not None and not Path(checkout).is_dir():
        raise NotADirectoryError(f"HaWoR checkout not found at {checkout}")
    images = seq_folder / "extracted_images"
    images.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(sorted(frames_dir.glob("*.jpg"))):
        target = images / f"{index:04d}.jpg"
        if target.exists():
            continue
        try:
            os.link(frame, target)
        except OSError:  # different filesystem - fall back to a symlink
            target.symlink_to(frame.resolve())
    files = find_weights_files(args.weights, args.third_party)
    missing = [name for name in ("checkpoint", "infiller") if name not in files]
    if missing:
        raise FileNotFoundError(
            f"HaWoR weight file(s) {missing} not found under {args.weights}; looked for "
            f"{[rel for name in missing for rel in HAWOR_FILE_CANDIDATES[name]]} - run "
            "scripts/download_weights.sh --only hawor"
        )
    checkpoint = Path(files["checkpoint"]).resolve()
    ensure_model_config_next_to(checkpoint, files.get("model_config"))
    # HaWoR's own hawor_video.load_hawor calls torch.load without weights_only,
    # but its infiller and our model load path go through torch>=2.6's strict
    # default, so allowlist before HaWoR touches the files.
    for name in ("checkpoint", "infiller"):
        allowed = allow_trusted_checkpoint_globals(files[name], label=f"HaWoR {name}")
        report_checkpoint_globals(files[name], allowed)
    return argparse.Namespace(
        video_path=str(seq_folder.parent / f"{seq_folder.name}.mp4"),
        input_type="file",
        checkpoint=str(checkpoint),
        infiller_weight=str(Path(files["infiller"]).resolve()),
        img_focal=args.focal,
        vis_mode="cam",
    )


def ensure_model_config_next_to(checkpoint: Path, config: Path | None) -> None:
    """Put ``model_config.yaml`` where HaWoR looks for it.

    HaWoR's ``load_hawor`` derives the config path as
    ``checkpoint.parent.parent / "model_config.yaml"``, so a checkpoint at
    ``weights/hawor/checkpoints/hawor.ckpt`` makes it look in ``weights/hawor/``.
    ``scripts/download_weights.sh`` writes the file *inside* ``checkpoints/``,
    which HaWoR never reads - it then dies with a bare ``FileNotFoundError`` that
    looks like a missing weight.

    The file is copied up (not moved) so both layouts keep working; an existing
    file is never overwritten.
    """
    expected = checkpoint.parent.parent / "model_config.yaml"
    if expected.is_file():
        return
    source = config if config is not None and Path(config).is_file() else None
    if source is None:
        return  # nothing to copy; HaWoR will report the real problem
    expected.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, expected)
    print(
        f"copied model config {source} -> {expected} (HaWoR reads it from there)",
        file=sys.stderr,
    )


def hawor_checkout_cwd(third_party: Path) -> Path:
    """The working directory HaWoR's functions must run in.

    ``hawor/configs/__init__.py`` hard-codes ``CACHE_DIR_HAWOR = "./_DATA"`` and
    resolves ``MANO.MODEL_PATH``/``MANO.MEAN_PARAMS`` against it, so all of
    HaWoR's MANO lookups are **relative to the current working directory** and
    only resolve when that directory is the checkout itself (``_DATA/data/mano``,
    ``_DATA/data_left/mano_left/``). Running from the repository root makes every
    one of them a missing file.

    Note this must happen *after* our own paths are absolutised, because our
    ``--frames``/``--out-dir`` are relative to the repository root.
    """
    checkout = Path(third_party) / "HaWoR"
    if not checkout.is_dir():
        raise NotADirectoryError(f"HaWoR checkout not found at {checkout}")
    return checkout


def _patched_loader(loader: object, *, half: bool) -> object:
    """Wrap HaWoR's ``load_hawor`` to control *where* and *how* it loads.

    ``hawor_motion_estimation`` calls the module-level ``load_hawor``, so
    rebinding that name patches every call site without touching the checkout.

    Two things are forced here, both measured on the server rather than guessed:

    **The checkpoint is restored to CPU.** HaWoR calls
    ``HAWOR.load_from_checkpoint(path)`` without ``map_location``, and the 3.0 GiB
    checkpoint was saved from a GPU, so torch materialises 4.3 GB of tensors on
    the *GPU* before the model is even moved there - that allocation is what
    OOMed on a shared 12 GB P100, and it did not shrink when the crop size or the
    window changed, because it has nothing to do with inference. Forcing
    ``map_location="cpu"`` during the load drops the peak by ~4.3 GB.

    **Half precision, when asked.** ``inference`` is wrapped in autocast because
    HaWoR feeds it fp32 image tensors while the weights are fp16.
    """

    def load_hawor_patched(checkpoint_path: str) -> tuple[object, object]:
        import torch  # noqa: PLC0415

        original_load = torch.load

        def cpu_load(*call_args: object, **call_kwargs: object) -> object:
            # Lightning does not pass None here: ``load_from_checkpoint`` hands
            # torch.load its own ``_default_map_location`` callable, which picks
            # the current device - CUDA on a GPU host - so the 3.35 GB state dict
            # is materialised on the GPU before Lightning's own ``model.to(device)``
            # copies it a second time. Only an explicit str/torch.device from the
            # caller counts as intent; None and callables are steered to CPU.
            map_location = call_kwargs.get("map_location")
            if map_location is None or callable(map_location):
                call_kwargs["map_location"] = "cpu"
            return original_load(*call_args, **call_kwargs)

        torch.load = cpu_load
        try:
            model, model_cfg = loader(checkpoint_path)  # type: ignore[operator]
        finally:
            torch.load = original_load

        if half:
            model = model.half()
        original_inference = model.inference

        def inference(*call_args: object, **call_kwargs: object) -> object:
            if half:
                with torch.autocast("cuda", dtype=torch.float16):
                    return original_inference(*call_args, **call_kwargs)
            return original_inference(*call_args, **call_kwargs)

        model.inference = inference
        return model, model_cfg

    return load_hawor_patched


def _patch_crop_size(crop_size: int) -> None:
    """Make HaWoR's hand crops ``crop_size`` px instead of the upstream 256.

    ``TrackDatasetEval`` takes ``crop_size`` as a keyword with a default, and
    ``HAWOR.inference`` never passes it, so re-defaulting it here changes every
    call site without copying (or forking) upstream code. Activation memory
    scales with the crop *area*: 192 px is 56 % of 256 px, which is the
    difference between fitting and not fitting next to another job on a 12 GB
    card. It is a quality/memory trade-off and the runner reports it.
    """
    from lib.datasets.track_dataset import TrackDatasetEval  # noqa: PLC0415

    original_init = TrackDatasetEval.__init__

    def init_with_crop_size(self: object, *call_args: object, **call_kwargs: object) -> None:
        call_kwargs.setdefault("crop_size", crop_size)
        original_init(self, *call_args, **call_kwargs)

    TrackDatasetEval.__init__ = init_with_crop_size  # type: ignore[method-assign]


def invalidate_stale_hand_cache(
    seq_folder: Path,
    focal: float | None,
    box_pad: float = 1.0,
    boxes_fingerprint: str | None = None,
) -> None:
    """Drop HaWoR caches when focal, box padding, or tracked boxes change."""
    if focal is None:
        return
    marker = seq_folder / "est_focal.txt"
    marker_state: dict[str, object] = {"focal": float(focal), "box_pad": float(box_pad)}
    if boxes_fingerprint is not None:
        marker_state["boxes"] = str(boxes_fingerprint)

    previous_state = None
    force_invalidate = False
    if marker.is_file():
        raw = marker.read_text()
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            try:
                previous = float(raw.strip())
            except ValueError:
                print(f"invalid HaWoR cache marker {marker}; clearing cached tracks", file=sys.stderr)
                force_invalidate = True
            else:
                previous_state = {"focal": previous, "box_pad": 1.0}
        else:
            if isinstance(loaded, dict) and "focal" in loaded:
                try:
                    previous_state = {
                        "focal": float(loaded["focal"]),
                        "box_pad": float(loaded.get("box_pad", 1.0)),
                    }
                    if "boxes" in loaded:
                        previous_state["boxes"] = str(loaded["boxes"])
                except (TypeError, ValueError):
                    print(f"invalid HaWoR cache marker {marker}; clearing cached tracks", file=sys.stderr)
                    force_invalidate = True
            elif isinstance(loaded, (int, float)):
                previous_state = {"focal": float(loaded), "box_pad": 1.0}
            else:
                print(f"invalid HaWoR cache marker {marker}; clearing cached tracks", file=sys.stderr)
                force_invalidate = True

    marker.write_text(json.dumps(marker_state))
    if not force_invalidate and (previous_state is None or previous_state == marker_state):
        return

    removed = 0
    for cache in sorted(seq_folder.glob("tracks_*")):
        if cache.is_dir():
            shutil.rmtree(cache)
            removed += 1
    if removed:
        print(
            f"reconstruction inputs changed: dropped {removed} cached HaWoR track(s) "
            f"(old={previous_state}, new={marker_state})",
            file=sys.stderr,
        )


def absolutize_paths(args: argparse.Namespace) -> None:
    """Resolve every path we pass to HaWoR *before* it chdirs into its checkout.

    ``run_model`` moves the process into ``third_party/HaWoR`` because HaWoR
    resolves ``./_DATA`` against the working directory. Any relative path that is
    read after that point is then resolved against the checkout instead of the
    repository - which is how ``--camera-windows data/<clip>/camera/windows``
    came back as "holds no *.npz" while the directory plainly had 39 of them.
    """
    for name in ("frames", "out_dir", "detection", "camera_windows"):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(Path(value).resolve()))


def to_camera_space(
    r_w2c: object,
    t_w2c: object,
    landmarks: object,
    *,
    valid: object,
    pred_valid: object,
    confidence: object,
    vertices: object | None = None,
) -> dict[str, np.ndarray]:
    """World landmarks + per-frame w2c pose -> the clip's camera-space contract.

    ``landmarks`` is ``(T, 2, 21, 3)`` (frame-major, as the contract wants),
    ``r_w2c``/``t_w2c`` are ``(T, 3, 3)`` / ``(T, 3)`` from the camera stage, and
    the two validity sources are "the tracker saw this hand" (``valid``,
    ``(T, 2)``) and "the infiller trusted this hand" (``pred_valid``, ``(2, T)``).
    Frames that fail either test stay ``NaN`` - this project never fabricates a
    missing 3D pose. ``vertices`` (``(T, 2, V, 3)``, optional) takes the same
    transform so the debug video can draw the MANO mesh.
    """
    rotation = np.asarray(r_w2c, dtype=np.float64)
    translation = np.asarray(t_w2c, dtype=np.float64)
    joints = np.asarray(landmarks, dtype=np.float64)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise ValueError(f"landmarks must be [T, 2, 21, 3], got {joints.shape}")
    if rotation.shape != (joints.shape[0], 3, 3):
        raise ValueError(
            f"r_w2c must be [{joints.shape[0]}, 3, 3], got {rotation.shape}"
        )
    if translation.shape != (joints.shape[0], 3):
        raise ValueError(
            f"t_w2c must be [{joints.shape[0]}, 3], got {translation.shape}"
        )
    # X_cam = R_w2c @ X_world + t_w2c: the matrix is applied on the LEFT, so the
    # einsum must contract the matrix's SECOND index with the point ("tij"),
    # never the first ("tji" = R^T). The "tji" form mixes R_c2w into a w2c
    # transform and measured 276 mm median error when recovering HOT3D
    # camera-space hands from their world positions; "tij" recovers them at
    # 0.0 mm. The old identity-rotation unit test could not see the difference.
    camera_space = (
        np.einsum("tij,thnj->thni", rotation, joints) + translation[:, None, None, :]
    )
    hand_valid = np.asarray(pred_valid, dtype=np.float64).T > 0.5
    all_valid = hand_valid & np.asarray(valid, dtype=bool)
    result = {
        "joints_camera": np.where(all_valid[:, :, None, None], camera_space, np.nan),
        "valid": all_valid,
        "confidence": np.where(all_valid, np.asarray(confidence, dtype=np.float64), 0.0),
    }
    if vertices is not None:
        verts = np.asarray(vertices, dtype=np.float64)
        if verts.shape[:2] != joints.shape[:2] or verts.shape[3] != 3:
            raise ValueError(
                f"vertices must be [{joints.shape[0]}, 2, V, 3], got {verts.shape}"
            )
        # same left-multiply rule as the joints above, but the vertex array's
        # last axis is the 3-D coordinate: it must carry the *contracted*
        # index j ("tvnj"), not the output index i. Getting this wrong - the
        # old "tji,tvni->tvni" did - scales each axis by a matrix row/column
        # sum instead of rotating (caught by the non-identity round-trip test).
        camera_vertices = (
            np.einsum("tij,tvnj->tvni", rotation, verts) + translation[:, None, None, :]
        )
        result["vertices_camera"] = np.where(
            all_valid[:, :, None, None], camera_vertices, np.nan
        ).astype(np.float32)
    return result


def run_model(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Run HaWoR over the clip and return camera-space hands for every frame."""
    absolutize_paths(args)
    third_party = Path(args.third_party)
    frames_dir = Path(args.frames).resolve()
    out_dir = Path(args.out_dir).resolve()
    seq_folder = out_dir.parent / "hawor_seq"
    seq_folder.mkdir(parents=True, exist_ok=True)
    # 1. our tracking becomes HaWoR's model_tracks.npy. Fingerprint detections
    # before writing this run's tracks, so changed boxes cannot reuse stale
    # model parameters from an earlier input.
    detection = load_npz(args.detection, required=("boxes", "confidence", "valid"))
    boxes = np.asarray(detection["boxes"], dtype=np.float64)
    confidence = np.asarray(detection["confidence"], dtype=np.float64)
    valid = np.asarray(detection["valid"], dtype=bool)
    boxes_fingerprint = hashlib.sha1(np.ascontiguousarray(boxes).tobytes()).hexdigest()[:16]
    invalidate_stale_hand_cache(
        seq_folder, args.focal, boxes_fingerprint=boxes_fingerprint
    )
    model_boxes, tracks = hawor_tracks_from_detection(
        boxes=boxes, confidence=confidence, valid=valid
    )
    start_idx, end_idx = 0, int(boxes.shape[0])
    save_hawor_tracks(
        seq_folder / f"tracks_{start_idx}_{end_idx}", model_boxes, tracks
    )
    print(
        f"HaWoR will reconstruct {int(valid.sum())} tracked hand-frames "
        f"(left {int(valid[:, 0].sum())}, right {int(valid[:, 1].sum())})",
        file=sys.stderr,
    )

    # 2. HaWoR itself - everything below runs from inside the checkout, so all
    #    relative paths handed to HaWoR (and its "./_DATA" cache dir) resolve.
    checkout = hawor_checkout_cwd(third_party)
    sys.path.insert(0, str(checkout))
    install_renderer_stub()
    # HaWoR's MANO wrapper pulls in chumpy, which still does
    # ``from numpy import bool, int, ...`` - removed from numpy 1.24.
    restore_legacy_numpy_aliases()
    from hawor.utils.process import run_mano, run_mano_left  # noqa: PLC0415
    from lib.eval_utils.custom_utils import load_slam_cam  # noqa: PLC0415
    from scripts.scripts_test_video.hawor_video import (  # noqa: PLC0415
        hawor_infiller,
        hawor_motion_estimation,
    )
    import torch  # noqa: PLC0415

    hawor_args = build_hawor_args(args, seq_folder, frames_dir, checkout)
    precision = getattr(args, "precision", "auto")
    import scripts.scripts_test_video.hawor_video as hawor_video  # noqa: PLC0415

    hawor_video.load_hawor = _patched_loader(
        hawor_video.load_hawor, half=precision == "fp16"
    )
    print(
        "HaWoR checkpoint restore forced to CPU (upstream leaves map_location unset, "
        "which materialises ~4.3 GB of the checkpoint on the GPU before the model moves)",
        file=sys.stderr,
    )
    if precision == "fp16":
        print(
            "HaWoR precision=fp16: hamer checkpoint half-precision "
            "(3.0 GiB -> 1.5 GiB resident); joint output is validated as finite",
            file=sys.stderr,
        )
    crop_size = int(getattr(args, "crop_size", 256) or 256)
    if crop_size != 256:
        _patch_crop_size(crop_size)
        print(
            f"HaWoR crop_size={crop_size} (upstream 256): reduces activation memory; "
            "record it as a degraded mode in the ablation table",
            file=sys.stderr,
        )
    previous_cwd = Path.cwd()
    os.chdir(checkout)
    try:
        # Our own video_path points at <repo>/data/<clip>/<clip>.mp4 but HaWoR
        # reads frames from <video_dir>/<stem>/extracted_images, so the video
        # path has to stay absolute for that derivation to land in seq_folder.
        frame_chunks_all, _img_focal = hawor_motion_estimation(
            hawor_args, start_idx, end_idx, seq_folder
        )
        # Upstream calls hawor_slam() (DROID-SLAM) here. This project takes the
        # camera from VGGT-Omega instead, so the SLAM npz is synthesised from the
        # VGGT windows - same file, same keys, no lietorch/nvcc requirement.
        slam_path = write_camera_trajectory(args, seq_folder, start_idx, end_idx)
        r_w2c, t_w2c, _, _ = load_slam_cam(str(slam_path))
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
            hawor_args, start_idx, end_idx, frame_chunks_all
        )

        # 3. MANO landmarks + mesh vertices, then back to camera space
        torch.set_grad_enabled(False)
        # Frame-major (T, hand, joint, xyz): the trajectory contract, the validity
        # mask and every consumer are frame-major. An earlier revision filled a
        # (hand, T, joint, xyz) array here, which cannot reach the einsum below.
        landmarks = np.zeros((end_idx, 2, 21, 3), dtype=np.float64)
        vertices = np.zeros((end_idx, 2, 778, 3), dtype=np.float64)
        for hand, run in ((0, run_mano_left), (1, run_mano)):
            sl = slice(hand, hand + 1)
            mano = run(pred_trans[sl], pred_rot[sl], pred_hand_pose[sl], betas=pred_betas[sl])
            joints = mano["joints"] if isinstance(mano, dict) else mano
            joints = np.asarray(getattr(joints, "cpu", lambda: joints)())
            landmarks[:, hand] = np.asarray(joints, dtype=np.float64).reshape(end_idx, 21, 3)
            verts = mano.get("vertices") if isinstance(mano, dict) else None
            if verts is not None:
                verts = np.asarray(getattr(verts, "cpu", lambda: verts)())
                vertices[:, hand] = verts.reshape(end_idx, -1, 3)
    finally:
        os.chdir(previous_cwd)

    return to_camera_space(
        r_w2c,
        t_w2c,
        landmarks,
        valid=valid,
        pred_valid=pred_valid,
        confidence=confidence,
        vertices=vertices,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HaWoR runner")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--detection", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--focal", type=float, default=None, help="pixel focal length for HaWoR")
    parser.add_argument(
        "--camera-windows",
        default=None,
        help="Phase 3 camera/windows dir; its VGGT poses replace HaWoR's DROID-SLAM step. "
        "Without it the camera stays constant (identity) and the run is degraded",
    )
    parser.add_argument("--third-party", default="third_party")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="hand-model weight precision: fp16 halves 3.0 GiB to 1.5 GiB (shared 12 GB GPUs)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="hand crop size in px (upstream 256); 192 trades accuracy for ~44%% less activation memory",
    )
    add_shard_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available, detail = backend_available(Path(args.third_party))
    if args.check:
        emit(
            {
                "status": "ok" if available else "error",
                "backend": "hawor",
                "available": available,
                "detail": detail,
                "device": args.device,
            }
        )
        return 0 if available else 1

    try:
        if not args.detection or not args.out_dir or not args.frames:
            raise ValueError("--detection, --out-dir and --frames are required")
        detection = load_npz(args.detection, required=("boxes", "confidence", "valid"))
        num_frames = int(np.asarray(detection["valid"]).shape[0])
        selection = selection_from_args(args)
        from ego3d_action.hand.hawor import HaworClipRequest

        request = HaworClipRequest(
            num_frames=num_frames,
            frames_dir=Path(args.frames),
            window=args.window,
            overlap=args.overlap,
        )
        chosen = select_windows(request.ranges(), selection)
        out_dir = Path(args.out_dir)
        params = {
            "num_frames": num_frames,
            "window": args.window,
            "overlap": args.overlap,
            "focal": args.focal,
            # fp16 changes the numbers, so it is part of the unit identity.
            "precision": args.precision,
            "crop_size": args.crop_size,
        }
        if args.skip_existing and partition_is_reusable(
            out_dir, stage="hand", ranges=chosen, params=params, selection=selection
        ):
            emit(
                {
                    "status": "ok",
                    "backend": "hawor",
                    "num_frames": num_frames,
                    "windows": [f"{s:06d}_{e - 1:06d}.npz" for s, e in chosen],
                    "output_dir": str(out_dir),
                    "reused": True,
                }
            )
            return 0

        result = run_model(args)
        written = hand_windows_from_joints(
            result["joints_camera"],
            result["valid"],
            result["confidence"],
            out_dir=args.out_dir,
            window=args.window,
            overlap=args.overlap,
            selection=selection,
            vertices_camera=result.get("vertices_camera"),
        )
        record_partition(
            out_dir, stage="hand", ranges=chosen, params=params, selection=selection
        )
        emit(
            {
                "status": "ok",
                "backend": "hawor",
                "num_frames": int(result["joints_camera"].shape[0]),
                "valid_frames": int(np.count_nonzero(result["valid"])),
                "precision": args.precision,
                "crop_size": args.crop_size,
                "windows": [path.name for path in written],
                "output_dir": str(args.out_dir),
                "frame": "camera (converted from HaWoR's world with the camera trajectory)",
                "camera_source": _LAST_TRAJECTORY["source"],
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - reported through the runner protocol
        # The JSON summary stays one line, but the log keeps the full traceback:
        # without it a dtype/shape mismatch inside HaWoR is undebuggable.
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        emit({"status": "error", "backend": "hawor", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

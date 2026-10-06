#!/usr/bin/env bash
# One-shot: video/clip -> the full pipeline -> visualization/viewer.mp4.
#
#   CLIP=hot3d_ep000 bash scripts/run_viewer_pipeline.sh              # run whatever is missing
#   CLIP=hot3d_ep000 FORCE=1 bash scripts/run_viewer_pipeline.sh      # re-run every stage
#   CLIP=my_clip VIDEO=/path/to/raw.mp4 bash scripts/run_viewer_pipeline.sh   # new clip from a video
#   HAND=hawor CLIP=hot3d_ep000 FORCE=1 bash scripts/run_viewer_pipeline.sh  # HaWoR instead of WiLoR
#   WITH_WORLD=1 WITH_GT=1 ...                                        # also render the extras
#
# Stages (each skipped when its output already exists, unless FORCE=1):
#   preprocess -> detection -> hand (WiLoR|HaWoR) -> camera -> stitch
#   -> fusion -> refine -> render viewer.mp4 (+ optional world / gt_vs_pred)
#
# Notes:
# - DEVICE picks the GPU for the model backends (default cuda:0; cuda:1/2
#   returned all-zero kernels on this host on 2026-10-01 - do not use until
#   re-verified).
# - The ego panel of viewer.mp4 shows the *phase-2* hands (mesh needs MANO
#   vertices, which the refined trajectory does not carry); the world panel
#   and gt_vs_pred show the post-refine trajectory.
set -euo pipefail

CONFIG="${CONFIG:-configs/hot3d_p100.yaml}"
CLIP="${CLIP:?set CLIP=<clip-id> (an existing clip, or a new id together with VIDEO=...)}"
VIDEO="${VIDEO:-}"
HAND="${HAND:-wilor}"                 # wilor | hawor
DEVICE="${DEVICE:-cuda:0}"
FORCE="${FORCE:-0}"
WITH_WORLD="${WITH_WORLD:-0}"
WITH_GT="${WITH_GT:-1}"
DATA_ROOT="${DATA_ROOT:-data/hot3d}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The ego3d env provides ffmpeg/ffprobe (frame decoding, H.264 transcode);
# the stage orchestrators run on the base env, the model backends spawn with
# the interpreters configured under backends.python.<name>.
export PATH="/home/pjwang/miniforge3/envs/ego3d/bin:$PATH"
PY="/home/pjwang/miniforge3/envs/ego3d_base/bin/python"

CLIP_DIR="$DATA_ROOT/$CLIP"
if [ ! -d "$CLIP_DIR" ] && [ -z "$VIDEO" ]; then
    echo "clip $CLIP_DIR does not exist and no VIDEO=... was given" >&2
    exit 1
fi

stage() {  # stage <name> <output-check> <command...>
    local name="$1" check="$2"
    shift 2
    echo "==> $name"
    if [ "$FORCE" != "1" ] && [ -e "$check" ]; then
        echo "    skip ($check exists; FORCE=1 to re-run)"
        return 0
    fi
    "$@"
}

echo "clip: $CLIP | config: $CONFIG | hand: $HAND | device: $DEVICE"

# ---- phase 0: preprocess (only for a brand-new clip from a video) ----------
if [ -n "$VIDEO" ]; then
    stage preprocess "$CLIP_DIR/metadata.json" \
        "$PY" scripts/run_preprocess.py "$VIDEO" --clip "$CLIP" --config "$CONFIG"
fi

# ---- phase 1: WiLoR detection + conservative tracking -----------------------
stage detection "$CLIP_DIR/detection/detection.npz" \
    "$PY" scripts/run_detection.py --config "$CONFIG" --clip "$CLIP" --device "$DEVICE"

# ---- phase 3: VGGT-Omega camera windows (BEFORE the hand stage: the hand
# stage resolves its focal from this estimate - the prediction path never
# reads the reference calibration, the contract is "RGB in") ----------------
if compgen -G "$CLIP_DIR/camera/windows/*.npz" > /dev/null; then
    if [ "$FORCE" != "1" ]; then
        echo "==> camera: skip (camera/windows exists; FORCE=1 to re-run)"
    else
        echo "==> camera"
        "$PY" scripts/run_camera.py --config "$CONFIG" --clip "$CLIP" --device "$DEVICE"
    fi
else
    echo "==> camera"
    "$PY" scripts/run_camera.py --config "$CONFIG" --clip "$CLIP" --device "$DEVICE"
fi

# ---- phase 4: stitch camera windows before HaWoR. Each raw VGGT window has
# its own world gauge; HaWoR's infiller needs one continuous World-0 trajectory.
stage stitch "$CLIP_DIR/camera/stitched_camera.npz" "$PY" scripts/run_stitch.py --config "$CONFIG" --clip "$CLIP"

# ---- phase 2: hand reconstruction (uses the camera estimate and stitched
# trajectory above) ----------------------------------------------------------
case "$HAND" in
    wilor) HAND_SCRIPT=scripts/run_hand_wilor.py ;;
    hawor) HAND_SCRIPT=scripts/run_hand.py ;;
    *) echo "HAND must be wilor or hawor (got $HAND)" >&2; exit 1 ;;
esac
stage "hand ($HAND)" "$CLIP_DIR/hand/hand_camera.npz" "$PY" "$HAND_SCRIPT" --config "$CONFIG" --clip "$CLIP" --device "$DEVICE"

# ---- phase 5: hand + camera -> world ---------------------------------------
stage fusion "$CLIP_DIR/trajectory/trajectory_raw.npz" \
    "$PY" scripts/run_fusion.py --config "$CONFIG" --clip "$CLIP"

# ---- phase 6: gap fill, camera filter, bone scale, wrist depth, UKF+RTS ----
stage refine "$CLIP_DIR/trajectory/trajectory.npz" \
    "$PY" scripts/run_refine.py --config "$CONFIG" --clip "$CLIP"

# ---- renders ---------------------------------------------------------------
echo "==> render viewer.mp4"
"$PY" scripts/render_viewer.py --config "$CONFIG" --clip "$CLIP" --device cpu

if [ "$WITH_WORLD" = "1" ]; then
    echo "==> render world_space"
    "$PY" scripts/render_world_space.py --config "$CONFIG" --clip "$CLIP" --video
fi

if [ "$WITH_GT" = "1" ] && [ -e "$CLIP_DIR/trajectory/ground_truth.npz" ]; then
    echo "==> render gt_vs_pred (aligned)"
    "$PY" scripts/render_gt_vs_pred.py --config "$CONFIG" --clip "$CLIP" \
        --prediction "$CLIP_DIR/trajectory/trajectory.npz" \
        --ground-truth "$CLIP_DIR/trajectory/ground_truth.npz" \
        --align-gt --skeleton
fi

echo
echo "done. outputs:"
ls -la "$CLIP_DIR/visualization/viewer.mp4"
[ "$WITH_WORLD" = "1" ] && ls -la "$CLIP_DIR/visualization/world_space_time.mp4" || true
[ "$WITH_GT" = "1" ] && ls -la "$CLIP_DIR/visualization/gt_vs_pred.mp4" || true

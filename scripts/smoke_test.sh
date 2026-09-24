#!/usr/bin/env bash
# Phase 0 smoke test: synthesise a clip, decode it, render a debug video.
set -euo pipefail

ENV_NAME="${ENV_NAME:-ego3d_base}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CLIP="${CLIP:-smoke01}"
WORK="${WORK:-$(mktemp -d)}"
VIDEO="$WORK/$CLIP.mp4"

echo "synthesising $VIDEO"
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc=size=320x240:rate=30:duration=2" \
  -pix_fmt yuv420p "$VIDEO"

conda run -n "$ENV_NAME" python scripts/run_preprocess.py \
  --config configs/macrodata_final.yaml --clip "$CLIP" --data-root "$WORK/data" "$VIDEO"

echo "frames written to $WORK/data/$CLIP/frames"
find "$WORK/data/$CLIP" -maxdepth 2 -type f | head -5
echo "smoke test OK"

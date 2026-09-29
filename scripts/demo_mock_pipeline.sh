#!/usr/bin/env bash
# Full Phases 0-7 run on a synthetic clip, using the deterministic mock backend.
#
#   bash scripts/demo_mock_pipeline.sh          # 300 frames
#   FRAMES=600 bash scripts/demo_mock_pipeline.sh
#
# No GPU, no checkpoints, no network. Every stage writes into outputs/mock_demo
# and the last step prints the Action-MPJPE report for the raw and refined
# trajectories.
set -euo pipefail

ENV_NAME="${ENV_NAME:-ego3d_base}"
FRAMES="${FRAMES:-300}"
FPS="${FPS:-30}"
OUT="${OUT:-outputs/mock_demo}"
CLIP="clip01"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

run() { conda run -n "$ENV_NAME" python "$@"; }

# `ffmpeg` often lives inside the conda env rather than on the login PATH
# (and `conda run` resets PATH, so the pipeline inside cannot rely on it
# either). Resolve it once here and export it so both this script and the
# stages it launches see the same binary.
if ! command -v ffmpeg >/dev/null 2>&1; then
    for candidate in \
        "$(conda run -n "$ENV_NAME" which ffmpeg 2>/dev/null)" \
        "${CONDA_PREFIX:-}/bin/ffmpeg" \
        "$(dirname "$(command -v conda)")/../envs/$ENV_NAME/bin/ffmpeg"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            export PATH="$(dirname "$candidate"):$PATH"
            break
        fi
    done
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "error: ffmpeg not found; install it (apt install ffmpeg) or pick an env that has it" >&2
    exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"

echo "== synthesising a ${FRAMES}-frame clip (ffmpeg $(ffmpeg -version | head -1 | awk '{print $3}'))"
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc=size=320x240:rate=${FPS}:duration=$(awk "BEGIN{printf \"%.2f\", $FRAMES/$FPS}")" \
  -pix_fmt yuv420p "$OUT/${CLIP}.mp4"

echo "== phases 0-6"
started=$(date +%s.%N)
run scripts/run_pipeline.py \
  --config configs/mock.yaml \
  --clip "$CLIP" \
  --data-root "$OUT/data" \
  --video "$OUT/${CLIP}.mp4"
finished=$(date +%s.%N)
pipeline_seconds=$(awk "BEGIN{printf \"%.2f\", $finished-$started}")
echo "== pipeline wall time: ${pipeline_seconds}s"

echo "== mock ground truth"
run backends/mock_backend.py truth \
  --out "$OUT/data/${CLIP}/trajectory/truth.npz" \
  --num-frames "$FRAMES" \
  --fps "$FPS"

for variant in trajectory_raw.npz trajectory.npz; do
  echo "== evaluation: $variant"
  run scripts/evaluate_hot3d.py \
    --config configs/mock.yaml \
    --clip "$CLIP" \
    --prediction "$OUT/data/${CLIP}/trajectory/$variant" \
    --ground-truth "$OUT/data/${CLIP}/trajectory/truth.npz" \
    --pipeline-seconds "$pipeline_seconds"
done

echo
echo "artefacts under $OUT/data/${CLIP}:"
find "$OUT/data/${CLIP}" -maxdepth 2 -name '*.npz' -o -maxdepth 2 -name '*.mp4' | sort

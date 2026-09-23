#!/usr/bin/env bash
# Real-data path: import the bundled HOT3D sample and inspect it.
#
#   bash scripts/demo_hot3d_sample.sh              # import + ground-truth viewer
#   EPISODE=3 bash scripts/demo_hot3d_sample.sh
#   MANO_MODEL=weights/mano bash scripts/demo_hot3d_sample.sh   # 21-joint reference
#   WITH_MOCK=1 bash scripts/demo_hot3d_sample.sh  # also run Phases 1-6 with the mock backend
#
# The mock run is a plumbing/robustness check on real 512x512 footage: its hands
# are synthetic, so its scores are meaningless as an accuracy result.
#
# Without MANO_MODEL the reference is wrist-only. If you point MANO_MODEL at a
# converted real MANO model (see scripts/convert_mano.py) the reference becomes
# full 21-joint; if you point it at *_synthetic the same path is exercised with
# a fabricated hand, which is fine for plumbing and meaningless for metrics.
set -euo pipefail

ENV_NAME="${ENV_NAME:-ego3d_base}"
EPISODE="${EPISODE:-0}"
SAMPLE_ROOT="${SAMPLE_ROOT:-data/samples/lerobot_v3}"
DATA_ROOT="${DATA_ROOT:-data/hot3d}"
CLIP="${CLIP:-hot3d_ep$(printf '%03d' "$EPISODE")}"
WITH_MOCK="${WITH_MOCK:-0}"
MANO_MODEL="${MANO_MODEL:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

run() { conda run -n "$ENV_NAME" python "$@"; }

echo "== importing episode $EPISODE from $SAMPLE_ROOT"
mano_args=()
if [ -n "$MANO_MODEL" ]; then
  mano_args=(--mano-model "$MANO_MODEL")
  echo "   using MANO model: $MANO_MODEL"
fi
run scripts/import_lerobot.py \
  --config configs/hot3d.yaml \
  --clip "$CLIP" \
  --data-root "$DATA_ROOT" \
  --root "$SAMPLE_ROOT" \
  --episode "$EPISODE" \
  "${mano_args[@]}" \
  --overwrite

echo "== rendering the ground-truth viewer (reference against itself)"
run scripts/render_gt_vs_pred.py \
  --config configs/hot3d.yaml \
  --clip "$CLIP" \
  --data-root "$DATA_ROOT" \
  --prediction "$DATA_ROOT/$CLIP/trajectory/ground_truth.npz" \
  --ground-truth "$DATA_ROOT/$CLIP/trajectory/ground_truth.npz" \
  --stills 4

if [ "$WITH_MOCK" = "1" ]; then
  echo "== phases 1-6 with the mock backend (plumbing check on real footage)"
  run scripts/run_pipeline.py \
    --config configs/mock.yaml \
    --clip "$CLIP" \
    --data-root "$DATA_ROOT" \
    --from-stage phase1-detection

  echo "== reference versus prediction"
  run scripts/render_gt_vs_pred.py \
    --config configs/hot3d.yaml \
    --clip "$CLIP" \
    --data-root "$DATA_ROOT" \
    --prediction "$DATA_ROOT/$CLIP/trajectory/trajectory.npz" \
    --ground-truth "$DATA_ROOT/$CLIP/trajectory/ground_truth.npz" \
    --stills 4
  run scripts/evaluate_hot3d.py \
    --config configs/hot3d.yaml \
    --clip "$CLIP" \
    --prediction "$DATA_ROOT/$CLIP/trajectory/trajectory.npz" \
    --ground-truth "$DATA_ROOT/$CLIP/trajectory/ground_truth.npz"
fi

echo
echo "artefacts under $DATA_ROOT/$CLIP:"
find "$DATA_ROOT/$CLIP" -maxdepth 2 \( -name '*.npz' -o -name '*.mp4' -o -name '*.json' \) | sort

#!/usr/bin/env bash
# Real-data path: import the bundled HOT3D sample and inspect it.
#
#   bash scripts/demo_hot3d_sample.sh              # import + ground-truth viewer
#   EPISODE=3 bash scripts/demo_hot3d_sample.sh
#   MANO_MODEL=weights/mano bash scripts/demo_hot3d_sample.sh   # override the model
#   NO_MANO=1 bash scripts/demo_hot3d_sample.sh                 # wrist-only reference
#   WITH_MOCK=1 bash scripts/demo_hot3d_sample.sh  # also run Phases 1-6 with the mock backend
#
# The mock run is a plumbing/robustness check on real 512x512 footage: its hands
# are synthetic, so its scores are meaningless as an accuracy result.
#
# The 21-joint reference is the default: configs/hot3d.yaml sets
# paths.mano_model: weights/mano and that directory holds both official models.
# Point MANO_MODEL at another directory to override it, or set NO_MANO=1 for the
# wrist-only reference. Pointing it at *_synthetic exercises the same code path
# with a fabricated hand - fine for plumbing, meaningless for metrics.
set -euo pipefail

ENV_NAME="${ENV_NAME:-ego3d_base}"
EPISODE="${EPISODE:-0}"
SAMPLE_ROOT="${SAMPLE_ROOT:-data/samples/lerobot_v3}"
DATA_ROOT="${DATA_ROOT:-data/hot3d}"
CLIP="${CLIP:-hot3d_ep$(printf '%03d' "$EPISODE")}"
WITH_MOCK="${WITH_MOCK:-0}"
MANO_MODEL="${MANO_MODEL:-}"
NO_MANO="${NO_MANO:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

run() { conda run -n "$ENV_NAME" python "$@"; }

echo "== importing episode $EPISODE from $SAMPLE_ROOT"
mano_args=()
if [ "$NO_MANO" = "1" ]; then
  mano_args=(--no-mano)
  echo "   wrist-only reference (NO_MANO=1)"
elif [ -n "$MANO_MODEL" ]; then
  mano_args=(--mano-model "$MANO_MODEL")
  echo "   using MANO model: $MANO_MODEL"
else
  echo "   using paths.mano_model from configs/hot3d.yaml"
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

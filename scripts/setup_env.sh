#!/usr/bin/env bash
# Create the orchestrator environment and verify the CPU-side stack.
#
#   bash scripts/setup_env.sh
#
# The model backends (WiLoR / HaWoR / VGGT-Omega) live in their own envs and
# are created separately on the GPU server - see environment-*.yml.
set -euo pipefail

ENV_NAME="${ENV_NAME:-ego3d_base}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env '$ENV_NAME' already exists - skipping creation"
else
  conda env create -f environment-base.yml
fi

echo "installing the package in editable mode"
conda run -n "$ENV_NAME" python -m pip install --no-build-isolation -e . >/dev/null

echo "verifying the environment"
conda run -n "$ENV_NAME" python - <<'PY'
import platform, sys
import numpy, scipy, cv2, yaml, pytest
import ego3d_action
from ego3d_action.runtime.device import describe_environment

print(f"python      : {platform.python_version()}")
print(f"numpy/scipy : {numpy.__version__} / {scipy.__version__}")
print(f"opencv      : {cv2.__version__}")
print(f"package     : ego3d_action {ego3d_action.__version__}")
print(f"environment : {describe_environment('auto').format()}")
if describe_environment("auto").resolved_device == "cpu":
    print("note: CPU-only machine - Phase 0/1-tracking/4/5/6 logic and the whole test suite run here;")
    print("      the model backends (Phases 1-3) need the GPU server.")
PY

echo "done. activate with: conda activate $ENV_NAME"

#!/usr/bin/env python
"""Write a **synthetic stand-in** for the MANO model (for plumbing only).

    python scripts/make_synthetic_mano.py --output-dir weights/mano_synthetic

The real MANO model is licence-gated and is not bundled. This script fabricates
a model with the same structure (``v_template``, ``shapedirs``, ``j_regressor``,
``weights``, ``posedirs``, ``f``) so the forward-kinematics path can be exercised
end to end without it.

The hands it produces are **not real**: the pose comes from the data, but the
shape and bone lengths come from this fixture. Never report a metric computed
with it - the output directory is deliberately named ``*_synthetic`` and every
converted reference records the model path in its metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.testing.synthetic import write_synthetic_mano_npz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="write a synthetic MANO stand-in")
    parser.add_argument("--output-dir", default="weights/mano_synthetic")
    parser.add_argument("--shape-scale", type=float, default=0.0)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    written = write_synthetic_mano_npz(
        output_dir / "MANO_RIGHT.npz", shape_scale=args.shape_scale
    )
    print(f"wrote {written}")
    print(
        "WARNING: this is a synthetic stand-in, not the licence-gated MANO model. "
        "It exists so the forward-kinematics path can be exercised; any metric computed "
        "from a reference built with it is meaningless."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

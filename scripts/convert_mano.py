#!/usr/bin/env python
"""Convert the official MANO pickles to ``.npz`` so this project can read them.

    # run this in an environment that has chumpy (e.g. ego3d_hawor)
    python scripts/convert_mano.py \
        --input /path/to/models/MANO_RIGHT.pkl /path/to/models/MANO_LEFT.pkl \
        --output-dir weights/mano

The official ``MANO_RIGHT.pkl`` / ``MANO_LEFT.pkl`` are chumpy objects; the
``.npz`` written here contains plain arrays (``v_template``, ``shapedirs``,
``j_regressor``, ``weights``, ``posedirs``, ``f``) and needs no chumpy at
runtime. MANO is licence-gated: download it from the official site and respect
its terms - this script only reformats what you already have.

**chumpy is not needed.** The official archive wraps only ``shapedirs`` in a
chumpy ``Select``; ``ego3d_action.hand.mano_model.read_mano_pickle`` unpickles it
with a tiny stand-in class and materialises the array, so this runs in the
normal base environment:

    conda run -n ego3d_base python scripts/convert_mano.py \\
        --input weights/mano/MANO_RIGHT.pkl --output-dir weights/mano
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.hand.mano_model import REQUIRED_KEYS, read_mano_pickle  # noqa: E402

ARRAY_KEYS = (*REQUIRED_KEYS, "posedirs", "f")


def convert(source: Path, destination: Path) -> dict[str, tuple[int, ...]]:
    """Read the official pickle (no chumpy) and write an ``.npz`` next to it."""
    if not source.is_file():
        raise SystemExit(f"input model not found: {source}")
    raw = read_mano_pickle(source)
    arrays = {key: np.asarray(raw[key]) for key in ARRAY_KEYS if key in raw}
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise SystemExit(f"{source} is missing {missing}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **arrays)
    return {key: np.asarray(value).shape for key, value in arrays.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="convert MANO pickles to npz")
    # append + "+" so that both `--input a.pkl b.pkl` and `--input a.pkl --input b.pkl`
    # work; plain nargs="+" silently keeps only the last occurrence.
    parser.add_argument(
        "--input",
        nargs="+",
        action="append",
        required=True,
        help="MANO_*.pkl files (may be repeated)",
    )
    parser.add_argument("--output-dir", default="weights/mano", help="where to write the npz")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    sources = [item for group in args.input for item in group]
    for item in sources:
        source = Path(item)
        destination = output_dir / f"{source.stem}.npz"
        shapes = convert(source, destination)
        print(f"{source} -> {destination}")
        for key, shape in shapes.items():
            print(f"    {key:12s} {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.hand.mano_model import REQUIRED_KEYS  # noqa: E402

ARRAY_KEYS = (*REQUIRED_KEYS, "posedirs", "f")


def convert(source: Path, destination: Path) -> dict[str, tuple[int, ...]]:
    """Read the pickle and write an ``.npz`` next to it."""
    try:
        import chumpy  # noqa: F401 - required to unpickle the official model
    except ImportError as exc:
        raise SystemExit(
            "chumpy is required to unpickle the official MANO model; run this script in an "
            "environment that has it (e.g. conda run -n ego3d_hawor python scripts/convert_mano.py)"
        ) from exc

    if not source.is_file():
        raise SystemExit(f"input model not found: {source}")
    with source.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")

    arrays: dict[str, np.ndarray] = {}
    for key in ARRAY_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        arrays[key] = np.asarray(getattr(value, "r", value))
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise SystemExit(f"{source} is missing {missing}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **arrays)
    return {key: np.asarray(value).shape for key, value in arrays.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="convert MANO pickles to npz")
    parser.add_argument("--input", nargs="+", required=True, help="MANO_*.pkl files")
    parser.add_argument("--output-dir", default="weights/mano", help="where to write the npz")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    for item in args.input:
        source = Path(item)
        destination = output_dir / f"{source.stem}.npz"
        shapes = convert(source, destination)
        print(f"{source} -> {destination}")
        for key, shape in shapes.items():
            print(f"    {key:12s} {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

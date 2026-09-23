#!/usr/bin/env python
"""Audit the environment and the assets this project needs.

    python scripts/doctor.py --config configs/macrodata_final.yaml
    python scripts/doctor.py --config configs/hot3d.yaml --clip hot3d_ep000 --runners
    python scripts/doctor.py --strict --json

Exit codes: ``0`` when the CPU path is ready (tests, mock pipeline, reference
import, evaluation); ``1`` when something CPU-required is missing, or when
``--strict`` is given and anything at all is missing or warned about.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.runtime.doctor import run_all, summarise  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="environment and asset audit")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--clip", default=None, help="also audit one clip's artefacts")
    parser.add_argument(
        "--runners", action="store_true", help="execute each backend runner's --check"
    )
    parser.add_argument("--strict", action="store_true", help="fail on warnings too")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    checks = run_all(args.config, clip=args.clip, with_runners=args.runners)
    text, exit_code = summarise(checks, strict=args.strict)
    if args.json:
        print(
            json.dumps(
                {
                    "checks": [
                        {
                            "name": c.name,
                            "section": c.section,
                            "status": c.status,
                            "detail": c.detail,
                            "fix": c.fix,
                            "required_for": c.required_for,
                        }
                        for c in checks
                    ],
                    "exit_code": exit_code,
                },
                indent=2,
            )
        )
    else:
        print(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

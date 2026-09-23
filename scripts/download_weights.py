#!/usr/bin/env python
"""Download the model weights this project needs into one folder.

    python scripts/download_weights.py --dry-run          # show the plan
    python scripts/download_weights.py                    # fetch everything fetchable
    python scripts/download_weights.py --only wilor,hawor # subset (id, alias or backend)
    python scripts/download_weights.py --verify-only      # audit, download nothing
    python scripts/download_weights.py --dest-root /mnt/weights --force

Everything is driven by ``weights.manifest.yaml`` (edit it, or override a single
entry with ``--url-override <id>=<url>``). Downloads are resumable, verified
(size, optional sha256, format sniffing) and atomic; licence-gated assets such
as MANO are never fetched - the script prints exactly how to get them.

Exit codes: 0 when everything not marked manual/optional is present and valid,
1 otherwise (so it can gate a CI job or a server bootstrap).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego3d_action.cli import configure_logging  # noqa: E402
from ego3d_action.errors import Ego3DActionError  # noqa: E402
from ego3d_action.runtime.weights import (  # noqa: E402
    audit,
    download_all,
    filter_assets,
    load_manifest,
    manual_steps,
)

DEFAULT_MANIFEST = Path("weights.manifest.yaml")


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def print_plan(assets: list, dest_root: Path) -> None:
    print(f"manifest : {len(assets)} asset(s)")
    print(f"dest root: {dest_root}")
    for asset in assets:
        tags = [asset.backend]
        if asset.manual:
            tags.append("manual")
        if asset.optional:
            tags.append("optional")
        target = dest_root / asset.dest
        state = "present" if target.is_file() else "-"
        print(f"  {asset.id:28s} [{', '.join(tags)}] {state:8s} {target}")
        print(f"      source: {asset.primary_source or '(none - manual)'}")
        if asset.note:
            print(f"      note  : {asset.note}")


def print_manual(assets: list, dest_root: Path) -> None:
    pending = manual_steps(assets, dest_root)
    if not pending:
        return
    print("\nmanual steps still required:")
    for asset, target in pending:
        print(f"  - {asset.id}: {asset.note or 'download by hand'}")
        if asset.page:
            print(f"      page     : {asset.page}")
        print(f"      save as  : {target}")
        if asset.post_step:
            for line in asset.post_step.strip().splitlines():
                print(f"      then     : {line.strip()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="download the model weights")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dest-root", default="weights", help="where the weights tree lives")
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated ids/aliases/backends (e.g. wilor,hawor,mano)",
    )
    parser.add_argument("--url-override", action="append", default=[], metavar="ID=URL")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--verify-only", action="store_true", help="audit what is on disk")
    parser.add_argument("--force", action="store_true", help="re-download even if valid")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    try:
        configure_logging(args.log_level)
        manifest_path = Path(args.manifest)
        dest_root = Path(args.dest_root)
        assets = load_manifest(manifest_path)
        selectors = [item for item in args.only.split(",") if item.strip()]
        assets = filter_assets(assets, selectors)

        overrides: dict[str, list[str]] = {}
        for pair in args.url_override:
            if "=" not in pair:
                raise Ego3DActionError(f"--url-override expects ID=URL, got '{pair}'")
            key, url = pair.split("=", 1)
            overrides.setdefault(key.strip(), []).append(url.strip())
        unknown = set(overrides) - {asset.id for asset in assets}
        if unknown:
            raise Ego3DActionError(
                f"--url-override for unknown asset(s) {sorted(unknown)}; known: "
                f"{sorted(asset.id for asset in assets)}"
            )

        if args.dry_run:
            print_plan(assets, dest_root)
            print_manual(assets, dest_root)
            return 0

        if args.verify_only:
            rows = audit(assets, dest_root)
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                print(f"{'id':28s} {'status':8s} detail")
                for row in rows:
                    print(f"{row['id']:28s} {row['status']:8s} {row['detail']}")
                print_manual(assets, dest_root)
            blocking = [
                row
                for row, asset in zip(rows, assets, strict=False)
                if row["status"] != "ok" and not asset.optional
            ]
            if blocking:
                print(f"\n{len(blocking)} asset(s) not usable yet", file=sys.stderr)
            return 1 if blocking else 0

        run = download_all(
            assets,
            dest_root,
            overrides=overrides,
            force=args.force,
            timeout=args.timeout,
            retries=args.retries,
        )
        if args.json:
            print(json.dumps(list(run.results), indent=2))
        else:
            for row in run.results:
                print(f"[{str(row['status']):7s}] {row['id']:28s} {row['detail']}")
            print(f"\n{len(run.ok)} ok, {len(run.unresolved)} still missing/errored")
            print_manual(assets, dest_root)

        blocking = [
            row
            for row in run.unresolved
            if not next(asset for asset in assets if asset.id == row["id"]).optional
        ]
        if blocking:
            print(
                "\nsome weights are still missing. Re-run after fixing the URLs "
                "(--url-override) or after the manual steps above.",
                file=sys.stderr,
            )
        return 1 if blocking else 0
    except Ego3DActionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

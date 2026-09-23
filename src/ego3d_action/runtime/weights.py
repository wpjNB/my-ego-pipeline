"""Manifest-driven downloader/verifier for the model weights.

The project deliberately ships no checkpoints (see ``doc_auto/setup.md``), so
there has to be one place that knows *what* to fetch, *where* it goes and *how
to tell whether the file is usable*. That is this module plus
``weights.manifest.yaml``.

Sources
-------

* ``https://`` / ``http://`` - plain download, resumable via ``Range`` when the
  server supports it.
* ``hf://<repo>/<path>`` - Hugging Face hub; uses ``huggingface_hub`` when
  installed and says exactly what to install when it is not.
* ``file://`` (or a bare path) - a local mirror, for air-gapped servers and for
  the test suite.

Integrity
---------

Every download is written to ``<dest>.part``, checked (size, optional sha256,
format sniffing) and only then moved into place, so a truncated file can never
look like a usable checkpoint. Format sniffing works without torch: a modern
``torch.save`` file is a zip, a legacy one is a pickle, ``safetensors`` starts
with a JSON header and ``npz`` is a zip containing ``.npy`` members.

Assets marked ``auth: manual`` (MANO is licence-gated) are never fetched - the
report prints the page, the exact filename, the destination and the post-step.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import yaml

from ..errors import Ego3DActionError, StageIOError

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
VALID_FORMATS = ("torch_zip", "pickle", "safetensors", "npz", "zip", "any")
VALID_AUTH = ("none", "manual")


class DownloadError(Ego3DActionError):
    """A weight could not be fetched or failed verification."""


@dataclass(frozen=True)
class WeightAsset:
    """One entry of the manifest."""

    id: str
    dest: str
    sources: tuple[str, ...]
    backend: str = "general"
    sha256: str | None = None
    size_bytes: int | None = None
    min_bytes: int = 1024
    format: str = "any"
    auth: str = "none"
    optional: bool = False
    note: str = ""
    page: str = ""
    post_step: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def manual(self) -> bool:
        return self.auth == "manual"

    @property
    def primary_source(self) -> str:
        return self.sources[0] if self.sources else ""


def load_manifest(path: str | Path) -> list[WeightAsset]:
    """Read ``weights.manifest.yaml``.

    Raises:
        StageIOError: missing file, bad YAML, or an entry with an unknown
            ``format``/``auth`` or no source.
    """
    source = Path(path)
    if not source.is_file():
        raise StageIOError(f"weight manifest not found: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StageIOError(f"{source} is not valid YAML: {exc}") from exc
    entries = (raw or {}).get("assets")
    if not isinstance(entries, list) or not entries:
        raise StageIOError(f"{source} must contain a non-empty 'assets' list")

    assets: list[WeightAsset] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise StageIOError(f"manifest entry must be a mapping, got {entry!r}")
        asset_id = str(entry.get("id", "")).strip()
        dest = str(entry.get("dest", "")).strip()
        if not asset_id or not dest:
            raise StageIOError(f"manifest entry needs 'id' and 'dest': {entry!r}")
        sources = entry.get("sources") or entry.get("url") or []
        if isinstance(sources, str):
            sources = [sources]
        sources = tuple(str(item) for item in sources if item)
        fmt = str(entry.get("format", "any"))
        auth = str(entry.get("auth", "none"))
        if fmt not in VALID_FORMATS:
            raise StageIOError(f"{asset_id}: unknown format '{fmt}' (expected {VALID_FORMATS})")
        if auth not in VALID_AUTH:
            raise StageIOError(f"{asset_id}: unknown auth '{auth}' (expected {VALID_AUTH})")
        # A source that is still a "<...>" placeholder is a deliberate "we could
        # not pin this URL": treat it as a manual step rather than fetching it.
        if sources and all("<" in item and ">" in item for item in sources):
            auth = "manual"
        if not sources and auth != "manual":
            raise StageIOError(f"{asset_id}: no source given")
        assets.append(
            WeightAsset(
                id=asset_id,
                dest=dest,
                sources=sources,
                backend=str(entry.get("backend", "general")),
                sha256=(str(entry["sha256"]).lower() if entry.get("sha256") else None),
                size_bytes=(
                    int(entry["size_bytes"]) if entry.get("size_bytes") is not None else None
                ),
                min_bytes=int(entry.get("min_bytes", 1024)),
                format=fmt,
                auth=auth,
                optional=bool(entry.get("optional", False)),
                note=str(entry.get("note", "")),
                page=str(entry.get("page", "")),
                post_step=str(entry.get("post_step", "")),
                aliases=tuple(str(item) for item in (entry.get("aliases") or [])),
            )
        )
    return assets


def sha256_of(path: str | Path, *, chunk: int = CHUNK) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sniff_format(path: str | Path) -> str:
    """Best-effort format detection by content (no torch required)."""
    target = Path(path)
    with target.open("rb") as handle:
        head = handle.read(64)
        if head.startswith(b"PK\x03\x04"):
            return "npz" if b".npy" in handle.read(8192) else "torch_zip"
    if head.startswith(b"\x80"):
        return "pickle"
    if len(head) >= 8:
        (length,) = struct.unpack("<Q", head[:8])
        if 0 < length <= 100 * 1024 * 1024:
            with target.open("rb") as handle:
                header = handle.read(8 + length)
            if len(header) == 8 + length:
                try:
                    parsed = json.loads(header[8:].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
                if isinstance(parsed, dict):
                    return "safetensors"
    return "unknown"


def verify_file(path: str | Path, asset: WeightAsset) -> tuple[bool, str]:
    """Size + checksum + format check; returns ``(ok, detail)``."""
    target = Path(path)
    if not target.is_file():
        return False, "not downloaded"
    size = target.stat().st_size
    if size < asset.min_bytes:
        return False, f"too small: {size} bytes < {asset.min_bytes}"
    if asset.size_bytes is not None and size != asset.size_bytes:
        return False, f"size mismatch: {size} != {asset.size_bytes}"
    if asset.sha256:
        actual = sha256_of(target)
        if actual != asset.sha256:
            return False, f"sha256 mismatch: {actual} != {asset.sha256}"
    if asset.format != "any":
        detected = sniff_format(target)
        if detected != asset.format:
            return False, f"format mismatch: looks like '{detected}', expected '{asset.format}'"
    return True, f"{size / (1 << 20):.1f} MiB"


def _open_source(url: str, *, timeout: float, offset: int = 0) -> tuple[object, int, int]:
    """Open ``url`` and return ``(handle, offset, total_size)``.

    Raises:
        DownloadError: unknown scheme, network failure or an unusable response.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "hf":
        handle = _resolve_hf(url).open("rb")
        total = Path(_resolve_hf(url)).stat().st_size
        if offset:
            handle.seek(min(offset, total))
        return handle, offset, total
    if parsed.scheme in {"", "file"}:
        path = Path(urllib.request.url2pathname(parsed.path)) if parsed.scheme else Path(url)
        if not path.is_file():
            raise DownloadError(f"local mirror file not found: {path}")
        handle = path.open("rb")
        total = path.stat().st_size
        if offset:
            handle.seek(min(offset, total))
        return handle, offset, total
    if parsed.scheme in {"http", "https"}:
        request = urllib.request.Request(url, headers={"User-Agent": "ego3d-action/0.1"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            response = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - http(s) only
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:
                return b"", offset, offset  # nothing left to fetch
            raise DownloadError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DownloadError(
                f"cannot reach {url} ({type(exc).__name__}: {exc}). If this machine has no "
                "internet access, download the file elsewhere and use a file:// mirror."
            ) from exc
        resumed = offset > 0 and getattr(response, "status", 200) == 206
        if offset and not resumed:
            offset = 0  # server ignored Range: start over
        length = response.headers.get("Content-Length")
        total = (offset + int(length)) if length else 0
        return response, offset, total
    raise DownloadError(f"unsupported source scheme '{parsed.scheme}' in {url}")


def _resolve_hf(url: str) -> Path:
    """Resolve ``hf://<repo>/<path>`` to a local file via ``huggingface_hub``.

    Raises:
        DownloadError: the package is missing or the download failed.
    """
    parsed = urllib.parse.urlparse(url)
    repo = parsed.netloc
    filename = parsed.path.lstrip("/")
    if not repo or not filename:
        raise DownloadError(f"hf:// sources look like hf://<repo>/<path>, got {url}")
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise DownloadError(
            "hf:// sources need huggingface_hub: pip install huggingface_hub (or download the "
            "file manually and point the manifest at a file:// mirror)"
        ) from exc
    try:
        return Path(hf_hub_download(repo_id=repo, filename=filename))
    except Exception as exc:  # noqa: BLE001 - the library raises several types
        raise DownloadError(f"huggingface_hub could not fetch {repo}/{filename}: {exc}") from exc


def fetch_asset(
    asset: WeightAsset,
    dest_root: str | Path,
    *,
    sources: Sequence[str] | None = None,
    force: bool = False,
    timeout: float = 60.0,
    retries: int = 3,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Download one asset into ``dest_root/<asset.dest>``.

    Returns a result dict (``status`` in ``ok`` / ``missing`` / ``error``) for
    expected conditions so a batch run can report every asset at once. Genuine
    failures are still surfaced as ``error`` with the last attempts attached.
    """
    target = Path(dest_root) / asset.dest
    if asset.manual and not sources:
        # An explicit source (``--url-override``) can rescue a placeholder entry,
        # which is exactly how a moved URL gets fixed without editing the manifest.
        return {
            "id": asset.id,
            "status": "missing",
            "detail": "manual download required (licence-gated)",
            "path": str(target),
        }
    if target.is_file() and not force:
        ok, detail = verify_file(target, asset)
        if ok:
            return {
                "id": asset.id,
                "status": "ok",
                "detail": f"already present ({detail})",
                "path": str(target),
            }
        logger.warning("%s exists but failed verification (%s); re-downloading", asset.id, detail)

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    candidate_sources = list(sources or asset.sources)
    errors: list[str] = []
    for source in candidate_sources:
        for attempt in range(1, max(1, retries) + 1):
            try:
                if force and part.is_file():
                    part.unlink()
                offset = part.stat().st_size if part.is_file() else 0
                handle, offset, total = _open_source(source, timeout=timeout, offset=offset)
                written = offset
                with part.open("ab" if offset else "wb") as out:
                    reader = getattr(handle, "read", None)
                    if reader is None:
                        raise DownloadError(f"{source} produced no readable stream")
                    while True:
                        block = reader(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        written += len(block)
                        if on_progress is not None:
                            on_progress(written, total)
                close = getattr(handle, "close", None)
                if close is not None:
                    close()
                if total and written != total:
                    raise DownloadError(f"truncated download: {written} of {total} bytes")
                ok, detail = verify_file(part, asset)
                if not ok:
                    bad = part.with_name(part.name + ".bad")
                    os.replace(part, bad)
                    raise DownloadError(f"{detail} (kept for inspection at {bad})")
                os.replace(part, target)
                return {
                    "id": asset.id,
                    "status": "ok",
                    "detail": f"downloaded from {source} ({detail})",
                    "path": str(target),
                }
            except DownloadError as exc:
                errors.append(f"{source} [attempt {attempt}]: {exc}")
                if attempt < max(1, retries):
                    time.sleep(min(2.0 * attempt, 5.0))
            except OSError as exc:
                errors.append(f"{source} [attempt {attempt}]: {type(exc).__name__}: {exc}")
    return {
        "id": asset.id,
        "status": "error",
        "detail": " | ".join(errors[-3:]) or "no source configured",
        "path": str(target),
    }


def filter_assets(assets: Sequence[WeightAsset], selectors: Sequence[str]) -> list[WeightAsset]:
    """Select assets by id, alias or backend (``--only wilor,hawor``)."""
    if not selectors:
        return list(assets)
    wanted = {item.strip().lower() for item in selectors if item.strip()}
    chosen = [
        asset
        for asset in assets
        if asset.id.lower() in wanted
        or asset.backend.lower() in wanted
        or bool(wanted & {alias.lower() for alias in asset.aliases})
    ]
    known = {asset.id.lower() for asset in assets}
    known |= {asset.backend.lower() for asset in assets}
    known |= {alias.lower() for asset in assets for alias in asset.aliases}
    unknown = wanted - known
    if unknown:
        raise StageIOError(
            f"unknown selector(s) {sorted(unknown)}; known ids: "
            f"{sorted(asset.id for asset in assets)}"
        )
    return chosen


def audit(assets: Sequence[WeightAsset], dest_root: str | Path) -> list[dict[str, object]]:
    """Check what is already on disk without downloading anything."""
    rows: list[dict[str, object]] = []
    for asset in assets:
        target = Path(dest_root) / asset.dest
        if asset.manual:
            rows.append(
                {
                    "id": asset.id,
                    "status": "manual",
                    "detail": "licence-gated: download by hand",
                    "path": str(target),
                }
            )
            continue
        ok, detail = verify_file(target, asset)
        rows.append(
            {
                "id": asset.id,
                "status": "ok" if ok else "missing",
                "detail": detail,
                "path": str(target),
            }
        )
    return rows


@dataclass(frozen=True)
class WeightRun:
    """Aggregated result of a batch run."""

    results: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    @property
    def ok(self) -> list[Mapping[str, object]]:
        return [row for row in self.results if row.get("status") == "ok"]

    @property
    def failed(self) -> list[Mapping[str, object]]:
        return [row for row in self.results if row.get("status") == "error"]

    @property
    def unresolved(self) -> list[Mapping[str, object]]:
        return [row for row in self.results if row.get("status") in {"missing", "error"}]


def download_all(
    assets: Sequence[WeightAsset],
    dest_root: str | Path,
    *,
    overrides: Mapping[str, Sequence[str]] | None = None,
    force: bool = False,
    timeout: float = 60.0,
    retries: int = 3,
) -> WeightRun:
    """Fetch every asset, honouring ``--url-override`` style replacements."""
    overrides = overrides or {}
    results: list[Mapping[str, object]] = []
    for asset in assets:
        result = fetch_asset(
            asset,
            dest_root,
            sources=overrides.get(asset.id),
            force=force,
            timeout=timeout,
            retries=retries,
        )
        results.append(result)
        logger.info("[%s] %s - %s", result["status"], asset.id, result["detail"])
    return WeightRun(results=tuple(results))


def manual_steps(
    assets: Sequence[WeightAsset], dest_root: str | Path
) -> list[tuple[WeightAsset, Path]]:
    """Assets a human still has to fetch by hand."""
    pending: list[tuple[WeightAsset, Path]] = []
    for asset in assets:
        target = Path(dest_root) / asset.dest
        if asset.manual or not target.is_file():
            pending.append((asset, target))
    return pending

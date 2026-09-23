"""The manifest-driven weight downloader (tested through file:// mirrors)."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from ego3d_action.errors import StageIOError
from ego3d_action.runtime.weights import (
    DownloadError,
    WeightAsset,
    audit,
    download_all,
    fetch_asset,
    filter_assets,
    load_manifest,
    manual_steps,
    sha256_of,
    sniff_format,
    verify_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_torch_zip(path: Path, *, payload: bytes = b"x" * 4096) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", "3")
    return path


def make_safetensors(path: Path, *, payload: bytes = b"y" * 4096) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"__metadata__": {}, "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    encoded = header.encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)
    return path


def make_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, array=np.zeros(4))
    return path


def make_pickle(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00" + b"z" * 2048)
    return path


def write_manifest(path: Path, assets: list[dict[str, object]]) -> Path:
    path.write_text(yaml.safe_dump({"assets": assets}), encoding="utf-8")
    return path


# ---------------------------------------------------------------- manifest


def test_shipped_manifest_is_valid() -> None:
    assets = load_manifest(REPO_ROOT / "weights.manifest.yaml")
    ids = [asset.id for asset in assets]
    assert "wilor_checkpoint" in ids
    assert "hawor_checkpoint" in ids
    assert "mano_right" in ids
    mano = next(asset for asset in assets if asset.id == "mano_right")
    assert mano.manual  # licence-gated: never fetched by the script
    vggt = next(asset for asset in assets if asset.id == "vggt_omega_reproduction")
    assert vggt.manual  # its URL is a <placeholder>, so it is a manual step
    assert vggt.page and vggt.post_step
    # Every non-manual entry must have a source and a sanity floor.
    for asset in assets:
        if not asset.manual:
            assert asset.sources and asset.min_bytes > 0


def test_manifest_validation(tmp_path: Path) -> None:
    with pytest.raises(StageIOError, match="not found"):
        load_manifest(tmp_path / "nope.yaml")

    broken = tmp_path / "broken.yaml"
    broken.write_text("assets: []\n", encoding="utf-8")
    with pytest.raises(StageIOError, match="non-empty"):
        load_manifest(broken)

    bad_format = write_manifest(
        tmp_path / "bad-format.yaml",
        [{"id": "a", "dest": "a.bin", "sources": ["file:///tmp/x"], "format": "exe"}],
    )
    with pytest.raises(StageIOError, match="unknown format"):
        load_manifest(bad_format)

    no_source = write_manifest(tmp_path / "no-source.yaml", [{"id": "a", "dest": "a.bin"}])
    with pytest.raises(StageIOError, match="no source"):
        load_manifest(no_source)


def test_filter_assets_by_id_alias_and_backend() -> None:
    assets = [
        WeightAsset(id="w1", dest="w1", sources=("file:///a",), backend="wilor"),
        WeightAsset(id="h1", dest="h1", sources=("file:///b",), backend="hawor", aliases=("hands",)),
    ]
    assert [a.id for a in filter_assets(assets, ["wilor"])] == ["w1"]
    assert [a.id for a in filter_assets(assets, ["hands"])] == ["h1"]
    assert [a.id for a in filter_assets(assets, ["w1", "h1"])] == ["w1", "h1"]
    assert len(filter_assets(assets, [])) == 2
    with pytest.raises(StageIOError, match="unknown selector"):
        filter_assets(assets, ["nope"])


# ------------------------------------------------------------- verification


def test_sniff_format_recognises_the_real_container_types(tmp_path: Path) -> None:
    assert sniff_format(make_torch_zip(tmp_path / "m.ckpt")) == "torch_zip"
    assert sniff_format(make_npz(tmp_path / "m.npz")) == "npz"
    assert sniff_format(make_pickle(tmp_path / "m.pth")) == "pickle"
    assert sniff_format(make_safetensors(tmp_path / "m.safetensors")) == "safetensors"
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not a model at all")
    assert sniff_format(junk) == "unknown"


def test_verify_file_checks_size_checksum_and_format(tmp_path: Path) -> None:
    model = make_torch_zip(tmp_path / "m.ckpt")
    good = WeightAsset(
        id="m", dest="m.ckpt", sources=("file:///x",), format="torch_zip", min_bytes=100
    )
    assert verify_file(model, good)[0]

    too_small = WeightAsset(id="m", dest="m.ckpt", sources=("file:///x",), min_bytes=10**9)
    assert not verify_file(model, too_small)[0]
    assert "too small" in verify_file(model, too_small)[1]

    wrong_format = WeightAsset(
        id="m", dest="m.ckpt", sources=("file:///x",), format="safetensors", min_bytes=10
    )
    assert "format mismatch" in verify_file(model, wrong_format)[1]

    wrong_sha = WeightAsset(
        id="m", dest="m.ckpt", sources=("file:///x",), min_bytes=10, sha256="0" * 64
    )
    assert "sha256 mismatch" in verify_file(model, wrong_sha)[1]

    right_sha = WeightAsset(
        id="m", dest="m.ckpt", sources=("file:///x",), min_bytes=10, sha256=sha256_of(model)
    )
    assert verify_file(model, right_sha)[0]
    assert not verify_file(tmp_path / "gone.ckpt", good)[0]


# --------------------------------------------------------------- fetching


def test_fetch_asset_from_a_file_mirror(tmp_path: Path) -> None:
    source = make_torch_zip(tmp_path / "mirror" / "wilor_final.ckpt")
    asset = WeightAsset(
        id="wilor_checkpoint",
        dest="wilor/wilor_final.ckpt",
        sources=(source.as_uri(),),
        format="torch_zip",
        min_bytes=100,
    )
    result = fetch_asset(asset, tmp_path / "weights", retries=1)
    assert result["status"] == "ok", result
    assert (tmp_path / "weights" / "wilor/wilor_final.ckpt").is_file()
    # A second run is a no-op that re-verifies.
    again = fetch_asset(asset, tmp_path / "weights", retries=1)
    assert "already present" in str(again["detail"])


def test_fetch_asset_resumes_a_partial_download(tmp_path: Path) -> None:
    source = make_torch_zip(tmp_path / "mirror" / "hawor.ckpt", payload=b"z" * 8192)
    payload = source.read_bytes()
    target = tmp_path / "weights" / "hawor" / "hawor.ckpt"
    target.parent.mkdir(parents=True)
    part = target.with_name("hawor.ckpt.part")
    part.write_bytes(payload[: len(payload) // 2])  # an interrupted download

    asset = WeightAsset(
        id="hawor_checkpoint",
        dest="hawor/hawor.ckpt",
        sources=(source.as_uri(),),
        format="torch_zip",
        min_bytes=100,
    )
    result = fetch_asset(asset, tmp_path / "weights", retries=1)
    assert result["status"] == "ok", result
    assert target.read_bytes() == payload
    assert not part.exists()


def test_fetch_asset_quarantines_a_corrupt_download(tmp_path: Path) -> None:
    source = make_torch_zip(tmp_path / "mirror" / "broken.ckpt")
    asset = WeightAsset(
        id="broken",
        dest="broken.ckpt",
        sources=(source.as_uri(),),
        format="torch_zip",
        min_bytes=100,
        sha256="1" * 64,  # deliberately wrong
    )
    result = fetch_asset(asset, tmp_path / "weights", retries=1)
    assert result["status"] == "error"
    assert "sha256 mismatch" in str(result["detail"])
    bad = tmp_path / "weights" / "broken.ckpt.part.bad"
    assert bad.is_file()  # kept for inspection, never installed
    assert not (tmp_path / "weights" / "broken.ckpt").exists()


def test_fetch_asset_falls_back_to_the_next_source(tmp_path: Path) -> None:
    good = make_torch_zip(tmp_path / "mirror" / "good.ckpt")
    asset = WeightAsset(
        id="multi",
        dest="multi.ckpt",
        sources=((tmp_path / "mirror" / "missing.ckpt").as_uri(), good.as_uri()),
        format="torch_zip",
        min_bytes=100,
    )
    result = fetch_asset(asset, tmp_path / "weights", retries=1)
    assert result["status"] == "ok"
    # The good mirror is the one that won.
    assert good.name in str(result["detail"])


def test_fetch_asset_reports_a_missing_mirror(tmp_path: Path) -> None:
    asset = WeightAsset(
        id="gone",
        dest="gone.ckpt",
        sources=((tmp_path / "mirror" / "nope.ckpt").as_uri(),),
        format="torch_zip",
        min_bytes=10,
    )
    result = fetch_asset(asset, tmp_path / "weights", retries=1)
    assert result["status"] == "error"
    assert "not found" in str(result["detail"])


def test_fetch_asset_reports_an_unreachable_http_source(tmp_path: Path) -> None:
    """No network here: the failure must be a clear, actionable message."""
    asset = WeightAsset(
        id="remote",
        dest="remote.ckpt",
        sources=("http://127.0.0.1:1/none.ckpt",),
        format="torch_zip",
        min_bytes=10,
    )
    result = fetch_asset(asset, tmp_path / "weights", retries=1, timeout=0.5)
    assert result["status"] == "error"
    detail = str(result["detail"])
    assert "cannot reach" in detail or "HTTP" in detail


def test_fetch_asset_rejects_an_unknown_scheme(tmp_path: Path) -> None:
    asset = WeightAsset(
        id="weird", dest="weird.bin", sources=("gopher://example.com/x",), min_bytes=1
    )
    result = fetch_asset(asset, tmp_path / "weights", retries=1)
    assert result["status"] == "error"
    assert "unsupported source scheme" in str(result["detail"])


def test_hf_source_without_huggingface_hub_is_reported(tmp_path: Path) -> None:
    pytest.importorskip  # keep the import-time semantics explicit
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        asset = WeightAsset(
            id="hf", dest="hf.bin", sources=("hf://org/repo/model.safetensors",), min_bytes=1
        )
        result = fetch_asset(asset, tmp_path / "weights", retries=1)
        assert result["status"] == "error"
        assert "huggingface_hub" in str(result["detail"])


def test_download_all_with_a_url_override(tmp_path: Path) -> None:
    mirror = make_torch_zip(tmp_path / "mirror" / "real.ckpt")
    manifest = write_manifest(
        tmp_path / "m.yaml",
        [
            {
                "id": "a",
                "dest": "a/a.ckpt",
                "sources": ["hf://<placeholder>/a.ckpt"],  # becomes manual
                "format": "torch_zip",
                "min_bytes": 100,
            },
            {
                "id": "b",
                "dest": "b/b.ckpt",
                "sources": [(tmp_path / "mirror" / "missing.ckpt").as_uri()],
                "format": "torch_zip",
                "min_bytes": 100,
            },
        ],
    )
    assets = load_manifest(manifest)
    assert assets[0].manual

    run = download_all(
        assets,
        tmp_path / "weights",
        overrides={"a": [mirror.as_uri()], "b": [mirror.as_uri()]},
        retries=1,
    )
    assert len(run.ok) == 2, run.results
    assert run.unresolved == []


def test_audit_and_manual_steps(tmp_path: Path) -> None:
    present = make_npz(tmp_path / "weights" / "present.npz")
    assets = [
        WeightAsset(id="present", dest="present.npz", sources=("file:///x",), format="npz", min_bytes=10),
        WeightAsset(id="absent", dest="absent.ckpt", sources=("file:///y",), min_bytes=10),
        WeightAsset(id="mano", dest="mano/MANO_RIGHT.pkl", sources=(), auth="manual", page="https://mano.is.tue.mpg.de/"),
    ]
    rows = {row["id"]: row for row in audit(assets, tmp_path / "weights")}
    assert rows["present"]["status"] == "ok"
    assert rows["absent"]["status"] == "missing"
    assert rows["mano"]["status"] == "manual"
    assert present.is_file()

    pending = manual_steps(assets, tmp_path / "weights")
    assert [asset.id for asset, _ in pending] == ["absent", "mano"]


# ------------------------------------------------------------------- CLI


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/download_weights.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_dry_run_lists_every_asset() -> None:
    result = run_cli("--dry-run")
    assert result.returncode == 0, result.stderr
    for asset_id in ("wilor_checkpoint", "hawor_infiller", "mano_right"):
        assert asset_id in result.stdout
    assert "manual steps still required" in result.stdout


def test_cli_verify_only_reports_missing_without_downloading() -> None:
    result = run_cli("--verify-only", "--dest-root", "weights")
    assert result.returncode == 1  # nothing is downloaded on this machine
    assert "missing" in result.stdout
    assert "manual" in result.stdout


def test_cli_rejects_unknown_selectors_and_overrides() -> None:
    bad_selector = run_cli("--dry-run", "--only", "nope")
    assert bad_selector.returncode == 2
    assert "unknown selector" in bad_selector.stderr

    bad_override = run_cli("--dry-run", "--url-override", "nope=http://x/y")
    assert bad_override.returncode == 2
    assert "unknown asset" in bad_override.stderr

    malformed = run_cli("--dry-run", "--url-override", "wilor_checkpoint")
    assert malformed.returncode == 2
    assert "ID=URL" in malformed.stderr


def test_cli_downloads_from_a_local_mirror(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    make_torch_zip(mirror_dir / "wilor_final.ckpt")
    manifest = write_manifest(
        tmp_path / "m.yaml",
        [
            {
                "id": "wilor_checkpoint",
                "dest": "wilor/wilor_final.ckpt",
                "sources": [(mirror_dir / "wilor_final.ckpt").as_uri()],
                "format": "torch_zip",
                "min_bytes": 100,
            }
        ],
    )
    dest = tmp_path / "weights"
    result = run_cli("--manifest", str(manifest), "--dest-root", str(dest), "--retries", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "wilor/wilor_final.ckpt").is_file()

    verified = run_cli(
        "--manifest", str(manifest), "--dest-root", str(dest), "--verify-only"
    )
    assert verified.returncode == 0, verified.stdout
    assert "ok" in verified.stdout

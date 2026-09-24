"""The bash weight scripts: plan, download from a mirror, verification."""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = REPO_ROOT / "scripts" / "download_weights.sh"
VERIFY = REPO_ROOT / "scripts" / "verify_weights.sh"
MANO = REPO_ROOT / "scripts" / "install_mano.sh"


def run(script: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    # The real floors are 4.58 GB (VGGT) / 10 MiB (WiLoR, HaWoR); tests use small
    # sparse files and lower the VGGT floor so the scripts stay fast.
    # The real floors are 2.4 GB / 3.1 GB / 400 MB / 4.3 GB (measured on the
    # hubs); tests use small fakes and lower every floor.
    merged = {
        **os.environ,
        "WILOR_MIN_BYTES": "1000000",
        "HAWOR_MIN_BYTES": "1000000",
        "INFILLER_MIN_BYTES": "1000000",
        "VGGT_MIN_BYTES": "1000000",
        **(env or {}),
    }
    return subprocess.run(
        [str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=merged,
    )


def torch_like(path: Path, *, megabytes: float = 1.0) -> Path:
    """A file that looks like a modern ``torch.save`` archive (a zip)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"x" * int(megabytes * 1024 * 1024))
        archive.writestr("archive/version", "3")
    return path


def fill_required_tree(root: Path, *, vggt_mb: float = 101.0) -> None:
    """Create every required file with the right magic and a passing size."""
    torch_like(root / "wilor" / "wilor_final.ckpt", megabytes=11.0)
    (root / "wilor" / "model_config.yaml").write_text("# config\n" + "# pad\n" * 40)
    torch_like(root / "hawor" / "checkpoints" / "hawor.ckpt", megabytes=11.0)
    torch_like(root / "hawor" / "checkpoints" / "infiller.pt", megabytes=2.0)
    torch_like(root / "vggt-omega" / "vggt_omega_1b_416_reproduce.pt", megabytes=vggt_mb)


# ------------------------------------------------------------------ syntax


@pytest.mark.parametrize("script", [DOWNLOAD, VERIFY])
def test_scripts_are_executable_and_parse(script: Path) -> None:
    assert script.is_file()
    assert os.access(script, os.X_OK), f"{script} is not executable"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_mano_installer_is_executable_and_parses() -> None:
    assert MANO.is_file() and os.access(MANO, os.X_OK)
    result = subprocess.run(["bash", "-n", str(MANO)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------ MANO install


def make_fake_mano(directory: Path, *, left: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, size in (("MANO_RIGHT.pkl", 2_000_000), ("MANO_LEFT.pkl", 1_500_000)):
        if name.endswith("LEFT.pkl") and not left:
            continue
        path = directory / name
        with path.open("wb") as handle:
            handle.write(b"\x80\x04\x95")
            handle.seek(size - 1)
            handle.write(b"\0")
    return directory


def test_mano_installer_requires_a_source() -> None:
    result = run(MANO)
    assert result.returncode == 2
    assert "--from" in result.stderr

    missing = run(MANO, "--from", "/tmp/definitely-not-here-mano")
    assert missing.returncode == 2
    assert "does not exist" in missing.stderr


def test_mano_installer_rejects_a_bogus_pickle(tmp_path: Path) -> None:
    source = tmp_path / "mano"
    source.mkdir()
    (source / "MANO_RIGHT.pkl").write_text("not a pickle at all")
    result = run(MANO, "--from", str(source), "--third-party", str(tmp_path / "tp"), "--dest-root", str(tmp_path / "w"))
    assert result.returncode == 1
    assert "not a MANO model" in result.stderr or "not a pickle" in result.stderr


def test_mano_installer_dry_run_lists_every_destination(tmp_path: Path) -> None:
    source = make_fake_mano(tmp_path / "mano")
    tp, dest = tmp_path / "third_party", tmp_path / "weights"
    result = run(
        MANO, "--from", str(source), "--dry-run",
        "--third-party", str(tp), "--dest-root", str(dest),
    )
    assert result.returncode == 0, result.stderr
    for needle in (
        "third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl",
        "third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl",
        "third_party/WiLoR/mano_data/MANO_RIGHT.pkl",
        "weights/mano/MANO_RIGHT.pkl",
    ):
        assert needle in result.stdout
    assert "plan complete" in result.stdout
    assert not tp.exists() and not dest.exists()  # a plan touches nothing


def test_mano_installer_copies_into_all_four_locations(tmp_path: Path) -> None:
    source = make_fake_mano(tmp_path / "mano")
    tp, dest = tmp_path / "third_party", tmp_path / "weights"
    result = run(
        MANO, "--from", str(source),
        "--third-party", str(tp), "--dest-root", str(dest),
    )
    assert result.returncode == 0, result.stderr
    for target in (
        tp / "HaWoR/_DATA/data/mano/MANO_RIGHT.pkl",
        tp / "HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl",
        tp / "WiLoR/mano_data/MANO_RIGHT.pkl",
        dest / "mano/MANO_RIGHT.pkl",
    ):
        assert target.is_file(), target
        assert target.stat().st_size > 1_000_000
    # A second run is a no-op.
    again = run(
        MANO, "--from", str(source),
        "--third-party", str(tp), "--dest-root", str(dest),
    )
    assert "already present" in again.stdout


def test_mano_installer_can_symlink_and_survives_a_missing_left(tmp_path: Path) -> None:
    source = make_fake_mano(tmp_path / "mano", left=False)
    tp, dest = tmp_path / "third_party", tmp_path / "weights"
    result = run(
        MANO, "--from", str(source), "--link",
        "--third-party", str(tp), "--dest-root", str(dest),
    )
    assert result.returncode == 0, result.stderr
    right = tp / "WiLoR/mano_data/MANO_RIGHT.pkl"
    assert right.is_symlink()
    assert right.resolve() == (source / "MANO_RIGHT.pkl").resolve()
    assert not (tp / "HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl").exists()
    assert "fix_shapedirs" in result.stdout


def test_mano_installer_accepts_the_right_pickle_directly(tmp_path: Path) -> None:
    source = make_fake_mano(tmp_path / "mano", left=False)
    tp, dest = tmp_path / "third_party", tmp_path / "weights"
    result = run(
        MANO, "--from", str(source / "MANO_RIGHT.pkl"),
        "--third-party", str(tp), "--dest-root", str(dest),
    )
    assert result.returncode == 0, result.stderr
    assert (dest / "mano/MANO_RIGHT.pkl").is_file()


# -------------------------------------------------------------- downloader


def test_dry_run_prints_the_plan_without_writing_anything(tmp_path: Path) -> None:
    dest = tmp_path / "weights"
    result = run(DOWNLOAD, "--dry-run", "--dest", str(dest))
    assert result.returncode == 0, result.stderr
    for needle in (
        "wilor_final.ckpt",
        "hawor/checkpoints/hawor.ckpt",
        "infiller.pt",
        "vggt_omega_1b_416_reproduce.pt",
        "MANO_RIGHT.pkl",
    ):
        assert needle in result.stdout
    assert "huggingface.co" in result.stdout
    # VGGT-Omega comes from ModelScope, with the API form as a fallback.
    assert "modelscope.cn/models/facebook/VGGT-Omega/resolve/master" in result.stdout
    assert "api/v1/models/facebook/VGGT-Omega/repo" in result.stdout
    assert "Plan complete (nothing was downloaded)" in result.stdout
    assert not dest.exists() or not any(dest.rglob("*"))


def test_only_filter_selects_one_backend(tmp_path: Path) -> None:
    result = run(DOWNLOAD, "--dry-run", "--only", "wilor", "--dest", str(tmp_path / "w"))
    assert result.returncode == 0, result.stderr
    assert "wilor_final.ckpt" in result.stdout
    assert "hawor/checkpoints/hawor.ckpt" not in result.stdout
    assert "vggt" not in result.stdout.lower()


def test_with_repos_prints_the_clone_plan(tmp_path: Path) -> None:
    # Point THIRD_PARTY at an empty directory so the clones are planned, not skipped.
    result = run(
        DOWNLOAD,
        "--dry-run",
        "--with-repos",
        "--only",
        "wilor",
        "--dest",
        str(tmp_path / "w"),
        env={"THIRD_PARTY": str(tmp_path / "third_party")},
    )
    assert result.returncode == 0, result.stderr
    assert "git clone" in result.stdout
    assert "WiLoR" in result.stdout and "HaWoR" in result.stdout
    assert "--recursive" in result.stdout  # upstream asks for it on WiLoR/HaWoR


def test_unknown_argument_is_rejected(tmp_path: Path) -> None:
    result = run(DOWNLOAD, "--nope", "--dest", str(tmp_path / "w"))
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


@pytest.mark.skipif(shutil.which("wget") is None, reason="wget is required")
def test_download_from_a_local_mirror_then_verify(tmp_path: Path) -> None:
    """The real download path, exercised through a file:// mirror."""
    mirror = tmp_path / "mirror"
    torch_like(mirror / "wilor" / "wilor_final.ckpt", megabytes=11.0)
    torch_like(mirror / "vggt" / "vggt_omega_1b_416_reproduce.pt", megabytes=101.0)

    dest = tmp_path / "weights"
    dest.mkdir()
    env = {
        "WILOR_BASE": (tmp_path / "mirror" / "wilor").as_uri(),
        "VGGT_URL": (mirror / "vggt" / "vggt_omega_1b_416_reproduce.pt").as_uri(),
    }
    result = run(DOWNLOAD, "--only", "wilor,vggt", "--dest", str(dest), env=env)

    # The mirror lacks HaWoR and the WiLoR config/detector, so the run reports
    # failures - but the files it did fetch must be in place and verified.
    assert (dest / "wilor" / "wilor_final.ckpt").is_file()
    assert (dest / "vggt-omega" / "vggt_omega_1b_416_reproduce.pt").is_file()
    assert "wilor_final.ckpt" in result.stdout
    assert not list(dest.rglob("*.part")), "no partial file may survive a finished run"


# -------------------------------------------------------------- verifier


def test_verify_reports_missing_files_and_exits_nonzero(tmp_path: Path) -> None:
    dest = tmp_path / "empty"
    dest.mkdir()
    result = run(VERIFY, "--dest", str(dest))
    assert result.returncode == 1
    assert "MISSING" in result.stdout
    assert "wilor/wilor_final.ckpt" in result.stdout
    assert str(dest) in result.stdout
    assert "./scripts/download_weights.sh" in result.stdout


def test_verify_passes_on_a_complete_tree(tmp_path: Path) -> None:
    dest = tmp_path / "weights"
    fill_required_tree(dest)
    result = run(VERIFY, "--dest", str(dest))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all required weights present and readable" in result.stdout
    assert "MISSING" not in result.stdout


def test_verify_catches_a_truncated_file(tmp_path: Path) -> None:
    dest = tmp_path / "weights"
    fill_required_tree(dest)
    torch_like(dest / "hawor" / "checkpoints" / "hawor.ckpt", megabytes=0.2)  # < 10 MiB floor
    result = run(VERIFY, "--dest", str(dest))
    assert result.returncode == 1
    assert "TOO SMALL" in result.stdout


def test_verify_catches_a_wrong_container(tmp_path: Path) -> None:
    """An HTML error page saved as .ckpt must not pass."""
    dest = tmp_path / "weights"
    fill_required_tree(dest)
    # Big enough for the size floor, wrong container magic.
    (dest / "wilor" / "wilor_final.ckpt").write_bytes(b"<html>404 not found</html>\n" * 500_000)
    result = run(VERIFY, "--dest", str(dest))
    assert result.returncode == 1
    assert "BAD FORMAT" in result.stdout
    assert "not a usable checkpoint" in result.stdout


def test_verify_quiet_prints_only_problems(tmp_path: Path) -> None:
    dest = tmp_path / "empty"
    dest.mkdir()
    result = run(VERIFY, "--dest", str(dest), "--quiet")
    assert result.returncode == 1
    assert "WiLoR checkpoint" in result.stdout  # a problem is still shown
    assert "MANO right hand" not in result.stdout  # optional absence is hidden


def test_verify_strict_counts_optional_assets(tmp_path: Path) -> None:
    dest = tmp_path / "weights"
    fill_required_tree(dest)
    assert run(VERIFY, "--dest", str(dest)).returncode == 0
    strict = run(VERIFY, "--dest", str(dest), "--strict")
    assert strict.returncode == 1  # optional files (detector, MANO) are missing
    assert "optional asset(s) absent" in strict.stdout


def test_verify_notes_an_unconverted_mano(tmp_path: Path) -> None:
    dest = tmp_path / "weights"
    fill_required_tree(dest)
    (dest / "mano").mkdir(parents=True)
    (dest / "mano" / "MANO_RIGHT.pkl").write_bytes(b"\x80" + b"p" * 2_000_000)
    result = run(VERIFY, "--dest", str(dest))
    assert "not converted yet" in result.stdout

    (dest / "mano" / "MANO_RIGHT.npz").write_bytes(b"PK\x03\x04" + b"n" * 4096)
    converted = run(VERIFY, "--dest", str(dest))
    assert "paths.mano_model" in converted.stdout


def test_verify_accepts_either_published_vggt_file(tmp_path: Path) -> None:
    """ModelScope ships the 416 reproduction, the 512 and the 256-text checkpoint."""
    for name in ("vggt_omega_1b_416_reproduce.pt", "vggt_omega_1b_512.pt", "vggt_omega_1b_256_text.pt"):
        dest = tmp_path / name.replace(".pt", "")
        fill_required_tree(dest)
        (dest / "vggt-omega" / "vggt_omega_1b_416_reproduce.pt").unlink()
        torch_like(dest / "vggt-omega" / name, megabytes=2.0)
        result = run(VERIFY, "--dest", str(dest))
        assert result.returncode == 0, f"{name}: {result.stdout}"
        assert name in result.stdout
